import tensorflow as tf
import os
import datetime
from tensorflow.python import debug as tf_debug


def _conv_block(input, conv_nums, channels_out, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv-block'):
    """[summary]

    Arguments:
        input {[type]} -- input
        conv_nums {int} -- the number of convolution layers
        filters {[type]} -- kernel size
        channels_out {[type]} -- number of output channels

    Keyword Arguments:
        padding {str} -- padding strategy (default: {'same'})
        activation {[type]} -- activation function (default: {tf.nn.relu})
        name {str} -- block name (default: {'conv-block'})

    Returns:
        [type] -- output
    """

    with tf.variable_scope(name):
        initializer = tf.keras.initializers.he_normal()

        conv_input = input
        for i in range(conv_nums):
            conv = tf.layers.Conv2D(
                filters=channels_out,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
                use_bias=True,
                bias_initializer=tf.zeros_initializer,
                kernel_initializer=initializer
            )
            conv_input = conv(conv_input)

            weights, bias = conv.trainable_weights

            tf.summary.histogram('weights', weights)
            tf.summary.histogram('biases', bias)
            tf.summary.histogram('activations', conv_input)
            tf.summary.histogram(
                'sparsity/conv', tf.nn.zero_fraction(conv_input))

        pool = tf.layers.max_pooling2d(
            conv_input, pool_size=(2, 2), strides=2, name='max-pooling')

        return pool


def fc_layer(input, channels_out, activation=None, name='fc'):
    with tf.variable_scope(name):
        # if activation == tf.nn.relu:
        he_initializer = tf.keras.initializers.he_normal()

        layer = tf.layers.Dense(channels_out, use_bias=True, activation=activation,
                                kernel_initializer=he_initializer, bias_initializer=tf.zeros_initializer)
        value = layer(input)

        weights, bias = layer.trainable_weights

        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', bias)
        tf.summary.histogram('activation', value)
        tf.summary.histogram('sparsity', tf.nn.zero_fraction(value))
        return value


def _my_vgg_model_fn(features, labels, mode, params):
    """[summary]

    Arguments:
        X {tensor[batch_size, width, height, channels]} -- a batch of images
        Y {tensor[batch_size,]} -- a batch of landmarks
        mode {ModeKeys} -- Estimator Mode
        params {dictionary} -- Params to customize the model
            - 'num_classes': the number of output classes
            - 'learning_rate': learning rate

    Returns:
        [EstimatorSpec] -- Estimator Spec
    """
    X = features['image']

    tf.summary.image('input', X, 3)

    conv_out = X

    conv_nums = [2, 2, 3, 3, 3]
    conv_channels_outs = [64, 128, 256, 512, 512]

    for i in range(len(conv_nums)):
        conv_out = _conv_block(input = conv_out, conv_nums=conv_nums[i], channels_out=conv_channels_outs[i], name='conv-block-' + str(i))

    flatten = tf.layers.flatten(conv_out)

    dense1 = fc_layer(flatten, 2048, activation=tf.nn.sigmoid, name='fc1')
    Y_hat = fc_layer(dense1, params['num_classes'],
                     activation=None, name='fc2')

    print('label shape: {}'.format(labels.shape))
    print('output shape: {}'.format(Y_hat.shape))
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=Y_hat), name='loss')

    tf.summary.scalar('loss', loss)

    predicted_classes = tf.argmax(Y_hat, 1)

    # prediction
    predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(Y_hat),
            'logits': Y_hat
        }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    accuracy = tf.metrics.accuracy(labels=labels,
                               predictions=predicted_classes,
                               name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    # evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    train_op = tf.train.GradientDescentOptimizer(
        learning_rate=params['learning_rate']).minimize(loss, global_step=tf.train.get_global_step())

    # train
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return tf.estimator.EstimatorSpec(
        mode,
        loss = loss,
        eval_metric_ops=metrics,
        predictions=predictions,
        train_op=train_op
    )

class MyVGG(object):
    def __init__(self, image_shape, model_dir, log_dir, num_classes):
        self.name = 'my-vgg'
        # self.batch_shape = batch_shape
        self.image_shape = image_shape
        self.model_dir = os.path.join(model_dir, self.name)
        self.log_dir = os.path.join(log_dir, self.name)
        self.num_classes = num_classes

    def import_data(self, dataset, batch_size, target_size, validation_split=0.1, output_type='dataset'):
        dataset_generator = dataset.get_train_validation_generator(
            batch_size=batch_size,
            target_size=target_size,
            validation_size=validation_split,
            output_type=output_type)

        train_dataset, val_dataset = next(dataset_generator)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)

        features, labels = iterator.get_next()
        # self.X = features['image']

        self.train_iter = iterator.make_initializer(train_dataset)
        self.eval_iter = iterator.make_initializer(val_dataset)

        return features, labels

    def build(self, dataset, batch_size, target_size, num_classes, learning_rate):
        features, labels = self.import_data(dataset, batch_size, target_size)

        self.global_step = tf.get_variable(
            name='global_step', trainable=False, initializer=tf.constant(0))

        self.spec = _my_vgg_model_fn(features, labels, mode=None, params={
            'num_classes': num_classes,
            'learning_rate': learning_rate
        })


    def train_one_epoch(self, input_fn, writer, current_step, session, steps=None):
        # train
        train_init = input_fn()
        loss = self.spec.loss
        train_op = self.spec.train_op
        merged_summary = tf.summary.merge_all()

        session.run(train_init)

        step = 0
        total_loss = 0
        try:
            while True:
                step += 1
                train_loss, _, s, current_step = session.run(
                    [loss, train_op, merged_summary, self.global_step])
                writer.add_summary(s, current_step)
                print('current step: {}'.format(current_step))
                print('{} - train loss: {}'.format(step, train_loss))
                total_loss += train_loss

                if(steps is not None and step >= steps):
                    break

        except tf.errors.OutOfRangeError as err:
            print('end epoch:')

        return total_loss, current_step

    def predict(self, input_fn):
        pass

    def evaluate(self, input_fn, writer, current_step, session, steps=None):
        merged_summary = tf.summary.merge_all()
        eval_init = input_fn()
        metrics_ops = self.spec.eval_metric_ops

        session.run(eval_init)

        step = 0
        total_accuracy = 0
        try:
            while True:
                step += 1
                s, metrics = session.run([merged_summary, metrics_ops])
                writer.add_summary(s, current_step)
                print('current step: {}'.format(current_step))
                print('metrics: {}'.format(metrics))
                total_accuracy += metrics['accuracy'][1]

                if(steps is not None and step >= steps):
                    break

        except tf.errors.OutOfRangeError as err:
            print('end epoch')

        return total_accuracy

    def save(self):
        pass

    def load(self):
        pass

    def train(self, dataset, epochs=10):
        writer_path = os.path.join(self.log_dir, str(datetime.datetime.now()))
        train_writer = tf.summary.FileWriter(writer_path + '-train')
        eval_writer = tf.summary.FileWriter(writer_path + '-eval')

        current_step = 0
        total_losses = []
        total_accuracies = []
        max_steps = 10
        self.build(dataset, 50, self.image_shape, self.num_classes, 0.001)

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            train_writer.add_graph(sess.graph)
            eval_writer.add_graph(sess.graph)

            for i in range(epochs):
                print("Traing epoch:{}".format(i))
                total_loss, current_step = self.train_one_epoch(
                    lambda: self.train_iter, current_step=current_step, writer=train_writer, steps=max_steps, session=sess)

                total_losses.append(total_loss)

                if i % 1 == 0:
                    print("Evaluating epoch: {}".format(i))
                    total_accuracy = self.evaluate(
                        lambda: self.eval_iter, current_step=current_step, writer=eval_writer, session=sess, steps=max_steps)

                total_accuracies.append(total_accuracy)

            return total_losses, total_accuracies

    def fit(self, dataset):
        learning_rates = [0.0001, 0.001, 0.01]
        conv_nums = []
        conv_channels_out = []

        pass