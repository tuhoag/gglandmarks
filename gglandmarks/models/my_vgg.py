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
            tf.summary.histogram('sparsity/conv', tf.nn.zero_fraction(conv_input))

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

    conv_out = _conv_block(input = X, conv_nums=2, kernel_size=(3, 3), channels_out=64)

    flatten = tf.layers.flatten(conv_out)

    dense1 = fc_layer(flatten, 4096, activation=tf.nn.relu, name='fc1')
    Y_hat = fc_layer(dense1, params['num_classes'],
                     activation=None, name='fc2')

    print('label shape: {}'.format(labels.shape))
    print('output shape: {}'.format(Y_hat.shape))
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=Y_hat), name='loss')

    tf.summary.scalar('loss', loss)

    predicted_classes = tf.argmax(Y_hat, 1)

    correct_prediction = tf.equal(predicted_classes, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    metrics = {
        'accuracy': accuracy
    }
    tf.summary.scalar('accuracy', accuracy)

    train_op = tf.train.GradientDescentOptimizer(
        learning_rate=params['learning_rate']).minimize(loss, global_step=tf.train.get_global_step())

    # prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': Y_hat,
            'logits': Y_hat
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # train
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

class MyVGG(object):
    def __init__(self, batch_shape, model_dir, log_dir, num_classes):
        self.name = 'my-vgg'
        self.batch_shape = batch_shape
        self.image_shape = (batch_shape[1], batch_shape[2])
        self.model_dir = os.path.join(model_dir, self.name)
        self.log_dir = os.path.join(log_dir, self.name)
        self.num_classes = num_classes

    def train(self, input_fn, steps=None):
        global_step = tf.get_variable(
            name='global_step', trainable=False, initializer=tf.constant(0))

        # train
        train_iter = input_fn()
        train_X, train_Y = train_iter.get_next()
        spec = _my_vgg_model_fn(
            train_X, train_Y,
            mode=tf.estimator.ModeKeys.TRAIN,
            params={'num_classes': self.num_classes, 'learning_rate': 0.001}
        )
        loss = spec.loss
        train_op = spec.train_op
        merged_summary = tf.summary.merge_all()

        with tf.Session() as sess:
            sess = tf_debug.TensorBoardDebugWrapperSession(
                sess, "localhost:7000")
            writer = tf.summary.FileWriter(os.path.join(
                self.log_dir, str(datetime.datetime.now())))
            sess.run(tf.global_variables_initializer())
            writer.add_graph(sess.graph)

            for i in range(10):
                sess.run(train_iter.initializer)
                sess.run(global_step.initializer)
                step = 0
                total_loss = 0
                num_batches = 0
                try:
                    while True:
                        step += 1
                        temp_X, temp_Y, train_loss, _, s, current_step = sess.run(
                            [train_X, train_Y, loss, train_op, merged_summary, global_step])
                        writer.add_summary(s, current_step)
                        print('current step: {}'.format(current_step))
                        print('{} - train loss: {}'.format(step, train_loss))
                        print(
                            'X shape: {} - Y shape: {}'.format(temp_X['image'].shape, temp_Y.shape))
                        total_loss += train_loss

                        if(steps is not None and step >= steps):
                            break

                except tf.errors.OutOfRangeError as err:
                    print('end epoch: {}'.format(i))

                print('{} - total loss: {}'.format(i, total_loss))

    def predict(self):
        pass

    def evaluate(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def fit(self, dataset):
        dataset_generator = dataset.get_train_validation_generator(
            batch_size=100,
            target_size=self.image_shape,
            validation_size=0.1,
            output_type='dataset')

        train_dataset, val_dataset = next(dataset_generator)
        train_iter = train_dataset.make_initializable_iterator()
        eval_iter = val_dataset.make_initializable_iterator()

        self.train(lambda: train_iter, steps=2)
