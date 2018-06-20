import tensorflow as tf
from .tf_base_model import TFBaseModel
import os

def _input_layer(features, name='input'):
    with tf.variable_scope(name):
        X = features['image']
        tf.summary.image('input', X, 3)

        return X


def _conv_layer(input, kernel_size, filters, padding='same', strides=(1, 1), activation=tf.nn.relu, batch_norm=True, name='conv'):
    with tf.variable_scope(name):
        conv_input = input
        conv = tf.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            activation=None,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.he_normal()
        )
        conv_out = conv(conv_input)

        if batch_norm:
            batch_out = tf.layers.batch_normalization(
                conv_out,
                axis=3,
                scale=False,
                name='batch_normalization')
        else:
            batch_out = conv_out

        if activation is not None:
            output = activation(batch_out)
        else:
            output = batch_out

        weights, bias = conv.trainable_weights

        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', bias)
        tf.summary.histogram(
            'sparsity/conv', tf.nn.zero_fraction(conv_out))
        tf.summary.histogram('batch_output', batch_out)
        tf.summary.histogram('output', output)

        return output


def _identity_block_layer(input, f, filters, name):
    """[summary]

    Arguments:
        input [tensor] -- input tensor
        f [integer] -- channel out size of the middle conv layer
        filters [python list of integers] -- the number of filters of each conv layer
        name [string] -- name of layer

    Returns:
        [tensor] -- output layer
    """

    with tf.variable_scope(name):
        X = input

        X = _conv_layer(input=X,
                        kernel_size=(1, 1),
                        filters=filters[0],
                        strides=(1, 1),
                        padding='valid',
                        activation=tf.nn.relu,
                        name='res2a')

        X = _conv_layer(input=X,
                        kernel_size=(f, f),
                        filters=filters[1],
                        strides=(1, 1),
                        padding='same',
                        activation=tf.nn.relu,
                        name='res2b')

        X = _conv_layer(input=X,
                        kernel_size=(1, 1),
                        filters=filters[2],
                        strides=(1, 1),
                        padding='valid',
                        activation=None,
                        name='res2c')

        X = tf.add(input, X)
        X = tf.nn.relu(X)

        return X


def _convolution_block_layer(input, f, filters, s, name):
    """[summary]

    Arguments:
        input [tensor] -- input tensor
        f [integer] -- channel out size of the middle conv layer
        filters [python list of integers] -- the number of filters of each conv layer
        s [integer] -- stride of first conv layer to use
        name [string] -- name of layer

    Returns:
        [tensor] -- output layer
    """

    with tf.variable_scope(name):
        X = input

        X = _conv_layer(input=X,
                        kernel_size=(1, 1),
                        filters=filters[0],
                        strides=(s, s),
                        padding='valid',
                        activation=tf.nn.relu,
                        name='res2a')

        X = _conv_layer(input=X,
                        kernel_size=(f, f),
                        filters=filters[1],
                        strides=(1, 1),
                        padding='same',
                        activation=tf.nn.relu,
                        name='res2b')

        X = _conv_layer(input=X,
                        kernel_size=(1, 1),
                        filters=filters[2],
                        strides=(1, 1),
                        padding='valid',
                        activation=None,
                        name='res2c')

        shortcut = _conv_layer(input=input,
                               kernel_size=(1, 1),
                               filters=filters[3],
                               strides=(s, s),
                               padding='valid',
                               activation=None,
                               name='shortcut')

        X = tf.add(input, shortcut)
        X = tf.nn.relu(X)

        return X


def _inference(features, classes):
    X = _input_layer(features)

    X = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(X)

    # Stage 1
    X = _conv_layer(input=X,
                    filters=64,
                    strides=(2, 2),
                    kernel_size=(7, 7),
                    padding='valid',
                    activation=tf.nn.relu)
    X = tf.layers.max_pooling2d(inputs=X, pool_size=(3, 3), strides=(2, 2))

    # Stage 2
    filters = [
        [64, 64, 256],
        [128, 128, 512],
        [256, 256, 1024],
        [512, 512, 2048],
    ]

    X = _convolution_block_layer(input=X,
                                 f=3,
                                 filters=filters[0],
                                 s=1,
                                 name='stage-2-0')
    for i in range(2):
        X = _identity_block_layer(input=X,
                                  f=3,
                                  filters=filters[0],
                                  name='stage-2-' + (str(i + 1)))

    # Stage 3
    X = _convolution_block_layer(input=X,
                                 f=3,
                                 filters=filters[1],
                                 s=1,
                                 name='stage-3-0')

    for i in range(3):
        X = _identity_block_layer(input=X,
                                  f=3,
                                  filters=filters[1],
                                  name='stage-3-' + (str(i + 1)))

    # Stage 4
    X = _convolution_block_layer(input=X,
                                 f=3,
                                 filters=filters[2],
                                 s=1,
                                 name='stage-4-0')

    for i in range(5):
        X = _identity_block_layer(input=X,
                                  f=3,
                                  filters=filters[2],
                                  name='stage-4-' + (str(i + 1)))

    # Stage 5
    X = _convolution_block_layer(input=X,
                                 f=3,
                                 filters=filters[3],
                                 s=1,
                                 name='stage-5-0')

    for i in range(2):
        X = _identity_block_layer(input=X,
                                  f=3,
                                  filters=filters[3],
                                  name='stage-5-' + (str(i + 1)))

    X = tf.layers.average_pooling2d(inputs=X,
                                    pool_size=(2, 2),
                                    strides=(2, 2))

    X = tf.layers.flatten(inputs=X)
    X = tf.layers.dense(inputs=X,
                        units=classes,
                        activation=None)

    return X


def _loss(labels, logits):
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits), name='loss')

        tf.summary.scalar('loss', loss)

        return loss


def _optimize(learning_rate):
    with tf.variable_scope('train'):
        train_op = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate).minimize(loss, global_step=tf.train.get_global_step())

        return train_op


def _model_fn(features, labels, mode, params):
    logits = _inference(features, params['classes'])
    loss = _loss(labels, logits)

    predicted_classes = tf.argmax(logits, 1)

    # prediction
    predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits),
        'logits': logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, eval_metric_ops=metrics)

    # train
    train_op = _optimize(learning_rate=params['learning_rate'])

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        eval_metric_ops=metrics,
        predictions=predictions,
        train_op=train_op
    )


class MyResNets(TFBaseModel):
    def __init__(self, image_shape, model_dir, log_dir, num_classes):
        self.name = 'my-resnets'
        self.model_dir = os.path.join(model_dir, self.name)
        self.log_dir = os.path.join(log_dir, self.name)

        super().__init__(image_shape, num_classes)


