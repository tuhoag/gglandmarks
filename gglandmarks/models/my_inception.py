import tensorflow as tf

"""Inception V3 for Tensorflow

# Reference
- [Rethinking the Inception Architecture for Computer Vision](
    http://arxiv.org/abs/1512.00567)

"""


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


def _base_inception_layer(input, name):
    with tf.variable_scope(name):
        with tf.variable_scope('filter-1x1'):
            filter1 = _conv_layer(input, (1, 1), 64, name='conv-1x1')

        with tf.variable_scope('filter-3x3'):
            filter2 = _conv_layer(input, (1, 1), 64, name='conv-1x1')
            filter2 = _conv_layer(filter2, (3, 3), 64, name='conv-3x3')

        with tf.variable_scope('filter-5x5'):
            filter3 = _conv_layer(input, (1, 1), 64, name='conv-1x1')
            filter3 = _conv_layer(filter3, (5, 5), 64, name='conv-5x5')

        with tf.variable_scope('max-pooling'):
            filter4 = tf.layers.max_pooling2d(
                input, pool_size=(3, 3), strides=1, padding='same', name='max-pooling')
            filter4 = _conv_layer(filter4, (1, 1), 64, name='conv-1x1')

        output = tf.concat([filter1, filter2, filter3, filter4], axis=3)


def _stem_layer(input, name):
    conv1 = _conv_layer(input, kernel_size=(
        7, 7), filters=64, strides=(2, 2), name='conv-1')
    pool1 = tf.layers.max_pooling2d(
        conv1, pool_size=(3, 3), strides=(2, 2), name='pool-1')

    conv2 = _conv_layer(pool1, kernel_size=(
        3, 3), filters=192, strides=(1, 1), name='conv-2')
    pool2 = tf.layers.max_pooling2d(
        conv2, pool_size=(3, 3), strides=(2, 2), name='pool-2')

    pass


def _input_layer(features, name='input'):
    with tf.variable_scope(name):
        X = features['image']
        tf.summary.image('input', X, 3)

        return X


def _inference(features, params):
    """Do inference from features to logits

    Arguments:
        features {[type]} -- [description]
        params {[type]} -- [description]
    """
    # 1. build input
    X = _input_layer(features)

    # 2.
    conv1 = _conv_layer(input=X,
                        kernel_size=(3, 3),
                        filters=32,
                        padding='valid',
                        strides=(2, 2),
                        batch_norm=True,
                        activation=tf.nn.relu,
                        name='conv1')

    conv2 = _conv_layer(input=conv1,
                        kernel_size=(3, 3),
                        filters=32,
                        padding='valid',
                        strides=(1, 1),
                        batch_norm=True,
                        activation=tf.nn.relu,
                        name='conv2')

    conv3 = _conv_layer(input=conv2,
                        kernel_size=(3, 3),
                        filters=64,
                        padding='same',
                        strides=(1, 1),
                        batch_norm=True,
                        activation=tf.nn.relu,
                        name='conv3')

    pool1 = tf.layers.max_pooling2d(inputs=conv3,
                                    strides=(2, 2),
                                    pool_size=(3, 3),
                                    padding='valid',
                                    name='pool1')

    conv4 = _conv_layer(input=pool1,
                        kernel_size=(1, 1),
                        filters=80,
                        padding='valid',
                        strides=(1, 1),
                        batch_norm=True,
                        activation=tf.nn.relu,
                        name='conv4')

    conv5 = _conv_layer(input=conv4,
                        kernel_size=(3, 3),
                        filters=192,
                        padding='valid',
                        strides=(1, 1),
                        batch_norm=True,
                        activation=tf.nn.relu,
                        name='conv5')

    pool2 = tf.layers.max_pooling2d(inputs=conv5,
                                    strides=(2, 2),
                                    pool_size=(3, 3),
                                    padding='valid',
                                    name='pool2')


def _loss(parameter_list):
    pass


def _optimize(parameter_list):
    pass


def _model_fn(features, labels, mode, params):
    # Y_hat = _inference(features, params)
    pass


class MyInception(object):
    def __init__(self):
        pass

    def import_data(self):
        pass

    def inference(self):
        pass

    def build(self):
        pass

    def train_one_epoch(self):
        pass

    def evaluate_one_epoch(self):
        pass

    def train(self):
        pass
