import tensorflow as tf
from .tf_base_model import TFBaseModel, _optimize, _input_layer
import os
from gglandmarks.datasets import GoogleLandmarkDataset
import datetime


def _depthwise_seperable_conv2d(input, pointwise_conv_filters, name):
    with tf.variable_scope(name):
        depthwise_out = _depthwise_conv2d_layer(input,
            pointwise_conv_filters=pointwise_conv_filters,
            name='depthwise')
        conv2d_out = _conv_layer(depthwise_out,
                                 filters=pointwise_conv_filters,
                                 kernel_size=(
                                     1, 1), padding='SAME',
                                 strides=(1, 1))

        return conv2d_out


def _depthwise_conv2d_layer(input, pointwise_conv_filters, activation=tf.nn.relu, kernel_size=(3, 3), multiplier=1, strides=(1, 1, 1, 1), batch_norm=True, name='depthwise_conv'):
    with tf.variable_scope(name):
        w = tf.get_variable('kernel', shape=(
            kernel_size[0], kernel_size[1], input.shape[-1], multiplier))

        conv_out = tf.nn.depthwise_conv2d(input,
                                          filter=w,
                                          strides=strides,
                                          padding='SAME')

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

        return output


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

        weights = conv.trainable_weights

        # tf.summary.histogram('weights', weights)
        # # tf.summary.histogram('biases', bias)
        # tf.summary.histogram(
        #     'sparsity/conv', tf.nn.zero_fraction(conv_out))
        # tf.summary.histogram('batch_output', batch_out)
        # tf.summary.histogram('output', output)
        # tf.summary.histogram(
        #     'sparsity/output', tf.nn.zero_fraction(output))

        return output

def fc_layer(input, channels_out, activation=None, name='fc'):
    with tf.variable_scope(name):
        # if activation == tf.nn.relu:
        he_initializer = tf.keras.initializers.he_normal()

        layer = tf.layers.Dense(channels_out, use_bias=True, activation=activation,
                                kernel_initializer=he_initializer, bias_initializer=tf.zeros_initializer)
        value = layer(input)

        weights, bias = layer.trainable_weights

        # tf.summary.histogram('weights', weights)
        # tf.summary.histogram('biases', bias)
        # tf.summary.histogram('activation', value)
        # tf.summary.histogram('sparsity', tf.nn.zero_fraction(value))
        return value


def _inference(features, params):
    X = _input_layer(features)

    X = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(X)

    # Stage 1
    X = _conv_layer(input=X,
                    filters=32,
                    strides=(2, 2),
                    kernel_size=(3, 3),
                    padding='valid',
                    activation=tf.nn.relu)
    X = tf.layers.max_pooling2d(inputs=X, pool_size=(3, 3), strides=(2, 2))

    layer_sizes = [64, 128, 256, 512, 1024]
    num_layers = [1, 2, 2, 6, 2]

    block_id = 0
    for i in range(len(num_layers)):
        num_layer = num_layers[i]
        for _ in range(num_layer):
            X = _depthwise_seperable_conv2d(X, layer_sizes[i], name='block-{}'.format(block_id))
            block_id += 1

    X = tf.layers.average_pooling2d(inputs=X,
                                    pool_size=(7, 7),
                                    strides=(2, 2))

    X = tf.layers.flatten(inputs=X)
    Y_hat = fc_layer(input=X,
                     channels_out=params['num_classes'],
                     activation=None)

    return Y_hat

def _loss(labels, logits):
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits), name='loss')

        tf.summary.scalar('loss', loss)

        return loss


def _model_fn(features, labels, mode, params):
    """[summary]

    Arguments:
        features {[type]} -- [description]
        labels {[type]} -- [description]
        mode {[type]} -- [description]
        params {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    logits = _inference(features, params)
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
    train_op = _optimize(loss, params)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        eval_metric_ops=metrics,
        predictions=predictions,
        train_op=train_op
    )


class MyMobileNet(TFBaseModel):
    def __init__(self, model_dir):
        super().__init__(name='my-resnets', model_dir=model_dir, model_fn=_model_fn)

    @staticmethod
    def finetune(data_path, image_original_size, model_dir):
        learning_rates = [0.0001]
        batch_size = 128
        target_size = 64

        for learning_rate in learning_rates:
            dataset = GoogleLandmarkDataset(
                data_path, (image_original_size[0], image_original_size[1]), images_count_min=500)
            print(dataset.train_df.shape)
            print(dataset.num_classes)

            model_params = {
                'num_classes': dataset.num_classes,
                'learning_rate': learning_rate,
                'stage4_identity_blocks': 8
                # 'decay_steps': dataset.train_df.shape[0] / batch_size
            }

            model = MyMobileNet(model_dir=model_dir)
            model.build(dataset=dataset,
                        batch_size=batch_size,
                        target_size=(target_size, target_size),
                        params=model_params)

            logname = 'slr={}-cls={}-l={}'.format(learning_rate, dataset.num_classes,
                                                  model_params['stage4_identity_blocks']).replace('[', '(').replace(']', ')')

            total_losses, total_accuracies = model.fit(train_iter=model.train_iter,
                                                       eval_iter=model.eval_iter,
                                                       logname=logname)

            print('accuracy:{} - losses: {}'.format(total_accuracies, total_losses))
