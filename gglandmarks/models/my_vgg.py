import tensorflow as tf
import os
import datetime
from tensorflow.python import debug as tf_debug
from gglandmarks.datasets import GoogleLandmarkDataset
import pandas as pd
import traceback
from .tf_base_model import TFBaseModel, _optimize, _loss


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

def _inference(features, params):
    X = features['image']

    tf.summary.image('input', X, 3)

    conv_out = X

    conv_nums = params['conv_nums']
    conv_channels_outs = params['conv_channels_outs']

    for i in range(len(conv_nums)):
        conv_out = _conv_block(
            input=conv_out, conv_nums=conv_nums[i], channels_out=conv_channels_outs[i], name='conv-block-' + str(i))

    flatten = tf.layers.flatten(conv_out)

    dense_count = params['dense_count']
    dense_weights_count = params['dense_weights_count']

    dense_out = flatten

    for i in range(dense_count):
        dense_out = fc_layer(dense_out, dense_weights_count,
                             activation=tf.nn.relu, name='fc'+str(i))    

    Y_hat = fc_layer(dense_out, params['num_classes'],
                     activation=None, name='logits')

    return Y_hat

def _my_vgg_model_fn(features, labels, mode, params):
    """[summary]

    Arguments:
        X {tensor[batch_size, width, height, channels]} -- a batch of images
        Y {tensor[batch_size,]} -- a batch of landmarks
        mode {ModeKeys} -- Estimator Mode
        params {dictionary} -- Params to customize the model
            - 'num_classes': the number of output classes
            - 'learning_rate': learning rate
            - 'conv_nums': the number of conv in each layers (vgg16 or vgg19)
            - 'conv_channels_out': the output channels of each layers
            - 'dense_count': the number of dense layer
            - 'dense_weights_count': the number of weights in each dense layer

    Returns:
        [EstimatorSpec] -- Estimator Spec
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

    # evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    train_op = _optimize(loss, params)

    # train
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        eval_metric_ops=metrics,
        predictions=predictions,
        train_op=train_op
    )


class MyVGG(TFBaseModel):
    def __init__(self, model_dir):
        super().__init__(name='my-vgg', model_dir=model_dir, model_fn=_my_vgg_model_fn)

    @staticmethod
    def finetune(data_path, image_original_size, model_dir):
        image_sizes = [32, 48, 84]
        learning_rates = [0.0001, 0.001, 0.01]
        conv_nums = [
            [2, 2, 3, 3, 3],
            [2, 2, 4, 4, 4]
        ]

        conv_channels_outs = [64, 128, 256, 512, 512]

        dense_weights_counts = [
            1024, 2048, 4096
        ]

        dense_counts = [1, 2]

        images_count_mins = [
            100, 500, 1000, 5000, 10000, 500000
        ]

        i = 0
        batch_size = 32
        stats_file = open('./output/fine_tune-' +
                          str(datetime.datetime.now()) + '.csv', 'w')
        stats_file.write(
            'learning_rate,images_count_min,num_classes,image_size,conv_num,dense_weights_count,dense_count,loss,accuracy\n')
        # for i in range(5):
        #     df.loc[i] = [np.random.randint(-1,1) for n in range(3)]
        try:

            for conv_num in conv_nums:
                # conv_num=conv_nums[1]
                image_size = 128
                images_count_min = 500
                learning_rate = 0.0001
                dense_weights_count = dense_weights_counts[1]
                dense_count = dense_counts[1]

                # tf.reset_default_graph()

                dataset = GoogleLandmarkDataset(
                    data_path, (image_original_size[0], image_original_size[1]), images_count_min=images_count_min)
                print(dataset.train_df.shape)
                print(dataset.num_classes)

                logname = 'lr={}-icm={}-c={}-s={}-cn={}-dwc={}-dc={}'.format(
                    learning_rate, images_count_min, dataset.num_classes, image_size, conv_num, dense_weights_count, dense_count).replace('[', '(').replace(']', ')')

                print(logname)
                model_params = {
                    'num_classes': dataset.num_classes,
                    'learning_rate': learning_rate,
                    'conv_nums': conv_num,
                    'conv_channels_outs': conv_channels_outs,
                    'dense_count': dense_count,
                    'dense_weights_count': dense_weights_count
                }
                model = MyVGG(model_dir=model_dir)
                model.build(
                    dataset=dataset,
                    batch_size=batch_size,
                    target_size=(image_size, image_size),
                    params=model_params)

                total_losses, total_accuracies = model.fit(train_iter=model.train_iter,
                                                           eval_iter=model.eval_iter,
                                                           logname=logname)
                stats_file.write('{},{},{},{},"{}",{},{},{},{}\n'.format(learning_rate, images_count_min, dataset.num_classes,
                                                                         image_size, conv_num, dense_weights_count, dense_count, total_losses, total_accuracies))
                i = i + 1
        except Exception as e:
            # print(e)
            traceback.print_exc(e)
            stats_file.close()
