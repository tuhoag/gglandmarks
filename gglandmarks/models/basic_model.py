import tensorflow as tf
import keras
from keras.applications import VGG16, DenseNet121
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import Sequential
from keras.layers import Dense, Input, Flatten, Activation, Dropout
from keras.models import Model, load_model
import glob
import os
import numpy as np
from tqdm import tqdm
from .abstract_model import AbstractModel
import timeit
import datetime


def conv_layer(input, channels_out, name='conv'):
    with tf.variable_scope(name):
        initializer = tf.keras.initializers.he_normal()

        conv = tf.layers.Conv2D(
            filters=channels_out,
            kernel_size=(5, 5),
            padding='same',
            activation=tf.nn.relu,
            use_bias=True,
            bias_initializer=tf.zeros_initializer,
            kernel_initializer=initializer
        )
        conv_value = conv(input)
        pool = tf.layers.max_pooling2d(conv_value, pool_size=(2, 2), strides=2)

        weights, bias = conv.trainable_weights

        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', bias)
        tf.summary.histogram('activations', conv_value)
        tf.summary.histogram('sparsity/conv', tf.nn.zero_fraction(conv_value))
        tf.summary.histogram('sparsity/pool', tf.nn.zero_fraction(pool))

        return pool


def fc_layer(input, channels_out, activation=None, name='fc'):
    with tf.variable_scope(name):
        # if activation == tf.nn.relu:
        initializer = tf.keras.initializers.he_normal()

        layer = tf.layers.Dense(channels_out, use_bias=True, activation=activation,
                                kernel_initializer=initializer, bias_initializer=tf.zeros_initializer)
        value = layer(input)

        weights, bias = layer.trainable_weights

        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', bias)
        tf.summary.histogram('activation', value)
        tf.summary.histogram('sparsity', tf.nn.zero_fraction(value))
        return value

def model_fn(features, labels, mode, params):
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

    conv1 = conv_layer(X, 32, name='conv1')
    conv2 = conv_layer(conv1, 64, name='conv2')

    flatten = tf.layers.flatten(conv2)

    dense1 = fc_layer(flatten, 1024, activation=tf.nn.relu, name='fc1')
    Y_hat = fc_layer(dense1, params['num_classes'], name='fc2')

    print('label shape: {}'.format(labels.shape))
    print('output shape: {}'.format(Y_hat.shape))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=Y_hat))

    tf.summary.scalar('loss', loss)

    predicted_classes = tf.argmax(Y_hat, 1)

    # prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.sigmoid(Y_hat),
            'logits': Y_hat
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # evaluation
    correct_prediction = tf.equal(predicted_classes, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    metrics = {
        'accuracy': accuracy
    }
    tf.summary.scalar('accuracy', accuracy)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    train_op = tf.train.GradientDescentOptimizer(
        learning_rate=params['learning_rate']).minimize(loss, global_step=tf.train.get_global_step())

    # train
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


class MyBasicModel(object):
    def __init__(self, batch_shape, num_classes, logdir):
        self.name = 'basic-model'
        self.batch_shape = batch_shape
        self.image_shape = (batch_shape[1], batch_shape[2])
        self.num_classes = num_classes
        self.logdir = os.path.join(logdir + self.name)

    def train(self, input_fn, steps=None):
        global_step = tf.get_variable(
            name='global_step', trainable=False, initializer=tf.constant(0))

        # train
        train_iter = input_fn()
        train_X, train_Y = train_iter.get_next()
        spec = model_fn(
            train_X, train_Y,
            mode=tf.estimator.ModeKeys.TRAIN,
            params={'num_classes': self.num_classes, 'learning_rate': 0.001}
        )
        loss = spec.loss
        train_op = spec.train_op
        merged_summary = tf.summary.merge_all()

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(os.path.join(
                self.logdir, str(datetime.datetime.now())))
            sess.run(tf.global_variables_initializer())
            writer.add_graph(sess.graph)

            for i in range(10):
                sess.run(train_iter.initializer)
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
                        print('X shape: {} - Y shape: {}'.format(temp_X['image'].shape, temp_Y.shape))
                        total_loss += train_loss

                        if(steps is not None and step >= steps):
                            break

                except tf.errors.OutOfRangeError as err:
                    print('end epoch: {}'.format(i))

                print('{} - total loss: {}'.format(i, total_loss))
                # writer.add_summary(total_loss, i * step)

    def fit4(self, dataset):
        dataset_generator = dataset.get_train_validation_generator(
            batch_size=1000,
            target_size=self.image_shape,
            validation_size=0.1,
            output_type='dataset')

        train_dataset, val_dataset = next(dataset_generator)
        train_iter = train_dataset.make_initializable_iterator()
        eval_iter = val_dataset.make_initializable_iterator()

        classifier = tf.estimator.Estimator(
            model_fn=model_fn,
            params={
                'learning_rate': 0.1,
                'num_classes': self.num_classes
            }
        )

        classifier.train(
            input_fn=lambda: train_dataset,
            steps=10
        )

    def fit3(self, dataset):
        dataset_generator = dataset.get_train_validation_generator(
            batch_size=1000,
            target_size=self.image_shape,
            validation_size=0.1,
            output_type='dataset')

        train_dataset, val_dataset = next(dataset_generator)
        train_iter = train_dataset.make_initializable_iterator()
        eval_iter = val_dataset.make_initializable_iterator()

        self.train(lambda: train_iter, steps=10)

    def fit2(self, dataset):
        dataset_generator = dataset.get_train_validation_generator(
            batch_size=1000,
            target_size=self.image_shape,
            validation_size=0.1,
            output_type='dataset')

        train_dataset, val_dataset = next(dataset_generator)
        train_iter = train_dataset.make_initializable_iterator()
        X, Y = train_iter.get_next()

        conv1 = conv_layer(X, 32, name='conv1')
        conv2 = conv_layer(conv1, 64, name='conv2')

        flatten = tf.layers.flatten(conv2)

        dense1 = fc_layer(
            flatten, 1024, activation=tf.nn.relu, name='fc1')
        Y_hat = fc_layer(dense1, self.num_classes,
                         activation=tf.nn.sigmoid, name='fc2')

        print('label shape: {}'.format(Y.shape))
        print('output shape: {}'.format(Y_hat.shape))
        loss = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=Y_hat)
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=0.001).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(Y_hat, 1), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # accuracy = tf.metrics.accuracy(labels=tf.argmax(Y, 1), predictions=tf.argmax(Y_hat, 1))

        merged_summary = tf.summary.merge_all()

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(self.logdir)
            writer.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())
            for i in range(10):
                sess.run(train_iter.initializer)
                step = 0
                total_loss = 0
                num_batches = 0
                try:
                    while True:
                        start = timeit.default_timer()
                        step += 1
                        num_batches += 1
                        if step % 10 == 0:
                            s = sess.run(merged_summary)
                            writer.add_summary(s)

                        _, train_loss, train_accuracy, train_X, train_Y = sess.run(
                            [optimizer, loss, accuracy, X, Y])
                        end = timeit.default_timer()
                        print('{} - X shape {} - Y shape {}'.format(step,
                                                                    train_X.shape, train_Y.shape))
                        print('{}, train loss: {}, train accuracy: {} - time: {:.2f}'.format(
                            i, train_loss, train_accuracy, end-start))

                except tf.errors.OutOfRangeError as err:
                    print('end epoch: {}'.format(i))

                total_loss += train_loss
                total_loss = total_loss / num_batches
                print('total loss: {}'.format(total_loss))

    def fit(self, dataset):
        X = tf.placeholder(dtype=tf.float32,
                           shape=self.batch_shape,
                           name='X')

        Y = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self.num_classes),
            name='Y'
        )

        conv1 = self.conv_layer(X, 32, name='conv1')
        conv2 = self.conv_layer(conv1, 64, name='conv2')

        flatten = tf.layers.flatten(conv2)

        dense1 = self.fc_layer(
            flatten, 1024, activation=tf.nn.relu, name='fc1')
        Y_hat = self.fc_layer(dense1, self.num_classes,
                              activation=tf.nn.sigmoid, name='fc2')

        print('label shape: {}'.format(Y.shape))
        print('output shape: {}'.format(Y_hat.shape))
        loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=Y_hat)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss
        )

        correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # accuracy = tf.metrics.accuracy(labels=tf.argmax(Y, 1), predictions=tf.argmax(Y_hat, 1))

        merged_summary = tf.summary.merge_all()
        dataset_generator = dataset.get_train_validation_generator(
            batch_size=1000,
            target_size=self.image_shape,
            validation_size=0.1)

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(self.logdir)
            writer.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())

            for i in range(10):
                train_gen, val_gen = next(dataset_generator)
                step = 0
                total_loss = 0
                train_iter = iter(train_gen)
                while(True):
                    start_total = start = timeit.default_timer()
                    batch = next(train_iter)
                    end = timeit.default_timer()
                    print('{} - X shape {} - Y shape {}, time={}'.format(step,
                                                                         batch[0].shape, batch[1].shape, end-start))
                    step += 1
                    start = timeit.default_timer()
                    s, train_loss, train_accuracy = sess.run(
                        [merged_summary, loss, accuracy], feed_dict={X: batch[0], Y: batch[1]})
                    writer.add_summary(s, step)
                    sess.run(train_op, feed_dict={X: batch[0], Y: batch[1]})
                    total_loss += train_loss
                    end = timeit.default_timer()

                    print('{}, train loss: {}, train accuracy: {} - time: {:.2f} - total time: {:.2f}'.format(
                        i, train_loss, train_accuracy, end-start, end-start_total))
