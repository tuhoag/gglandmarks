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

class MyBasicModel(object):
    def __init__(self, batch_shape, num_classes, logdir):
        self.name = 'basic-model'
        self.batch_shape = batch_shape        
        self.image_shape = (batch_shape[1], batch_shape[2])
        self.num_classes = num_classes
        self.logdir = os.path.join(logdir + self.name)        

    def conv_layer(self, input, channels_out, name='conv'):
        with tf.name_scope(name):
            channels_in = input.get_shape().as_list()[-1]
            print(channels_in)
            w = tf.Variable(tf.truncated_normal([5, 5, channels_in, channels_out]), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name='B')
            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME')
            act = tf.nn.relu(conv + b)

            tf.summary.histogram('weights', w)
            tf.summary.histogram('biases', b)
            tf.summary.histogram('activations', act)

            return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def fc_layer(self, input, channels_out, activation=None, name='fc'):
        with tf.name_scope(name):
            channels_in = input.get_shape().as_list()[-1]
            w = tf.Variable(tf.truncated_normal([channels_in, channels_out]), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name='B')
            act = tf.matmul(input, w) + b

            if activation is not None:
                act = activation(act)

            tf.summary.histogram('weights', w)
            tf.summary.histogram('biases', b)
            tf.summary.histogram('activations', act)

            return act

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

        dense1 = self.fc_layer(flatten, 1024, activation= tf.nn.relu, name='fc1')
        Y_hat = self.fc_layer(dense1, self.num_classes, activation=tf.nn.sigmoid, name='fc2')
        
        print('label shape: {}'.format(Y.shape))
        print('output shape: {}'.format(Y_hat.shape))
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=Y, logits=Y_hat)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss
        )

        correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # accuracy = tf.metrics.accuracy(labels=tf.argmax(Y, 1), predictions=tf.argmax(Y_hat, 1))
        
        merged_summary = tf.summary.merge_all()
        dataset_generator = dataset.get_train_validation_generator(
            batch_size=32,
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
                for batch in train_gen:
                    print('{} - X shape {} - Y shape {}'.format(step, batch[0].shape, batch[1].shape))
                    step += 1
                    s, train_loss, train_accuracy = sess.run([merged_summary, loss, accuracy], feed_dict={X: batch[0], Y: batch[1]})
                    writer.add_summary(s, step)

                    print('{}, train loss: {}, train accuracy: {}'.format(i, train_loss, train_accuracy))

                    sess.run(train_op, feed_dict={X: batch[0], Y: batch[1]})
                    total_loss += train_loss