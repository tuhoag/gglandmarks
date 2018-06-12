import tensorflow as tf
import os
import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Reshape, Flatten, Activation
from keras.initializers import TruncatedNormal, Constant
from keras.models import Model
import keras.backend as K

LOGDIR = './data/mnist_tutorial_keras/'
SPRITES = os.path.join(os.getcwd(), 'sprite_1024.png')
LABELS = os.path.join(os.getcwd(), 'labels_1024.tsv')

mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + "data", one_hot=True)

def conv_layer(input, channels_in, channels_out, name='conv'):
    with tf.name_scope(name):
        conv = Conv2D(channels_out, 
            kernel_size=(5, 5),
            strides=(1, 1), 
            padding='same',
            use_bias=True,
            kernel_initializer=TruncatedNormal(),
            bias_initializer=Constant(0.1),
            activation='relu')(input)
        pool = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv)
        
        return pool        

def fc_layer(input, channels_in, channels_out, name='fc'):
    with tf.name_scope(name):
        act = Dense(channels_out,
            use_bias=True,
            kernel_initializer=TruncatedNormal(),
            bias_initializer=Constant(0.1))(input)
        
        return act

def mnist_model(learning_rate, two_conv_layer, two_fc_layer, writer):    
    x = Input(shape=(784,))
    x_image = Reshape((28, 28, 1))(x)

    # tf.summary.image('input', x_image, 3)

    if two_conv_layer:
        conv1 = conv_layer(x_image, 1, 32, 'conv1')
        conv_out = conv_layer(conv1, 32, 64, 'conv2')
    else:
        conv_out = conv_layer(x_image, 1, 16, 'conv1')

    flattened = Flatten()(conv_out)    

    if two_fc_layer:
        fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, name='fc1')
        relu = Activation('relu')(fc1)        
        logits = fc_layer(relu, 1024, 10, 'fc2')
    else:        
        logits = fc_layer(flattened, 7 * 7 * 64, 10, 'fc1')

    logits = Activation('softmax')(logits)

    model = Model(inputs=x, outputs=logits)
    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    sess = K.get_session()
    writer.add_graph(sess.graph)

    for i in range(100):
        batch = mnist.train.next_batch(100)
        
        results = model.train_on_batch(batch[0], batch[1])

        if i % 5 == 0:
            print(results)

def make_hparam_string(learning_rate, two_conv_layer, two_fc_layer):
    return 'l={}-fc={}-conv={}'.format(learning_rate, two_fc_layer, two_conv_layer)

def main():
    for learning_rate in [1E-3, 1E-4, 1E-5]:
        for two_conv_layer in [True, False]:
            for two_fc_layer in [True, False]:
                hparam_str = make_hparam_string(learning_rate, two_conv_layer, two_fc_layer)
                print(hparam_str)
                writer = tf.summary.FileWriter('./logs/mnist_demo/' + hparam_str)

                mnist_model(learning_rate, two_conv_layer, two_fc_layer, writer)

if __name__ == '__main__':
    main()