import tensorflow as tf

LOGDIR = './data/mnist_tutorial/'

mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + "data", one_hot=True)

def conv_layer(input, channels_in, channels_out, name='conv'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, channels_in, channels_out]), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name='B')
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME')
        act = tf.nn.relu(conv + b)

        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)

        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def fc_layer(input, channels_in, channels_out, name='fc'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channels_in, channels_out]), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name='B')
        act = tf.matmul(input, w) + b

        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)

        return act

def mnist_model(learning_rate, two_conv_layer, two_fc_layer, writer):
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)

    if two_conv_layer:
        conv1 = conv_layer(x_image, 1, 32, 'conv1')
        conv_out = conv_layer(conv1, 32, 64, 'conv2')
    else:
        conv_out = conv_layer(x_image, 1, 32, 'conv1')

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    if two_fc_layer:
        fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, name='fc1')
        logits = fc_layer(fc1, 1024, 10, 'fc2')
    else:
        logits = fc_layer(flattened, 7 * 7 * 64, 10, 'fc1')

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits = logits, labels=y
        ))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    with tf.Session() as sess:
        merged_summary = tf.summary.merge_all()
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        for i in range(500):
            batch = mnist.train.next_batch(100)

            if i % 5 == 0:
                s = sess.run(merged_summary, feed_dict={x: batch[0], y: batch[1]})
                writer.add_summary(s, i)

            if i % 10 == 0:
                [train_loss, train_accuracy] = sess.run([loss, accuracy], feed_dict={x: batch[0], y: batch[1]})
                print('step %d, loss %g, training accuracy %g' % (i, train_loss, train_accuracy))

            sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

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