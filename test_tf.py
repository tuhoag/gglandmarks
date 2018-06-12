import tensorflow as tf
import os

LOGDIR = './data/mnist_tutorial/'
SPRITES = os.path.join(os.getcwd(), 'sprite_1024.png')
LABELS = os.path.join(os.getcwd(), 'labels_1024.tsv')

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
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)

    if two_conv_layer:
        conv1 = conv_layer(x_image, 1, 32, 'conv1')
        conv_out = conv_layer(conv1, 32, 64, 'conv2')
    else:
        conv_out = conv_layer(x_image, 1, 16, 'conv1')

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    if two_fc_layer:
        fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, name='fc1')
        relu = tf.nn.relu(fc1)
        embedding_input = relu
        embedding_size = 1024
        logits = fc_layer(relu, 1024, 10, 'fc2')
    else:
        embedding_input = flattened
        embedding_size = 7 * 7 * 64
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

    merged_summary = tf.summary.merge_all()

    embedding = tf.Variable(tf.zeros([1024, embedding_size], name='test_embedding'))
    assignment = embedding.assign(embedding_input)
    saver = tf.train.Saver()

    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = SPRITES
    embedding_config.metadata_path = LABELS
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    with tf.Session() as sess:
        
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        for i in range(100):
            batch = mnist.train.next_batch(100)
            # print(batch[0].shape)
            # print(batch[1].shape)
            if i % 5 == 0:
                s, train_loss, train_accuracy = sess.run([merged_summary, loss, accuracy], feed_dict={x: batch[0], y: batch[1]})
                writer.add_summary(s, i)
                print('step %d, loss %g, training accuracy %g' % (i, train_loss, train_accuracy))

            if i % 10 == 0:
                sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})                
                saver.save(sess, os.path.join(writer.get_logdir(), 'my-model.ckpt'), i)

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