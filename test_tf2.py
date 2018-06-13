import tensorflow as tf

writer1 = tf.summary.FileWriter('./logs/tf_tutorial/1')
with tf.Graph().as_default():
    a = tf.Variable(1, name="a")
    a = a + 1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a))
        print(a.eval())
        print(a.eval())
        writer1.add_graph(sess.graph)

writer2 = tf.summary.FileWriter('./logs/tf_tutorial/2')
with tf.Graph().as_default():
    a = tf.Variable(1, name="a")
    a = tf.assign(a, tf.add(a, 1))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a))
        print(a.eval())
        print(a.eval())
        writer2.add_graph(sess.graph)
