import tensorflow as tf

# with tf.Graph().as_default():
#     a = tf.Variable(1, name="a")
#     a = a + 1
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         print(sess.run(a))
#         print(a.eval())
#         print(a.eval())

with tf.Graph().as_default():
  a = tf.Variable(1, name="a")
  a = tf.assign(a, tf.add(a,1))
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(a.eval())
    print(a.eval())