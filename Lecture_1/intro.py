import tensorflow as tf

g = tf.Graph()

with g.as_default():
    a = tf.add(5,7)

with tf.compat.v1.Session(graph = g) as sess:
    w = sess.run(a)
    print(w)
