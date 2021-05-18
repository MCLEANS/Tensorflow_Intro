import tensorflow as tf

project_graph = tf.Graph()
with project_graph.as_default():
    a = tf.add(3,5,)
    b = tf.add(3.0,4.6)

with tf.compat.v1.Session(graph = project_graph) as sess:
    a,b = sess.run([a,b])
    print(a)
    print(b)
