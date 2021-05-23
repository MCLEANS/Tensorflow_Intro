import tensorflow as tf 

graph = tf.Graph()

with graph.as_default():
    a = tf.add(3,5)



with tf.compat.v1.Session(graph = graph) as sess:
    print(sess.run(a))