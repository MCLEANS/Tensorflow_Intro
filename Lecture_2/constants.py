import tensorflow as tf 

graph = tf.Graph()

with graph.as_default():
    a = tf.constant([3,4], name = "input_data")
    b = tf.constant([[2,3],[4,6]], name = "variables")
    x = tf.add(a,b, name = 'Addition')
    y = tf.multiply(a,b, name = "multiplication")

with tf.compat.v1.Session(graph = graph) as sess:
    writer = tf.compat.v1.summary.FileWriter("./graphs",sess.graph)
    x,y = sess.run([x,y])
    print(x)
    print(y)

writer.close()