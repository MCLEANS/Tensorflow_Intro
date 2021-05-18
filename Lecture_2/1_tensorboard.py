
import tensorflow as tf 

project_graph = tf.Graph()

with project_graph.as_default():
    x = 2
    y = 23
    result1 = tf.add(x,y, name = "add")
    result2 = tf.multiply(result1,x, name = "Multiply")

with tf.compat.v1.Session(graph = project_graph) as sess:
    writer = tf.compat.v1.summary.FileWriter("./graphs",sess.graph)
    #result2 = sess.run(result2)

writer.close()

