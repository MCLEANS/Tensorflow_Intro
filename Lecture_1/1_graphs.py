import tensorflow as tf

#Create a graph and declare it as the default graph
project_graph = tf.Graph()
with project_graph.as_default():
    x = 2
    y = 4
    result1 = tf.add(x,y)
    result2 = tf.multiply(result1,5)
    result3 = tf.multiply(result1,result2)

with tf.compat.v1.Session(graph = project_graph) as sess:
    result3 , result2 = sess.run([result3,result2])
    print(result3)
