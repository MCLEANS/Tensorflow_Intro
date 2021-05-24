import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv");
df=df.dropna()
freq = df["freq"]
phAngle = df["phAngle"]
power = df["power"];
reacPower = df["reacPower"]
rmsCur = df["rmsCur"]
rmsVolt = df["rmsVolt"]

freq = np.array(freq)
phAngle = np.array(phAngle)
power = np.array(power)
reacPower = np.array(reacPower)
rmsCur = np.array(rmsCur)
rmsVolt = np.array(rmsVolt)

project_graph = tf.Graph()
with project_graph.as_default():
    #create placeholders for inputs X and the output Y
    X1 = tf.compat.v1.placeholder(tf.float32,name = "X1")
    X2 = tf.compat.v1.placeholder(tf.float32,name = "X2")
    X3 = tf.compat.v1.placeholder(tf.float32,name = "X3")
    X4 = tf.compat.v1.placeholder(tf.float32,name = "X4")
    X5 = tf.compat.v1.placeholder(tf.float32,name = "X5")
    Y = tf.compat.v1.placeholder(tf.float32,name = "Y")

    #create weights and biases initialized to 0
    w1 = tf.Variable(0.0,name = "w1")
    w2 = tf.Variable(0.0,name = "w2")
    w3 = tf.Variable(0.0,name = "w3")
    w4 = tf.Variable(0.0,name = "w4")
    w5 = tf.Variable(0.0,name = "w5")
    b = tf.Variable(0.0, name = "bias")

    #build model to predict Y
    Y_predicted = (X1*w1)+(X2*w2)+(X3*w3)+(X4*w4)+(X5*w5)+b

    #Use the square error as the loss
    error = Y - Y_predicted
    #loss = tf.reduce_mean(error, name = "loss")
    loss =  tf.reduce_mean(tf.pow(Y_predicted - Y, 2)) / (2 * len(phAngle))
    #loss = tf.keras.losses.mean_squared_error(Y,Y_predicted)

    #Use gradient descent with learning rate of 0.01 to minimize loss
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.0001).minimize(loss)
    summary_op = tf.compat.v1.summary.merge_all()
    loss_sum = tf.compat.v1.summary.scalar('Loss', loss)
    #initialize variables
    init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session(graph = project_graph) as sess:
    sess.run(init)
    writer = tf.compat.v1.summary.FileWriter("./graphs",sess.graph)
    #train the model
    for i in range(50000):
        _,l = sess.run([optimizer,loss], feed_dict = {X1:phAngle,X2:reacPower,X3:freq,X4:rmsCur,X5:rmsVolt,Y:power})
        writer.add_summary(sess.run(loss_sum,feed_dict = {X1:phAngle,X2:reacPower,X3:freq,X4:rmsCur,X5:rmsVolt,Y:power}),i)
        print("Epoch  {} : {} ".format(i,l))
    writer.close()
    _w1,_w2,_w3,_w4,_w5,_b = sess.run([w1,w2,w3,w4,w5,b])

plt.title("Coffee Machine")
plt.scatter(reacPower,power, color = "green",label = "Reactive Power")
plt.scatter(phAngle,power, color = "yellow",label = "Phase Angle")
plt.scatter(rmsVolt,power, color = "grey", label = "Rms Voltage")
plt.scatter(rmsCur,power, color = "red" , label = "Rms Current")
plt.scatter(freq,power, color = "blue", label = "Frequency")
plt.xlabel("Itteration")
plt.ylabel("Power (Watt)")
plt.plot(list(range(0,(freq.size))),(phAngle*_w1)+(reacPower*_w2)+(freq*_w3)+(rmsCur*_w4)+(rmsVolt*_w5)+_b, color = "black", label = "Line of best fit")
plt.colorbar()
plt.legend()
plt.show()

_X1 = 274
_X2 = -12.489
_X3 = 50
_X4 = 0.053
_X5 = 233.84
Y_= (_X1*_w1)+(_X2*_w2)+(_X3*_w3)+(_X4*_w4)+(_X5*_w5)+_b
print("Predicted Y =  {}".format(Y_))



