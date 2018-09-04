import cv2, os
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import math
from data_processing import get_images_and_labels,get_emotion
from utils import *

##Include the next three lines if training on GPU(s)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
###


data=get_images_and_labels("C:\\Users\\Hiep Nguyen\\Desktop\\Emotion_Detector\\data\\train_set")
(x_train,y_train)=data
x_train=np.array(x_train)
#print(x_train.shape)
x_train=x_train.reshape(161,151,151,-1)
x_train=x_train/255
y_train=get_emotion(y_train)
y_train=np.array(y_train)
#print(y_train.shape)
y_train=y_train.reshape(161,-1)


test_set=data=get_images_and_labels("C:\\Users\\Hiep Nguyen\\Desktop\\Emotion_Detector\\data\\dev_set")
(x_test,y_test)=test_set
x_test=np.array(x_test)
x_test=x_test.reshape(10,151,151,-1)
x_test=x_test/255
y_test=get_emotion(y_test)
y_test=np.array(y_test)
y_test.reshape(10,-1)


y_train=convert_to_one_hot(y_train.T,4).T
y_test=convert_to_one_hot(y_test.T,4).T

#fl1=40
output = 4
#keep_prob= 0.5


def create_placeholders(n_H0, n_W0, n_C0, n_y):

    X = tf.placeholder(tf.float32,[None,n_H0,n_W0,n_C0])
    Y = tf.placeholder(tf.float32,[None,n_y])

    return X, Y

def initialize_parameters():

   #tf.set_random_seed(1)
    W1 = tf.get_variable('W1',shape=(5,5,1,16),initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W2',shape=(2,2,16,32),initializer=tf.contrib.layers.xavier_initializer(seed=0))
    WL1 = tf.get_variable('WL1',shape=(output,1568),initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b1=tf.get_variable('b1',shape=(output,1),initializer=tf.zeros_initializer())
    parameters = {"W1": W1,
                  "W2": W2,
                  "WL1":WL1,
                  "b1":b1}

    return parameters


def forward_prop(X, parameters):

    W1 = parameters['W1']
    W2 = parameters['W2']
    WL1 = parameters['WL1']
    b1= parameters['b1']

    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1,ksize=[1,6,6,1],strides=[1,6,6,1],padding='SAME')
    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.add(tf.matmul(WL1,tf.transpose(P2)),b1)
    Z3 = tf.transpose(Z3)

    return Z3

def compute_cost(Z3, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001,
          num_epochs = 1000, minibatch_size = 64, print_cost = True):


    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0,n_W0,n_C0,n_y)
    parameters = initialize_parameters()
    Z3 = forward_prop(X,parameters)
    cost = compute_cost(Z3,Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) 
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,
                                                                    Y:minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches



            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        save_path = saver.save(sess, "./saved_model/CNN_model.ckpt")
        print("Model is saved to ", save_path)       


        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        #print(sess.run(parameters))
        return train_accuracy, test_accuracy, parameters


 
if __name__ == '__main__':
	model(x_train, y_train, x_test, y_test)
