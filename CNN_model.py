import cv2, os
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import math
from helper import get_images_and_labels
from utils import *


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



data=get_images_and_labels("C:\\Users\\Hiep Nguyen\\Desktop\\CV_project\\data\\train_set")
(x_train,y_train)=data
x_train=np.array(x_train)
x_train=x_train.reshape(161,151,151,-1)
x_train=x_train/255
y_train=np.array(y_train)
y_train=y_train.reshape(161,-1)


test_set=data=get_images_and_labels("C:\\Users\\Hiep Nguyen\\Desktop\\CV_project\\data\\test_set")
(x_test,y_test)=test_set
x_test=np.array(x_test)
x_test=x_test.reshape(5,151,151,-1)
x_test=x_test/255
y_test=np.array(y_test)
y_test.reshape(5,-1)


y_train=convert_to_one_hot(y_train.T,16).T
y_test=convert_to_one_hot(y_test.T,16).T

def create_placeholders(n_H0, n_W0, n_C0, n_y):

    X = tf.placeholder(tf.float32,[None,n_H0,n_W0,n_C0])
    Y = tf.placeholder(tf.float32,[None,n_y])

    return X, Y

def initialize_parameters():


    tf.set_random_seed(1)
    W1 = tf.get_variable('W1',shape=(5,5,1,8),initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W2',shape=(2,2,8,16),initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


def forward_propagation(X, parameters):

    W1 = parameters['W1']
    W2 = parameters['W2']

    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1,ksize=[1,6,6,1],strides=[1,6,6,1],padding='SAME')
    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2,16,activation_fn=None)

    return Z3

def compute_cost(Z3, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))

    return cost


    return mini_batches

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):


    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0,n_W0,n_C0,n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)
    cost = compute_cost(Z3,Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
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
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters

_, _, parameters = model(x_train, y_train, x_test, y_test)
