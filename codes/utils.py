import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from PIL import Image

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    #Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    #Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def forward_prop_predict(X, parameters):
    keep_prob=tf.placeholder_with_default(1.0,shape=())
    W1 = parameters['W1']
    W2 = parameters['W2']
    WL1 = parameters['WL1']
    b1= parameters['b1']

    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
    #Z1=tf.layers.batch_normalization(Z1)
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1,ksize=[1,6,6,1],strides=[1,6,6,1],padding='SAME')
    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')
    #Z2 = tf.layers.batch_normalization(Z2)
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')
    P2 = tf.contrib.layers.flatten(P2)
    P2 = tf.nn.dropout(P2,keep_prob)
    Z3 = tf.add(tf.matmul(WL1,tf.transpose(P2)),b1)
    Z3 = tf.transpose(Z3)
    Z3 = tf.nn.softmax(Z3)
    return Z3,keep_prob


def predict(X, parameters):

    W1 = tf.convert_to_tensor(parameters["W1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    WL1 = tf.convert_to_tensor(parameters["WL1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "WL1": WL1,
                  }


    x = tf.placeholder("float", [None, 151,151,1])

    z3,keep_prob = forward_prop_predict(x, parameters)
    #z3 = tf.nn.softmax(z3)
    pred = tf.argmax(z3,1)

    sess = tf.Session()
    prediction = sess.run(pred, feed_dict = {x: X})

    return prediction
