import tensorflow as tf 

def restore_model():
	output=4
	W1 = tf.get_variable('W1',shape=(5,5,1,16),initializer=tf.contrib.layers.xavier_initializer())
	W2 = tf.get_variable('W2',shape=(2,2,16,32),initializer=tf.contrib.layers.xavier_initializer())
	WL1 = tf.get_variable('WL1',shape=(output,1568),initializer=tf.contrib.layers.xavier_initializer())
	b1=tf.get_variable('b1',shape=(output,1),initializer=tf.zeros_initializer())


	saver=tf.train.Saver()
	with tf.Session() as sess:

		saver.restore(sess,"./saved_model/CNN_model.ckpt")
		parameters={'W1':sess.run(W1),
					'W2':sess.run(W2),
					'WL1': sess.run(WL1),
					'b1' :sess.run(b1)
					}

	#print(parameters)
	return parameters

if __name__ == '__main__':
	restore_model()




	
