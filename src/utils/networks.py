import tensorflow as tf


def mlp(layers):
	def network(X):
		h = tf.layers.flatten(X)
		for unit in layers:
			h = tf.layers.dense(h, unit, activation=tf.nn.relu)
		return h

	return network
