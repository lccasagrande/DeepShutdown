import tensorflow as tf
import numpy as np


def mlp(layers, activations):
	assert len(layers) == len(activations)

	def network(X):
		h = tf.layers.flatten(X)
		for units, activation in zip(layers, activations):
			h = tf.layers.dense(h, units, activation=activation)
		return h

	return network




# def cnn_small():
#	def network_fn(X, nenv=1):
#		nbatch = X.shape[0]
#		nsteps = nbatch // nenv
#		h = tf.cast(X, tf.float32)
#
#		h = tf.nn.relu(conv(h, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
#		# h = tf.nn.relu(conv(h, 'c2', nf=16, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
#		h = conv_to_fc(h)
#		# h = activ(fc(h, 'fc1', nh=128, init_scale=np.sqrt(2)))
#
#		M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
#		S = tf.placeholder(tf.float32, [nenv, 2 * 128])  # states
#
#		xs = batch_to_seq(h, nenv, nsteps)
#		ms = batch_to_seq(M, nenv, nsteps)
#
#		h5, snew = lstm(xs, ms, S, scope='lstm', nh=128)
#
#		h = seq_to_batch(h5)
#		initial_state = np.zeros(S.shape.as_list(), dtype=float)
#
#		return h, {'S': S, 'M': M, 'state': snew, 'initial_state': initial_state}
#
#	return network_fn
