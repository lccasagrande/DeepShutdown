import tensorflow as tf
import numpy as np


def batch_to_seq(h, nbatch, nsteps, flat=False):
	if flat:
		h = tf.reshape(h, [nbatch, nsteps])
	else:
		h = tf.reshape(h, [nbatch, nsteps, -1])
	return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]

def mask_softmax(logits, mask):
	masked_e = tf.exp(logits) * tf.cast(mask, dtype=tf.float32)
	return masked_e / tf.reduce_sum(masked_e, axis=-1, keepdims=True)

def sparse_softmax_cross_entropy_with_logits(logits, labels):
	encoded = tf.one_hot(labels, logits.get_shape()[-1])
	return tf.nn.softmax_cross_entropy_with_logits_v2(labels=encoded, logits=logits)


def seq_to_batch(h, flat=False):
	shape = h[0].get_shape().as_list()
	if not flat:
		assert (len(shape) > 1)
		nh = h[0].get_shape()[-1].value
		return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
	else:
		return tf.reshape(tf.stack(values=h, axis=1), [-1])


def KL(old_logits, logits):
	a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
	a1 = old_logits - tf.reduce_max(old_logits, axis=-1, keepdims=True)
	ea0 = tf.exp(a0)
	ea1 = tf.exp(a1)
	z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
	z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
	p0 = ea0 / z0
	return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)


def entropy(logits):
	a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
	ea0 = tf.exp(a0)
	z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
	p0 = ea0 / z0
	return tf.reduce_mean(tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1))


def max_min_mean_summary(var, family, collections=None):
	mean = tf.reduce_mean(var)
	tf.summary.scalar('mean', mean, family=family, collections=collections)
	tf.summary.scalar('max', tf.reduce_max(var), family=family, collections=collections)
	tf.summary.scalar('min', tf.reduce_min(var), family=family, collections=collections)


def variable_summaries(var, family, plot_hist=False, collections=None):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	mean = tf.reduce_mean(var)
	tf.summary.scalar('mean', mean, family=family, collections=collections)
	with tf.name_scope('stddev'):
		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	tf.summary.scalar('stddev', stddev, family=family, collections=collections)
	tf.summary.scalar('max', tf.reduce_max(var), family=family, collections=collections)
	tf.summary.scalar('min', tf.reduce_min(var), family=family, collections=collections)
	if plot_hist:
		tf.summary.histogram('histogram', var, family=family, collections=collections)


class TfRunningMeanStd(object):
	# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
	'''
	TensorFlow variables-based implmentation of computing running mean and std
	Benefit of this implementation is that it can be saved / loaded together with the tensorflow model
	'''

	def __init__(self, x, epsilon=1e-4, shape=(), scope=''):
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			self.mean = tf.get_variable('mean', initializer=np.zeros(shape, 'float64'), dtype=tf.float64,
			                            trainable=False)
			self.var = tf.get_variable('std', initializer=np.ones(shape, 'float64'), dtype=tf.float64, trainable=False)
			self.count = tf.get_variable('count', initializer=np.full((), epsilon, 'float64'), dtype=tf.float64,
			                             trainable=False)

		tf.reduce_mean(x, axis=0)
		self.batch_mean, self.batch_var = tf.nn.moments(x, axes=[0])
		self.batch_count = tf.to_double(tf.shape(x)[0])

		delta = tf.to_double(self.batch_mean) - self.mean
		tot_count = tf.add(self.count, self.batch_count)

		new_mean = self.mean + delta * self.batch_count / tot_count
		m_a = self.var * self.count
		m_b = tf.to_double(self.batch_var) * self.batch_count
		M2 = m_a + m_b + tf.square(delta) * self.count * self.batch_count / tot_count
		new_var = M2 / tot_count
		new_count = tot_count

		self.update_ops = tf.group([self.var.assign(new_var), self.mean.assign(new_mean), self.count.assign(new_count)])


class RunningMeanStd(object):
	# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
	'''
	TensorFlow variables-based implmentation of computing running mean and std
	Benefit of this implementation is that it can be saved / loaded together with the tensorflow model
	'''

	def __init__(self, sess, epsilon=1e-4, shape=(), scope=''):
		self._new_mean = tf.placeholder(shape=shape, dtype=tf.float64)
		self._new_var = tf.placeholder(shape=shape, dtype=tf.float64)
		self._new_count = tf.placeholder(shape=(), dtype=tf.float64)

		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			self._mean = tf.get_variable('mean', initializer=np.zeros(shape, 'float64'), dtype=tf.float64,
			                             trainable=False)
			self._var = tf.get_variable('std', initializer=np.ones(shape, 'float64'), dtype=tf.float64, trainable=False)
			self._count = tf.get_variable('count', initializer=np.full((), epsilon, 'float64'), dtype=tf.float64,
			                              trainable=False)

		self.update_ops = tf.group([
			self._var.assign(self._new_var),
			self._mean.assign(self._new_mean),
			self._count.assign(self._new_count)
		])

		sess.run(tf.variables_initializer([self._mean, self._var, self._count]))
		self.sess = sess

	# self._set_mean_var_count()

	def _set_mean_var_count(self):
		self.mean, self.var, self.count = self.sess.run([self._mean, self._var, self._count])

	def update(self, x):
		batch_mean = np.mean(x, axis=0)
		batch_var = np.var(x, axis=0)
		batch_count = x.shape[0]

		delta = batch_mean - self.mean
		tot_count = self.count + batch_count

		new_mean = self.mean + delta * batch_count / tot_count
		m_a = self.var * self.count
		m_b = batch_var * batch_count
		M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
		new_var = M2 / tot_count
		new_count = tot_count

		self.sess.run(self.update_ops, feed_dict={
			self._new_mean: new_mean,
			self._new_var: new_var,
			self._new_count: new_count
		})

# self._set_mean_var_count()
