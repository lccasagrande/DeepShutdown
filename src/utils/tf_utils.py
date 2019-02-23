import tensorflow as tf


def batch_to_seq(h, nbatch, nsteps, flat=False):
	if flat:
		h = tf.reshape(h, [nbatch, nsteps])
	else:
		h = tf.reshape(h, [nbatch, nsteps, -1])
	return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]


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
