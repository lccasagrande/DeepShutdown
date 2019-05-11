import numpy as np
from collections import defaultdict, deque


class MulRunningMeanStd(object):
	# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
	def __init__(self, epsilon=1e-4):
		self.history = defaultdict(lambda: deque(maxlen=10))
		self.count = defaultdict(lambda: epsilon)
		self.var = defaultdict(lambda: 1)
		self.mean = defaultdict(lambda: 0)

	def add(self, key, value):
		self.history[key].append(value)

	def update(self):
		#while self.history:
		#	k, values = self.history.popitem()
		for k, values in self.history.items():
			self.mean[k] = np.mean(values)
			self.var[k] = np.var(values)
			#self._update_from_moments(np.mean(values), np.var(values), len(values), k)

	def _update_from_moments(self, batch_mean, batch_var, batch_count, k):
		delta = batch_mean - self.mean[k]
		tot_count = self.count[k] + batch_count

		new_mean = self.mean[k] + delta * batch_count / tot_count
		m_a = self.var[k] * self.count[k]
		m_b = batch_var * batch_count
		M2 = m_a + m_b + np.square(delta) * self.count[k] * batch_count / tot_count
		new_var = M2 / tot_count
		new_count = tot_count

		self.mean[k] = new_mean
		self.var[k] = new_var
		self.count[k] = new_count


def safemean(xs):
	return np.nan if len(xs) == 0 else np.mean(xs)


def softmax(x, axis=-1):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=axis, keepdims=True)


def discount_with_dones(rewards, dones, gamma):
	discounted, r = [], 0
	for reward, done in zip(rewards[::-1], dones[::-1]):
		r = reward + gamma * r * (1. - done)  # fixed off by one bug
		discounted.append(r)
	return discounted[::-1]


def pad_sequence(seqs):
	maxlen = max(len(s) for s in seqs)
	seq, seqs_len = [], []
	for s in seqs:
		seqlen = len(s)
		seqs_len.append(seqlen)
		if seqlen < maxlen:
			seq.append(np.pad(s, ((0, maxlen - seqlen), (0, 0)), mode='constant', constant_values=0))
		else:
			seq.append(s)
	return np.asarray(seq), np.asarray(seqs_len)


def discount(rewards, gamma):
	discounted, r = [], 0
	for reward in rewards[::-1]:
		r = reward + gamma * r
		discounted.append(r)
	return np.asarray(discounted[::-1])


def explained_variance(ypred, y):
	"""
	Computes fraction of variance that ypred explains about y.
	Returns 1 - Var[y-ypred] / Var[y]

	interpretation:
		ev=0  =>  might as well have predicted zero
		ev=1  =>  perfect prediction
		ev<0  =>  worse than just predicting zero

	"""
	assert y.ndim == 1 and ypred.ndim == 1
	vary = np.var(y)
	return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def tile_images(img_nhwc):
	"""
	Tile N images into one big PxQ image
	(P,Q) are chosen to be as close as possible, and if N
	is square, then P=Q.

	input: img_nhwc, list or array of images, ndim=4 once turned into array
		n = batch index, h = height, w = width, c = channel
	returns:
		bigim_HWc, ndarray with ndim=3
	"""
	img_nhwc = np.asarray(img_nhwc)
	N, h, w, c = img_nhwc.shape
	H = int(np.ceil(np.sqrt(N)))
	W = int(np.ceil(float(N) / H))
	img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
	img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
	img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
	img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
	return img_Hh_Ww_c


def normalize(dt):
	return (dt - dt.mean()) / (dt.std() + 1e-8)
