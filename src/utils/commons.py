import numpy as np


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
	avg = dt.mean()
	std = dt.std()
	return (dt - avg) / std if std != 0 else dt
