import tensorflow as tf
import numpy as np
import os
import json
import random
import time as tm
from collections import defaultdict
from gym.utils import colorize


class Agent:
	def __init__(self, name, seed):
		self.name = name
		self.seed = seed
		np.random.seed(seed)
		random.seed(seed)

	def act(self, state):
		raise NotImplemented()

	def evaluate(self, env, n_episodes, visualize=False, verbose=True):
		reward_history, steps_history = list(), list()
		t0 = tm.time()
		for epi in range(1, n_episodes + 1):
			ob = env.reset()
			reward,steps = 0.0, 0
			while True:
				if visualize:
					env.render()
				action = self.act(ob)
				ob, r, done, info = env.step(action)
				reward += r
				steps += 1
				if done:
					break
			reward_history.append(reward)
			steps_history.append(steps)
			if verbose:
				results = " ".join([" [{}: {}]".format(k, v) for k, v in info.items()])
				print("[EVALUATE {}] {}".format(epi, results))

		env.close()
		tend = tm.time() - t0
		print("[EVALUATE] Avg. reward {:.4f} - Avg. steps {:.4f} over {} episodes in {:.4f} seconds".format(np.mean(reward_history), np.mean(steps_history), n_episodes, tend))


class LearningAgent(Agent):
	def __init__(self, input_shape, nb_actions, **kwargs):
		super(LearningAgent, self).__init__(**kwargs)
		self.input_shape = input_shape
		self.nb_actions = nb_actions

	def build_model(self):
		raise NotImplementedError()

	def fit(self):
		raise NotImplementedError()


class TFAgent(LearningAgent):
	def __init__(self, saver_max_to_keep=1, save_path=None, **kwargs):
		super(TFAgent, self).__init__(**kwargs)
		tf.set_random_seed(self.seed)
		self._saver_max_to_keep = saver_max_to_keep
		self._log = defaultdict(list)
		self._summary_writer = None
		self._saver = None
		self._session = None
		self._log_fn = None
		self._writer = None
		self._save_path = self.checkpoint_dir if save_path is None else save_path

	def save_log(self):
		print(colorize(" [*] Saving logs...", "green"))
		with open(self.log_fn, "w+") as f:
			json.dump(self._log, f)

	def save_model(self, step=None):
		print(colorize(" [*] Saving checkpoints...", "green"))
		ckpt_file = os.path.join(self._save_path, self.name)
		self.saver.save(self.session, ckpt_file, global_step=step)

	def load_model(self):
		print(colorize(" [*] Loading checkpoints...", "green"))

		ckpt = tf.train.get_checkpoint_state(self._save_path)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			fname = os.path.join(self._save_path, ckpt_name)
			self.saver.restore(self.session, fname)
			print(colorize(" [*] Load SUCCESS: %s" % fname, "green"))
			return True
		else:
			print(colorize(" [!] Load FAILED: %s" % self.checkpoint_dir, "red"))
			return False

	@property
	def checkpoint_dir(self):
		root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
		ckpt_path = os.path.join(root, "checkpoints", self.name)
		os.makedirs(ckpt_path, exist_ok=True)
		return ckpt_path

	@property
	def saver(self):
		if self._saver is None:
			self._saver = tf.train.Saver(max_to_keep=self._saver_max_to_keep)
		return self._saver

	@property
	def log_fn(self):
		if self._log_fn is None:
			path = os.path.join(self._save_path, self.name)
			os.makedirs(path, exist_ok=True)
			self._log_fn = path + ".log"
		return self._log_fn

	@property
	def writer(self):
		if self._writer is None:
			writer_path = os.path.join(self._save_path, self.name)
			os.makedirs(writer_path, exist_ok=True)
			self._writer = tf.summary.FileWriter(writer_path, self.session.graph)
		return self._writer

	@property
	def session(self):
		if self._session is None:
			config = tf.ConfigProto()
			config.intra_op_parallelism_threads = 1
			config.inter_op_parallelism_threads = 1
			config.gpu_options.allow_growth = True
			self._session = tf.Session(config=config)

		return self._session
