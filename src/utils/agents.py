import tensorflow as tf
import os
import joblib
import numpy as np
import random
from gym.utils import colorize
from abc import ABC, abstractmethod
import multiprocessing


class Agent(ABC):
	def __init__(self, *, env_id):
		self.env_id = env_id

	@abstractmethod
	def act(self, obs):
		raise NotImplementedError

	@abstractmethod
	def play(self, *args):
		raise NotImplementedError


class LearningAgent(Agent):
	@abstractmethod
	def compile(self, *args):
		raise NotImplementedError

	@abstractmethod
	def fit(self, *args):
		raise NotImplementedError

	@abstractmethod
	def load(self, fn):
		raise NotImplementedError

	@abstractmethod
	def save(self, fn):
		raise NotImplementedError


class TFAgent(LearningAgent):
	def __init__(self, env_id, seed=None):
		super().__init__(env_id=env_id)
		self._session = None
		self.seed = seed
		if seed is not None:
			tf.set_random_seed(seed)
			np.random.seed(seed)
			random.seed(seed)

	@property
	def session(self):
		if self._session is None:
			num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
			config = tf.ConfigProto()
			config.intra_op_parallelism_threads = num_cpu
			config.inter_op_parallelism_threads = num_cpu
			config.gpu_options.allow_growth = True
			config.allow_soft_placement = True
			self._session = tf.Session(config=config)

		return self._session

	def load(self, fn):
		print(colorize(" [*] Loading variables...", "green"))
		variables = tf.trainable_variables()
		values = joblib.load(os.path.expanduser(fn))
		restores = [v.assign(values[v.name]) for v in variables]
		self.session.run(restores)

	def save(self, fn):
		print(colorize(" [*] Saving variables...", "green"))
		variables = tf.trainable_variables()
		values = self.session.run(variables)
		model = {v.name: value for v, value in zip(variables, values)}
		os.makedirs(os.path.dirname(fn), exist_ok=True)
		joblib.dump(model, fn)
