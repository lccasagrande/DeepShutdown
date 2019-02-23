import tensorflow as tf
import os
import joblib
from gym.utils import colorize
from abc import ABC, abstractmethod
import multiprocessing


class Agent(ABC):
	def __init__(self, *, env_id, input_shape, nb_actions):
		self.env_id = env_id
		self.input_shape = input_shape
		self.nb_actions = nb_actions

	@abstractmethod
	def act(self, obs):
		raise NotImplementedError

	@abstractmethod
	def play(self, *args):
		raise NotImplementedError


class LearningAgent(Agent):
	def __init__(self, env_id, input_shape, nb_actions, network):
		super().__init__(env_id=env_id, input_shape=input_shape, nb_actions=nb_actions)
		self._network = network

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
	def __init__(self, env_id, input_shape, nb_actions, network):
		super().__init__(env_id=env_id, input_shape=input_shape, nb_actions=nb_actions, network=network)
		self._session = None
		self._writer = None
		#self.output_dir = output_dir
		#if output_dir is not None:
		#	os.makedirs(output_dir, exist_ok=True)

	#@property
	#def writer(self):
	#	if self._writer is None:
	#		assert self.output_dir is not None, "Output dir should be set to dump summaries"
	#		self._writer = tf.summary.FileWriter(self.output_dir + "summaries", tf.get_default_graph())
	#	return self._writer

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
