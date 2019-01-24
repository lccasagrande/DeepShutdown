import numpy as np
from abc import ABC, abstractmethod

class AbstractEnvRunner(ABC):
	def __init__(self, *, env, model, nsteps):
		self.env = env
		self.model = model
		self.nenv = env.num_envs if hasattr(env, 'num_envs') else 1
		self.obs = env.reset()
		self.nsteps = nsteps
		self.done = [False for _ in range(self.nenv)]

	@abstractmethod
	def run(self):
		raise NotImplementedError

