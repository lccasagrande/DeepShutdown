import numpy as np
from multiprocessing import Process, Pipe
from gym import Wrapper, wrappers


# This class is to run multiple environments at the same time.

def worker(remote, env_fn_wrapper):
	print("Worker ALIVE!")
	env = env_fn_wrapper.x()
	while True:
		cmd, data = remote.recv()
		if cmd == 'step':
			ob, reward, done, info = env.step(data)
			total_info = info.copy()  # Very important for passing by value instead of reference
			if done: ob = env.reset()
			remote.send((ob, reward, done, total_info))
		elif cmd == 'reset':
			ob = env.reset()
			remote.send(ob)
		elif cmd == 'close':
			env.close()
			remote.close()
			break
		elif cmd == 'get_spaces':
			remote.send((env.action_space, env.observation_space))
		#elif cmd == 'monitor':
		#	is_monitor, is_train, experiment_dir, record_video_every = data
		#	env.monitor(is_monitor, is_train, experiment_dir, record_video_every)
		elif cmd == 'render':
			env.render()
		else:
			raise NotImplementedError


class CloudpickleWrapper(object):
	"""
	Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
	"""

	def __init__(self, x):
		self.x = x

	def __getstate__(self):
		import cloudpickle
		return cloudpickle.dumps(self.x)

	def __setstate__(self, ob):
		import pickle
		self.x = pickle.loads(ob)


class SubprocVecEnv():
	def __init__(self, env_fns):
		"""
		env_fns: list of environments to run in sub-processes
		"""
		self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(len(env_fns))])
		#self.ps = [Process(target=test_worker, args=(work_remote,), daemon=True) for work_remote in self.work_remotes]
		self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn))) for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
		for p in self.ps: p.start()
		self.remotes[0].send(('get_spaces', None))
		self.action_space, self.observation_space = self.remotes[0].recv()

	def step(self, actions):
		for remote, action in zip(self.remotes, actions):
			remote.send(('step', action))
		results = [remote.recv() for remote in self.remotes]
		obs, rews, dones, infos = zip(*results)
		return np.stack(obs), np.stack(rews), np.stack(dones), infos

	def reset(self):
		for remote in self.remotes:
			remote.send(('reset', None))
		return np.stack([remote.recv() for remote in self.remotes])

	def close(self):
		for remote in self.remotes:
			remote.send(('close', None))
		for p in self.ps:
			p.join()

	#def monitor(self, is_monitor=True, is_train=True, experiment_dir="", record_video_every=10):
	#	for remote in self.remotes:
	#		remote.send(('monitor', (is_monitor, is_train, experiment_dir, record_video_every)))

	def render(self):
		for remote in self.remotes:
			remote.send(('render', None))

	@property
	def num_envs(self):
		return len(self.remotes)


class Monitor(Wrapper):
	def __init__(self, env, rank=0):
		Wrapper.__init__(self, env=env)
		self.rank = rank
		self.rewards = []
		self.current_metadata = {}  # extra info that gets injected into each log entry
		self.summaries_dict = {'reward': 0, 'episode_length': 0}

	def reset(self):
		self.summaries_dict['reward'] = -1
		self.summaries_dict['episode_length'] = -1
		self.rewards = []
		return self.env.reset()

	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		self.rewards.append(reward)
		if done:
			self.summaries_dict['reward'] = sum(self.rewards)
			self.summaries_dict['episode_length'] = len(self.rewards)
		info = self.summaries_dict
		return observation, reward, done, info

	def get_summaries_dict(self):
		return self.summaries_dict

	def monitor(self, is_monitor, is_train, experiment_dir="", record_video_every=10):
		if is_monitor:
			if is_train:
				self.env = wrappers.Monitor(self.env, experiment_dir + 'output', resume=True,
											video_callable=lambda count: count % record_video_every == 0)
			else:
				self.env = wrappers.Monitor(self.env, experiment_dir + 'test', resume=True,
											video_callable=lambda count: count % record_video_every == 0)
		else:
			self.env = wrappers.Monitor(self.env, experiment_dir + 'output', resume=True,
										video_callable=False)
		self.env.reset()
