import numpy as np
import time
import gym
import json
import csv
import os.path as osp
from multiprocessing import Process, Pipe
from gym import Wrapper
from gym.envs.classic_control import rendering
from src.utils.commons import tile_images


def worker(remote, env_fn_wrapper):
	env = env_fn_wrapper.x()
	try:
		while True:
			cmd, data = remote.recv()
			if cmd == 'step':
				ob, reward, done, info = env.step(data)
				if done:
					ob = env.reset()
				remote.send((ob, reward, done, info))
			elif cmd == 'reset':
				ob = env.reset()
				remote.send(ob)
			elif cmd == 'close':
				remote.close()
				break
			elif cmd == 'get_spaces':
				remote.send((env.action_space, env.observation_space))
			elif cmd == 'render':
				remote.send(env.render(mode='rgb_array', close=True))
			else:
				raise NotImplementedError
	except KeyboardInterrupt:
		print('SubprocVecEnv worker: got KeyboardInterrupt')
	finally:
		env.close()


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


class SubprocVecEnv:
	def __init__(self, env_fns):
		"""
		env_fns: list of environments to run in sub-processes
		"""
		self.viewer = None
		self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(len(env_fns))])
		# self.ps = [Process(target=test_worker, args=(work_remote,), daemon=True) for work_remote in self.work_remotes]
		self.ps = [
			Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)), daemon=True)
			for (work_remote, env_fn) in zip(self.work_remotes, env_fns)
		]
		for p in self.ps:
			p.start()

		for remote in self.work_remotes:
			remote.close()

		self.remotes[0].send(('get_spaces', None))
		self.action_space, self.observation_space = self.remotes[0].recv()
		self.num_envs = len(self.remotes)

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

		if self.viewer is not None:
			self.viewer.close()

	# def monitor(self, is_monitor=True, is_train=True, experiment_dir="", record_video_every=10):
	#	for remote in self.remotes:
	#		remote.send(('monitor', (is_monitor, is_train, experiment_dir, record_video_every)))

	def get_viewer(self):
		if self.viewer is None:
			self.viewer = rendering.SimpleImageViewer()
		return self.viewer

	def render(self):
		raise NotImplementedError()
# for remote in self.remotes:
#	remote.send(('render', None))


#
# imgs = [remote.recv() for remote in self.remotes]
# bigimg = tile_images(imgs)
# self.get_viewer().imshow(bigimg)


class Monitor(Wrapper):
	def __init__(self, env, info_keywords=()):
		super().__init__(env)
		self.tstart = time.time()
		self.info_keywords = info_keywords
		self.rewards = None
		self.needs_reset = True
		self.episode_rewards = []
		self.episode_lengths = []
		self.episode_times = []
		self.total_steps = 0

	def step(self, action):
		if self.needs_reset:
			raise RuntimeError("Tried to step environment that needs reset")
		ob, rew, done, info = self.env.step(action)
		self._update(rew, done, info)
		return ob, rew, done, info

	def reset(self, **kwargs):
		self.rewards = []
		self.needs_reset = False
		return self.env.reset(**kwargs)

	def _update(self, rew, done, info):
		self.rewards.append(rew)
		if done:
			self.needs_reset = True
			eprew = sum(self.rewards)
			eplen = len(self.rewards)
			epinfo = {"score": round(eprew, 6), "nsteps": eplen, "time": round(time.time() - self.tstart, 6)}
			for k in self.info_keywords:
				epinfo[k] = info[k]
			self.episode_rewards.append(eprew)
			self.episode_lengths.append(eplen)
			self.episode_times.append(time.time() - self.tstart)

			if isinstance(info, dict):
				info['episode'] = epinfo

		self.total_steps += 1

	def get_total_steps(self):
		return self.total_steps

	def get_episode_rewards(self):
		return self.episode_rewards

	def get_episode_lengths(self):
		return self.episode_lengths

	def get_episode_times(self):
		return self.episode_times


class CSVMonitor(Monitor):
	EXT = "monitor.csv"
	f = None

	def __init__(self, env, filename, info_keywords=()):
		super().__init__(env, info_keywords)
		if not filename.endswith(CSVMonitor.EXT):
			if osp.isdir(filename):
				filename = osp.join(filename, CSVMonitor.EXT)
			else:
				filename = filename + "." + CSVMonitor.EXT
		self.f = open(filename, "wt")
		self.logger = csv.DictWriter(self.f, fieldnames=('score', 'nsteps', 'time') + tuple(info_keywords))
		self.logger.writeheader()
		self.f.flush()

	def _write_row(self, epinfo):
		if self.logger:
			self.logger.writerow(epinfo)
			self.f.flush()

	def step(self, action):
		ob, rew, done, info = super().step(action)
		if 'episode' in info:
			self._write_row(info['episode'])
		return ob, rew, done, info

	def close(self):
		super().close()
		if self.f is not None:
			self.f.close()


def make_vec_env(env_id, nenv, seed, monitor_dir=None):
	def make_env(rank):
		def _thunk():
			env = gym.make(env_id)
			env.seed(seed + rank if seed is not None else None)
			env = Monitor(env) if monitor_dir is None else CSVMonitor(env, filename=monitor_dir+str(rank))
			return env

		return _thunk

	return SubprocVecEnv([make_env(i) for i in range(nenv)])
