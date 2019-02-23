import gym
import gridgym.envs.grid_env
import tensorflow as tf
import os.path as osp
import numpy as np
import os
import pandas as pd
import random
from utils.env_wrappers import VecFrameStack
from collections import defaultdict
from src.utils.env_wrappers import make_vec_env, VecFrameStack
from src.baselines.baselines.ppo2 import ppo2
from src.baselines.baselines.a2c.utils import conv, fc, conv_to_fc, lstm, batch_to_seq, seq_to_batch
from src.baselines.baselines.common import misc_util
from src.baselines.baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from src.baselines.baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from src.baselines.baselines.bench.monitor import Monitor
from src.baselines.baselines.common.retro_wrappers import RewardScaler


class PPOAgent:
	def __init__(self, network, env_id, num_env, seed, rew_scale, nb_frames, incl_action=True):
		self.network = network
		self.num_env = num_env
		self.env_id = env_id
		self.seed = seed
		self.incl_action = incl_action

		tf.set_random_seed(self.seed)
		np.random.seed(self.seed)
		random.seed(self.seed)
		self.rew_scale = rew_scale
		self.nb_frames = nb_frames
		self.model = None

	def train(self, n_timesteps, env=None, nsteps=512, gamma=.98, noptepochs=4, nminibatches=6, ent_coef=0.00, lr=1e-3,
	          # lr=lambda f: 3e-4 * f,
	          cliprange=0.2, weights=None, save_path=None, monitor_dir=None):
		if env is None:
			env = VecFrameStack(make_vec_env(self.env_id, self.num_env, monitor_dir=monitor_dir), self.nb_frames,
			                    self.incl_action)

		self.model = ppo2.learn(
			env=env,
			seed=self.seed,
			network=lstm22(),
			total_timesteps=n_timesteps,
			nsteps=nsteps,
			lam=.95,  # 0.95,
			gamma=gamma,
			lr=lr,  # 1.e-3,#1.e-3, # f * 2.5e-4,
			noptepochs=noptepochs,
			log_interval=10,
			nminibatches=nminibatches,  # 8,  # 8
			ent_coef=ent_coef,
			# estimate_q=True,
			cliprange=cliprange,  # 0.2 value_network='copy' normalize_observations=True estimate_q=True
			# value_network="copy",
			load_path=weights
		)
		env.close()

		if save_path is not None:
			save_path = osp.expanduser(save_path)
			self.model.save(save_path)

	def evaluate(self, weights, nb_episodes=1, output_fn=None, verbose=True, render=False):
		def get_trajectory(env, render):
			def initialize_placeholders(nlstm=64, **kwargs):
				return np.zeros((1, 2 * nlstm)), np.zeros((1))

			state, dones = initialize_placeholders()
			score, steps, results, done, info, obs = 0, 0, defaultdict(float), False, [{}], env.reset()
			while not done:
				if render:
					env.render('console')
				actions, _, state, _ = self.model.step(obs, S=state, M=dones)
				print(actions)
				# print("State: {} - Action: {}".format(obs, actions))
				obs, rew, done, info = env.step(actions)
				done = done.any() if isinstance(done, np.ndarray) else done
				results["steps"] += 1
				results["score"] += rew[0]

			for k, v in info[0].items():
				results[k] = v

			return results

		env = VecFrameStack(make_vec_env(self.env_id, 1), self.nb_frames, self.incl_action)
		self.train(n_timesteps=0, env=env, weights=weights, nminibatches=1)
		results = defaultdict(list)
		rewards = list()

		for i in range(1, nb_episodes + 1):
			ep_results = get_trajectory(env, render)
			ep_results['episode'] = i
			rewards.append(ep_results["score"])
			for k, v in ep_results.items():
				results[k].append(v)

			if verbose:
				print("[EVALUATE][PPO] {}".format(
					" ".join([" [{}: {}]".format(k, v) for k, v in sorted(results.items())])))

		print("Avg reward: {}".format(np.mean(rewards)))
		if output_fn is not None:
			pd.DataFrame([results]).to_csv(output_fn, index=False)


def mlp(num_layers=2, num_hidden=64, activation=tf.nn.leaky_relu, layer_norm=False):
	def network_fn(X):
		h = tf.layers.flatten(X)
		for i in range(num_layers):
			h = fc(h, "mlp_fc{}".format(i), nh=num_hidden, init_scale=np.sqrt(2))
			if layer_norm:
				h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
			h = activation(h)

		return h

	return network_fn


def lstm22(nlstm=64):
	def network_fn(X, nenv=1):
		nbatch = X.shape[0]
		nsteps = nbatch // nenv

		h = tf.layers.flatten(X)

		M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
		S = tf.placeholder(tf.float32, [nenv, 2 * nlstm])  # states

		xs = batch_to_seq(h, nenv, nsteps)
		ms = batch_to_seq(M, nenv, nsteps)

		h5, snew = lstm(xs, ms, S, scope='lstm', nh=nlstm)

		h = seq_to_batch(h5)
		initial_state = np.zeros(S.shape.as_list(), dtype=float)

		return h, {'S': S, 'M': M, 'state': snew, 'initial_state': initial_state}

	return network_fn


def cnn_small(**conv_kwargs):
	def network_fn(X, nenv=1):
		nbatch = X.shape[0]
		nsteps = nbatch // nenv
		h = tf.cast(X, tf.float32)

		h = tf.nn.relu(conv(h, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
		# h = tf.nn.relu(conv(h, 'c2', nf=16, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
		h = conv_to_fc(h)
		# h = activ(fc(h, 'fc1', nh=128, init_scale=np.sqrt(2)))

		M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
		S = tf.placeholder(tf.float32, [nenv, 2 * 128])  # states

		xs = batch_to_seq(h, nenv, nsteps)
		ms = batch_to_seq(M, nenv, nsteps)

		h5, snew = lstm(xs, ms, S, scope='lstm', nh=128)

		h = seq_to_batch(h5)
		initial_state = np.zeros(S.shape.as_list(), dtype=float)

		return h, {'S': S, 'M': M, 'state': snew, 'initial_state': initial_state}

	return network_fn
