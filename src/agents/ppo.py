import gym
import gridgym.envs.grid_env
import tensorflow as tf
import os.path as osp
import numpy as np
import os
import pandas as pd
from collections import defaultdict
from src.baselines.baselines import logger
from src.baselines.baselines.ppo2 import ppo2
from src.baselines.baselines.a2c.utils import conv, fc, conv_to_fc, lstm, batch_to_seq, seq_to_batch
from src.baselines.baselines.common import misc_util
from src.baselines.baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from src.baselines.baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from src.baselines.baselines.bench.monitor import Monitor
from src.baselines.baselines.common.retro_wrappers import RewardScaler

try:
	from mpi4py import MPI
except ImportError:
	MPI = None


class PPOAgent:
	def __init__(self, network, env_id, num_env, seed, rew_scale):
		self.network = network
		self.num_env = num_env
		self.env_id = env_id
		self.seed = seed
		self.rew_scale = rew_scale
		self.model = None

	def _build_env(self, nenv):
		def make_vec_env(nenv):
			mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

			def make_env(rank):  # pylint: disable=C0111
				def _thunk():
					env = gym.make(self.env_id)
					env.seed(self.seed + 10000 * mpi_rank + rank)  # if self.seed is not None else None)
					env = Monitor(
						env,
						logger.get_dir()
						and os.path.join(logger.get_dir(), str(mpi_rank) + "." + str(rank)),
						allow_early_resets=True,
					)

					return RewardScaler(env, self.rew_scale) if self.rew_scale != 1 else env

				return _thunk

			misc_util.set_global_seeds(self.seed)
			return (
				SubprocVecEnv([make_env(i) for i in range(nenv)])
				if nenv > 1
				else DummyVecEnv([make_env(0)])
			)

		return make_vec_env(nenv)

	def train(self, n_timesteps, env=None, nsteps=40, lr=5e-4, gamma=.99, noptepochs=2, nminibatches=4, ent_coef=0.01,
	          cliprange=0.1, weights=None, save_path=None):
		def config_log():
			# configure logger, disable logging in child MPI processes (with rank > 0)
			if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
				rank = 0
				logger.configure()
			else:
				logger.configure(format_strs=[])
				rank = MPI.COMM_WORLD.Get_rank()
			return rank

		rank = config_log()
		env = self._build_env(self.num_env) if env is None else env
		self.model = ppo2.learn(
			env=env,
			seed=self.seed,
			network=lstm22(),
			total_timesteps=n_timesteps,
			nsteps=nsteps,
			lam=.99,#0.95,
			gamma=gamma,
			lr=lr,  # 1.e-3,#1.e-3, # f * 2.5e-4,
			noptepochs=noptepochs,
			log_interval=10,
			nminibatches=nminibatches,  # 8,  # 8
			ent_coef=ent_coef,
			cliprange=cliprange,  # 0.2 value_network='copy' normalize_observations=True estimate_q=True
			#value_network="copy",
			load_path=weights,
		)
		env.close()

		if save_path is not None and rank == 0:
			save_path = osp.expanduser(save_path)
			self.model.save(save_path)

	def evaluate(self, weights, nb_episodes=1, output_fn=None, verbose=True, render=False):
		def get_trajectory(env, render):
			def initialize_placeholders(nlstm=128, **kwargs):
				return np.zeros((1, 2 * nlstm)), np.zeros((1))

			state, dones = initialize_placeholders()
			score, steps, results, done, info, obs = 0, 0, defaultdict(float), False, [{}], env.reset()
			while not done:
				if render:
					env.render()
				actions, _, state, _ = self.model.step(obs, S=state, M=dones)
				print("State: {} - Action: {}".format(obs, actions))
				obs, rew, done, info = env.step(actions)
				done = done.any() if isinstance(done, np.ndarray) else done
				results["steps"] += 1
				results["score"] += rew[0]

			for k, v in info[0].items():
				results[k] = v

			return results

		env = self._build_env(1)
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
				print("PPO - " + " ".join("{}: {}".format(k, v) for k, v in results.items()))

		print("Avg reward: {}".format(np.mean(rewards)))
		if output_fn is not None:
			pd.DataFrame([results]).to_csv(output_fn, index=False)


def mlp(num_layers=1, num_hidden=32, activation=tf.nn.relu, layer_norm=False):
	def network_fn(X):
		h = tf.layers.flatten(X)
		for i in range(num_layers):
			h = fc(h, "mlp_fc{}".format(i), nh=num_hidden, init_scale=np.sqrt(2))
			if layer_norm:
				h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
			h = activation(h)

		return h

	return network_fn


def lstm22(nlstm=128):
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
