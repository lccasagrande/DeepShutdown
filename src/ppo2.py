import gym
import gym_grid.envs.grid_env as g
import multiprocessing
import os.path as osp
import tensorflow as tf
import numpy as np
import os
import argparse
import pandas as pd
from collections import defaultdict
from baselines import logger
from baselines.acer import acer
from baselines.ppo2 import ppo2
from baselines.deepq import deepq
from baselines.a2c import a2c
from baselines.a2c.utils import conv, fc, conv_to_fc, lstm, batch_to_seq, seq_to_batch
from baselines.common import misc_util
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.bench import Monitor
from baselines.common.retro_wrappers import RewardScaler
from baselines.common.models import register
from utils.common import print_episode_result

try:
	from mpi4py import MPI
except ImportError:
	MPI = None


@register("cnn_1")
def cnn_small(**conv_kwargs):
	def network_fn(X):
		# h = tf.cast(X, tf.float32) / 255.

		activ = tf.nn.relu
		h = activ(
			conv(X, "c1", nf=16, rf=2, stride=1, init_scale=np.sqrt(2), **conv_kwargs)
		)
		h = conv_to_fc(h)
		h = activ(fc(h, 'fc1', nh=20, init_scale=np.sqrt(2)))
		return h

	return network_fn


@register("mlp_small")
def mlp(num_layers=5, num_hidden=32, activation=tf.nn.relu, layer_norm=False):
	def network_fn(X):
		h = tf.layers.flatten(X)
		for i in range(num_layers):
			h = fc(h, "mlp_fc{}".format(i), nh=num_hidden, init_scale=np.sqrt(2))
			if layer_norm:
				h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
			h = activation(h)

		return h

	return network_fn


def build_env(args):
	def make_vec_env(env_id, nenv, seed, reward_scale):
		mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

		def make_env(rank):  # pylint: disable=C0111
			def _thunk():
				env = gym.make(env_id)
				env.seed(seed + 10000 * mpi_rank + rank if seed is not None else None)
				env = Monitor(
					env,
					logger.get_dir()
					and os.path.join(logger.get_dir(), str(mpi_rank) + "." + str(rank)),
					allow_early_resets=True,
				)

				return RewardScaler(env, reward_scale) if reward_scale != 1 else env

			return _thunk

		misc_util.set_global_seeds(seed)
		return (
			SubprocVecEnv([make_env(i) for i in range(nenv)])
			if nenv > 1
			else DummyVecEnv([make_env(0)])
		)

	return make_vec_env(
		args.env,
		args.num_env or multiprocessing.cpu_count(),
		args.seed,
		reward_scale=args.reward_scale,
	)


def train(args):
	print("Training with {}".format(args.model))
	env = build_env(args)

	if args.model == "A2C":
		model = a2c.learn(
			env=env,
			seed=args.seed,
			network=args.network,
			total_timesteps=int(args.num_timesteps),
			nsteps=8,
			lr=1e-3,
			ent_coef=0.01,
			gamma=1,
			log_interval=10,
			value_network="copy",
			load_path=args.load_path,
		)

	elif args.model == "PPO":
		model = ppo2.learn(
			env=env,
			seed=args.seed,
			network=args.network,
			total_timesteps=int(args.num_timesteps),
			nsteps=64,
			lam=0.95,
			gamma=1,
			lr=1e-3,  # 1.e-3,#1.e-3, # f * 2.5e-4,
			noptepochs=4,
			log_interval=10,
			nminibatches=8,
			ent_coef=0.01,
			cliprange=0.1,  # 0.2 value_network='copy' normalize_observations=True estimate_q=True
			value_network="copy",
			load_path=args.load_path,
		)
	elif args.model == "ACER":
		model = acer.learn(
			network=args.network,
			env=env,
			seed=args.seed,
			nsteps=16,
			lrschedule='constant',
			total_timesteps=int(args.num_timesteps),
			ent_coef=0.01,
			max_grad_norm=0.5,
			# c=0.5,
			lr=1e-3,
			gamma=1,
			replay_ratio=4,
			replay_start=1000,
			log_interval=100,
			value_network="copy",
			load_path=args.load_path)
	elif args.model == "DQN":
		model = deepq.learn(env,
		                    network='mlp',
		                    seed=args.seed,
		                    lr=1e-3,
		                    total_timesteps=int(args.num_timesteps),
		                    gamma=1.0,
		                    buffer_size=10000,
		                    exploration_final_eps=0.05,
		                    train_freq=4,
		                    target_network_update_freq=1000,
		                    exploration_fraction=0.1,
		                    hiddens=[])
	else:
		raise NotImplementedError("Model not implemented")

	return model, env


def train_model(args):
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
	model, env = train(args)
	env.close()

	if args.save_path is not None and rank == 0:
		save_path = osp.expanduser(args.save_path)
		model.save(save_path)


def test_model(args, metrics=[], verbose=True):
	def get_trajectory(env, results, metrics, visualize=False):
		def initialize_placeholders(nlstm=128, **kwargs):
			return np.zeros((1, 2 * nlstm)), np.zeros((1))

		state, dones = initialize_placeholders()
		score, steps, obs = 0, 0, env.reset()
		while True:
			if visualize:
				env.render()

			actions, _, state, _ = model.step(obs, S=state, M=dones)
			obs, rew, done, info = env.step(actions)
			done = done.any() if isinstance(done, np.ndarray) else done

			steps += 1
			score += rew[0]

			if done:
				for m in metrics:
					results[m] += info[0][m]
				results["score"] += score
				results["steps"] += steps
				return

	args.load_path = args.save_path if args.load_path == None else args.load_path
	args.save_path = None
	args.num_timesteps = 0
	args.num_env = 1
	results = defaultdict(float)
	model, _ = train(args)
	env = build_env(args)
	env.close()

	for _ in range(args.test_epi):
		get_trajectory(env, results, metrics, args.render)

	for k in results.keys():
		results[k] = results[k] / args.test_epi

	if verbose:
		print_episode_result("PPO2", results)

	if args.test_outputfn is not None:
		dt = pd.DataFrame([results])
		dt.to_csv(args.test_outputfn, index=False)


def run(args):
	batsim_metrics = [
		"makespan",
		"mean_slowdown",
		"total_slowdown",
		"total_turnaround_time",
		"mean_turnaround_time",
		"total_waiting_time",
		"mean_waiting_time",
		"max_waiting_time",
		"max_turnaround_time",
		"max_slowdown",
	]
	sim_metrics = [
		"energy_consumed",
		"makespan",
		"total_slowdown",
		"mean_slowdown",
		"total_turnaround_time",
		"mean_turnaround_time",
		"total_waiting_time",
		"mean_waiting_time",
	]
	if args.test:
		test_model(args, sim_metrics)
	else:
		train_model(args)

	print("Done!")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", type=str, default="grid-v0")
	parser.add_argument("--network", help="Network", default="mlp_small", type=str)
	parser.add_argument("--model", help="Model", default="PPO", type=str)
	parser.add_argument("--num_timesteps", type=int, default=9e6)
	parser.add_argument("--num_env", default=24, type=int)
	parser.add_argument("--reward_scale", default=1.0, type=float)
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--save_path", default="../weights/acer_training", type=str)
	parser.add_argument("--load_path", default=None, type=str)
	parser.add_argument("--test", default=False, action="store_true")
	parser.add_argument("--test_epi", default=1, type=int)
	parser.add_argument("--test_outputfn", default=None, type=str)
	parser.add_argument("--render", default=False, action="store_true")
	args = parser.parse_args()
	run(args)
