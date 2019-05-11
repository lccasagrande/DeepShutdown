import tensorflow as tf
import gym
import gridgym.envs.grid_env as g
from gridgym.envs.simulator.utils.graphics import plot_simulation_graphics
from gridgym.envs.off_reservation_env import OffReservationGridEnv
from evalys.visu.legacy import *
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from collections import defaultdict
from plotly.offline import plot
from src.utils import loggers as log
from src.utils.networks import *
from src.agents.myppo import PPOAgent


def plot_hist(dt, interval, title, with_error=False):
	traces = []
	for (data, name) in dt:
		means, maxs, mins = [], [], []
		for i in range(0, len(data), interval):
			means.append(np.mean(data[i:i + interval]))
			maxs.append(np.max(data[i:i + interval]))
			mins.append(np.min(data[i:i + interval]))
		maxs, mins = np.array(maxs), np.array(mins)
		x = list(range(0, len(means) * interval, interval))
		error_y = dict(type='data', symmetric=False, array=maxs - means,
		               arrayminus=means - mins) if with_error else dict()
		traces.append(go.Scatter(x=x, y=means, mode='lines+markers', name=name, error_y=error_y))
	fig = go.Figure(data=traces, layout=go.Layout(title=title))
	plot(fig, filename=title + ".html")


def run(args):
	loggers = log.LoggerWrapper()
	if args.log_interval != 0:
		loggers.append(log.CSVLogger(args.log_dir + "ppo_log.csv"))
	if args.verbose:
		loggers.append(log.ConsoleLogger())

	agent = PPOAgent(args.env_id, args.seed, args.nb_frames, args.log_dir, normalize_obs=False, clip_obs=None)
	lstm_shape = (args.nb_frames, agent.input_shape[-1] // args.nb_frames)
	agent.compile(
		p_network=mlp([64, 64], tf.nn.leaky_relu),
		#p_network=lstm_mlp(64, lstm_shape, [64], activation=tf.nn.leaky_relu, layer_norm=False),
		#p_network=lstm(100, lstm_shape),
		batch_size=(args.nsteps * args.num_envs) // args.nb_batches,
		epochs=args.epochs,
		lr=1e-4,
		end_lr=1e-6,
		ent_coef=0.0,
		vf_coef=1.,
		decay_steps=1000,  # 150,  ## args.nb_timesteps / (args.nsteps * args.num_envs)
		max_grad_norm=None,
		shared=False,
		summ_dir=args.summary_dir)

	if not args.test and not OffReservationGridEnv.EXPORT_RESULTS:
		if args.weights is not None and args.continue_learning:
			agent.load(args.weights)

		if args.weights is not None and args.load_vf:
			agent.load_value(args.weights)

		history = agent.fit(
			clip_value=.2,
			lam=args.lam,
			timesteps=args.nb_timesteps,
			nsteps=args.nsteps,
			num_envs=args.num_envs,
			gamma=args.discount,
			log_interval=args.log_interval,
			loggers=loggers,
			nb_batches=args.nb_batches)

		if args.weights is not None:
			agent.save(args.weights)
	else:
		agent.load(args.weights)

	# plot_hist([(hist['score'], 'score1')], 100, 'Scores', with_error=True)
	# plot_hist([(hist['policy_loss'], 'score1')], 100, 'Policy Loss', with_error=True)
	# plot_hist([(hist['value_loss'], 'score1')], 100, 'Value Loss', with_error=True)
	# plot_hist([(hist['entropy'], 'score1')], 100, 'Entropy', with_error=True)

	ep = agent.play(render=args.render, verbose=args.verbose, n=args.nb_workloads)

	if OffReservationGridEnv.EXPORT_RESULTS:
		fig = plot_simulation_graphics("GridGym/gridgym/envs/simulator/files/output/")
		#fig.savefig("../results/20_drl_generic.png")
		#bench = pd.read_csv("../results/benchmark.csv")
		#ep['policy'] = 'drl_generic'
		#ep['load'] = '20'
		#hist = pd.DataFrame(ep)#.to_csv("../../../results/benchmark.csv", index=False)
		#bench = pd.concat([hist, bench], ignore_index=True)
		#bench.to_csv("../results/benchmark.csv", index=False)

	ep.pop('workload', None)
	print("[EVALUATE][INFO] {}".format(" ".join([" [{}: {}]".format(k, np.mean(v)) for k, v in sorted(ep.items())])))


# if args.output_fn is not None and results:
#	pd.DataFrame([results]).to_csv(args.output_fn, index=False)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_id", default="off_reservation-v0", type=str)
	parser.add_argument("--weights", default="../weights/ppo_value2", type=str)
	parser.add_argument("--output_fn", default="../weights/ppo_results2.csv", type=str)
	parser.add_argument("--log_dir", default="../weights/", type=str)
	parser.add_argument("--summary_dir", default="../weights/summaries/t", type=str)
	parser.add_argument("--seed", default=123, type=int)
	parser.add_argument("--nb_workloads", default=1, type=int)
	parser.add_argument("--nb_batches", default=9, type=int)
	parser.add_argument("--nb_timesteps", default=10e6, type=int)
	parser.add_argument("--nb_frames", default=10, type=int)
	parser.add_argument("--nsteps", default=540,  type=int)
	parser.add_argument("--num_envs", default=12, type=int)
	parser.add_argument("--epochs", default=9,  type=int)
	parser.add_argument("--discount", default=.98, type=float)
	parser.add_argument("--lam", default=.99, type=float)
	parser.add_argument("--log_interval", default=1,  type=int)
	parser.add_argument("--verbose", default=True, action="store_true")
	parser.add_argument("--render", default=0, action="store_true")
	parser.add_argument("--test", default=0, action="store_true")
	parser.add_argument("--continue_learning", default=False, action="store_true")
	parser.add_argument("--load_vf", default=False, action="store_true")
	return parser.parse_args()


if __name__ == "__main__":
	run(parse_args())
