import tensorflow as tf
import gym
import gridgym.envs.grid_env as g
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from collections import defaultdict
from plotly.offline import plot
from src.utils import loggers as log
from src.utils.networks import mlp
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

	agent.compile(
		p_network=mlp([64, 64], tf.nn.leaky_relu),
		# v_network=mlp([64, 64], tf.nn.leaky_relu),
		lr=1e-3,
		ent_coef=0.01,
		vf_coef=.25,
		decay_steps=200,
		max_grad_norm=None,
		shared=False)

	if not args.test:
		if args.weights is not None and args.continue_learning:
			agent.load(args.weights)

		if args.v_weights is not None:
			agent.load_value(args.v_weights)

		history = agent.fit(
			clip_value=.3,
			lam=.95,
			timesteps=args.nb_timesteps,
			nsteps=args.nsteps,
			num_envs=args.num_envs,
			gamma=args.discount,
			log_interval=args.log_interval,
			epochs=args.epochs,
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

	hist = defaultdict(list)
	for _ in range(args.nb_workloads):
		ep = agent.play(render=args.render, verbose=args.verbose)
		for k, v in ep.items():
			hist[k].append(v)

	hist.pop('workload', None)
	print("[EVALUATE][INFO] {}".format(" ".join([" [{}: {}]".format(k, np.mean(v)) for k, v in sorted(hist.items())])))


# if args.output_fn is not None and results:
#	pd.DataFrame([results]).to_csv(args.output_fn, index=False)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_id", default="shutdown-v0", type=str)
	# parser.add_argument("--weights", default="../results/5/myppo2", type=str)
	# parser.add_argument("--output_fn", default="../results/5/ppo_results.csv", type=str)
	# parser.add_argument("--log_dir", default="../results/5/", type=str)
	parser.add_argument("--weights", default="../weights/myppo2", type=str)
	parser.add_argument("--output_fn", default="../weights/ppo_results2.csv", type=str)
	parser.add_argument("--log_dir", default="../weights/", type=str)
	parser.add_argument("--v_weights", default=None, type=str)
	parser.add_argument("--seed", default=123, type=int)
	parser.add_argument("--nb_timesteps", default=1e6, type=int)
	parser.add_argument("--nsteps", default=256, action="store_true")
	parser.add_argument("--nb_frames", default=15, type=int)
	parser.add_argument("--num_envs", default=12, type=int)
	parser.add_argument("--epochs", default=8, action="store_true")
	parser.add_argument("--discount", default=.99, action="store_true")
	parser.add_argument("--nb_batches", default=4, action="store_true")
	parser.add_argument("--log_interval", default=1, action="store_true")
	parser.add_argument("--verbose", default=True, action="store_true")
	parser.add_argument("--render", default=0, action="store_true")
	parser.add_argument("--test", default=0, action="store_true")
	parser.add_argument("--nb_workloads", default=1, action="store_true")
	parser.add_argument("--continue_learning", default=False, action="store_true")
	return parser.parse_args()


if __name__ == "__main__":
	run(parse_args())
