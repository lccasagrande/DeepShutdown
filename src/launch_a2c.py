import tensorflow as tf
import argparse
import numpy as np
import gym
import plotly.graph_objs as go
from plotly.offline import plot
from src.utils import loggers as log
from src.utils.networks import mlp
from src.agents.a2c import A2CAgent
import gridgym.envs.grid_env as g


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
	layers = [64, 64, 64]
	activations = [tf.nn.relu] * len(layers)
	LR = 5e-4

	loggers = log.LoggerWrapper()
	if args.log_interval != 0:
		loggers.append(log.JSONLogger(args.log_dir + "/log.json"))
	if args.verbose:
		loggers.append(log.ConsoleLogger())

	env = gym.make(args.env_id)
	nb_actions = env.action_space.n
	input_shape = env.observation_space.shape
	agent = A2CAgent(args.env_id, input_shape, nb_actions, mlp(layers, activations))

	if args.test:
		agent.compile()
		agent.load(args.weights)
		agent.play(render=args.render, verbose=args.verbose)
	else:
		agent.compile(lr=LR, ent_coef=.01)
		hist = agent.fit(
			timesteps=args.nb_timesteps,
			nsteps=args.nsteps,
			num_envs=args.num_envs,
			discount=args.discount,
			log_interval=args.log_interval,
			loggers=loggers)

		agent.play(render=args.render, verbose=args.verbose)
		agent.save(args.weights)
		# plot_hist([(hist['score'], 'score1')], 100, 'Scores', with_error=True)
		# plot_hist([(hist['policy_loss'], 'score1')], 100, 'Policy Loss', with_error=True)
		# plot_hist([(hist['value_loss'], 'score1')], 100, 'Value Loss', with_error=True)
		# plot_hist([(hist['entropy'], 'score1')], 100, 'Entropy', with_error=True)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_id", type=str, default="shutdown-v0")
	parser.add_argument("--num_envs", default=12, type=int)
	parser.add_argument("--weights", default="../weights/a2c_shutdown", type=str)
	parser.add_argument("--log_dir", default="../weights", type=str)
	parser.add_argument("--verbose", default=True, action="store_true")
	parser.add_argument("--render", default=True, action="store_true")
	parser.add_argument("--nb_timesteps", type=int, default=1e6)
	parser.add_argument("--discount", default=.99, action="store_true")
	parser.add_argument("--nsteps", default=50, action="store_true")
	parser.add_argument("--log_interval", default=10, action="store_true")
	parser.add_argument("--test", default=0, action="store_true")
	return parser.parse_args()


if __name__ == "__main__":
	run(parse_args())