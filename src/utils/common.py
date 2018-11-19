import pandas as pd
import random
import plotly.graph_objs as go
import plotly.tools as tools
import math
import sys
import time as t
import os
import shutil
import numpy as np
import gym
import tensorflow as tf
from matplotlib.colors import XKCD_COLORS as allcolors
from plotly.offline import plot
from .env_wrappers import SubprocVecEnv


def print_episode_result(name, result):
	msg = name
	for k, value in result.items():
		if isinstance(value, list):
			msg += " {} [{}] ".format(k, value[-1])
		else:
			msg += " {} [{}] ".format(k, value)
	print(msg)


class LinearAnnealEpsGreedy():
	def __init__(self, value_max, value_min, nb_steps):
		self.value_max = value_max
		self.value_min = value_min
		self.nb_steps = nb_steps

	def get_current_value(self, step):
		a = -float(self.value_max - self.value_min) / float(self.nb_steps)
		b = float(self.value_max)
		return max(self.value_min, a * float(step) + b)


def overwrite_dir(dir):
	if os.path.exists(dir):
		shutil.rmtree(dir)
	create_dir(dir)


def create_dir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)


def export_rewards(n_ep, avg_scores, output_dir):
	x, y = np.linspace(0, n_ep, len(avg_scores),
	                   endpoint=False), np.asarray(avg_scores)
	dt = pd.DataFrame({'ep': list(x), 'avg_reward': list(y)})
	dt.to_csv(output_dir + 'rewards.csv', index=False)


def export_max_q_values(Q, output_dir):
	hist = {}
	for k, value in Q.items():
		actions = [i for i, act in enumerate(value) if act == np.amax(value)]
		actions = "-".join(str(q) for q in actions)
		hist[k] = actions

	q_max_values = pd.DataFrame.from_dict(hist, orient='index')
	q_max_values.index.name = "state"
	q_max_values.to_csv(output_dir + "max_q_values.csv")


def dict_to_csv(data, fn):
	dt = pd.DataFrame(data)
	dt.to_csv(fn, index=False)


def plot_reward(avg_scores, n_ep, title, output_dir):
	x, y = np.linspace(0, n_ep, len(avg_scores),
	                   endpoint=False), np.asarray(avg_scores)

	dict_to_csv({'ep': list(x), 'avg_reward': list(y)},
	            output_dir + 'rewards.csv')

	trace = go.Scatter(x=x, y=y)
	layout = go.Layout(
		title=title,
		xaxis={'title': 'Episode'},
		yaxis={'title': 'Avg Reward'},
	)
	fig = go.Figure(data=[trace], layout=layout)
	plot(fig, filename=output_dir + 'rewards.html')


def print_progress(t_start, ep, num_episodes):
	total_time = t.time() - t_start
	avg_ep_time = round(total_time / ep, 2)
	rem_time = round(avg_ep_time * (num_episodes - ep), 2)
	print("\nEpisode {}/{}: (Avg:{} s - Remaining: {} s)".format(ep,
	                                                             num_episodes, avg_ep_time, rem_time), flush=True)
	sys.stdout.flush()


def merge_results(path, scheds, output_fn=None):
	results = pd.DataFrame()
	for policy in scheds:
		df = pd.read_csv(
			"{}/{}_schedule.csv".format(path, policy))
		df['policy'] = policy
		results = results.append(df, ignore_index=True)

	if output_fn is not None:
		results.to_csv(output_fn)
	return results


def plot_results(data, metrics, output_fn, main_title, max_cols=3, y_labels=None, colors=None):
	def get_cols_and_rows():
		tot_metrics = len(metrics)
		rows = math.ceil(tot_metrics / max_cols)
		cols = tot_metrics if tot_metrics < max_cols else max_cols
		return rows, cols

	def update_rows_and_cols(row, col):
		if col == max_cols:
			row += 1
			col = 1
		else:
			col += 1
		return row, col

	def calc_diffs(dt):
		min_value = dt.min()
		perc = round(((dt * 100) / min_value) - 100, 2)
		return [str(p) + "%" for p in perc]

	random_colors = colors if colors is not None else random.sample(
		list(allcolors.values()), k=len(data['policy']))
	rows, cols = get_cols_and_rows()
	fig = tools.make_subplots(rows=rows, cols=cols, subplot_titles=metrics)

	row, col = 1, 1
	for metric in metrics:
		graph = go.Bar(
			showlegend=False,
			name=metric,
			text=calc_diffs(data[metric]),
			textposition='auto',
			textfont=dict(size=20, color='#ffffff'),
			x=data['policy'],
			y=data[metric],
			marker={'color': random_colors})

		fig.append_trace(graph, row, col)
		row, col = update_rows_and_cols(row, col)

	if y_labels is not None:
		assert len(y_labels) == len(metrics)
		for i, title in enumerate(y_labels):
			axis_ind = "yaxis" + str(i + 1)
			fig['layout'][axis_ind].update(title=title)
	fig['layout'].update(title=main_title)
	plot(fig, filename=output_fn)


def plot_job_stats(data, title, total, metrics, output_fn, colors=None):
	def calc_diffs(dt):
		perc = round(((dt * 100) / total), 2)
		return [str(p) + "%" for p in perc]

	dt = []
	for i, metric in enumerate(metrics):
		color = None if colors is None else colors[i]
		dt.append(
			go.Bar(
				x=data['policy'],
				y=data[metric],
				name=metric,
				text=calc_diffs(data[metric]),
				textfont=dict(size=20, color='#ffffff'),
				marker={'color': color}
			)
		)
	layout = go.Layout(barmode='stack', title=title)
	fig = go.Figure(data=dt, layout=layout)
	plot(fig, filename=output_fn)


def make_vec_env(env_id, nenv, seed):
	def make_env(rank):  # pylint: disable=C0111
		def _thunk():
			env = gym.make(env_id)
			env.seed(seed + rank if seed is not None else None)
			# env = Monitor(env, rank)
			return env

		return _thunk

	return SubprocVecEnv([make_env(i) for i in range(nenv)])


def variable_summaries(var, family):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean, family=family)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev, family=family)
		tf.summary.scalar('max', tf.reduce_max(var), family=family)
		tf.summary.scalar('min', tf.reduce_min(var), family=family)
		tf.summary.histogram('histogram', var, family=family)


def discount(rewards, gamma):
	steps, retur = len(rewards), 0
	discounted_rewads = np.zeros(steps)
	for i in reversed(range(steps)):
		retur = rewards[i] + gamma * retur
		discounted_rewads[i] = retur
	return discounted_rewads


def normalize(dt):
	avg = dt.mean()
	std = dt.std()
	return (dt - avg) / std if std != 0 else dt


def get_avail_res_from_img2(state):
	res = state[0, 0:-2]
	free = 0
	for i in range(1, res.shape[0], 2):
		if res[i] == 0:
			free += 1
	return free


def get_jobs_from_img2(state):
	slot, jobs = 1, []
	jobs_state = state[:, 10:12]
	for j in range(jobs_state.shape[0]):
		if jobs_state[j, 0] != 0:
			res = jobs_state[j, 0] * 10
			time = jobs_state[j, 1] * 20
			jobs.append((res, time, slot))
		slot += 1
	return jobs, slot



def get_avail_res_from_img(state):
	res = state[0, 0:10]
	occuped = np.count_nonzero(res)
	return len(res) - occuped


def get_jobs_from_img(state):
	slot = 1
	jobs_state = state[:, 10:110]
	jobs = []
	for i in range(0, 100, 10):
		if jobs_state[0, i] != 0:
			res = np.count_nonzero(jobs_state[0, i:i + 10])
			time = np.count_nonzero(jobs_state[:, i])
			jobs.append((res, time, slot))
		slot += 1
	return jobs, slot
