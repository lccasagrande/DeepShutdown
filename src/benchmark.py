from gym_grid.envs.grid_env import *
from policies import Tetris, SJF, LJF, Random, FirstFit, User, Packer
from shutil import copyfile
from multiprocessing import Process, Manager
from collections import defaultdict
import os
from main import run, parse_args
from utils.common import overwrite_dir, print_episode_result
import time as t
import gym
import pandas as pd
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BATSIM_WORKLOAD = "src/gym_grid/envs/batsim/files/workloads"
BATSIM_PLATFORM = "src/gym_grid/envs/batsim/files/platforms/platform_hg_10.xml"
PYTHON = "python"


def run_experiment(env_type, policy, metrics, results, episodes=1, verbose=False, visualize=False):
	def get_trajectory(env, policy, metrics, results, visualize=False, nb_steps=None):
		score, steps, state = 0, 0, env.reset()
		while True:
			if visualize:
				env.render()

			action = policy.select_action(state)
			state, reward, done, info = env.step(action)

			steps += 1
			score += reward

			if done or (steps == nb_steps if nb_steps != None else False):
				for metric in metrics:
					results[metric] += info[metric]
				results['score'] += score
				results['steps'] += steps
				return

	policy_name = policy.__class__.__name__
	env = gym.make(env_type)
	ep_results = defaultdict(float)
	for _ in range(episodes):
		get_trajectory(env, policy, metrics, ep_results, visualize=visualize)

	for k in ep_results.keys():
		ep_results[k] = ep_results[k] / episodes

	if verbose:
		print_episode_result(policy_name, ep_results)

	results[policy_name] = ep_results


def exec_heuristics(env_type, policies, metrics, episodes, verbose=False):
	manager = Manager()
	process = []
	manager_result = manager.dict()
	for policy in policies:
		p = Process(target=run_experiment, args=(env_type, policy,
		                                         metrics, manager_result, episodes, verbose, False,))
		p.start()
		process.append(p)

	for p in process:
		p.join()

	return manager_result


def load_workloads(workloads_path):
	nb_workloads = 0
	names = []
	overwrite_dir(BATSIM_WORKLOAD)
	for w in os.listdir(workloads_path):
		if w.endswith('.json'):
			file = "/{}".format(w)
			copyfile(workloads_path + file, BATSIM_WORKLOAD + file)
			nb_workloads += 1
			names.append(w)
	return nb_workloads, names

def exec_reinforce(env_type, weight_fn):
	args = parse_args()
	args.env = env_type
	args.save_path = weight_fn
	args.test = True
	return run(args)

def train_reinforce(workload_path, timesteps, save_path, name, log_fn):
	load_workloads(workload_path)
	args = "--nb_iteration {} --save_path {} --name {} --save_interval {} --log_interval {}".format(int(timesteps), save_path, name, int(timesteps), int(timesteps))
	cmd = "{} src/main.py {} > {}".format(PYTHON, args, log_fn)
	process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
	process.wait()


def exec(model, env_type, weight_fn, episodes, output_dir):
	result_fn = output_dir + "/a2c_results.csv"
	args = "--model {} --env {} --load_path {} --test --test_outputfn {} --test_epi {}".format(model, env_type,
	                                                                                           weight_fn,
	                                                                                           result_fn,
	                                                                                           episodes)
	cmd = "{} src/ppo2.py {}".format(PYTHON, args)

	process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
	process.wait()

	return pd.read_csv(result_fn).to_dict(orient='records')[0]


def train(model, workload_path, timesteps, save_path, log_fn):
	load_workloads(workload_path)
	args = "--model {} --num_timesteps {} --save_path {}".format(model, int(timesteps), save_path)
	cmd = "{} src/ppo2.py {} > {}".format(PYTHON, args, log_fn)
	os.environ['OPENAI_LOGDIR'] = os.path.abspath(
		os.path.join(os.path.dirname(__file__), "../" + workload_path + "/" + model))
	process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
	process.wait()


def exec_batsched(output_dir, policy, workload_path):
	def run_batsim(output):
		cmd = "batsim -p {} -w {} -v {} -E -e {}".format(BATSIM_PLATFORM,
		                                                 workload_path,
		                                                 "quiet",
		                                                 output)

		return subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=False)

	def run_batsched():
		cmd = "batsched -d 1 -v {}".format(policy)
		return subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=False)

	output = output_dir + "/" + policy
	batsim = run_batsim(output)
	batsched = run_batsched()
	batsim.wait()
	return pd.read_csv(output + "_schedule.csv").to_dict(orient='records')[0]


def run_benchmark(env_type, workload_path, weight_fn, policies, metrics, output_fn):
	def get_ppo_results(env_type, weight_fn, metrics, episodes, output_dir, workload_name, results):
		result = exec("PPO", env_type, weight_fn, episodes, output_dir)
		results['policy'].append("PPO")
		results['workload'].append(workload_name)
		for metric, value in result.items():
			results[metric].append(value)

	def get_a2c_results(env_type, weight_fn, metrics, episodes, output_dir, workload_name, results):
		result = exec("A2C", env_type, weight_fn, episodes, output_dir)
		results['policy'].append("A2C")
		results['workload'].append(workload_name)
		for metric, value in result.items():
			results[metric].append(value)

	def get_reinforce_results(env_type, weight_fn, workload_name, results):
		result = exec_reinforce(env_type, weight_fn)
		results['policy'].append("REINFORCE")
		results['workload'].append(workload_name)
		for metric, value in result.items():
			results[metric].append(value)

	def get_acer_results(env_type, weight_fn, metrics, episodes, output_dir, workload_name, results):
		result = exec("ACER", env_type, weight_fn, episodes, output_dir)
		results['policy'].append("ACER")
		results['workload'].append(workload_name)
		for metric, value in result.items():
			results[metric].append(value)

	def get_heur_results(env_type, policies, metrics, episodes, workload_name, results):
		heur_results = exec_heuristics(env_type, policies, metrics, episodes)

		for pol, res in heur_results.items():
			results['policy'].append(pol)
			results['workload'].append(workload_name)
			for met, val in res.items():
				results[met].append(val)

	def get_batsched_results(output_dir, policies, metrics, workload_path, workload_name, results):
		tmp_results = {k: 0 for k in metrics}
		for p in policies:
			bat_results = exec_batsched(output_dir, p, workload_path)

			for k, value in bat_results.items():
				if k in tmp_results:
					tmp_results[k] = value

			results['policy'].append(p)
			results['workload'].append(workload_name)
			results['score'].append(0)
			results['steps'].append(0)
			for met, val in tmp_results.items():
				results[met].append(val)

	results = defaultdict(list)
	nb_workloads, names = load_workloads(workload_path)

	print("*** BENCHMARK *** Starting")
	start_time = t.time()

	# print("*** BATSCHED *** TEST *** START ***")
	# get_batsched_results(workload_path, ['easy_bf'], metrics, workload_path + "/" + names[0], 1, results)
	# print("*** BATSCHED *** TEST *** END ***")

	# print("*** A2C *** TEST *** START ***")
	# get_a2c_results(env_type, weight_fn, metrics, nb_workloads, workload_path, 1, results)
	# print("*** A2C *** TEST *** END ***")

	#print("*** ACER *** TEST *** START ***")
	#get_acer_results(env_type, weight_fn, metrics, nb_workloads, workload_path, 1, results)
	#print("*** ACER *** TEST *** END ***")

	print("*** ACER *** TEST *** START ***")
	get_reinforce_results(env_type, weight_fn, 1, results)
	print("*** ACER *** TEST *** END ***")

	# print("*** PPO *** TEST *** START ***")
	# get_ppo_results(env_type, weight_fn, metrics,
	#                nb_workloads, workload_path, 1, results)
	# print("*** PPO *** TEST *** END ***")

	# print("*** HEU *** TEST *** START ***")
	# get_heur_results(env_type, policies, metrics, nb_workloads, 1, results)
	# print("*** HEU *** TEST *** END ***")

	print("*** BENCHMARK *** Done in {} s.".format(round(t.time() - start_time, 4)))
	dt = pd.DataFrame.from_dict(results)
	dt.to_csv(output_fn, index=False)


def run_battery(env_type, workloads_path, train_steps, policies, metrics, output):
	workloads = [t for t in os.listdir(workloads_path) if os.path.isdir(os.path.join(workloads_path, t))]
	for workload in workloads:
		workload_path = os.path.join(workloads_path, workload)
		output_fn = "{}/{}_{}.csv".format(output, "benchmark", workload)
		#weight_fn = "{}/{}_{}.hdf5".format(output, "acer", workload)
		log_fn = "{}/{}_{}.txt".format(output, "reinforce_log", workload)
		# train("PPO", workload_path, train_steps[workload], weight_fn, log_fn)
		# train("A2C", workload_path, train_steps[workload], weight_fn, log_fn)
		#train("ACER", workload_path, train_steps[workload], weight_fn, log_fn)
		train_reinforce(workload_path, train_steps[workload], workload_path, "reinforce_"+workload, log_fn)

		run_benchmark(env_type, workload_path, workload_path, policies, metrics, output_fn)


if __name__ == "__main__":
	metrics = ['makespan', 'mean_slowdown', 'total_slowdown', 'total_turnaround_time', 'mean_turnaround_time',
	           'total_waiting_time', 'mean_waiting_time', 'max_waiting_time', 'max_turnaround_time', 'max_slowdown']
	workloads = 'Benchmark/workloads'
	output = "Benchmark"
	#train_steps = {"10": 5e5, "40": 1e6, "70": 1.5e6, "100": 2.5e6, "130": 5e6, "160": 7e6, "190": 9e6}
	train_steps = {"10": 3e2, "40": 3e3, "70": 4e3, "100": 5e3, "130": 6e3, "160": 7e3, "190": 8e3}
	policies = [Random(), Tetris(), SJF(), Packer()]
	eval_env = "batsim-v0"
	train_env = "grid-v0"

	# exec_ppo(eval_env, "Benchmark/ppo_190.hdf5", 1, "tmp")
	run_battery(eval_env, workloads, train_steps, policies, metrics, output)
# exec_ppo("weights/ppo_.4.hdf5", 1, "tmp")
# run_experiment(eval_env, SJF(), [], dict(), episodes=1, verbose=True, visualize=False)
# run_experiment(Random(), metrics,{}, episodes=100, verbose=True, visualize=False)
