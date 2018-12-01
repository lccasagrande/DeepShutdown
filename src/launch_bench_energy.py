import GridGym.gridgym.envs.grid_env
from src.agents.heuristics import FirstFitAgent,PackerAgent
from shutil import copyfile
from multiprocessing import Process, Manager
from collections import defaultdict
import os
from utils.common import overwrite_dir, print_episode_result
import time as t
import gym
import pandas as pd
import subprocess

BATSIM_ENV_OUTPUT = "results/bat_1_schedule.csv"
BATSIM_WORKLOAD = "src/GridGym/gridgym/envs/batsim/files/workloads"
BATSIM_PLATFORM = "src/GridGym/gridgym/envs/batsim/files/platforms/platform_hg_100.xml"
PYTHON = "python"


def run_experiment(env_type, policy, results, verbose=False):
	def get_trajectory(env, policy):
		score, steps, state = 0, 0, env.reset()
		while True:
			action = policy.act(state)
			state, reward, done, info = env.step(action)

			steps += 1
			score += reward

			if done:
				return

	policy_name = policy.__class__.__name__
	env = gym.make(env_type)
	#ep_results = defaultdict(float)

	get_trajectory(env, policy)

	ep_results = pd.read_csv(BATSIM_ENV_OUTPUT).to_dict(orient='records')[0]

	if verbose:
		print_episode_result(policy_name, ep_results)

	results[policy_name] = ep_results


def exec_heuristics(env_type, policies, verbose=False):
	#manager = Manager()
	#process = []
	#manager_result = manager.dict()
	results = dict()
	for policy in policies:
		run_experiment(env_type, policy, results, verbose)
		#p = Process(target=run_experiment, args=(env_type, policy, manager_result, episodes, verbose,))
		#p.start()
		#process.append(p)

	#for p in process:
	#	p.join()

	return results

def get_workloads(workloads_path):
	workloads = []
	for w in os.listdir(workloads_path):
		if w.endswith('.json'):
			workloads.append(w)
	return workloads

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


def run_benchmark(env_type, workload_path, policies, output_fn):
	def get_heur_results(env_type, policies, workload_name, results):
		heur_results = exec_heuristics(env_type, policies)

		for pol, res in heur_results.items():
			results['policy'].append(pol)
			results['workload'].append(workload_name)
			for met, val in res.items():
				results[met].append(val)

	def get_batsched_results(output_dir, policies, workload_path, workload_name, results):
		for p in policies:
			bat_results = exec_batsched(output_dir, p, workload_path)
			results['policy'].append(p)
			results['workload'].append(workload_name)
			for met, val in bat_results.items():
				results[met].append(val)

	results = defaultdict(list)
	workloads = get_workloads(workload_path)

	print("*** BENCHMARK *** Starting")
	start_time = t.time()

	for workload in workloads:
		overwrite_dir(BATSIM_WORKLOAD)
		copyfile(workload_path + "/{}".format(workload), BATSIM_WORKLOAD + "/{}".format(workload))

		print("*** HEU *** TEST *** START ***")
		get_heur_results(env_type, policies, workload, results)
		print("*** HEU *** TEST *** END ***")

		print("*** BATSCHED *** TEST *** START ***")
		get_batsched_results(workload_path, ['easy_bf', 'conservative_bf'], BATSIM_WORKLOAD + "/{}".format(workload), workload, results)
		print("*** BATSCHED *** TEST *** END ***")

		print("*** BENCHMARK *** Done in {} s.".format(round(t.time() - start_time, 4)))
	dt = pd.DataFrame.from_dict(results)
	dt.to_csv(output_fn, index=False)


def run_battery(env_type, workloads_path, policies, output):
	workloads = [t for t in os.listdir(workloads_path) if os.path.isdir(os.path.join(workloads_path, t))]
	for workload in workloads:
		workload_path = os.path.join(workloads_path, workload)
		output_fn = "{}/{}_{}.csv".format(output, "benchmark", workload)
		run_benchmark(env_type, workload_path, policies, output_fn)


if __name__ == "__main__":
	workloads = 'EnergyBench/workloads'
	output = "EnergyBench"
	policies = [PackerAgent("",123), FirstFitAgent("",123)]
	env = "batsim-v0"

	# exec_ppo(eval_env, "Benchmark/ppo_190.hdf5", 1, "tmp")
	run_battery(env, workloads, policies, output)
# exec_ppo("weights/ppo_.4.hdf5", 1, "tmp")
# run_experiment(eval_env, SJF(), [], dict(), episodes=1, verbose=True, visualize=False)
# run_experiment(Random(), metrics,{}, episodes=100, verbose=True, visualize=False)
