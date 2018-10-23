from gym_grid.envs.grid_env import GridEnv
from collections import deque
from policies import Tetris, SJF, LJF, Random, FirstFit, User, Packer
from shutil import copyfile
from plotly.offline import plot
from multiprocessing import Process, Manager
from collections import defaultdict
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import os
import utils
import shutil
import time as t
import gym
import csv
import numpy as np
import pandas as pd
import ppo2 as ppo
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
BATSIM_WORKLOAD = "src/gym_grid/envs/batsim/files/workloads"
PYTHON = "python"

def run_experiment(policy, metrics, results, episodes=1, verbose=False, visualize=False):
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
    env = gym.make('grid-v0')
    ep_results = defaultdict(float)
    for _ in range(episodes):
        get_trajectory(env, policy, metrics, ep_results, visualize=visualize)

    for k in ep_results.keys():
        ep_results[k] = ep_results[k] / episodes

    if verbose:
        utils.print_episode_result(policy_name, ep_results)

    results[policy_name] = ep_results


def exec_heuristics(policies, metrics, episodes, verbose=False):
    manager = Manager()
    process = []
    manager_result = manager.dict()
    for policy in policies:
        p = Process(target=run_experiment, args=(
            policy, metrics, manager_result, episodes, verbose, False, ))
        p.start()
        process.append(p)

    for p in process:
        p.join()

    return manager_result

def load_workloads(workloads_path):
    nb_workloads = 0
    utils.overwrite_dir(BATSIM_WORKLOAD)
    for w in os.listdir(workloads_path):
        if w.endswith('.json'):
            file = "/{}".format(w)
            copyfile(workloads_path+file, BATSIM_WORKLOAD + file)
            nb_workloads += 1
    return nb_workloads

def train_ppo(workload_path, timesteps, save_path, log_fn):
    load_workloads(workload_path)
    args = "--num_timesteps {} --save_path {}".format(int(timesteps), save_path)
    cmd = "{} src/ppo2.py {} > {}".format(PYTHON, args, log_fn)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()

def run_benchmark(workload_path, weight_fn, policies, metrics, output_fn):    
    def get_ppo_results(weight_fn, metrics, episodes, output_dir, workload_name, results):
        result_fn = output_dir+"/ppo_results.csv"
        args = "--load_path {} --test --test_outputfn {} --test_epi {}".format(weight_fn, result_fn, episodes)        
        cmd = "{} src/ppo2.py {}".format(PYTHON, args)

        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()

        ppo_result = pd.read_csv(result_fn).to_dict(orient='records')[0]
        results['policy'].append("PPO")
        results['workload'].append(workload_name)
        for metric, value in ppo_result.items():
            results[metric].append(value)

    def get_heur_results(policies, metrics, episodes, workload_name, results):
        heur_results = exec_heuristics(policies, metrics, episodes)

        for pol, res in heur_results.items():
            results['policy'].append(pol)
            results['workload'].append(workload_name)
            for met, val in res.items():
                results[met].append(val)

    results = defaultdict(list)
    #workloads = [t for t in os.listdir(workloads_path) if os.path.isdir(os.path.join(workloads_path, t))]
    nb_workloads = load_workloads(workload_path)

    print("*** BENCHMARK *** Starting")
    start_time = t.time()
    #for workload in workloads:
        #workload_path = os.path.join(workloads_path, workload)

    print("*** PPO *** TEST *** START ***")
    get_ppo_results(weight_fn, metrics, nb_workloads, workload_path, 1, results)
    print("*** PPO *** TEST *** END ***")

    print("*** HEU *** TEST *** START ***")
    get_heur_results(policies, metrics, nb_workloads, 1, results)
    print("*** HEU *** TEST *** END ***")

    print("*** BENCHMARK *** Done in {} s.".format(round(t.time() - start_time, 4)))
    dt = pd.DataFrame.from_dict(results)
    dt.to_csv(output_fn, index=False)


if __name__ == "__main__":
    metrics = ['total_slowdown', 'makespan','energy_consumed', 'mean_slowdown','total_turnaround_time','total_waiting_time']
    test_workloads = 'Benchmark/test'
    train_workloads = 'Benchmark/train'
    weight_fn = "Benchmark/ppo.hdf5"
    log_fn = "Benchmark/log.txt"
    output_fn = "Benchmark/benchmark_results.csv"

    policies = [Random(), Tetris(), SJF(), Packer()]
    utils.overwrite_dir('results')
    results = defaultdict(list)

    train_ppo(train_workloads, 60e6, weight_fn, log_fn)

    run_benchmark(test_workloads, weight_fn, policies, metrics, output_fn)
    #run_experiment(Random(), metrics,{}, episodes=100, verbose=True, visualize=False)
