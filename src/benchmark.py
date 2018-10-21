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

def run_experiment(policy, metrics, results, verbose=False, visualize=False):
    def get_trajectory(env, policy, metrics, visualize=False, nb_steps=None):
        result = defaultdict(float)
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
                    result[metric] = info[metric]
                result['score'] = score
                result['steps'] = steps
                return result

    policy_name = policy.__class__.__name__
    env = gym.make('grid-v0')

    traj_result = get_trajectory(env, policy, metrics, visualize)
    traj_result['Episode'] = 1
    if verbose:
        utils.print_episode_result(policy_name, traj_result)

    results[policy_name] = traj_result


def exec_heuristics(policies, metrics, verbose=False):
    manager = Manager()
    process = []
    manager_result = manager.dict()
    for policy in policies:
        p = Process(target=run_experiment, args=(
            policy, metrics, manager_result, verbose, False, ))
        p.start()
        process.append(p)

    for p in process:
        p.join()

    return manager_result


def exec_ppo(timesteps, metrics, output_dir):
    def train(timesteps, weight_fn, log):
        args = "--num_timesteps {} --save_path {}".format(int(timesteps), weight_fn)
        cmd = "{} src/ppo2.py {} > {}".format(PYTHON, args, log)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()

    def test(weight_fn, output_fn):
        args = "--save_path {} --test --test_outputfn {}".format(weight_fn, output_fn)
        cmd = "{} src/ppo2.py {}".format(PYTHON, args)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()

    weight_fn = output_dir+"/ppo.hdf5"
    result_fn = output_dir+"/ppo_results.csv"
    log_fn = output_dir+"/log"

    train(timesteps, weight_fn, log_fn)
    test(weight_fn, result_fn)
    return pd.read_csv(result_fn).to_dict(orient='records')[0]


def run_benchmark(input_dir, timesteps, metrics, policies):
    def load_workloads(workloads_path):
        utils.overwrite_dir(BATSIM_WORKLOAD)
        for w in os.listdir(workloads_path):
            if w.endswith('.json'):
                file = "/{}".format(w)
                copyfile(workloads_path+file, BATSIM_WORKLOAD + file)

    def get_ppo_results(timesteps, metrics, output_dir, test_name, workload_name, results):
        ppo_result = exec_ppo(timesteps, metrics, output_dir)
        results['policy'].append("PPO")
        results['workload'].append(workload_name)
        results['test'].append(test_name)
        for metric, value in ppo_result.items():
            results[metric].append(value)

    def get_heur_results(policies, metrics, test_name, workload_name, results):
        heur_results = exec_heuristics(policies, metrics)

        for pol, res in heur_results.items():
            results['policy'].append(pol)
            results['workload'].append(workload_name)
            results['test'].append(test_name)
            for met, val in res.items():
                results[met].append(val)

    results = defaultdict(list)
    tests = [t for t in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, t))]
    print("*** BENCHMARK *** Starting")
    start_time = t.time()
    for test in tests:
        workloads_path = os.path.join(input_dir, test, "workloads")
        workloads = os.listdir(workloads_path)
        assert len(workloads) == len(timesteps)
        for i, workload in enumerate(workloads):
            workload_path = os.path.join(workloads_path, workload)
            load_workloads(workload_path)

            print("*** PPO *** test {} - workload {} *** START ***".format(test, workload))
            get_ppo_results(timesteps[i], metrics, workload_path, test, workload, results)
            print("*** PPO *** test {} - workload {} *** END ***".format(test, workload))

            print("*** HEURISTIC *** test {} - workload {}  *** START ***".format(test, workload))
            get_heur_results(policies, metrics, test, workload, results)
            print("*** HEURISTIC *** test {} - workload {}  *** END ***".format(test, workload))

    print("*** BENCHMARK *** Done in {} s.".format(round(t.time() - start_time, 4)))
    dt = pd.DataFrame.from_dict(results)
    dt.to_csv(input_dir+"/benchmark.csv", index=False)


if __name__ == "__main__":
    metrics = ['total_slowdown', 'makespan','energy_consumed', 'mean_slowdown','total_turnaround_time','total_waiting_time']
    timesteps = [5e5, 1e6, 2e6, 4e6, 7e6, 9e6, 11e6, 13e6, 15e6, 19e6]
    input_dir = 'Benchmark'
    policies = [FirstFit(), Tetris(), SJF(), Random(), Packer()]
    utils.overwrite_dir('results')

    run_benchmark(input_dir, timesteps, metrics, policies)
    #exec_heuristics(policies, metrics, True)
    #run_experiment(Tetris(), metrics,{}, verbose=True, visualize=True)
