from gym_grid.envs.grid_env import GridEnv
from collections import deque
from policies import Tetris, SJF, LJF, Random, FirstFit, User
from shutil import copyfile
from plotly.offline import plot
from multiprocessing import Process, Manager
from collections import defaultdict
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import utils
import shutil
import gym
import csv
import numpy as np
import pandas as pd
import ppo2 as ppo
import subprocess


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


def run_experiment(policy, n_ep, seed, metrics, results, verbose=False, visualize=False):
    policy_name = policy.__class__.__name__
    env = gym.make('grid-v0')
    env.seed(seed)

    traj_result = get_trajectory(env, policy, metrics, visualize)
    traj_result['Episode'] = 1
    if verbose:
        utils.print_episode_result(policy_name, traj_result)

    results[policy_name] = traj_result


def run_heuristics(output_dir, policies, n_ep, seed, results, workloads, metrics=[], plot=True, verbose=False, visualize=False):
    np.random.seed(seed)
    manager = Manager()
    for w in workloads:
        process = []
        manager_result = manager.dict()
        load_workload(w)

        for policy in policies:
            p = Process(target=run_experiment, args=(
                policy, n_ep, seed, metrics, manager_result, verbose, visualize, ))
            p.start()
            process.append(p)

        for p in process:
            p.join()

        # PRINT
        for pol, res in manager_result.items():
            results['policy'].append(pol)
            results['workload'].append(w)
            for met, val in res.items():
                results[met].append(val)


def plot_results(data, name):
    def get_bar_plot(dt):
        traces = []
        for policy, values in dt.items():
            traces.append(go.Bar(
                x=[0],  # list(range(len(values))),
                y=[np.mean(values)],
                text=[np.mean(values)],
                textposition='auto',
                name=policy
            ))
        return traces

    fig = go.Figure(data=get_bar_plot(data), layout=go.Layout(title=name))
    plot(fig, filename=name+'.html')


def load_workload(workload):
    dest = "src/gym_grid/envs/batsim/files/workload"
    shutil.rmtree(dest)
    utils.create_dir(dest)
    w_path = "src/gym_grid/envs/batsim/files/{}/{}.json".format(
        workload, workload)
    copyfile(w_path, dest + "/{}.json".format(workload))


def run_ppo(output_dir, workloads, timesteps, metrics, results):
    def exec_ppo(workload, timesteps, metrics):
        def train(timesteps, weight_fn, log):
            args = "--num_timesteps {} --save_path {}".format(
                int(timesteps), weight_fn)
            cmd = "python src/ppo2.py {} > {}".format(args, log)
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            process.wait()

        def test(weight_fn, output_fn):
            args = "--save_path {} --test --test_outputfn {}".format(
                weight_fn, output_fn)
            cmd = "python src/ppo2.py {}".format(args)
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            process.wait()

        output_dir = "tests/{}".format(workload)
        utils.create_dir(output_dir)
        weight_fn = output_dir+"/ppo.hdf5"
        result_fn = output_dir+"/results.csv"

        train(timesteps, weight_fn, output_dir+"/log")
        test(weight_fn, result_fn)
        return pd.read_csv(result_fn).to_dict(orient='records')[0]

    for i, workload in enumerate(workloads):
        load_workload(workload)
        result = exec_ppo(workload, timesteps[i], metrics)

        result['policy'] = "PPO"
        result['workload'] = workload
        for metric, value in result.items():
            results[metric].append(value)


if __name__ == "__main__":
    metrics = ['total_slowdown', 'makespan','energy_consumed', 'mean_slowdown']
    workloads = ['10']#, '3', '4', '5', '6', '7', '8', '9', '10']
    timesteps = [5e6]#, 2.5e6, 3e6, 3.5e6, 4e6, 6e6, 7e6, 8e6]
    output_dir = 'benchmark/'
    policies = [FirstFit(), Tetris(), SJF(), LJF(), Random()]
    n_episodes = 1
    seed = 123
    results = defaultdict(list)

    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    utils.create_dir(output_dir)
    utils.create_dir('results')

    run_ppo(output_dir, workloads, timesteps, metrics, results)

    run_heuristics(output_dir, policies, n_episodes, seed, results, workloads, metrics=metrics, plot=False, verbose=False)

    dt = pd.DataFrame.from_dict(results)
    dt.to_csv(output_dir+"ppo_benchmark.csv", index=False)
    #run_experiment(SJF(), n_episodes, seed, metrics, {}, verbose=True, visualize=True)


# %%

#import plotly.plotly as py
#import plotly.graph_objs as go
#import plotly.tools as tls
#from plotly.offline import plot
#import numpy as np
#
#
# metrics = ["nb_jobs", "nb_jobs_finished", "nb_jobs_killed", "success_rate", "makespan",
#               "mean_waiting_time", "mean_turnaround_time", "mean_slowdown", "energy_consumed",
#               'total_slowdown', 'total_turnaround_time', 'total_waiting_time']
# def plot_results(data, name):
#    def get_bar_plot(dt):
#        traces = []
#        for policy, values in dt.items():
#            traces.append(go.Bar(
#                x=[0],  # list(range(len(values))),
#                y=[np.mean(values)],
#                text=[np.mean(values)],
#                textposition='auto',
#                name=policy
#            ))
#        return traces
#
#    fig = go.Figure(data=get_bar_plot(data), layout=go.Layout(title=name))
#    plot(fig, filename="results/"+name+'.html')
#
# for metric in metrics:
#
#    dt1 = pd.read_csv("benchmark/"+metric+".csv", index_col=False)
#    dt1.reset_index(drop=True, inplace=True)
#    dt2 = pd.read_csv("benchmark/ppo_"+metric+".csv", index_col=False)
#    dt2.columns = ['PPO2']
#    dt2.reset_index(drop=True, inplace=True)
#    x = pd.concat([dt1, dt2],axis=1)
#    print(x.head(5))
#    x.to_csv("benchmark/all_"+metric+".csv", index=False)
#    plot_results(x, metric)
