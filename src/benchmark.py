from gym_grid.envs.grid_env import GridEnv
from collections import deque
from policies import Tetris, SJF, LJF, Random, FirstFit, User
import utils
import shutil
import gym
import csv
import numpy as np
import pandas as pd

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import plot
from multiprocessing import Process, Manager
from collections import defaultdict


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
    result = defaultdict(list)
    env = gym.make('grid-v0')
    env.seed(seed)

    for i in range(1, n_ep + 1):
        traj_result = get_trajectory(env, policy, metrics, visualize)
        traj_result['Episode'] = i

        for k, v in traj_result.items():
            result[k].append(v)

        if verbose:
            utils.print_episode_result(policy_name, traj_result)
    results[policy_name] = result


def run(output_dir, policies, n_ep, seed, metrics=[], plot=True, verbose=True, visualize=False):
    np.random.seed(seed)
    manager = Manager()
    manager_result = manager.dict()
    process = []

    for policy in policies:
        p = Process(target=run_experiment, args=(
            policy, n_ep, seed, metrics, manager_result, verbose, visualize, ))
        p.start()
        process.append(p)

    for p in process:
        p.join()

    # PRINT
    for metric in metrics:
        tmp = dict()
        for key, value in manager_result.items():
            tmp[key] = value[metric]

        if plot:
            plot_results(tmp, metric)

        dt = pd.DataFrame.from_dict(tmp)
        dt.to_csv(output_dir+metric+".csv", index=False)


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


if __name__ == "__main__":
    metrics = ['total_slowdown', 'makespan', 'energy_consumed']
    output_dir = 'benchmark/'
    policies = [FirstFit(), Tetris(), SJF(), LJF()]
    n_episodes = 1
    seed = 123
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    utils.create_dir(output_dir)
    utils.create_dir('results')
    #run_experiment(SJF(), n_episodes, seed, metrics, {}, verbose=False, visualize=False)
    run(output_dir, policies, n_episodes, seed, metrics=metrics, plot=False, verbose=True)


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
