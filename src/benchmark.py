from gym_grid.envs.grid_env import GridEnv
from collections import deque
from policies import Tetris, SJF, LJF, Random, FirstFit
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


def run_experiment(policy, n_ep, seed, results):
    policy_name = policy.__class__.__name__
    policy_results = dict(score=[], slowdown=[], makespan=[], energy=[])
    env = gym.make('grid-v0')
    env.seed(seed)

    for i in range(1, n_ep + 1):
        score, state = 0,  env.reset()
        while True:
            act = policy.select_action(state)

            state, reward, done, info = env.step(act)

            score += reward

            if done:
                policy_results['score'].append(score)
                policy_results['slowdown'].append(info['mean_slowdown'])
                policy_results['makespan'].append(info['makespan'])
                policy_results['energy'].append(info['energy_consumed'])
                #print("\n{} - Episode {:7}, Score: {:7} - Slowdown Sum {:7} Mean {:3} - Makespan {:7}"
                #      .format(policy_name, i, score, info['total_slowdown'], info['mean_slowdown'], info['makespan']))
                break

    results[policy_name] = policy_results


def run(output_dir, policies, n_ep, seed):
    np.random.seed(seed)
    manager = Manager()
    manager_result = manager.dict()
    metrics = ['score','slowdown','makespan','energy']
    process = []
    utils.clean_or_create_dir(output_dir)

    for policy in policies:
        p = Process(target=run_experiment, args=(policy, n_ep, seed, manager_result, ))
        p.start()
        process.append(p)

    for p in process:
        p.join()

    # PRINT
    for metric in metrics:
        tmp = dict()
        for key, value in manager_result.items():
            tmp[key] = value[metric]

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
    output_dir = 'benchmark/'
    policies = [FirstFit(), Tetris(), Random(), SJF(), LJF()]
    n_episodes = 100
    seed = 123
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)

    run(output_dir, policies, n_episodes, seed)
    #plot_results(rewards, 'Episode Rewards')
    #plot_results(slowdowns, 'Mean Slowdown')
    #plot_results(makespans, 'Makespan')
