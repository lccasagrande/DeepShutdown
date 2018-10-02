from gym_grid.envs.grid_env import GridEnv
from collections import deque
from policies import Tetris, SJF, LJF, Random, FirstFit
import utils
import gym
import csv
import numpy as np
import pandas as pd

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import plot


def run_experiment(output_dir, policy, n_ep, out_freq, plot=False, render=False):
    env = gym.make('grid-v0')
    episodic_scores = []
    episodic_slowdowns = []
    episodic_makespans = []
    episodic_energy = []

    for i in range(1, n_ep + 1):
        score, state = 0,  env.reset()
        while True:
            if render:
                env.render()

            act = policy.select_action(state)

            state, reward, done, info = env.step(act)

            score += reward

            if done:
                episodic_scores.append(score)
                episodic_slowdowns.append(info['mean_slowdown'])
                episodic_makespans.append(info['makespan'])
                episodic_energy.append(info['energy_consumed'])
                print("Episode {:7}, Score: {:7} - Mean Slowdown {:3} - Makespan {:7}"
                      .format(i, score, info['mean_slowdown'], info['makespan']))
                break

    if plot:
        utils.plot_reward(episodic_scores,
                          n_ep,
                          title=policy.__class__.__name__ + ' Policy',
                          output_dir=output_dir)

    return episodic_scores, episodic_slowdowns, episodic_makespans, episodic_energy


def run(output_dir, policies):
    rewards = dict()
    slowdowns = dict()
    makespans = dict()
    energy = dict()

    for policy in policies:
        output = output_dir + policy.__class__.__name__ + "/"
        utils.clean_or_create_dir(output)
        eps_rewards, eps_slowdowns, eps_makespans, eps_energy = run_experiment(output_dir=output,
                                                                               policy=policy,
                                                                               n_ep=10,
                                                                               out_freq=100,
                                                                               plot=False,
                                                                               render=False)

        rewards[policy.__class__.__name__] = list(eps_rewards)
        slowdowns[policy.__class__.__name__] = list(eps_slowdowns)
        makespans[policy.__class__.__name__] = list(eps_makespans)
        energy[policy.__class__.__name__] = list(eps_energy)

    dt = pd.DataFrame.from_dict(rewards)
    dt.to_csv(output_dir+"rewards.csv", index=False)
    dt = pd.DataFrame.from_dict(slowdowns)
    dt.to_csv(output_dir+"slowdowns.csv", index=False)
    dt = pd.DataFrame.from_dict(makespans)
    dt.to_csv(output_dir+"makespans.csv", index=False)
    dt = pd.DataFrame.from_dict(energy)
    dt.to_csv(output_dir+"energy.csv", index=False)
    return rewards, slowdowns, makespans, energy


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
    rewards, slowdowns, makespans, energy = run(output_dir, policies)
    plot_results(rewards, 'Episode Rewards')
    plot_results(slowdowns, 'Mean Slowdown')
    plot_results(makespans, 'Makespan')
