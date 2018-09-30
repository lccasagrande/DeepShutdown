import utils
import gym
from gym_grid.envs.grid_env import GridEnv
import sys
import csv
import pandas as pd
import numpy as np
import shutil

from random import choice
import time as t
from collections import defaultdict, deque
from policies import FirstFitPolicy


def run(output_dir, n_ep=3, out_freq=100, plot=False):
    env = gym.make('grid-v0')
    episodic_scores = deque(maxlen=out_freq)
    avg_scores = deque(maxlen=n_ep)
    policy = FirstFitPolicy()
    action_history = []
    t_start = t.time()
    for i in range(1, n_ep + 1):
        score = 0
        epi_actions = np.zeros(env.action_space.n)
        utils.print_progress(t_start, i, n_ep)
        state = env.reset()
        while True:
            act = policy.select_valid_action(state)
            epi_actions[act] += 1
            state, reward, done, _ = env.step(act)
            score += reward
            env.render()

            if done:
                episodic_scores.append(score)
                action_history.append(epi_actions)
                print(
                    "\nScore: {:7} - Max Score: {:7}".format(score, max(episodic_scores)))
                break
        if i % out_freq == 0:
            avg_scores.append(np.mean(episodic_scores))

    pd.DataFrame(action_history,
                 columns=range(0, env.action_space.n),
                 dtype=np.int).to_csv(output_dir+"actions_hist.csv")

    if plot:
        utils.plot_reward(avg_scores,
                          n_ep,
                          title='FCFS Policy',
                          output_dir=output_dir)


if __name__ == "__main__":
    output_dir = 'results/fcfs/'
    utils.clean_or_create_dir(output_dir)

    run(output_dir)
