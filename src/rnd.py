import utils
import gym
from gym_grid.envs.grid_env import GridEnv
import sys
import csv
import pandas as pd
import numpy as np
from random import choice
import time as t
from collections import defaultdict, deque


def sample_valid_action(env, state):
    valid_actions = [0] + [(i+1) for i in range(env.action_space.n-1) if np.any(state[i] == 0)]
    return choice(valid_actions)


def run(output_dir, n_ep=10, out_freq=2, plot=True):
    env = gym.make('grid-v0')
    episodic_scores = deque(maxlen=out_freq)
    avg_scores = deque(maxlen=n_ep)
    action_history = []
    t_start = t.time()
    for i in range(1, n_ep + 1):
        score = 0
        epi_actions = np.zeros(env.action_space.n)
        utils.print_progress(t_start, i, n_ep)
        state = env.reset()
        while True:
            act = sample_valid_action(env, state)
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
                          title='Random Policy',
                          output_dir=output_dir)

    print(('\nBest Average Reward over %d Episodes: ' %
           out_freq), np.max(avg_scores))


if __name__ == "__main__":
    output_dir = 'results/random/'
    run(output_dir)
