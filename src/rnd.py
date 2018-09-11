import utils
import gym
from gym_grid.envs.grid_env import GridEnv
import sys
import numpy as np
import matplotlib.pyplot as plt
from random import sample
import time as t
from collections import defaultdict, deque


def run(n_ep=10, out_freq=2, plot=True):
    env = gym.make('grid-v0')
    episodic_scores = deque(maxlen=out_freq)
    avg_scores = deque(maxlen=n_ep)
    t_start = t.time()
    for i in range(1, n_ep + 1):
        utils.print_progress(t_start, i, n_ep)
        env.reset()
        score = 0
        while True:
            act = env.action_space.sample()
            _, reward, done, _ = env.step(act)
            score += reward
            if done:
                episodic_scores.append(score)
                break
        if i % out_freq == 0:
            avg_scores.append(np.mean(episodic_scores))

    if plot:
        utils.plot_reward(avg_scores, n_ep, title='Random Policy')

    print(('Best Average Reward over %d Episodes: ' % out_freq), np.max(avg_scores))

if __name__ == "__main__":
    run()
