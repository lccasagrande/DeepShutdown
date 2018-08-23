import gym
import gym_grid.envs.grid_env as g
import sys
import numpy as np
import matplotlib.pyplot as plt
import utils
from random import sample
from collections import defaultdict, deque


def select_random_valid_action(state):
    req_res = state['job']['res']
    available_res = [i for i, res in enumerate(
        state['resources']) if res == 'idle']
    selected_resources = sample(available_res, req_res)
    selected_resources = [
        1 if i in selected_resources else 0 for i, _ in enumerate(state['resources'])]

    n_possib_actions = len(available_res) + 1
    sched_prob = len(available_res) / n_possib_actions
    void_prob = 1 / n_possib_actions

    possible_actions = [[], selected_resources]
    return np.random.choice(possible_actions, size=1, p=[void_prob, sched_prob])[0]


def run(n_ep=1000, plot=True):
    env = gym.make('grid-v0')
    episodic_scores = deque(maxlen=100)
    avg_scores = deque(maxlen=(n_ep//2))

    for i in range(1, n_ep + 1):
        utils.print_progress(i, n_ep)
        score = 0
        state = env.reset()
        while True:
            act = select_random_valid_action(state)
            next_state, reward, done, _ = env.step(act)
            score += reward
            state = next_state
            if done:
                episodic_scores.append(score)
                break
        if (i % 2 == 0):
            avg_scores.append(np.mean(episodic_scores))

    if plot:
        utils.plot_graph(avg_scores, n_ep)


if __name__ == "__main__":
    run()
