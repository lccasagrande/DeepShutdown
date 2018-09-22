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




class EpsGreedyPolicy:
    def __init__(self, min_epsilon=0.05, eps_decay=100):
        self.min_epsilon = min_epsilon
        self.epsilon = 1
        self.epsilon_decay_steps = eps_decay
        self.epsilons = np.linspace(self.epsilon, min_epsilon, eps_decay)
        self.step = 0

    def select_valid_action(self, q_values, state):
        eps = min(self.step, self.epsilon_decay_steps-1)
        self.epsilon = self.epsilons[eps]
        self.step += 1

        if np.random.uniform() >= self.epsilon:
            return self.select_best_action(q_values, state)
        else:
            valid_actions = [0] + [(i+1) for i in range(q_values.shape[0]-1) if state[i][0] == False]
            return choice(valid_actions)

    def select_best_action(self, q_values, state):
        actions = [0]
        q_max = q_values[0]
        for i in range(q_values.shape[0]-1):
            if state[i][0] == True:
                continue
            act = i+1
            if q_values[act] > q_max:
                q_max = q_values[act]
                actions = [act]
            elif q_values[act] == q_max:
                actions.append(act)

        return choice(actions)



class Processor():
    def process_observation(self, observation):
        obs = []
        for i in range(observation.shape[0]-1):
            resource = observation[i][0]
            resource_queue = observation[i][1]
            job_remaining_time = int(resource_queue[0].remaining_time) if resource_queue[0] != None else 0
            obs.append([resource.is_computing])

        jobs_in_queue = observation[-1]['jobs']
        job_req_time = jobs_in_queue[0].requested_time if len(jobs_in_queue) != 0 else 0
        obs.append(job_req_time)
        obs.append(observation[-1]['lenght'])

        return obs


def run(output_dir, n_ep=250, out_freq=10, plot=True, alpha=.1, gamma=.99):
    env = gym.make('grid-v0')
    policy = EpsGreedyPolicy(min_epsilon=0.05, eps_decay=1000)
    processor = Processor()
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    episodic_scores = deque(maxlen=out_freq)
    avg_scores = deque(maxlen=n_ep)
    t_start = t.time()
    for i in range(1, n_ep + 1):
        score = 0
        utils.print_progress(t_start, i, n_ep)
        state = env.reset()
        state = processor.process_observation(state)
        while True:
            state_hash = tuple(np.hstack(state))

            act = policy.select_valid_action(Q[state_hash], state)

            next_state, reward, done, _ = env.step(act)
            next_state = processor.process_observation(next_state)
            next_state_hash = tuple(np.hstack(next_state))

            best_action = policy.select_best_action(Q[next_state_hash], next_state)

            td_target = reward + gamma * \
                Q[next_state_hash][best_action] - Q[state_hash][act]

            Q[state_hash][act] += alpha * td_target
            state = next_state
            score += reward
            if done:
                episodic_scores.append(score)
                print(
                    "\nScore: {:7} - Max Score: {:7} - Eps {}".format(score, max(episodic_scores), policy.epsilon))
                break
        if i % out_freq == 0:
            avg_scores.append(np.mean(episodic_scores))

    utils.export_q_values(Q, output_dir)
    utils.export_max_q_values(Q, output_dir)
    utils.export_rewards(n_ep, avg_scores, output_dir)

    # TEST
    state = env.reset()
    state = processor.process_observation(state)
    score = 0
    actions = []
    print("--- TESTING ---")
    while True:
        act = policy.select_best_action(Q[tuple(np.hstack(state))], state)
        env.render()
        next_state, reward, done, _ = env.step(act)
        next_state = processor.process_observation(next_state)
        state = next_state
        score += reward
        actions.append(act)
        if done:
            ac = "-".join(str(act) for act in actions)
            print("\nTest Score: {:7} - Actions: {}".format(score, actions))
            break

    if plot:
        utils.plot_reward(avg_scores, n_ep, 'QLEARNING', 'img/q_')

    print("Done!")

if __name__ == "__main__":
    output_dir = 'results/qlearning/'
    utils.clean_or_create_dir(output_dir)
    run(output_dir)
