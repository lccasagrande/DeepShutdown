from gym_grid.envs.grid_env import GridEnv
from random import choice
from collections import defaultdict, deque
from rl.policy import Policy as RLPolicy
import utils
import gym
import sys
import csv
import pandas as pd
import numpy as np
import shutil
import time as t


class Policy(object):
    def select_action(self, **kwargs):
        raise NotImplementedError()


class EpsGreedy(RLPolicy):
    def __init__(self, nb_res, job_slots, eps=.1):
        super(EpsGreedy, self).__init__()
        self.greedy_policy = Greedy(nb_res, job_slots)
        self.eps = eps
        self.nb_res = nb_res
        self.job_slots = job_slots

    def get_config(self):
        config = super(EpsGreedy, self).get_config()
        config['eps'] = self.eps
        return config

    def select_action(self, q_values, state):
        if np.random.uniform() >= self.eps:
            return self.greedy_policy.select_action(q_values, state)
        else:
            return self._select_random_action(q_values, state)

    def _select_random_action(self, q_values, state):
        valid_actions = [0]
        total_job_slots = self.nb_res+(self.job_slots*self.nb_res)

        action = 1
        for i in range(self.nb_res, total_job_slots, self.nb_res):
            if state[-1][0][i] > 0:
                valid_actions.append(action)
                action += 1

        return choice(valid_actions)


class Greedy(RLPolicy):
    def __init__(self, nb_res, job_slots):
        super(Greedy, self).__init__()
        self.nb_res = nb_res
        self.job_slots = job_slots

    def select_action(self, q_values, state):
        q_pos = 0
        actions = [q_pos]
        q_max = q_values[q_pos]
        total_job_slots = self.nb_res+(self.job_slots*self.nb_res)

        for i in range(self.nb_res, total_job_slots, self.nb_res):
            if state[-1][0][i] > 0:
                q_pos += 1

                if q_values[q_pos] > q_max:
                    q_max = q_values[q_pos]
                    actions = [q_pos]
                elif q_values[q_pos] == q_max:
                    actions.append(q_pos)

        return choice(actions)


class Random(Policy):
    def select_action(self, state):
        jobs = state['job_queue']['jobs']
        actions = [0] + list(range(1, len(jobs)+1))

        return choice(actions)


class SJF(Policy):
    def select_action(self, state):
        action, shortest_job = 0, np.inf
        jobs = state['job_queue']['jobs']
        avail_res = len([res_data['resource'].id for res_data in state['gantt']
                         if res_data['resource'].is_available])

        for i, job in enumerate(jobs):
            if job.requested_resources <= avail_res and job.requested_time < shortest_job:
                shortest_job = job.requested_time
                action = i+1

        return action


class LJF(Policy):
    def select_action(self, state):
        action, shortest_job = 0, -1
        jobs = state['job_queue']['jobs']
        avail_res = len([res_data['resource'].id for res_data in state['gantt']
                         if res_data['resource'].is_available])

        for i, job in enumerate(jobs):
            if job.requested_resources <= avail_res and job.requested_time > shortest_job:
                shortest_job = job.requested_time
                action = i+1

        return action


class Tetris(Policy):
    def select_action(self, state):
        action, score = 0, 0
        jobs = state['job_queue']['jobs']
        avail_res = len([res_data['resource'].id for res_data in state['gantt']
                         if res_data['resource'].is_available])

        for i, job in enumerate(jobs):
            if job.requested_resources <= avail_res and job.requested_resources > score:
                score = job.requested_resources
                action = i+1

        return action


class FirstFit(Policy):
    def select_action(self, state):
        action = 0
        gantt = state['gantt']
        jobs = state['job_queue']['jobs']
        avail_res = len(
            [res_data['resource'].id for res_data in gantt if res_data['resource'].is_available])

        for i, job in enumerate(jobs):
            if job.requested_resources <= avail_res:
                action = i+1
                break

        return action
