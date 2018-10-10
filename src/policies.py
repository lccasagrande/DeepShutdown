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


class User(Policy):
    def select_action(self, state):
        return int(input("Action: "))

class Random(Policy):
    def select_action(self, state):
        jobs = state['queue']
        actions = [0] + list(range(1, len(jobs)+1))

        return choice(actions)


def is_available(time_window, req_res):
    nb_res = len(time_window[0])
    for r in range(0, nb_res - req_res+1):
        if not np.any(time_window[:, r:r+req_res] == 255):
            return True
    return False


class FCFS(Policy):
    def select_action(self, state):
        jobs = state['queue']
        action = 0
        for i, j in enumerate(jobs):
            if j != None:
                action = i+1
                break
        return action


class SJF(Policy):
    def select_action(self, state):
        action, shortest_job = 0, np.inf
        jobs = state['queue']
        res_spaces = state['resources_spaces']

        for i, job in enumerate(jobs):
            if job == None: continue
            requested_time_window = res_spaces[:job.requested_time]
            if job.requested_time < shortest_job and is_available(requested_time_window, job.requested_resources):
                shortest_job = job.requested_time
                action = i+1

        return action


class LJF(Policy):
    def select_action(self, state):
        action, largest_job = 0, -1
        jobs = state['queue']
        res_spaces = state['resources_spaces']

        for i, job in enumerate(jobs):
            if job == None: continue
            requested_time_window = res_spaces[:job.requested_time]
            if job.requested_time > largest_job and is_available(requested_time_window, job.requested_resources):
                largest_job = job.requested_time
                action = i+1

        return action


class Tetris(Policy):
    def select_action(self, state):
        action, score = 0, 0
        jobs = state['queue']
        res_spaces = state['resources_spaces']

        for i, job in enumerate(jobs):
            if job == None: continue
            req_res_space = res_spaces[:job.requested_time]
            if job.requested_resources > score and is_available(req_res_space, job.requested_resources):
                score = job.requested_resources
                action = i+1

        return action


class FirstFit(Policy):
    def select_action(self, state):
        action = 0
        jobs = state['queue']
        res_spaces = state['resources_spaces']

        for i, job in enumerate(jobs):
            if job == None: continue
            requested_time_window = res_spaces[:job.requested_time]
            if is_available(requested_time_window, job.requested_resources):
                action = i+1
                break

        return action
