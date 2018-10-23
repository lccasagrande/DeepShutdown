from gym_grid.envs.grid_env import GridEnv
from random import choice
from collections import defaultdict, deque, namedtuple
from rl.policy import Policy as RLPolicy
import utils
import gym
import sys
import csv
import pandas as pd
import numpy as np
import shutil
import time as t


def get_avail_res_from_img(state):
    res = state[0, 0:10]
    occuped = np.count_nonzero(res)
    return len(res) - occuped


def get_jobs_from_img(state):
    slot = 1
    jobs_state = state[:, 10:110]
    jobs = []
    for i in range(0, 100, 10):
        if jobs_state[0, i] != 0:
            res = np.count_nonzero(jobs_state[0, i:i+10])
            time = np.count_nonzero(jobs_state[:, i])
            jobs.append((res, time, slot))
        slot += 1
    return jobs, slot


class Policy(object):
    def select_action(self, **kwargs):
        raise NotImplementedError()


class User(Policy):
    def select_action(self, state):
        return int(input("Action: "))


class Random(Policy):
    def select_action(self, state):
        _, slots = get_jobs_from_img(state)
        return np.random.randint(slots+1)


class SJF(Policy):
    def select_action(self, state):
        action, shortest_job = 0, np.inf
        nb_res = get_avail_res_from_img(state)
        jobs, _ = get_jobs_from_img(state)
        for j in jobs:
            if j[0] <= nb_res and j[1] < shortest_job:
                shortest_job = j[1]
                action = j[2]

        return action


class LJF(Policy):
    def select_action(self, state):
        action, largest_job = 0, -1

        nb_res = get_avail_res_from_img(state)
        jobs, _ = get_jobs_from_img(state)
        for j in jobs:
            if j[0] <= nb_res and j[1] > largest_job:
                largest_job = j[1]
                action = j[2]

        return action


class Packer(Policy):
    def select_action(self, state):
        action, score, = 0, 0

        nb_res = get_avail_res_from_img(state)
        jobs, _ = get_jobs_from_img(state)
        for j in jobs:
            if j[0] <= nb_res and j[0] > score:
                score = j[0]
                action = j[2]

        return action


class Tetris(Policy):
    def __init__(self, knob=0.5):
        self.knob = knob

    def select_action(self, state):
        action, score, = 0, 0

        nb_res = get_avail_res_from_img(state)
        jobs, _ = get_jobs_from_img(state)
        for j in jobs:
            if j[0] <= nb_res:
                sjf_score = 1 / float(j[1])
                align_score = j[0]

                combined_score = (self.knob * align_score) + \
                    ((1-self.knob) * sjf_score)
                if combined_score > score:
                    score = combined_score
                    action = j[2]

        return action


class FirstFit(Policy):
    def select_action(self, state):
        action = 0

        nb_res = get_avail_res_from_img(state)
        jobs, _ = get_jobs_from_img(state)
        for j in jobs:
            if j[0] <= nb_res:
                return j[2]

        return action
