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


def count_resources_avail(state):
    #resources = state[0:5]
    #res_occuped = np.count_nonzero(resources)
    return state[0]  # len(resources) - res_occuped


def get_jobs(state):
    jobs = []
    slot = 0
    jobs_state = state[1:]
    for i in range(0, len(jobs_state), 2):
        slot += 1
        if jobs_state[i] == 0:
            continue
        jobs.append(dict(requested_resources=jobs_state[i],
                         requested_time=jobs_state[i+1],
                         slot=slot))
    return jobs


class Policy(object):
    def select_action(self, **kwargs):
        raise NotImplementedError()


class User(Policy):
    def select_action(self, state):
        return int(input("Action: "))


class Random(Policy):
    def select_action(self, state):
        return choice(list(range(11)))


def get_avail_res_from_img(state):
    res = state[0,0:10]
    occuped = np.count_nonzero(res)
    return len(res) - occuped

def get_jobs_from_img(state):
    slot = 1
    jobs_state = state[:, 10:110]
    jobs = []
    for i in range(0, 100, 10):
        if jobs_state[0,i] != 0:
            res = np.count_nonzero(jobs_state[0,i:i+10])
            time = np.count_nonzero(jobs_state[:,i])
            jobs.append((res, time, slot))
        slot += 1
    return jobs

def get_available_res(state):
    count = 0
    for i in range(10):
        if state[i] == 0:
            count+=1
    return count


class SJF(Policy):
    def select_action(self, state):
        action, shortest_job = 0, np.inf
        nb_res = get_avail_res_from_img(state)
        jobs = get_jobs_from_img(state)
        for j in jobs:
            if j[0] <= nb_res and j[1] < shortest_job:
                shortest_job = j[1]
                action = j[2]

        return action


class LJF(Policy):
    def select_action(self, state):
        action, largest_job = 0,-1

        nb_res = get_avail_res_from_img(state)
        jobs = get_jobs_from_img(state)
        for j in jobs:
            if j[0] <= nb_res and j[1] > largest_job:
                largest_job = j[1]
                action = j[2]

        return action


class Tetris(Policy):
    def select_action(self, state):
        action, score, = 0, 0

        nb_res = get_avail_res_from_img(state)
        jobs = get_jobs_from_img(state)
        for j in jobs:
            if j[0] <= nb_res and j[0] > score:
                score = j[0]
                action = j[2]

        return action


class FirstFit(Policy):
    def select_action(self, state):
        action = 0
        
        nb_res = get_avail_res_from_img(state)
        jobs = get_jobs_from_img(state)
        for j in jobs:
            if j[0] <= nb_res:
                return j[2]

        return action
