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
        return choice(list(range(0, len(state[2:])+1)))


def get_available_res(state):
    count = 0
    for i in range(1,10,2):
        if state[i] == 0:
            count+=1
    return count / 5


class SJF(Policy):
    def select_action(self, state):
        action, slot, shortest_job = 0, 1, np.inf

        nb_res = get_available_res(state)
        jobs_state = state[10:-1]
        for i in range(0, len(jobs_state), 2):
            req_res = jobs_state[i]
            req_time = jobs_state[i+1]
            if req_res != 0 and req_res <= nb_res and req_time <= shortest_job:
                shortest_job = req_time
                action = slot
            slot += 1

        return action


class LJF(Policy):
    def select_action(self, state):
        action, slot, largest_job = 0, 1, -1
        nb_res = get_available_res(state)
        jobs_state = state[5:]
        for i in range(0, len(jobs_state), 2):
            req_res = jobs_state[i]
            req_time = jobs_state[i+1]
            if req_res != 0 and req_res <= nb_res and req_time >= largest_job:
                largest_job = req_time
                action = slot
            slot += 1

        return action


class Tetris(Policy):
    def select_action(self, state):
        action, score, slot = 0, 0, 1
        
        nb_res = get_available_res(state)
        jobs_state = state[5:]
        for i in range(0, len(jobs_state), 2):
            req_res = jobs_state[i]
            if req_res != 0 and req_res <= nb_res and req_res >= score:
                score = req_res
                action = slot
            slot += 1

        return action


class FirstFit(Policy):
    def select_action(self, state):
        action, slot = 0, 1
        
        nb_res = get_available_res(state)
        jobs_state = state[5:]
        for i in range(0, len(jobs_state), 2):
            req_res = jobs_state[i]
            if req_res != 0 and req_res <= nb_res:
                return slot
            slot += 1
        return action
