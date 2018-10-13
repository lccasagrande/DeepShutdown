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


def count_resources_inrow_avail(state):
    count, res_avail = 0, 0
    resources = state[0:5]
    for r in resources:
        if r == 1:
            count += 1
        else:
            if res_avail < count:
                res_avail = count
            count = 0

    return res_avail if res_avail > count else count


def get_jobs(state):
    jobs = []
    slot = 0
    jobs_state = state[5:]
    for i in range(0, len(jobs_state), 2):
        slot += 1
        if jobs_state[i] == 0:
            continue
        job = namedtuple('job', 'requested_resources,requested_time, slot')
        job.requested_resources = jobs_state[i]
        job.requested_time = jobs_state[i+1]
        job.slot = slot
        jobs.append(job)
    return jobs


class Policy(object):
    def select_action(self, **kwargs):
        raise NotImplementedError()


class User(Policy):
    def select_action(self, state):
        return int(input("Action: "))


class Random(Policy):
    def select_action(self, state):
        jobs = get_jobs(state)
        return choice(list(range(0, len(jobs)+1)))


class SJF(Policy):
    def select_action(self, state):
        action, shortest_job = 0, np.inf
        nb_res = count_resources_inrow_avail(state)
        if nb_res == 0:
            return action

        jobs = get_jobs(state)

        for i, job in enumerate(jobs):
            if job.requested_time < shortest_job and nb_res >= job.requested_resources:
                shortest_job = job.requested_time
                action = job.slot

        return action


class LJF(Policy):
    def select_action(self, state):
        action, largest_job = 0, -1
        nb_res = count_resources_inrow_avail(state)
        if nb_res == 0:
            return action

        jobs = get_jobs(state)

        for i, job in enumerate(jobs):
            if job.requested_time > largest_job and nb_res >= job.requested_resources:
                largest_job = job.requested_time
                action = job.slot

        return action


class Tetris(Policy):
    def select_action(self, state):
        action, score = 0, 0
        nb_res = count_resources_inrow_avail(state)
        if nb_res == 0:
            return action

        jobs = get_jobs(state)

        for i, job in enumerate(jobs):
            if job.requested_resources > score and nb_res >= job.requested_resources:
                score = job.requested_resources
                action = job.slot

        return action


class FirstFit(Policy):
    def select_action(self, state):
        action = 0
        nb_res = count_resources_inrow_avail(state)
        if nb_res == 0:
            return action

        jobs = get_jobs(state)

        for i, job in enumerate(jobs):
            if nb_res >= job.requested_resources:
                action = job.slot
                break

        return action
