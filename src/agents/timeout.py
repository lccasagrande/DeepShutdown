import itertools
import os

import gym
import numpy as np
import pandas as pd


from src.utils.agents import Agent
from gridgym.envs.simulator.utils.graphics import plot_simulation_graphics
from gridgym.envs.grid_env import GridEnv
from gridgym.envs.simulator.resource import PowerStateType


class TimeoutWithOracleAgent(Agent):
    def __init__(self, nb_resources, nb_nodes, t_turnoff, t_turnon, min_idle_time=None):
        self.t_turnoff = t_turnoff
        self.t_turnon = t_turnon
        self.nb_resources = nb_resources
        self.nodes_state = np.zeros(shape=nb_nodes)
        self.current_time = 0
        self.time_to_compensate_turning_off = min_idle_time

    def get_resources_that_compensate_turning_off(self, nb_resources, info):
        if not self.time_to_compensate_turning_off:
            nodes = info['nodes_profile']
            sleep_power = sum(
                ps['min'] for n in nodes for ps in n if ps['type'] == PowerStateType.sleep)
            idle_power = sum(
                ps['min'] for n in nodes for ps in n if ps['type'] == PowerStateType.computation)
            turnon_power = sum(
                ps['min'] for n in nodes for ps in n if ps['type'] == PowerStateType.switching_on)
            turnoff_power = sum(
                ps['min'] for n in nodes for ps in n if ps['type'] == PowerStateType.switching_off)
            self.time_to_compensate_turning_off = (
                (turnoff_power * self.t_turnoff) + (turnon_power * self.t_turnon) - sleep_power) / (idle_power - sleep_power)

        for j in info['next_submissions']:
            if nb_resources <= 0:
                break
            if j['subtime'] - self.current_time <= self.time_to_compensate_turning_off:
                nb_resources -= j['res']
        return max(0, nb_resources)

    def act(self, obs, info=None):
        reservation_size = 0
        nb_available = sum(1 if a == 0 else 0 for a in obs['agenda'])
        queue = list(itertools.takewhile(lambda j: j[0] != -1, obs['queue']))
        if len(queue) > 0:
            _, res, _, pjob_expected_time_to_start = queue[0]
            if pjob_expected_time_to_start == 0:
                nb_available -= res
                pjob_expected_time_to_start = -1
                queue = queue[1:]
            for (_, res, walltime, _) in queue:
                if pjob_expected_time_to_start == -1 and res <= nb_available:
                    nb_available -= res
                elif res <= nb_available and walltime <= pjob_expected_time_to_start:
                    nb_available -= res
        if info:
            nb_available = self.get_resources_that_compensate_turning_off(
                nb_available, info)

        for i, node in enumerate(obs['platform']):
            if nb_available >= len(node) and all(r == 0 or r == 2 or r == 3 for r in node):
                reservation_size += 1
                nb_available -= len(node)

        self.current_time = obs['time']
        return reservation_size

    def play(self, env, render=False, verbose=False):
        self.current_time = 0
        obs, done, score, info = env.reset(), False, 0, None
        while not done:
            if render:
                env.render()
            obs, reward, done, info = env.step(self.act(obs, info))
            score += reward
        if verbose:
            results = pd.read_csv(os.path.join(GridEnv.OUTPUT, '_schedule.csv')).to_dict('records')[0]
            results['score'] = score
            m = " - ".join("[{}: {}]".format(k, v) for k, v in results.items())
            print("[RESULTS] {}".format(m))

        plot_simulation_graphics(GridEnv.OUTPUT, show=True)
        env.close()
        return score


class TimeoutAgent(Agent):
    def __init__(self, timeout, nb_resources, nb_nodes):
        assert timeout > 0
        self.timeout = timeout
        self.nb_resources = nb_resources
        self.nodes_state = np.zeros(shape=nb_nodes)
        self.current_time = 0

    def act(self, obs):
        nb_available = sum(1 if a == 0 else 0 for a in obs['agenda'])
        queue = obs['queue']
        if len(queue) > 0:
            _, res, _, pjob_expected_time_to_start, _ = queue[0]
            if pjob_expected_time_to_start == 0 or res <= nb_available:
                nb_available -= res
                pjob_expected_time_to_start = -1
                queue = queue[1:]
            for (_, res, walltime, _, _) in queue:
                if pjob_expected_time_to_start == -1 and res <= nb_available:
                    nb_available -= res
                elif res <= nb_available and walltime <= pjob_expected_time_to_start:
                    nb_available -= res

        reservation_size = 0
        for i, node in enumerate(obs['platform']):
            if nb_available >= len(node):
                if all(r == 0 for r in node):
                    self.nodes_state[i] += obs['time'] - self.current_time
                else:
                    self.nodes_state[i] = 0
                if self.nodes_state[i] >= self.timeout or all(r == 2 or r == 3 for r in node):
                    reservation_size += 1
                    nb_available -= len(node)

        self.current_time = obs['time']
        return reservation_size

    def play(self, env, render=False, verbose=False):
        self.current_time = 0
        obs, done, score = env.reset(), False, 0
        while not done:
            if render:
                env.render()
            obs, reward, done, info = env.step(self.act(obs))
            score += reward
        results = pd.read_csv(os.path.join(GridEnv.OUTPUT, '_schedule.csv')).to_dict('records')[0]
        results['score'] = score
        if verbose:
            m = " - ".join("[{}: {}]".format(k, v) for k, v in results.items())
            print("[RESULTS] {}".format(m))
        env.close()
        return results
