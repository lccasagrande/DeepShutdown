import itertools
import os

import gym
import numpy as np
import pandas as pd


from src.utils.agents import Agent
from gridgym.envs.simulator.utils.graphics import plot_simulation_graphics
from gridgym.envs.grid_env import GridEnv
from gridgym.envs.simulator.resource import PowerStateType


class TimeoutAgent(Agent):
    def __init__(self, timeout, nb_resources, nb_nodes, queue_sz):
        assert timeout >= 0
        self.timeout = timeout
        self.nb_resources = nb_resources
        self.nodes_state = np.zeros(shape=nb_nodes)
        self.current_time = 0
        self.queue_sz = queue_sz

    def _get_available_resources(self, obs):
        nb_available = sum(1 if a == 0 else 0 for a in obs['agenda'])
        queue = obs['queue'][:self.queue_sz]
        if len(queue) > 0:
            (sub, res, wall, pjob_expected_t_start, user, profile) = queue[0]
            if pjob_expected_t_start == 0 or res <= nb_available:
                nb_available -= res
                pjob_expected_t_start = -1
                queue = queue[1:]
            for (sub, res, wall, expected_t_start, user, profile) in queue:
                if pjob_expected_t_start == -1 and res <= nb_available:
                    nb_available -= res
                elif res <= nb_available and wall <= pjob_expected_t_start:
                    nb_available -= res
        return nb_available

    def act(self, obs):
        reservation_size, sim_time, platform = 0, obs['time'], obs['platform']
        nb_available = self._get_available_resources(obs)

        for i, node in enumerate(platform):
            if nb_available >= len(node):
                if all(r == 0 for r in node):
                    self.nodes_state[i] += sim_time - self.current_time
                else:
                    self.nodes_state[i] = -1
                if self.nodes_state[i] >= self.timeout or all(r == 2 or r == 3 for r in node):
                    reservation_size += 1
                    nb_available -= len(node)

        self.current_time = sim_time
        return reservation_size

    def play(self, env, render=False, verbose=False):
        self.nodes_state = np.zeros(shape=self.nodes_state.shape)
        self.current_time = 0
        obs, done, score = env.reset(), False, 0
        while not done:
            if render:
                env.render()
            obs, reward, done, info = env.step(self.act(obs))
            score += reward
        results = pd.read_csv(os.path.join(
            GridEnv.OUTPUT, '_schedule.csv')).to_dict('records')[0]
        results['score'] = score
        results['workload'] = info['workload_name']
        if verbose:
            m = " - ".join("[{}: {}]".format(k, v) for k, v in results.items())
            print("[RESULTS] {}".format(m))
        env.close()
        return results
