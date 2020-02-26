import argparse
import math
import multiprocessing
import itertools
import os


import numpy as np
import pandas as pd
import gym
from gym import spaces
from sortedcontainers import SortedList

from gridgym.envs.off_reservation_env import OffReservationEnv
from gridgym.envs.grid_env import GridEnv
from batsim_py.utils.graphics import plot_simulation_graphics


class OffReservationAgent():
    def __init__(self, max_stretch, nb_resources, nb_nodes, queue_sz, timeout=0):
        assert timeout >= 0
        self.timeout = timeout
        self.nb_resources = nb_resources
        self.max_stretch = max_stretch
        self.queue_sz = queue_sz

        self.nodes_state = np.full(shape=nb_nodes, fill_value=-1, dtype=np.int)
        self.current_time = 0

    def _get_next_releases(self, obs):
        releases = []
        for (t_start, res, walltime, user, profile) in obs['jobs_running']:
            runtime = obs['time'] - t_start
            t_release = profile - runtime
            for _ in range(res):
                releases.append(t_release)
        return sorted(releases)

    def is_possible_to_delay(self, time, next_releases, job):
        is_possible_to_delay = True
        (sub, res, wall, pjob_expected_t_start, user, profile) = job
        if (time - sub) / wall >= self.max_stretch:
            is_possible_to_delay = False
        elif len(next_releases) == 0:
            is_possible_to_delay = False
        elif len(next_releases) > 0:
            expected_t_start = next_releases[:res][-1] + time
            if (expected_t_start - sub) / wall >= self.max_stretch:
                is_possible_to_delay = False
        return is_possible_to_delay

    def _get_available_resources(self, obs):
        curr_time = obs['time']
        nb_available = sum(1 if a == 0 else 0 for a in obs['agenda'])
        next_rel = self._get_next_releases(obs)
        queue = obs['queue'][:self.queue_sz]
        if len(queue) > 0:
            (sub, res, wall, pjob_expected_t_start, user, profile) = queue[0]
            schedule = False

            if pjob_expected_t_start == 0:
                nb_available -= res
                pjob_expected_t_start = -1
                next_rel.extend([wall] * res)
            elif res <= nb_available:
                pjob_expected_t_start = -1
                if not self.is_possible_to_delay(curr_time, next_rel, queue[0]):
                    nb_available -= res
                    next_rel.extend([wall] * res)

            next_rel.sort()
            for job in queue[1:]:
                schedule = False
                (sub, res, wall, expected_t_start, user, profile) = job
                if pjob_expected_t_start == -1 and res <= nb_available:
                    schedule = True
                elif res <= nb_available and wall <= pjob_expected_t_start:
                    schedule = True

                if schedule:
                    if not self.is_possible_to_delay(curr_time, next_rel, job):
                        nb_available -= res
                        next_rel.extend([wall] * res)
                        next_rel.sort()

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

    def play(self, env, render=False):
        self.nodes_state = np.full(
            shape=self.nodes_state.shape, fill_value=-1, dtype=np.int)
        self.current_time = 0
        obs, done, score = env.reset(), False, 0
        while not done:
            if render:
                env.render()
            obs, reward, done, info = env.step(self.act(obs))
            score += reward

        env.close()
        return score, info


def run(args):
    env = gym.make(args.env_id,
                   max_queue_sz=args.max_queue_sz,
                   act_interval=args.act_interval,
                   simulation_time=args.simulation_time,
                   qos_stretch=args.qos_stretch,
                   export=True,
                   use_batsim=args.use_batsim)

    nb_resources = env.observation_space.spaces['agenda'].shape[0]
    nb_nodes = env.observation_space.spaces['platform'].shape[0]
    agent = OffReservationAgent(
        args.max_strecth, nb_resources, nb_nodes, args.max_queue_sz, args.timeout)
    score, info = agent.play(env, render=args.render)

    print("Score: {}".format(score))

    if args.plot_results:
        results_dir = "/tmp/GridGym/{}".format(info['workload_name'])
        plot_simulation_graphics(results_dir, show=True)


def parse_args():
    parser = argparse.ArgumentParser()
    # Agent specific args
    agent = parser.add_argument_group(title="Agent options")
    agent.add_argument("--timeout", default=1, type=int)
    agent.add_argument("--max_strecth", default=0.5, type=float)

    # Environment options
    env = parser.add_argument_group(title="Environment options")
    env.add_argument("--env_id", default="OffReservation-v0", type=str)
    env.add_argument("--max_queue_sz", default=10, type=int)
    env.add_argument("--act_interval", default=1, type=int)
    env.add_argument("--simulation_time", default=1440, type=int)
    env.add_argument("--qos_stretch", default=0.5, type=float)
    env.add_argument("--use_batsim", action="store_true")

    #
    parser.add_argument("--plot_results", default=False, action="store_true")
    parser.add_argument("--render", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
