import gym
from copy import deepcopy
from collections import defaultdict
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from .batsim import BatsimHandler, InsufficientResourcesError, UnavailableResourcesError
import numpy as np


class GridEnv(gym.Env):
    def __init__(self):
        self.job_slots = 21
        self.simulator = BatsimHandler(self.job_slots, time_window=1)
        self._update_state()
        #self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.max_slowdown = 1

    @property
    def nb_resources(self):
        return self.simulator.nb_resources

    @property
    def max_time(self):
        return 15

    @property
    def max_speed(self):
        return self.simulator.max_resource_speed

    @property
    def max_energy_consumption(self):
        return self.simulator.max_resource_energy_cost

    def step(self, action):
        assert self.simulator.running_simulation, "Simulation is not running."

        tmp_reward = self._get_reward()
        try:
            reward = self._get_reward() if action == -1 else 0
            self.simulator.schedule(action)
        except (UnavailableResourcesError):
            self.simulator.schedule(-1)
            reward = tmp_reward

        self._update_state()

        done = not self.simulator.running_simulation

        return self.state, reward, done, {}

    def _get_reward(self):
        reward = 0
        for j in self.simulator.sched_manager.jobs_queue:
            reward += -1 / j.requested_time

        for j in self.simulator.sched_manager.jobs_waiting:
            reward += -1 / j.requested_time

        for j in self.simulator.sched_manager.jobs_running:
            reward += -1 / j.requested_time

        return reward

    def reset(self):
        self.simulator.close()
        self.simulator.start()
        self._update_state()
        return self.state

    def render(self, mode='human'):
       stats = "\rSubmitted: {:5} Completed: {:5} | Running: {:5} In Queue: {:5}".format(
           self.simulator.nb_jobs_submitted,
           self.simulator.nb_jobs_completed,
           self.simulator.nb_jobs_running,
           self.simulator.nb_jobs_in_queue)
       print(stats, end="", flush=True)

    def close(self):
        self.simulator.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _update_state(self):
        self.state = self.simulator.current_state

    def _get_action_space(self):
        return spaces.Discrete(self.simulator.queue_slots+1)

    def _get_observation_space(self):
        raise NotImplementedError()
        # obs_space = spaces.Box(low=0,
        #                       high=self.max_time,
        #                       shape=self.state.shape,
        #                       dtype=np.int16)
#
        # return obs_space
