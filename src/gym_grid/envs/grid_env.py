import gym
from copy import deepcopy
from collections import defaultdict
from gym import error, spaces, utils
from gym.utils import seeding
from sklearn.preprocessing import MinMaxScaler
from .batsim import BatsimHandler, InsufficientResourcesError, UnavailableResourcesError
import numpy as np


class GridEnv(gym.Env):
    def __init__(self):
        self.simulator = BatsimHandler()
        self._update_state()
        #self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.max_waiting_time = 360

    @property
    def max_time(self):
        return self.simulator.max_walltime

    @property
    def max_speed(self):
        return self.simulator.max_resource_speed

    @property
    def max_energy_consumption(self):
        return self.simulator.max_resource_energy_cost

    def step(self, action):
        assert self.simulator.running_simulation, "Simulation is not running."

        alloc_resources = self._prepare_input(action)

        energy_consumed_est = self.simulator.estimate_energy_consumption(
            alloc_resources)
        energy_consumed_est /= self.max_energy_consumption

        queue_jobs = self.simulator.nb_jobs_waiting + self.simulator.nb_jobs_in_queue
        queue_jobs = queue_jobs - 1 if action != 0 else queue_jobs
        queue_load = min(self.simulator.nb_resources,
                         queue_jobs) / self.simulator.nb_resources
        waiting_time = min(self.max_waiting_time, int(
            self.simulator.lookup_first_job().waiting_time)) / self.max_waiting_time

        try:
            self.simulator.schedule_job(alloc_resources)
            reward = -1 * (self.simulator.nb_jobs_in_queue +
                           self.simulator.nb_jobs_running +
                           self.simulator.nb_jobs_waiting)
            # reward = -1 * (energy_consumed_est +
            #               waiting_time + .5*queue_load) / 3
        except (InsufficientResourcesError, UnavailableResourcesError):
            reward = -1

        self._update_state()

        done = not self.simulator.running_simulation

        return self.state, reward, done, {}

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

    def _prepare_input(self, action):
        return [] if action == 0 else [action-1]

    def _update_state(self):
        self.state = self.simulator.current_state

    def _get_action_space(self):
        return spaces.Discrete(self.simulator.nb_resources+1)

    def _get_observation_space(self):
        raise NotImplementedError()
        # obs_space = spaces.Box(low=0,
        #                       high=self.max_time,
        #                       shape=self.state.shape,
        #                       dtype=np.int16)
#
        # return obs_space
