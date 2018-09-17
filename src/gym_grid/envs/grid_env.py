import gym
from collections import defaultdict
from gym import error, spaces, utils
from gym.utils import seeding
from .batsim.batsim import BatsimHandler, InsufficientResourcesError, UnavailableResourcesError
import numpy as np


class GridEnv(gym.Env):
    def __init__(self):
        self.simulator = BatsimHandler(output_freq=1)
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.state = np.zeros(
            shape=self.simulator.current_state.shape, dtype=np.float)
        self.seed()

    @property
    def max_speed(self):
        return self.simulator.resource_manager.max_speed

    @property
    def max_watt(self):
        return self.simulator.resource_manager.max_watts

    @property
    def max_time(self):
        return self.simulator.max_walltime

    def step(self, action):
        assert self.simulator.running_simulation, "Simulation is not running."
        self._take_action(action)
        self._update_state()
        reward = self._get_reward()
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

    def _get_reward(self):
        if self.simulator.running_simulation:
            return -1

        return self.simulator.metrics['makespan']

    def _take_action(self, action):
        resources = [] if action == 0 else [action-1]
        self.simulator.schedule_job(resources)

    def _update_state(self):
        res_shape = self.simulator.resource_manager.shape
        simulator_state = self.simulator.current_state
        for row in range(simulator_state.shape[0]):
            # Host properties
            for col in range(res_shape[1]):
                self.state[row][col] = simulator_state[row][col]

            # Gantt jobs
            for col in range(res_shape[1], simulator_state.shape[1]):
                job = simulator_state[row][col]
                if job is None:
                    self.state[row][col] = 0
                else:
                    self.state[row][col] = job.remaining_time

    def _get_action_space(self):
        return spaces.Discrete(self.simulator.nb_resources+1)

    def _get_observation_space(self):
        obs_space = spaces.Box(low=0,
                               high=self.max_time,
                               shape=self.simulator.state_shape,
                               dtype=np.int16)

        return obs_space
