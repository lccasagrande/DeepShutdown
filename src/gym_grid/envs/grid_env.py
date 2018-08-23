import gym
from collections import defaultdict
from gym import error, spaces, utils
from gym.utils import seeding
from .batsim.batsim import BatsimHandler, InsufficientResourcesError, UnavailableResourcesError
import numpy as np


class GridEnv(gym.Env):
    MAX_WALLTIME = 7200

    def __init__(self):
        self.simulator = BatsimHandler(output_freq=1, verbose='quiet')
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.seed()

    def step(self, action):
        assert self.simulator.running_simulation, "Simulation is not running."
        reward = self._take_action(action)
        obs = self._get_state()
        done = not self.simulator.running_simulation

        if done:
            reward = 1 / self.simulator.metrics['mean_exectime']

        return obs, reward, done, {}

    def reset(self):
        self.simulator.close()
        self.simulator.start()
        return self._get_state()

    def render(self):
        raise NotImplementedError()

    def close(self):
        self.simulator.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _take_action(self, action):
        res = [] if action == 0 else [action - 1]

        try:
            self.simulator.schedule_job(res)
            reward = 0
        except (InsufficientResourcesError, UnavailableResourcesError):
            reward = -1

        return reward

    def _get_reward(self, success):
        if self.simulator.running_simulation:
            if not success:
                return -1

        return 0

    def _get_state(self):
        res_info = self.simulator.get_resources_info()
        job = self.simulator.get_job_info()
        job_info = {
            'res': 0 if job is None else job.requested_resources,
            'requested_time': 0 if job is None else job.requested_time,
            'waiting_time': 0 if job is None else job.waiting_time
        }

        state = {'resources': res_info, 'job': job_info}

        return state

    def _get_action_space(self):
        return spaces.Tuple([spaces.Discrete(2) for _ in range(self.simulator.nb_resources+1)])

    def _get_observation_space(self):
        space = spaces.Dict({
            'resources': spaces.MultiDiscrete(range(self.simulator.nb_resources)),
            'job': spaces.Dict({
                'res': spaces.Discrete(self.simulator.nb_resources),
                'wall_time': spaces.Box(low=0, high=GridEnv.MAX_WALLTIME, shape=()),
                'waiting_time': spaces.Box(low=0, high=GridEnv.MAX_WALLTIME, shape=())
            })
        })

        return space
