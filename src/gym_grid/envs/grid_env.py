import gym
from collections import defaultdict
from gym import error, spaces, utils
from gym.utils import seeding
from .batsim.batsim import BatsimHandler, InsufficientResourcesError, UnavailableResourcesError
import numpy as np


class GridEnv(gym.Env):
    MAX_WALLTIME = 7200

    def __init__(self):
        self.simulator = BatsimHandler(output_freq=5)
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.nb_invalid_action = 0
        self.seed()

    def step(self, action):
        assert self.simulator.running_simulation, "Simulation is not running."
        reward = self._take_action(action)
        obs = self._get_state()
        done = not self.simulator.running_simulation

        return obs, reward, done, {}

    def reset(self):
        self.nb_invalid_action = 0
        self.simulator.close()
        self.simulator.start()
        return self._get_state()

    def render(self):
        stats = "\rJobs Submitted: {:7} Completed: {:7} | Running: {:5} In Queue: {:5}".format(
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

    def _take_action(self, action):
        resources = [] if action == 0 else [action-1]

        self.simulator.schedule_job(resources)

        return 0

    def _get_state(self):
        simulator_state = self.simulator.current_state
        state = np.zeros(shape=simulator_state.shape, dtype=np.float)
        for row in range(simulator_state.shape[0]):
            for col in range(simulator_state.shape[1]):
                job = simulator_state[row][col]
                if job != None:
                    state[row][col] = job.remaining_time

        return state

    def _get_action_space(self):
        # return spaces.Tuple([spaces.Discrete(2) for _ in range(self.simulator.nb_resources+1)])
        return spaces.Discrete(self.simulator.nb_resources+1)

    @property
    def max_time(self):
        return GridEnv.MAX_WALLTIME

    def _get_observation_space(self):
        obs_space = spaces.Box(low=0,
                               high=self.max_time,
                               shape=self.simulator.state_shape,
                               dtype=np.int16)

        return obs_space
