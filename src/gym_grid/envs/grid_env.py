import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .batsim.batsim import BatsimHandler
from collections import defaultdict
import os

class GridEnv(gym.Env):
    def __init__(self):
        self.simulator = BatsimHandler("tcp://*:28000")
        self.current_time = 0
        self.observation_space = self.__get_observation_space()
        self.action_space = self.__get_action_space()

    def step(self, action):
        success_alloc = self.__take_action(action)
        reward = self.__get_reward(success_alloc)
        obs = self.__get_state()
        done = not self.simulator.running

        return obs, reward, done, {}

    def reset(self):
        self.simulator.close()
        self.simulator.start()
        return self.__get_state()

    def __take_action(self, action):
        assert self.action_space.contains(action)
        return self.simulator.alloc_job(action-1)  # -1 = void

    def __get_reward(self, success_alloc):
        if self.simulator.running:
            if not success_alloc:
                return -1
            return 0
        statistics = self.simulator.statistics
        makespan = self.simulator.current_time
        return statistics['makespan'] + (0.5 * statistics['waiting_time'])

    def __get_action_space(self):
        nb_res = self.simulator.nb_resources
        return spaces.Discrete(nb_res+1)

    def __get_observation_space(self):
        nb_res = self.simulator.nb_resources
        resources = spaces.Tuple([spaces.Discrete(5) for _ in range(nb_res)])
        job = spaces.Discrete(nb_res+1)
        return spaces.Dict({"job": job, "resources": resources})

    def __get_state(self):
        state = self.__get_job_state()
        state.extend(self.__get_resources_state())
        return tuple(state)

    def __get_resources_state(self):
        resources = self.simulator.get_resources()
        if resources is None:
            return [0 for _ in range(self.simulator.nb_resources)]

        return [res.state_value for _, res in resources.items()]

    def __get_job_state(self):
        job = self.simulator.get_job()
        if job is None:
            return [0]
        return [job.res]
