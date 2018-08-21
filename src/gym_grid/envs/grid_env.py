import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

try:
    from .batsim.batsim import BatsimHandler
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install PYBATSIM (https://pypi.org/project/pybatsim/) and BATSIM (https://github.com/oar-team/batsim)".format(e))


class GridEnv(gym.Env):
    MAX_WALLTIME = 7200
    PLATFORM = "platform.xml"
    WORKLOAD = "workload.json"

    def __init__(self):
        self.simulator = BatsimHandler(
            "tcp://*:28000", GridEnv.PLATFORM, GridEnv.WORKLOAD)
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.seed()

    def step(self, action):
        success = self._take_action(action)
        reward = self._get_reward(success)
        obs = self._get_state()
        done = not self.simulator.running_simulation

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
        res = []
        for i, act in enumerate(action):
            if act == 1:
                res.append(i)
        success = self.simulator.manager.schedule_job(res)
        return success

    def _get_reward(self, success):
        if self.simulator.running_simulation:
            if not success:
                return -1

        return 0

    def _get_state(self):
        # def one_hot(a, num_classes):
        #    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

        res_info = self.simulator.manager.get_resources_info()
        job = self.simulator.manager.get_job_info()
        if job == None:
            job = {'res': 0, 'requested_time': 0, 'waiting_time': 0}
        else:
            job = {
                'res': job.requested_resources,
                'requested_time': job.requested_time,
                'waiting_time': int(self.simulator.time() - job.submit_time)
            }

        state = {
            'resources': res_info,
            'job': job
        }
        
        return state

    def _get_action_space(self):
        return spaces.Tuple([spaces.Discrete(2) for _ in range(self.simulator.nb_resources)])

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
