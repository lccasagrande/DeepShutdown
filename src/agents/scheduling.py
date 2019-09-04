import itertools
import os

import gym
import numpy as np
import pandas as pd


from src.utils.agents import Agent
from gridgym.envs.simulator.utils.graphics import plot_simulation_graphics
from gridgym.envs.grid_env import GridEnv
from gridgym.envs.simulator.resource import PowerStateType


class FirstComeFirstServed(Agent):
    def act(self, obs):
        nb_available = sum(1 if a[0] == 0 else 0 for a in obs['agenda'])
        priority_job = obs['queue'][0] if obs['queue'][0][0] != -1 else None
        return int(priority_job is not None and priority_job[1] <= nb_available)

    def play(self, env, render=False, verbose=False):
        obs, done, score, info = env.reset(), False, 0, None
        while not done:
            if render:
                env.render()
            obs, reward, done, info = env.step(self.act(obs))
            score += reward
        if verbose:
            info['score'] = score
            m = " - ".join("[{}: {}]".format(k, v) for k, v in info.items())
            print("[RESULTS] {}".format(m))
        env.close()
        return score
