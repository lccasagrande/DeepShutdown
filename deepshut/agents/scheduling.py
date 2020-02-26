import itertools
import os

import gym
import numpy as np
import pandas as pd


from src.utils.agents import Agent


class FirstComeFirstServed(Agent):
    def act(self, obs):
        nb_available = sum(1 if a[0] == 0 else 0 for a in obs['agenda'])
        priority_job = obs['queue'][0] if obs['queue'][0][0] != -1 else None
        return int(priority_job is not None and priority_job[1] <= nb_available)

    def play(self, env, render=False):
        obs, done, score, info = env.reset(), False, 0, None
        while not done:
            if render:
                env.render()
            obs, reward, done, info = env.step(self.act(obs))
            score += reward
        env.close()
        return score, info
