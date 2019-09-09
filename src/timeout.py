import argparse
import math
import multiprocessing
import itertools
import os

import gym
import argparse
from gym import spaces

from gridgym.envs.grid_env import GridEnv
from gridgym.envs.simulator.utils.graphics import plot_simulation_graphics
from src.agents.timeout import TimeoutAgent, TimeoutWithOracleAgent


def run(args):
    env = gym.make(args.env_id)
    nb_resources = env.observation_space.spaces['platform'].shape[0] * \
        env.observation_space.spaces['platform'].shape[1]
    nb_nodes = env.observation_space.spaces['platform'].shape[0]
    #agent = TimeoutWithOracleAgent(nb_resources, nb_nodes, t_turnoff=3, t_turnon=1, min_idle_time=60*5)
    agent = TimeoutAgent(args.timeout, nb_resources, nb_nodes)
    results = agent.play(env, verbose=args.verbose, render=args.render)

    if args.plot_results:
        plot_simulation_graphics(GridEnv.OUTPUT, show=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="OffReservation-v0", type=str)
    parser.add_argument("--log_dir", default="../weights/", type=str)
    parser.add_argument("--timeout", default=1, type=int)
    parser.add_argument("--plot_results", default=False, action="store_true")
    parser.add_argument("--verbose", default=True, action="store_true")
    parser.add_argument("--render", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
