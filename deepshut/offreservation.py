import argparse
import math
import multiprocessing
import itertools
import os

import pandas as pd
import gym
from gym import spaces

from gridgym.envs.off_reservation_env import OffReservationEnv
from gridgym.envs.grid_env import GridEnv
from gridgym.envs.simulator.utils.graphics import plot_simulation_graphics
from src.agents.offreservation import OffReservationAgent


def run(args):
    OffReservationEnv.TRACE = True
    env = gym.make(args.env_id)
    nb_resources = env.observation_space.spaces['agenda'].shape[0]
    nb_nodes = env.observation_space.spaces['platform'].shape[0]
    agent = OffReservationAgent(args.max_strecth, nb_resources, nb_nodes, args.queue_sz, args.timeout)
    results = agent.play(env, verbose=args.verbose, render=args.render)

    results['policy'] = 'OR'
    if args.output_fn is not None and results:
        pd.DataFrame([results]).to_csv(args.output_fn, index=False)

    if args.plot_results:
        plot_simulation_graphics(GridEnv.OUTPUT, show=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="OffReservation-v0", type=str)
    parser.add_argument("--plot_results", default=False, action="store_true")
    parser.add_argument("--output_fn", default=None, type=str)
    parser.add_argument("--verbose", default=True, action="store_true")
    parser.add_argument("--render", default=False, action="store_true")
    # Agent specific args
    parser.add_argument("--timeout", default=0, type=int)
    parser.add_argument("--queue_sz", default=10, type=int)
    parser.add_argument("--max_strecth", default=0.5, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())