
import argparse
import math
import multiprocessing
import itertools

import tensorflow as tf
import gym
import numpy as np
import pandas as pd
from gym import spaces

import gridgym.envs.grid_env as g
from gridgym.envs.simulator.utils.graphics import plot_simulation_graphics
from gridgym.envs.grid_env import GridEnv
from src.utils import loggers as log
from src.utils.networks import *
from src.utils.env_wrappers import make_vec_env, VecFrameStack
from src.agents.ppo import PPOAgent
from src.agents.timeout import TimeoutAgent


class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, queue_sz, max_walltime, max_nb_jobs):
        super().__init__(env)
        self.queue_sz = queue_sz
        self.max_walltime = max_walltime
        self.max_nb_jobs = max_nb_jobs
        obs = self.reset()
        self.observation_space = spaces.Box(
            low=0, high=1., shape=obs.shape, dtype=np.float)

    def get_queue_state(self, observation):
        queue = observation['queue']
        nb_resources = observation['agenda'].shape[0]
        queue_state = np.zeros(2 * self.queue_sz, dtype=np.float)
        for i, job in enumerate(list(itertools.takewhile(lambda j: j[0] != -1, queue[:self.queue_sz]))):
            idx = i * 2
            queue_state[idx] = job[1] / nb_resources
            queue_state[idx + 1] = np.log(job[2]) / np.log(self.max_walltime)
        return queue_state

    def get_promise(self, observation):
        promise = observation['queue'][0][-1]
        minimum = -1
        promise = np.log(1 + promise - minimum) / \
            np.log(1 + self.max_walltime - minimum)
        return promise

    def get_resources_state(self, observation):
        platform = observation['platform']
        state = np.zeros(5)
        for node in platform:
            for r_state in node:
                state[r_state] += 1
        state /= observation['agenda'].shape[0]
        return state

    def get_queue_size(self, observation):
        nb_jobs = len(list(itertools.takewhile(
            lambda j: j[0] != -1, observation['queue'])))
        nb_jobs = min(nb_jobs, self.max_nb_jobs)
        nb_jobs = np.log(1+nb_jobs) / np.log(1+self.max_nb_jobs)
        return nb_jobs

    def get_agenda(self, observation):
        a = [1-((observation['time'] - r[0]) / (r[1] - r[0]))
             if r[1] != 0 else 0 for r in observation['agenda']]
        return a

    def get_reservation_size(self, observation):
        r = observation['reservation_size'] / observation['platform'].shape[0]
        return r

    def get_day_and_time(self, observation):
        time = np.zeros(4)
        days = math.floor(observation['time'] / float(60*60*24))
        weeks = math.floor(days / 7)
        curr_time = observation['time'] - (days * 60*60*24)
        curr_day = days - (weeks * 7)

        curr_time = (2 * np.pi * curr_time) / (60*60*24)
        curr_day = (2 * np.pi * curr_day) / 7
        time[0] = (np.cos(curr_time) + 1) / 2.
        time[1] = (np.sin(curr_time) + 1) / 2.
        time[2] = (np.cos(curr_day) + 1) / 2.
        time[3] = (np.sin(curr_day) + 1) / 2.
        return time

    def observation(self, observation):
        obs = list(self.get_resources_state(observation))
        obs.extend(self.get_agenda(observation))
        obs.append(self.get_reservation_size(observation))
        obs.append(self.get_promise(observation))
        obs.extend(self.get_queue_state(observation))
        obs.append(self.get_queue_size(observation))
        obs.extend(self.get_day_and_time(observation))
        obs = np.asarray(obs, dtype=np.float32)
        return obs


def get_agent(input_shape, nb_actions, seed, num_frames, nsteps, num_envs, nb_batches, epochs, summary_dir=None):
    agent = PPOAgent(seed)
    lstm_shape = (num_frames, input_shape[-1] // num_frames)
    agent.compile(
        input_shape=input_shape,
        nb_actions=nb_actions,
        p_network=lstm_mlp(128, lstm_shape, [64], activation=tf.nn.leaky_relu),
        batch_size=(nsteps * num_envs) // nb_batches,
        epochs=epochs,
        lr=5e-4,
        end_lr=1e-6,
        ent_coef=0.005,
        vf_coef=1.,
        decay_steps=300,  # args.nb_timesteps / (args.nsteps * args.num_envs)
        max_grad_norm=None,
        shared=True,
        summ_dir=summary_dir)

    return agent


def build_env(env_id, num_envs=1, num_frames=1, seed=None, monitor_dir=None, info_kws=()):
    def create_wrapper(wrapper, **kwargs):
        def _thunk(env):
            return wrapper(env=env, **kwargs)
        return _thunk

    wrappers = [create_wrapper(
        ObsWrapper, queue_sz=20, max_walltime=345600, max_nb_jobs=3000)]

    env = make_vec_env(env_id=env_id,
                       nenv=num_envs,
                       seed=seed,
                       monitor_dir=monitor_dir,
                       info_kws=info_kws,
                       wrappers=wrappers)
    if num_frames > 1:
        env = VecFrameStack(env, num_frames, is_episodic=True)
    return env


def run(args):
    env = gym.make(args.env_id)
    nb_resources = env.observation_space.spaces['platform'].shape[0] * \
        env.observation_space.spaces['platform'].shape[1]
    nb_nodes = env.observation_space.spaces['platform'].shape[0]
    agent = TimeoutAgent(1, nb_resources, nb_nodes)
    agent.play(env, verbose=True)
    return
    test_env = build_env(args.env_id, 1, args.nb_frames)
    input_shape, nb_actions = test_env.observation_space.shape, test_env.action_space.n
    agent = get_agent(input_shape, nb_actions, args.seed, args.nb_frames,
                      args.nsteps, args.num_envs, args.nb_batches, args.epochs, args.summary_dir)
    if not args.test:
        training_env = build_env(args.env_id, args.num_envs,
                                 args.nb_frames, args.seed, args.log_dir)
        loggers = log.LoggerWrapper()
        if args.log_interval != 0:
            loggers.append(log.CSVLogger(args.log_dir + "ppo_log.csv"))
        if args.verbose:
            loggers.append(log.ConsoleLogger())

        if args.weights is not None and args.continue_learning:
            agent.load(args.weights)

        if args.weights is not None and args.load_vf:
            agent.load_value(args.weights)

        history = agent.fit(
            env=training_env,
            clip_value=.2,
            lam=args.lam,
            timesteps=args.nb_timesteps,
            nsteps=args.nsteps,
            gamma=args.discount,
            log_interval=args.log_interval,
            loggers=loggers,
            nb_batches=args.nb_batches,
            checkpoint=args.log_dir+"checkpoints/")

        if args.weights is not None:
            agent.save(args.weights)
    else:
        agent.load(args.weights)

    results = agent.play(env=test_env, render=args.render,
                         verbose=args.verbose)
    plot_simulation_graphics(GridEnv.OUTPUT, show=True)
    if args.output_fn is not None and results:
        pd.DataFrame([results]).to_csv(args.output_fn, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="OffReservation-v0", type=str)
    parser.add_argument(
        "--weights", default="../weights/checkpoints/clrlcgce02_2", type=str)
    parser.add_argument(
        "--output_fn", default="../weights/clrlcgce02_2.csv", type=str)
    parser.add_argument("--log_dir", default="../weights/", type=str)
    parser.add_argument("--summary_dir", default=None, type=str)
    parser.add_argument("--seed", default=48238, type=int)
    parser.add_argument("--nb_workloads", default=2, type=int)
    parser.add_argument("--nb_batches", default=64, type=int)
    parser.add_argument("--nb_timesteps", default=2e6, type=int)
    parser.add_argument("--nb_frames", default=20, type=int)
    parser.add_argument("--nsteps", default=1440,  type=int)
    parser.add_argument("--num_envs", default=16, type=int)
    parser.add_argument("--epochs", default=4,  type=int)
    parser.add_argument("--discount", default=.99, type=float)
    parser.add_argument("--lam", default=.95, type=float)
    parser.add_argument("--log_interval", default=1,  type=int)
    parser.add_argument("--verbose", default=True, action="store_true")
    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--continue_learning",
                        default=False, action="store_true")
    parser.add_argument("--load_vf", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
