import argparse
import math
import multiprocessing
import itertools
import os
from collections import defaultdict

import gym
import pandas as pd
import numpy as np
import tensorflow as tf
from gym import spaces
from gridgym.envs.off_reservation_env import OffReservationEnv

from gridgym.envs.grid_env import GridEnv
from gridgym.envs.simulator.utils.graphics import plot_simulation_graphics
from src.agents.ppo import PPOAgent
from src.utils.env_wrappers import make_vec_env, VecFrameStack
from src.utils.networks import *
from src.utils import loggers as log

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, queue_sz, max_walltime, max_nb_jobs, max_job_user_count):
        super().__init__(env)
        self.queue_sz = queue_sz
        self.max_job_user_count = max_job_user_count
        self.max_walltime = max_walltime
        self.max_nb_jobs = max_nb_jobs
        shape = (5 + (4*self.queue_sz) + 1 + 1 + 1,)
        self.observation_space = spaces.Box(
            low=0, high=1., shape=shape, dtype=np.float)

    def get_resources_state(self, observation):
        state = np.zeros(5)
        for n in observation['platform']:
            for r in n:
                state[int(r)] += 1
        state /= observation['agenda'].shape[0]
        return state

    def get_queue_state(self, obs):
        queue = obs['queue'][:self.queue_sz]
        nb_resources = obs['agenda'].shape[0]
        #[j.subtime, j.res, j.walltime, j.expected_time_to_start, j.user, j.profile] for j in self._get_queue()
        # [[j.subtime, j.res, j.walltime, j.expected_time_to_start, j.user] for j in self.rjms.jobs_queue])
        usr_count = defaultdict(int)
        for j in itertools.chain(*[obs['jobs_running'], obs['queue']]):
            usr_count[j[-2]] += 1

        queue_state = np.zeros(4 * self.queue_sz, dtype=np.float)
        for i, j in enumerate(queue):
            idx = 4 * i
            queue_state[idx+0] = j[1] / nb_resources
            queue_state[idx+1] = np.log(1+j[2]) / np.log(1+self.max_walltime)
            queue_state[idx+2] = min(1, (obs['time'] - j[0]) / (j[2] / 2.))
            queue_state[idx+3] = min(1, usr_count[j[4]] / self.max_job_user_count)
        return queue_state

    def get_queue_size(self, observation):
        nb_jobs = min(len(observation['queue']), self.max_nb_jobs)
        nb_jobs = np.log(1+nb_jobs) / np.log(1+self.max_nb_jobs)
        return nb_jobs

    def get_queue_load(self, observation):
        load = sum(job[1] for job in observation['queue'])
        max_load = 3*observation['agenda'].shape[0]
        load = min(max_load, load)
        load /= max_load
        return load

    def get_promise(self, observation):
        promise = -1
        if len(observation['queue']) > 0:
            promise = observation['queue'][0][3]
        return np.log(2 + promise) / np.log(2+self.max_walltime)

    def get_agenda(self, observation):
        return observation['agenda']

    def get_reservation_size(self, observation):
        r = observation['reservation_size'] / observation['platform'].shape[0]
        return r

    def get_day_and_time(self, observation):
        time = []
        days = math.floor(observation['time'] / float(60*24))
        curr_time = observation['time'] - (days * 60*24)
        curr_time = (2 * np.pi * curr_time) / (60*24)
        time.append((np.cos(curr_time) + 1) / 2.)
        time.append((np.sin(curr_time) + 1) / 2.)

        #weeks = math.floor(days / 7)
        #curr_day = days - (weeks * 7)
        #curr_day = (2 * np.pi * curr_day) / 7
        #time.append((np.cos(curr_day) + 1) / 2.)
        #time.append((np.sin(curr_day) + 1) / 2.)
        return np.asarray(time)

    def observation(self, observation):
        obs = list(self.get_resources_state(observation))
        obs.append(observation['time'] / 1440)
        obs.append(self.get_queue_size(observation))
        obs.append(self.get_promise(observation))
        obs.extend(self.get_queue_state(observation))
        return np.asarray(obs)


def get_agent(input_shape, nb_actions, nb_timesteps, seed, num_frames, nsteps, num_envs, nb_batches, epochs, summary_dir=None):
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
        vf_coef=.5,
        decay_steps=nb_timesteps / (nsteps * num_envs),  # 300
        max_grad_norm=None,
        shared=False,
        summ_dir=summary_dir)

    return agent


def build_env(env_id, num_envs=1, num_frames=1, seed=None, monitor_dir=None, info_kws=()):
    def create_wrapper(wrapper, **kwargs):
        def _thunk(env):
            return wrapper(env=env, **kwargs)
        return _thunk

    wrappers = [create_wrapper(
        ObsWrapper, queue_sz=10, max_walltime=1440, max_nb_jobs=1500, max_job_user_count=50)]

    env = make_vec_env(env_id=env_id,
                       nenv=num_envs,
                       seed=seed,
                       monitor_dir=monitor_dir,
                       info_kws=info_kws,
                       wrappers=wrappers)
    if num_frames > 1:
        env = VecFrameStack(env, num_frames)
    return env


def run(args):
    test_env = build_env(
        args.env_id, num_frames=args.nb_frames, info_kws=['workload_name'])

    input_shape, nb_actions = test_env.observation_space.shape, test_env.action_space.n
    agent = get_agent(input_shape, nb_actions, args.nb_timesteps, args.seed, args.nb_frames,
                      args.nsteps, args.num_envs, args.nb_batches, args.epochs, args.summary_dir)
    if not args.test:
        training_env = build_env(
            args.env_id,
            num_envs=args.num_envs,
            num_frames=args.nb_frames,
            seed=args.seed,
            monitor_dir=args.log_dir,
            info_kws=['workload_name'])

        loggers = log.LoggerWrapper()
        if args.log_interval != 0:
            loggers.append(log.CSVLogger(args.log_dir + "ppo_log.csv"))
        if args.verbose:
            loggers.append(log.ConsoleLogger())

        if args.weights is not None and args.cont_lr:
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
            checkpoint=None)

        if args.weights is not None:
            agent.save(args.weights)
    else:
        agent.load(args.weights)

    OffReservationEnv.TRACE = True
    test_env = build_env(args.env_id, num_frames=args.nb_frames, info_kws=['workload_name'])
    results = agent.play(
            env=test_env,
            render=args.render,
            verbose=args.verbose)

    results['policy'] = 'DeepShutdown'
    if args.output_fn is not None and results:
        pd.DataFrame([results]).to_csv(args.output_fn, index=False)
        #plot_simulation_graphics(GridEnv.OUTPUT, show=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="OffReservation-v0", type=str)
    #parser.add_argument("--env_id", default="Scheduling-v0", type=str)
    parser.add_argument("--weights", default="../weights/weights", type=str)
    parser.add_argument("--log_dir", default="../weights/", type=str)
    parser.add_argument("--output_fn", default=None, type=str)
    parser.add_argument("--summary_dir", default=None, type=str)
    parser.add_argument("--seed", default=48238, type=int)
    parser.add_argument("--nb_batches", default=16, type=int)
    parser.add_argument("--nb_timesteps", default=40e6, type=int)
    parser.add_argument("--nb_frames", default=20, type=int)
    parser.add_argument("--nsteps", default=1440,  type=int)
    parser.add_argument("--num_envs", default=16, type=int)
    parser.add_argument("--epochs", default=4,  type=int)
    parser.add_argument("--discount", default=.99, type=float)
    parser.add_argument("--lam", default=.95, type=float)
    parser.add_argument("--log_interval", default=1,  type=int)
    parser.add_argument("--verbose", default=True, action="store_true")
    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument("--cont_lr", default=False, action="store_true")
    parser.add_argument("--load_vf", default=False, action="store_true")
    parser.add_argument("--test", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
