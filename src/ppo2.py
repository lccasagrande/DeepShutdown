import gym
import gym_grid.envs.grid_env as g
import sys
import multiprocessing
import os.path as osp
import tensorflow as tf
import numpy as np
import os
import argparse
import pandas as pd
import benchmark
import utils
from collections import defaultdict
from baselines.common.tf_util import get_session
from baselines import bench, logger
from baselines.acer import acer
from baselines.a2c import a2c
from baselines.acer import acer
from baselines.ppo2 import ppo2
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lnlstm, lstm
from baselines.common import misc_util
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.bench import Monitor
from baselines.common.models import cnn_lstm
from baselines.common.retro_wrappers import RewardScaler
from baselines.common.models import register
try:
    from mpi4py import MPI
except ImportError:
    MPI = None



@register("cnn_1")
def cnn_small(**conv_kwargs):
    def network_fn(X):
        #h = tf.cast(X, tf.float32) / 255.

        activ = tf.nn.relu
        h = activ(conv(X, 'c1', nf=16, rf=2, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
        h = conv_to_fc(h)
        #h = activ(fc(h, 'fc1', nh=20, init_scale=np.sqrt(2)))
        return h
    return network_fn

@register("mlp_small")
def mlp(num_layers=1, num_hidden=20, activation=tf.nn.relu, layer_norm=False):
    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        return h

    return network_fn


def build_env(args):
    nenv = args.num_env or multiprocessing.cpu_count()

    get_session(tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1))

    env = make_vec_env(args.env, nenv, args.seed,
                       reward_scale=args.reward_scale)

    return env


def make_vec_env(env_id, nenv, seed, reward_scale):
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

    def make_env(rank):  # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + 10000*mpi_rank +
                     rank if seed is not None else None)
            env = Monitor(env,
                          logger.get_dir() and os.path.join(
                              logger.get_dir(), str(mpi_rank) + '.' + str(rank)),
                          allow_early_resets=True)

            return RewardScaler(env, reward_scale) if reward_scale != 1 else env
        return _thunk

    misc_util.set_global_seeds(seed)
    return SubprocVecEnv([make_env(i) for i in range(nenv)]) if nenv > 1 else DummyVecEnv([make_env(0)])


def train(args):
    print('Training with PPO2')
    env = build_env(args)

    #model = a2c.learn(env=env,
    #                 network=args.network,
    #                 seed=args.seed,
    #                 nsteps=5,
    #                 total_timesteps=int(args.num_timesteps),
    #                 lr=1.e-3,
    #                 load_path=args.load_path)

    model = ppo2.learn(
        env=env,
        seed=args.seed,
        total_timesteps=args.num_timesteps,
        nsteps=64,
        lam=0.95,
        gamma=0.99,
        network=args.network,
        lr=1.e-3, # f * 2.5e-4,
        noptepochs=4,
        log_interval=1,
        nminibatches=4,
        ent_coef=.01,
        normalize_observations=True,
        value_network='copy',
        cliprange=0.4, #0.2 value_network='copy' normalize_observations=True estimate_q=True
        load_path=args.load_path)
#
    return model, env


def train_model(args):
    def config_log():
        # configure logger, disable logging in child MPI processes (with rank > 0)
        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            rank = 0
            logger.configure()
        else:
            logger.configure(format_strs=[])
            rank = MPI.COMM_WORLD.Get_rank()
        return rank

    rank = config_log()
    model, env = train(args)
    env.close()

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

def play(model, env, n_ep=1, render=False, reward_scale=1.):
    def initialize_placeholders(nlstm=128, **kwargs):
        return np.zeros((1, 2*nlstm)), np.zeros((1))

    result = defaultdict(float)
    state, dones = initialize_placeholders()
    env.close()
    obs = env.reset()
    for i in range(1, n_ep+1):        
        score, steps = 0, 0
        while True:
            if render:
                env.render(mode='image')

            actions, _, state, _ = model.step(obs, S=state, M=dones)
            obs, rew, done, info = env.step(actions)
            done = done.any() if isinstance(done, np.ndarray) else done

            score += rew[0] * (1/reward_scale)
            steps += 1

            if done:
                result['total_slowdown'] = info[0]['total_slowdown']
                result['makespan'] = info[0]['makespan']
                result['energy_consumed'] = info[0]['energy_consumed']
                result['mean_slowdown'] = info[0]['mean_slowdown']
                result['score'] = score
                result['steps'] = steps
                result['Episode'] = i
                utils.print_episode_result("PPO2", result)
                break

def test_model(args):
    args.load_path = args.save_path if args.load_path == None else args.load_path
    args.save_path = None
    args.num_timesteps = 0
    args.num_env = 1

    model, env = train(args)
    env.close()
    env = build_env(args)
    play(model, env, args.test_ep, args.render, args.reward_scale)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='grid-v0')
    parser.add_argument('--network', help='Network', default='mlp_small', type=str)
    parser.add_argument('--num_timesteps', type=int, default=5000000)
    parser.add_argument('--num_env', default=12, type=int)
    parser.add_argument('--reward_scale', default=1., type=float)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--save_path', default='weights/ppo', type=str)
    parser.add_argument('--load_path', default=None, type=str) #"weights/ppo_slowdown"
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--test_ep', default=1, type=int)
    parser.add_argument('--render', default=False, action='store_true')
    args = parser.parse_args()
    return args


def main(args):
    if args.train:
        train_model(args)
    else:
        test_model(args)

    print("Done!")


if __name__ == '__main__':
    main(arg_parser())
