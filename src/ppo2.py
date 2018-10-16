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

    # model = a2c.learn(env=env,
    #                  network=network,
    #                  seed=seed,
    #                  nsteps=5,
    #                  total_timesteps=total_timesteps,
    #                  lr=2.5e-3,
    #                  load_path=args.load_path)

    model = ppo2.learn(
        env=env,
        seed=args.seed,
        total_timesteps=args.num_timesteps,
        nsteps=256,
        lam=0.95,
        gamma=0.99,
        network=args.network,
        lr=1.e-3,
        noptepochs=4,
        log_interval=1,
        nminibatches=4,
        ent_coef=.01,
        cliprange=0.2,
        load_path=args.load_path)

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


def test_model(args):
    def initialize_placeholders(nlstm=128, **kwargs):
        return np.zeros((args.num_env or 1, 2*nlstm)), np.zeros((1))

    assert args.load_path != None
    args.num_timesteps = 0
    args.num_env = 1
    result = defaultdict(float)

    model, env = train(args)
    env.close()
    env = build_env(args)
    env.close()
    state, dones = initialize_placeholders()
    obs = env.reset()
    steps, score = 0, 0
    while True:
        if args.render:
            env.render(mode='image')

        actions, _, state, _ = model.step(obs, S=state, M=dones)
        obs, rew, done, info = env.step(actions)
        done = done.any() if isinstance(done, np.ndarray) else done

        result['score'] += rew[0] * (1/args.reward_scale)
        result['steps'] += 1

        if done:
            result['Episode'] = 1
            result['Slowdown'] = info[0]['total_slowdown']
            result['Makespan'] = info[0]['makespan']
            utils.print_episode_result("PPO2", result)
            break

   #pd.DataFrame.from_dict(result).to_csv("benchmark/ppo.csv", index='Episode')


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='grid-v0')
    parser.add_argument('--network', help='Network', default='mlp', type=str)
    parser.add_argument('--num_timesteps', type=int, default=50e6)
    parser.add_argument('--num_env', default=None, type=int)
    parser.add_argument('--reward_scale', default=.1, type=float)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--save_path', default='weights/ppo_slowdown_200', type=str)
    parser.add_argument('--load_path', default='weights/ppo_slowdown_200', type=str) #"weights/ppo_slowdown"
    parser.add_argument('--train', default=True, action='store_true')
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
