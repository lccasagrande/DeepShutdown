import gym
import gym_grid.envs.grid_env as g
import sys
import multiprocessing
import os.path as osp
import tensorflow as tf
import numpy as np
import os
import argparse
from baselines.common.tf_util import get_session
from baselines import bench, logger
from baselines.a2c import a2c
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common import misc_util
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.bench import Monitor
from baselines.common.retro_wrappers import RewardScaler

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def ann(scaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2,
                    init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=2, stride=1,
                    init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


def build_env(args):
    nenv = args.num_env or multiprocessing.cpu_count()

    get_session(tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1))

    env = make_vec_env(args.env, nenv, args.seed, reward_scale=args.reward_scale)

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
    total_timesteps = int(args.num_timesteps)
    seed = args.seed
    network = ann
    env = build_env(args)

    print('Training with a2c')

    model = a2c.learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        network=network
    )

    return model, env


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = train(args)
    env.close()

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    #if args.play:
    #    logger.log("Running trained model")
    #    env = build_env(args)
    #    obs = env.reset()
#
    #    def initialize_placeholders(nlstm=128, **kwargs):
    #        return np.zeros((args.num_env or 1, 2*nlstm)), np.zeros((1))
    #    state, dones = initialize_placeholders(**extra_args)
    #    while True:
    #        actions, _, state, _ = model.step(obs, S=state, M=dones)
    #        obs, _, done, _ = env.step(actions)
    #        env.render()
    #        done = done.any() if isinstance(done, np.ndarray) else done
#
    #        if done:
    #            obs = env.reset()
#
    #    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='grid-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=123)
    parser.add_argument('--num_timesteps', type=float, default=1e2),
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=0.0001, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default="weights", type=str)
    args = parser.parse_args()
    main(args)
