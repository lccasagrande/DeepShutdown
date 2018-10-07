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
from baselines.acer import acer
from baselines.a2c import a2c
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lnlstm
from baselines.common import misc_util
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.bench import Monitor
from baselines.common.retro_wrappers import RewardScaler

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def cnn_lstm(nlstm=128, **conv_kwargs):
    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        h = ann(X, **conv_kwargs)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        h5, snew = lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn

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
    ac = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
    x = activ(ac)
    return x, None


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

    model = acer.learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        network=network,
        load_path=args.load_path
    )

    return model, env


def main(args):
    argstrain = True
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
        
    logger.log("Running trained model")
    env = build_env(args)
    obs = env.reset()

    def initialize_placeholders(nlstm=128, **kwargs):
        return np.zeros((args.num_env or 1, 2*nlstm)), np.zeros((1))
    state, dones = initialize_placeholders()
    while True:
        actions, _, state, _ = model.step(obs, S=state, M=dones)
        obs, _, done, _ = env.step(actions)
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done

        if done:
            obs = env.reset()

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='grid-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=123)
    parser.add_argument('--num_timesteps', type=float, default=0),
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--load_path', help='Path to save trained model to', default="weights/a2c", type=str)
    args = parser.parse_args()
    main(args)
