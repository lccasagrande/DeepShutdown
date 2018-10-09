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
from collections import defaultdict
from baselines.common.tf_util import get_session
from baselines import bench, logger
from baselines.acer import acer
from baselines.a2c import a2c
from baselines.ppo2 import ppo2
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


@register("ann_lstm")
def ann_lstm(nlstm=256, layer_norm=False, **conv_kwargs):
    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        h = ann(X, **conv_kwargs)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn

def ann(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c3', nf=32, rf=2, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h1 = conv_to_fc(h)
    return activ(fc(h1, 'fc1', nh=64, init_scale=np.sqrt(2)))


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
    total_timesteps = int(args.num_timesteps)
    seed = int(args.seed)
    network = "ann_lstm"
    env = build_env(args)

    print('Training with PPO2')

    model = ppo2.learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        nsteps=1000,
        gamma=1,
        network=network,
        lr=2.5e-4,
        noptepochs=4,
        log_interval=1,
        nminibatches=4,
        ent_coef=.01,
        cliprange=lambda f: f * 0.1,
        load_path=args.load_path)

    return model, env


def train_model(args):
    args.load_path = None
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

def test_model(args):
    def initialize_placeholders(nlstm=128, **kwargs):
        return np.zeros((args.num_env or 1, 2*nlstm)), np.zeros((1))

    metrics = ["nb_jobs", "nb_jobs_finished", "nb_jobs_killed", "success_rate", "makespan",
               "mean_waiting_time", "mean_turnaround_time", "mean_slowdown", "energy_consumed",
               'total_slowdown', 'total_turnaround_time', 'total_waiting_time']

    assert args.load_path != None
    args.num_timesteps = 0
    args.num_env = 1
    result = defaultdict(list)
    model, env = train(args)
    env.close()

    env = build_env(args)
    env.close()

    state, dones = initialize_placeholders()

    for i in range(1, args.test_ep + 1):
        obs = env.reset()
        steps, score = 0, 0
        while True:
            #env.render(mode='image')
            actions, _, state, _ = model.step(obs, S=state, M=dones)
            obs, rew, done, info = env.step(actions)
            done = done.any() if isinstance(done, np.ndarray) else done
            steps += 1
            score += rew[0] * (1/args.reward_scale)

            if done:
                print("\nPPO Steps {} - Episode {:7}, Score: {:7} - Slowdown Sum {:7} Mean {:3} - Makespan {:7}".format(
                    steps, i, score, info[0]['total_slowdown'], info[0]['mean_slowdown'], info[0]['makespan']))
                for metric in metrics:
                    result[metric].append(info[0][metric])
                result['score'].append(score)
                result['steps'].append(steps)
                break

    # PRINT
    for metric in metrics:
        if args.plot:
            benchmark.plot_results(result[metric], metric)
        pd.DataFrame.from_dict(result[metric]).to_csv("benchmark/ppo_"+metric+".csv", index=False)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='grid-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=123)
    parser.add_argument('--num_timesteps', type=float, default=10e6),
    parser.add_argument('--num_env', default=None, type=int)
    parser.add_argument('--reward_scale', default=1., type=float)
    parser.add_argument('--save_path', default='weights/ppo_grid', type=str)
    parser.add_argument('--network', help='Network', default='cnn', type=str)
    parser.add_argument('--load_path', default="weights/ppo_grid", type=str)
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--plot', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--test_ep', default=1, type=int)
    args = parser.parse_args()
    return args

def main(args):
    if args.train:
        train_model(args)

    if args.test:
        test_model(args)

if __name__ == '__main__':
    args = arg_parser()
    main(args)
