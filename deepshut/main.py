import argparse
from collections import defaultdict
import itertools
import math
import os
import numpy as np
from typing import Sequence
from typing import List

import gym
from gym import spaces
from gym import ObservationWrapper
from gridgym.envs.off_reservation_env import OffReservationEnv
import joblib
import numpy as np
import tensorflow as tf

from simple_rl.PolicyGradient.ppo import ProximalPolicyOptimization
import simple_rl.utils.loggers as log
from simple_rl.utils.wrappers import VecFrameStack, make_vec_env

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# multiprocessing.set_start_method('spawn', True)


class ObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super(ObsWrapper, self).__init__(env)
        self.queue_sz = 10
        self.max_job_user_count = 50
        self.max_walltime = 1440
        self.max_sim_t = 1440
        self.max_nb_jobs = 4500
        self.qos_tresh = 0.5
        shape = (5 + 1 + 1 + 1 + (4*self.queue_sz),)
        self.observation_space = spaces.Box(
            low=0, high=1., shape=shape, dtype=np.float)

    def get_resources_state(self, obs) -> Sequence[float]:
        state = np.zeros(5)
        for server in obs['platform']['status']:
            for h_status in server:
                state[int(h_status) - 1] += 1
        state /= obs['platform']['agenda'].shape[0]
        return state

    def get_queue_state(self, obs) -> Sequence[float]:
        queue = [j for j in obs['queue']['jobs'] if j[1] != 0]
        nb_resources = obs['platform']['agenda'].shape[0]

        # Count the number of jobs per user
        usr_count: dict = defaultdict(int)
        for sub, res, wall, user in queue:
            if user != -1:
                usr_count[user] += 1

        jobs_running = [j for j in obs['platform']['agenda'] if j[1] > 0]
        for stat_t, wall_t, user in jobs_running:
            if user != -1:
                usr_count[user] += 1

        # Get jobs in queue
        queue_state = np.zeros(4 * self.queue_sz, dtype=np.float)
        for i, (sub, res, wall, user) in enumerate(queue):
            idx = 4 * i
            queue_state[idx+0] = res / nb_resources
            queue_state[idx+1] = wall / self.max_walltime
            queue_state[idx+2] = min(1,
                                     (obs['current_time'] - sub) / (wall * self.qos_tresh))
            if user != -1:
                queue_state[idx+3] = min(1, usr_count[user] /
                                         self.max_job_user_count)
        return queue_state

    def get_queue_size(self, obs) -> float:
        return obs['queue']['size'] / self.max_nb_jobs

    def get_promise(self, obs) -> float:
        return min(0, obs['queue']['promise'] / self.max_walltime)

    def get_current_time(self, obs) -> float:
        return obs['current_time'] / self.max_sim_t

    def observation(self, observation) -> Sequence[float]:
        obs = list(self.get_resources_state(observation))
        obs.append(self.get_queue_size(observation))
        obs.append(self.get_promise(observation))
        obs.append(self.get_current_time(observation))
        obs.extend(self.get_queue_state(observation))
        return np.asarray(obs)


class DeepShutdown(ProximalPolicyOptimization):
    def __init__(self, env, learning_rate, discount_factor, gae, steps_per_update, refresh_rate=50):
        if not isinstance(env, VecFrameStack):
            raise ValueError(
                "Expected an environment wrapped by a VecFrameStack")

        lstm_shape = (
            env.nstack, env.observation_space.shape[-1] // env.nstack)
        network = DeepShutdown.build_network(lstm_shape)

        super().__init__(env=env,
                         p_network=network,
                         learning_rate=learning_rate,
                         discount_factor=discount_factor,
                         gae=gae,
                         steps_per_update=steps_per_update,
                         vf_network=network,
                         refresh_rate=refresh_rate)

    @staticmethod
    def build_network(input_shape):
        def _build(X):
            h = tf.reshape(X, (-1,) + input_shape)
            h = tf.compat.v1.keras.layers.CuDNNLSTM(128)(h)
            h = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)(h)
            return h
        return _build

    def tf1_load_model(self, fn):
        variables = self.model.trainable_variables
        values = joblib.load(os.path.expanduser(fn))
        for v in variables:
            var_name = v.name
            spl = var_name.split("/")
            if spl[0] == "cu_dnnlstm":
                var_name = "policy_network/cu_dnnlstm/" + spl[1]
            elif spl[0] == "cu_dnnlstm_1":
                var_name = "value_network/cu_dnnlstm_1/" + spl[1]
            elif spl[0] == 'dense':
                var_name = "policy_network/dense/" + spl[1]
            elif spl[0] == 'dense_1':
                var_name = "value_network/dense_1/" + spl[1]
            elif spl[0] == 'vf':
                var_name = "value_logits/" + spl[1]
            elif spl[0] == 'policy':
                var_name = "policy_logits/" + spl[1]
            vl = values[var_name]
            v.assign(vl)

    def play(self, render=False):
        states, scores = self.env.reset(), np.zeros(self.env.num_envs)
        epscore = np.zeros_like(scores)
        infos = []
        while True:
            if render:
                self.env.render()
            states, rewards, dones, info = self.env.step(self.act(states))
            epscore += rewards
            if dones.any():
                for i in np.nonzero(dones)[0]:
                    scores[i] = epscore[i]
                    infos.append(info[i])
                    epscore[i] = 0

            if scores.all():
                return scores, infos


def make_env(env_id, num_envs=1, num_frames=1, seed=None, monitor_dir=None, sequential=True, info_kws=(), **env_args):
    env = make_vec_env(env_id=env_id,
                       num_envs=num_envs,
                       sequential=sequential,
                       seed=seed,
                       monitor_dir=monitor_dir,
                       info_kws=info_kws,
                       env_wrappers=[ObsWrapper],
                       **env_args)

    env = VecFrameStack(env, num_frames)
    return env


def run(args):
    if not args.test_only:
        training_env = make_env(env_id=args.env_id,
                                num_envs=args.num_envs,
                                num_frames=args.nb_frames,
                                seed=args.seed,
                                monitor_dir=args.log_dir,
                                sequential=True,
                                info_kws=['workload'],
                                platform_fn=args.platform,
                                workloads_dir=args.workloads_dir,
                                t_action=args.act_interval,
                                queue_max_len=args.max_queue_sz,
                                qos_treshold=args.qos_stretch,
                                hosts_per_server=args.hosts_per_server,
                                simulation_time=args.simulation_time)

        ds_agent = DeepShutdown(env=training_env,
                                learning_rate=5e-4,
                                discount_factor=args.discount,
                                gae=args.lam,
                                steps_per_update=args.nsteps,
                                refresh_rate=100)

        logger = log.LoggersWrapper()
        if args.log_dir != "":
            logger.append(log.CSVLogger(args.log_dir + "ppo_log.csv"))
        if args.verbose:
            logger.append(log.ConsoleLogger())

        if args.continue_learning:
            ds_agent.load(args.weights)

        ds_agent.train(timesteps=int(args.nb_timesteps),
                       batch_size=int(
                           (args.nsteps * args.num_envs) / args.nb_batches),
                       clip_vl=0.2,
                       entropy_coef=0.005,
                       vf_coef=.5,
                       epochs=args.epochs,
                       logger=logger)

        ds_agent.save(args.weights)

    # TEST
    test_env = make_env(env_id=args.env_id,
                        num_envs=1,
                        num_frames=args.nb_frames,
                        seed=args.seed,
                        sequential=False,
                        info_kws=['workload'],
                        platform_fn=args.platform,
                        workloads_dir=args.workloads_dir,
                        t_action=args.act_interval,
                        queue_max_len=args.max_queue_sz,
                        hosts_per_server=args.hosts_per_server,
                        simulation_time=args.simulation_time,
                        qos_treshold=args.qos_stretch)

    ds_agent = DeepShutdown(env=test_env,
                            learning_rate=5e-4,
                            discount_factor=args.discount,
                            gae=args.lam,
                            steps_per_update=args.nsteps,
                            refresh_rate=100)

    ds_agent.load(args.weights)

    scores, infos = ds_agent.play()

    if args.verbose:
        scores = [info['episode']['score'] for info in infos]
        print("[SCORE] \tAvg: {}\tMax: {}\tMin: {}".format(
            np.mean(scores), np.max(scores), np.min(scores)))


def parse_args():
    parser = argparse.ArgumentParser()
    # Agent options
    agent_group = parser.add_argument_group(title="Agent options")
    agent_group.add_argument(
        "--weights", default="../misc/weights/checkpoints/weights", type=str)
    agent_group.add_argument("--log_dir", default="../misc/weights/", type=str)
    agent_group.add_argument("--seed", default=48238, type=int)
    agent_group.add_argument("--nb_batches", default=16, type=int)
    agent_group.add_argument("--nb_timesteps", default=1e5, type=int)
    agent_group.add_argument("--nsteps", default=1440,  type=int)
    agent_group.add_argument("--num_envs", default=16, type=int)
    agent_group.add_argument("--epochs", default=4,  type=int)
    agent_group.add_argument("--discount", default=.99, type=float)
    agent_group.add_argument("--lam", default=.95, type=float)
    agent_group.add_argument("--continue_learning",
                             default=False, action="store_true")
    agent_group.add_argument("--test_only", default=False, action="store_true")

    # Environment options
    env_group = parser.add_argument_group(title="Environment options")
    env_group.add_argument("--env_id", default="OffReservation-v0", type=str)
    env_group.add_argument("--max_queue_sz", default=10, type=int)
    env_group.add_argument("--act_interval", default=1, type=int)
    env_group.add_argument("--simulation_time", default=1440, type=int)
    env_group.add_argument("--qos_stretch", default=0.5, type=float)
    env_group.add_argument("--use_batsim", action="store_true")
    env_group.add_argument("--nb_frames", default=20, type=int)
    env_group.add_argument("--hosts_per_server", default=12, type=int)

    env_group.add_argument(
        "--platform", default="../misc/platforms/platform.xml", type=str)
    env_group.add_argument(
        "--workloads_dir", default="../misc/workloads/", type=str)

    #
    parser.add_argument("--verbose", default=True, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
