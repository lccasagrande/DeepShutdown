import argparse
import math
import multiprocessing
import itertools
from collections import defaultdict
import joblib
import os

import gym
import numpy as np
import tensorflow as tf
from gym import spaces
from gym import ObservationWrapper

import simple_rl.utils.loggers as log
from simple_rl.PolicyGradient.ppo import ProximalPolicyOptimization
from simple_rl.utils.wrappers import make_vec_env, VecFrameStack
from gridgym.envs.off_reservation_env import OffReservationEnv
from gridgym.envs.grid_env import GridEnv
from batsim_py.utils.graphics import plot_simulation_graphics

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# multiprocessing.set_start_method('spawn', True)


class ObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super(ObsWrapper, self).__init__(env)
        self.queue_sz = 10
        self.max_job_user_count = 50
        self.max_walltime = 1440
        self.max_nb_jobs = 4500
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
        # [j.subtime, j.res, j.walltime, j.expected_time_to_start, j.user, j.profile] for j in self._get_queue()
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
            queue_state[idx+3] = min(1, usr_count[j[4]] /
                                     self.max_job_user_count)
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
        return np.asarray(time)

    def observation(self, observation):
        obs = list(self.get_resources_state(observation))
        obs.append(observation['time'] / 1440)
        obs.append(self.get_queue_size(observation))
        obs.append(self.get_promise(observation))
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
                                info_kws=['workload_name'],
                                max_queue_sz=args.max_queue_sz,
                                act_interval=args.act_interval,
                                simulation_time=args.simulation_time,
                                qos_stretch=args.qos_stretch,
                                export=False,
                                use_batsim=args.use_batsim)

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
                        info_kws=['workload_name'],
                        max_queue_sz=args.max_queue_sz,
                        act_interval=args.act_interval,
                        simulation_time=args.simulation_time,
                        qos_stretch=args.qos_stretch,
                        export=True,
                        use_batsim=args.use_batsim)

    ds_agent = DeepShutdown(env=test_env,
                            learning_rate=5e-4,
                            discount_factor=args.discount,
                            gae=args.lam,
                            steps_per_update=args.nsteps,
                            refresh_rate=100)

    ds_agent.load2(args.weights)

    scores, infos = ds_agent.play()

    if args.verbose:
        scores = [info['episode']['score'] for info in infos]
        print("[SCORE] \tAvg: {}\tMax: {}\tMin: {}".format(
            np.mean(scores), np.max(scores), np.min(scores)))

    if args.plot_results:
        results_dir = "/tmp/GridGym/{}".format(infos[-1]['workload_name'])
        plot_simulation_graphics(results_dir, show=True)


def parse_args():
    parser = argparse.ArgumentParser()
    # Agent options
    agent_group = parser.add_argument_group(title="Agent options")
    agent_group.add_argument(
        "--weights", default="../weights/checkpoints/weights", type=str)
    agent_group.add_argument("--log_dir", default="../weights/", type=str)
    agent_group.add_argument("--seed", default=48238, type=int)
    agent_group.add_argument("--nb_batches", default=16, type=int)
    agent_group.add_argument("--nb_timesteps", default=50e6, type=int)
    agent_group.add_argument("--nsteps", default=1440,  type=int)
    agent_group.add_argument("--num_envs", default=16, type=int)
    agent_group.add_argument("--epochs", default=4,  type=int)
    agent_group.add_argument("--discount", default=.99, type=float)
    agent_group.add_argument("--lam", default=.95, type=float)
    agent_group.add_argument("--continue_learning",
                             default=False, action="store_true")
    agent_group.add_argument(
        "--test_only", default=True, action="store_true")

    # Environment options
    env_group = parser.add_argument_group(title="Environment options")
    env_group.add_argument(
        "--env_id", default="OffReservation-v0", type=str)
    env_group.add_argument("--max_queue_sz", default=10, type=int)
    env_group.add_argument("--act_interval", default=1, type=int)
    env_group.add_argument("--simulation_time", default=1440, type=int)
    env_group.add_argument("--qos_stretch", default=0.5, type=float)
    env_group.add_argument("--use_batsim", action="store_true")
    env_group.add_argument("--nb_frames", default=20, type=int)

    #
    parser.add_argument("--plot_results", default=True,
                        action="store_true")
    parser.add_argument("--verbose", default=True, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
