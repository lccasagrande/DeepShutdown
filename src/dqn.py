from __future__ import division
from PIL import Image
import argparse
import numpy as np
import random
import gym
import math
import time
import gym_grid.envs.grid_env as g
import keras.backend as K
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation, Flatten, Convolution2D, BatchNormalization, Permute, GRU, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from policies import Greedy, EpsGreedy
from dqn_agent import DQNAgent
from rl.policy import LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class GridProcessor(Processor):
    def __init__(self, job_slots, backlog, time_window, nb_res, max_slowdown, max_efficiency):
        super().__init__()
        self.max_slowdown = max_slowdown
        self.max_efficiency = max_efficiency
        self.time_window = time_window
        self.job_slots = job_slots
        self.backlog = backlog
        self.nb_res = nb_res
        self.output_shape = (self.time_window, self.nb_res +
                             (self.job_slots*self.nb_res) + self.backlog, 1)

    def process_state_batch(self, batch):
        return np.asarray([v[0] for v in batch])

    def process_observation(self, observation):
        gantt = observation['gantt']
        jobs_in_queue = observation['job_queue']['jobs']
        nb_jobs_in_queue = observation['job_queue']['nb_jobs_in_queue']
        nb_jobs_waiting = observation['job_queue']['nb_jobs_waiting']

        obs = np.zeros(shape=self.output_shape, dtype=np.int8)

        for res, resource_space in enumerate(gantt):
            job = resource_space['queue'][0]
            if job != None:
                time_window = min(self.time_window, int(job.remaining_time))
                #tmp = obs[0:time_window]
                obs[0:time_window, res] = [1]

        index = self.nb_res
        job_slots = min(self.job_slots, nb_jobs_in_queue)
        for job_slot in range(job_slots):
            job = jobs_in_queue[job_slot]
            time_window = min(self.time_window, job.requested_time)
            start_idx = (job_slot*self.nb_res) + index
            end_idx = start_idx + job.requested_resources
            obs[0:time_window, start_idx:end_idx] = [1]

        nb_jobs_in_queue -= len(jobs_in_queue) + nb_jobs_waiting
        backlog_slot = 0
        index = self.nb_res + (self.job_slots*self.nb_res)
        while nb_jobs_in_queue != 0 and backlog_slot != self.backlog:
            time_window = min(self.time_window, nb_jobs_in_queue)
            obs[0:time_window, index+backlog_slot] = [1]
            backlog_slot += 1
            nb_jobs_in_queue -= time_window

        assert obs.ndim == 3  # (height, width, channel)
        return obs


def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(Permute((1, 2, 3), input_shape=input_shape))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (2, 2), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(output_shape))
    model.add(Activation('linear'))
    print(model.summary())
    return model


if __name__ == "__main__":
    train = True
    weights_nb = 0
    name = "dqn_1"
    seed = 123

    env = gym.make('grid-v0')
    np.random.seed(seed)
    env.seed(seed)

    processor = GridProcessor(job_slots=env.job_slots,
                              backlog=env.backlog,
                              time_window=env.time_window,
                              nb_res=env.nb_resources,
                              max_slowdown=env.max_slowdown,
                              max_efficiency=env.max_energy_consumption)

    model = build_model(processor.output_shape, env.action_space.n)

    memory = SequentialMemory(limit=100000, window_length=32)

    test_policy = Greedy(env.nb_resources, env.job_slots)
    train_policy = LinearAnnealedPolicy(inner_policy=EpsGreedy(env.nb_resources, env.job_slots),
                                        attr='eps',
                                        value_max=1.,
                                        value_min=.1,
                                        value_test=.05,
                                        nb_steps=500000)

    dqn = DQNAgent(model=model,
                   nb_actions=env.action_space.n,
                   policy=train_policy,
                   test_policy=test_policy,
                   memory=memory,
                   processor=processor,
                   nb_steps_warmup=50000,
                   gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)

    dqn.compile(Adam(lr=.0001), metrics=['mae'])

    callbacks = [
        ModelIntervalCheckpoint('weights/'+name+'/' + name+'_weights_{step}.h5f', interval=100000),
        FileLogger('log/'+name+'/'+name+'_log.json', interval=1),
        TensorBoard(log_dir='log/'+name+'/'+name)
    ]
    if train:
        dqn.fit(env=env,
                callbacks=callbacks,
                nb_steps=1750000,
                log_interval=10000,
                visualize=False,
                verbose=1,
                nb_max_episode_steps=2000)

        dqn.save_weights('weights/'+name+'/'+name +
                         '_weights_0.h5f', overwrite=True)
    else:
        dqn.load_weights('weights/'+name+'/'+name +
                         '_weights_'+weights_nb+'.h5f')
        dqn.test(env, nb_episodes=100, visualize=True)
