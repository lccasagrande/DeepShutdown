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
    def __init__(self):
        super().__init__()

    def process_state_batch(self, batch):
        return np.asarray([v[0] for v in batch])

    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        return observation


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

    model = build_model(input_shape=env.observation_space.shape,
                        output_shape=env.action_space.n)

    memory = SequentialMemory(limit=1000000, window_length=10)

    test_policy = Greedy(env.nb_resources, env.job_slots)
    train_policy = LinearAnnealedPolicy(inner_policy=EpsGreedy(env.nb_resources, env.job_slots),
                                        attr='eps',
                                        value_max=1.,
                                        value_min=.1,
                                        value_test=.05,
                                        nb_steps=1000000)

    dqn = DQNAgent(model=model,
                   nb_actions=env.action_space.n,
                   policy=train_policy,
                   test_policy=test_policy,
                   processor=GridProcessor(),
                   memory=memory,
                   nb_steps_warmup=50000,
                   gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)

    dqn.compile(Adam(lr=.0001), metrics=['mae'])

    callbacks = [
        ModelIntervalCheckpoint('weights/'+name+'/' +
                                name+'_weights_{step}.h5f', interval=100000),
        FileLogger('log/'+name+'/'+name+'_log.json', interval=1),
        TensorBoard(log_dir='log/'+name+'/'+name)
    ]
    if train:
        dqn.fit(env=env,
                callbacks=callbacks,
                nb_steps=3000000,
                log_interval=10000,
                visualize=False,
                verbose=2,
                nb_max_episode_steps=2000)

        dqn.save_weights('weights/'+name+'/'+name +
                         '_weights_0.h5f', overwrite=True)
    else:
        dqn.load_weights('weights/'+name+'/'+name +
                         '_weights_'+weights_nb+'.h5f')
        dqn.test(env, nb_episodes=100, visualize=True)
