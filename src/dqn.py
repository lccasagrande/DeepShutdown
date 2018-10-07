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
from rl.agents import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


WINDOW_LENGTH = 4


class GridProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 2  # (height, width)
        return observation.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch


def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (2, 2), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
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

    model = build_model(input_shape= (WINDOW_LENGTH,) + env.observation_space.shape, output_shape=env.action_space.n)

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

    train_policy = LinearAnnealedPolicy(inner_policy=EpsGreedyQPolicy(),
                                        attr='eps',
                                        value_max=1.,
                                        value_min=.1,
                                        value_test=.05,
                                        nb_steps=3000000)

    dqn = DQNAgent(model=model,
                   nb_actions=env.action_space.n,
                   policy=train_policy,
                   processor=GridProcessor(),
                   memory=memory,
                   nb_steps_warmup=100000,
                   gamma=1,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)

    dqn.compile(Adam(lr=.00001), metrics=['mae'])

    callbacks = [
        ModelIntervalCheckpoint('weights/'+name+'/' +
                                name+'_weights_{step}.h5f', interval=250000),
        FileLogger('log/'+name+'/'+name+'_log.json', interval=1),
        TensorBoard(log_dir='log/'+name+'/'+name)
    ]
    if train:
        dqn.fit(env=env,
                callbacks=callbacks,
                nb_steps=9000000,
                log_interval=10000,
                visualize=False,
                verbose=1,
                nb_max_episode_steps=4000)

        dqn.save_weights('weights/'+name+'/'+name +
                         '_weights_0.h5f', overwrite=True)
    else:
        dqn.load_weights('weights/'+name+'/'+name +
                         '_weights_'+weights_nb+'.h5f')
        dqn.test(env, nb_episodes=100, visualize=True)
