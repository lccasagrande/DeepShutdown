import argparse
import numpy as np
import random
import gym
import math
import time
import gym_grid.envs.grid_env as g
import utils
import keras.backend as K
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation, Flatten, Convolution2D, BatchNormalization, Permute, LSTM, TimeDistributed, GRU, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from rl.agents import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


WINDOW_LENGTH = 8


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
    model.add(Convolution2D(32, (2, 2), strides=(1, 1),
                            padding='same', activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(output_shape, activation='linear'))
    print(model.summary())
    return model


if __name__ == "__main__":
    train = True
    weights_nb = 0
    name = "ddqn_1"
    seed = 123
    weight_path = "weights/" + name
    log_path = "log/" + name
    utils.clean_or_create_dir(weight_path)
    utils.clean_or_create_dir(log_path)

    env = gym.make('grid-v0')
    np.random.seed(seed)
    env.seed(seed)

    model = build_model(input_shape=(
        WINDOW_LENGTH,) + env.observation_space.shape, output_shape=env.action_space.n)

    memory = SequentialMemory(limit=4000000, window_length=WINDOW_LENGTH)

    train_policy = LinearAnnealedPolicy(inner_policy=EpsGreedyQPolicy(),
                                        attr='eps',
                                        value_max=1.,
                                        value_min=.1,
                                        value_test=.05,
                                        nb_steps=4e6)

    dqn = DQNAgent(model=model,
                   nb_actions=env.action_space.n,
                   policy=train_policy,
                   processor=GridProcessor(),
                   memory=memory,
                   dueling_type='max',
                   enable_dueling_network=True,
                   nb_steps_warmup=100000,
                   gamma=.99,
                   target_model_update=10000,
                   train_interval=8,
                   delta_clip=1.)

    dqn.compile(Adam(lr=.00001), metrics=['mae', 'mse'])

    callbacks = [
        ModelIntervalCheckpoint(weight_path + '/weights_{step}.h5f', interval=5e5),
        FileLogger(log_path+'/log.json', interval=100),
        TensorBoard(log_dir=log_path,
                    write_graph=True, write_images=True)
    ]
    if train:
        dqn.fit(env=env,
                callbacks=callbacks,
                nb_steps=10e6,
                log_interval=10000,
                visualize=False,
                verbose=1,
                nb_max_episode_steps=4000)

        dqn.save_weights(weight_path+'/weights_0.h5f', overwrite=True)
    else:
        dqn.load_weights(weight_path+'/weights_'+weights_nb+'.h5f')
        dqn.test(env, nb_episodes=100, visualize=True)
