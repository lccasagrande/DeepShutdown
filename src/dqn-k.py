from __future__ import division
from PIL import Image
import argparse
import numpy as np
import random
import gym
import gym_grid.envs.grid_env as g
import keras.backend as K
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, GRU
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from dqn_utils import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, Policy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class CustomGreedyQPolicy(Policy):
    def select_action(self, q_values, state):
        actions = [0]
        q_max = q_values[0]

        for i in range(q_values.shape[0]-1):
            if state[0][i][3] != 0:
                continue
            act = i+1
            if q_values[act] > q_max:
                q_max = q_values[act]
                actions = [act]
            elif q_values[act] == q_max:
                actions.append(act)

        return random.choice(actions)


class CustomEpsGreedyQPolicy(Policy):
    def __init__(self, eps=.1):
        super(CustomEpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values, state):
        assert q_values.ndim == 1

        if np.random.uniform() >= self.eps:
            return self.select_best_action(q_values, state)
            #act = np.amax(np.take(q_values, valid_actions))
            #valid_actions = np.where(q_values == act)[0]

        valid_actions = [
            0] + [(i+1) for i in range(q_values.shape[0]-1) if state[0][i][3] == 0]
        return random.choice(valid_actions)

    def get_config(self):
        config = super(CustomEpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config

    def select_best_action(self, q_values, state):
        actions = [0]
        q_max = q_values[0]

        for i in range(q_values.shape[0]-1):
            if state[0][i][3] != 0:
                continue
            act = i+1
            if q_values[act] > q_max:
                q_max = q_values[act]
                actions = [act]
            elif q_values[act] == q_max:
                actions.append(act)

        return random.choice(actions)


class GridProcessor(Processor):
    def __init__(self, max_time, max_speed, max_watt, input_shape, nb_res):
        super().__init__()
        self.time_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
        self.speed_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
        self.energy_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
        self.time_scaler.fit([[0], [max_time]])
        self.speed_scaler.fit([[0], [max_speed]])
        self.energy_scaler.fit([[0], [max_watt]])
        self.nb_res = nb_res
        self.input_shape = input_shape

    def process_state_batch(self, batch):
        return np.asarray([v[0] for v in batch])

    def process_observation(self, observation):
        self.speed_scaler.transform(
            observation[:self.nb_res, 0].reshape(-1, 1))
        self.energy_scaler.transform(observation[:self.nb_res, 1:3])
        self.time_scaler.transform(observation[:self.nb_res, 4:])
        self.time_scaler.transform(observation[-1, :].reshape(-1, 1))
        observation = observation.reshape(self.input_shape)
        assert observation.ndim == 2  # (height, width, channel)
        return observation


def build_model(output_shape, input_shape):
    model = Sequential()
    model.add(Permute((1, 2, 3), input_shape=input_shape))
    model.add(Convolution2D(32, (4, 4), strides=(
       1, 1), data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (2, 2), strides=(
       1, 1), data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (1, 1), strides=(
       1, 1), data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(output_shape))
    model.add(Activation('linear'))
    print(model.summary())
    return model


if __name__ == "__main__":
    K.set_image_dim_ordering('tf')
    env = gym.make('grid-v0')
    name = "dqn_cnn"
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    input_shape = env.observation_space.shape  # + (1,)  # add channel

    model = build_model(nb_actions, input_shape)

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    processor = GridProcessor(
        env.max_time, env.max_speed, env.max_watt, input_shape, env.nb_res)

    # Select a policy. We use eps-greedy action selection, which means that a random action is selected
    # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
    # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
    # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
    policy = LinearAnnealedPolicy(CustomEpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=50000)

    test_policy = CustomGreedyQPolicy()

    # The trade-off between exploration and exploitation is difficult and an on-going research topic.
    # If you want, you can experiment with the parameters or use a different policy. Another popular one
    # is Boltzmann-style exploration:
    # policy = BoltzmannQPolicy(tau=1.)
    # Feel free to give it a try!

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, test_policy=test_policy, memory=memory,
                   processor=processor, nb_steps_warmup=10000, gamma=.1, target_model_update=20000,
                   train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.0001), metrics=['mae'])

    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    callbacks = [ModelIntervalCheckpoint(
        'weights/'+name+'_1_weights_{step}.h5f', interval=50000)]
    callbacks += [FileLogger('log/'+name+'/'+name+'_1_log.json', interval=1)]
    callbacks += [TensorBoard(log_dir='log/'+name)]
    dqn.fit(env, callbacks=callbacks, nb_steps=200000,
            log_interval=10000, visualize=False)

    # After training is done, we save the final weights one more time.
    dqn.save_weights('weights/'+name+'_1_weights.h5f', overwrite=True)

    #dqn.test(env, nb_episodes=1, visualize=False)
