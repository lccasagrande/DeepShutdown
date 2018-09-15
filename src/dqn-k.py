from __future__ import division
import argparse

from PIL import Image
import numpy as np
import random
import gym
import gym_grid.envs.grid_env as g
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K

from dqn_utils import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, Policy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (31, 31, 1)
WINDOW_LENGTH = 4


class CustomEpsGreedyQPolicy(Policy):
    """Implement the epsilon greedy policy with valid actions only

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """

    def __init__(self, eps=.1):
        super(CustomEpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values, state):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
            state (np.ndarray): The current state of the environment

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        valid_actions = [
            0] + [(i+1) for i in range(q_values.shape[0]-1) if np.any(state[0][i] == 0)]

        if np.random.uniform() >= self.eps:
            act = np.amax(np.take(q_values, valid_actions))
            valid_actions = np.where(q_values == act)[0]

        return random.choice(valid_actions)

    def get_config(self):
        """Return configurations of EpsGreedyPolicy

        # Returns
            Dict of config
        """
        config = super(CustomEpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config


class GridProcessor(Processor):
    def __init__(self, max_time, max_speed, max_watt, input_shape):
        super().__init__()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.cpu_scaler = MinMaxScaler(feature_range=(0, 1))
        self.energy_scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit([[0], [max_time]])
        self.cpu_scaler.fit([[0], [max_speed]])
        self.energy_scaler.fit([[0], [max_watt]])
        self.input_shape = input_shape

    def process_state_batch(self, batch):
        return np.asarray([v[0] for v in batch])

    def process_observation(self, observation):
        obs = self.scaler.transform(observation[:,3:])
        cpu = self.cpu_scaler.transform(observation[:,0].reshape(-1, 1))
        energy = self.energy_scaler.transform(observation[:,1:3])
        obs = np.concatenate((cpu, energy, obs), axis=1)
        obs = obs.reshape(self.input_shape)
        assert obs.ndim == 3  # (height, width, channel)
        return obs


def build_model(output_shape, input_shape):
    model = Sequential()
    model.add(Permute((1, 2, 3), input_shape=input_shape))
    model.add(Convolution2D(32, (8, 8), strides=(
        4, 4), data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (2, 2), strides=(1, 1)))
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
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    input_shape = env.observation_space.shape + (1,)

    model = build_model(nb_actions, input_shape)

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    processor = GridProcessor(
        env.max_time, env.max_speed, env.max_watt, input_shape)

    # Select a policy. We use eps-greedy action selection, which means that a random action is selected
    # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
    # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
    # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
    policy = LinearAnnealedPolicy(CustomEpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)

    # The trade-off between exploration and exploitation is difficult and an on-going research topic.
    # If you want, you can experiment with the parameters or use a different policy. Another popular one
    # is Boltzmann-style exploration:
    # policy = BoltzmannQPolicy(tau=1.)
    # Feel free to give it a try!

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.0001), metrics=['mae'])

    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    callbacks = [ModelIntervalCheckpoint(
        'weights/dqn_1_weights_{step}.h5f', interval=250000)]
    callbacks += [FileLogger('log/dqn_1_log.json', interval=100)]
    callbacks += [TensorBoard(log_dir='log/dqn')]
    dqn.fit(env, callbacks=callbacks, nb_steps=1750000,
            log_interval=10000, visualize=False)

    # After training is done, we save the final weights one more time.
    dqn.save_weights('weights/dqn_1_weights.h5f', overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
    # elif args.mode == 'test':
    #    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    #    if args.weights:
    #        weights_filename = args.weights
    #    dqn.load_weights(weights_filename)
    #    dqn.test(env, nb_episodes=10, visualize=True)
