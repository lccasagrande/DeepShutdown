import gym
import gym_grid.envs.grid_env as g
import sys
import random
from random import randrange, sample
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from keras import metrics
from keras import backend as K
from keras.models import Sequential
from keras.losses import mean_squared_error
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
import utils
import time as t
import pandas as pd


class LRTensorBoard(TensorBoard):
    # add other arguments to __init__ if you need
    def __init__(self, log_dir, write_graph=False, write_grads=False, write_images=False):
        super().__init__(log_dir=log_dir, write_graph=write_graph,
                         write_grads=write_grads, write_images=write_images)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'learning_rate': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


class Memory:
    def __init__(self, size):
        self.__buffer = deque(maxlen=size)

    def record(self, experience):
        self.__buffer.append(experience)

    def remember(self, lenght):
        memories = random.sample(self.__buffer, lenght)
        return memories

    @property
    def len(self):
        return len(self.__buffer)


class DQNAgent:
    def __init__(self, input_shape, output_shape, mem_len, log_dir, lr=0.00025, gamma=0.99):
        # add channel dim
        self.input_shape = (input_shape[0], input_shape[1], 1)
        self.output_shape = output_shape
        self.log_dir = log_dir
        self.memory = Memory(mem_len)
        self.min_memory_to_train = 50000
        self.gamma = gamma
        self.lr = lr
        self.epsilon = 1
        self.min_epsilon = 0.1
        self.epsilon_decay_steps = 400000
        self.total_steps = 0
        self.target_model_update_count = 0
        self.target_model_update_freq = 5000
        self.callbacks = self._get_callbacks()
        self.epsilons = np.linspace(
            self.epsilon, self.min_epsilon, self.epsilon_decay_steps)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Convolution2D(32, (8, 8), strides=(4, 4), input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (2, 2), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.output_shape))
        model.add(Activation('linear'))
        model.compile(loss=mean_squared_error,
                      optimizer=Adam(lr=self.lr),
                      metrics=[metrics.mean_absolute_error])
        return model

    def predict(self, state):
        x = state.reshape(
            1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        return self.model.predict(x)

    def train(self, bath_size):
        assert self.min_memory_to_train >= bath_size
        if self.memory.len < self.min_memory_to_train:
            return

        samples = self.memory.remember(bath_size)
        input_states = np.zeros(
            (bath_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        target_states = np.zeros(
            (bath_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        actions, rewards, dones = [], [], []

        for i, (s, a, r, n_s, done) in enumerate(samples):
            input_states[i] = s.reshape(self.input_shape)
            target_states[i] = n_s.reshape(self.input_shape)
            actions.append(a)
            rewards.append(r)
            dones.append(done)

        cummrewards = self.model.predict(input_states)
        targets = self.target_model.predict(target_states)

        for i in range(bath_size):
            if dones[i]:
                cummrewards[i][actions[i]] = rewards[i]
            else:
                cummrewards[i][actions[i]] = rewards[i] + \
                    self.gamma * (np.amax(targets[i]))

        self.model.fit(input_states,
                       cummrewards,
                       batch_size=bath_size,
                       epochs=1,
                       verbose=0,
                       callbacks=self.callbacks)

    def _get_callbacks(self):
        tensor = LRTensorBoard(log_dir=self.log_dir)
        return [tensor]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def store(self, state, action, reward, next_state, done):
        memory = (state, action, reward, next_state, done)
        self.memory.record(memory)
        self.total_steps += 1

        if self.total_steps - self.target_model_update_count == self.target_model_update_freq:
            self.update_target_model()
            self.target_model_update_count = self.total_steps

        self.epsilon = self.epsilons[min(
            self.total_steps, self.epsilon_decay_steps-1)]

    def get_action(self, state):
        valid_actions = [0] + [(i+1) for i in range(self.output_shape-1) if np.any(state[i] == 0)]

        if np.random.uniform() < self.epsilon:
            return random.choice(valid_actions)

        pred = self.predict(self.input_shape)[0]
        pred = np.take(pred, valid_actions)
        return np.argmax(valid_actions)


def run(output_dir, n_ep=60, out_freq=2, plot=True):
    K.set_image_dim_ordering('tf')
    env = gym.make('grid-v0')
    np.random.seed(123)
    env.seed(123)

    episodic_scores = deque(maxlen=out_freq)
    avg_scores = deque(maxlen=n_ep)
    action_history = []

    # Create agent
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit([[0], [env.max_time]])
    agent = DQNAgent(input_shape=env.observation_space.shape,
                     output_shape=env.action_space.n,
                     mem_len=1000000,
                     log_dir='log/cnn1')

    t_start = t.time()
    for i in range(1, n_ep + 1):
        utils.print_progress(t_start, i, n_ep)
        score = 0
        epi_actions = np.zeros(env.action_space.n)
        state = scaler.transform(env.reset())
        while True:
            # Choose action
            action = agent.get_action(state)
            epi_actions[action] += 1
            next_state, reward, done, _ = env.step(action)
            next_state = scaler.transform(next_state)

            # Record experience
            agent.store(state, action, reward, next_state, done)
            agent.train(32)

            # Update State and Score
            state = next_state
            score += reward
            env.render()
            if done:
                episodic_scores.append(score)
                action_history.append(epi_actions)
                print(
                    "\nScore: {:7} - Max Score: {:7} - Epsilon: {:7} - Memory: {:7}"
                        .format(score, max(episodic_scores), agent.epsilon, agent.memory.len))
                break

        if (i % out_freq == 0):
            avg_scores.append(np.mean(episodic_scores))

    pd.DataFrame(action_history,
                 columns=range(0, env.action_space.n),
                 dtype=np.int).to_csv(output_dir+"actions_hist.csv")

    if plot:
        utils.plot_reward(avg_scores,
                          n_ep,
                          title='DQN CNN Policy',
                          output_dir=output_dir)

    print(('Best Average Reward over %d Episodes: ' %
           out_freq), np.max(avg_scores))


if __name__ == "__main__":
    output_dir = 'results/dqn/'
    run(output_dir)
