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


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

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
    def __init__(self, input_shape, output_shape, mem_len, log_dir, lr=0.00001, gamma=0.99):
        self.input_shape = (input_shape[0], input_shape[1], 1) # add channel dim
        self.output_shape = output_shape
        self.log_dir = log_dir
        self.memory = Memory(mem_len)
        self.min_memory_to_train = 1000
        self.gamma = gamma
        self.lr = lr
        self.epsilon = 1
        self.min_epsilon = 0.1
        self.epsilon_decay_steps = 500
        self.total_steps = 0
        self.target_model_update_count = 0
        self.target_model_update_freq = 10
        self.callbacks = self._get_callbacks()
        self.epsilons = np.linspace(
            self.epsilon, self.min_epsilon, self.epsilon_decay_steps)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()

        model.add(Convolution2D(64, (3, 3), activation='relu',
                                input_shape=self.input_shape))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.output_shape, activation='linear'))

        model.compile(loss=mean_squared_error,
                      optimizer=Adam(lr=self.lr),
                      metrics=[metrics.mean_absolute_error, metrics.mean_absolute_percentage_error])
        return model

    def predict(self, state):
        x = state.reshape(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        return self.model.predict(x)

    def train(self, bath_size):
        assert self.min_memory_to_train >= bath_size
        if self.memory.len < self.min_memory_to_train:
            return

        samples = self.memory.remember(bath_size)
        input_states = np.zeros((bath_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        target_states = np.zeros((bath_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
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
        tensor = TensorBoard(log_dir=self.log_dir,write_graph=False)
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
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.output_shape))

        act_values = self.predict(state)
        return np.argmax(act_values[0])


def prep_input(encoder, scaler, state):
    res_state = state['resources']
    res_state = np.array(encoder.transform(res_state)).ravel()
    job_wall = scaler.transform(state['job']['requested_time'])[0][0]
    job_wait = scaler.transform(state['job']['waiting_time'])[0][0]
    return res_state


def run(n_ep=10, out_freq=2, plot=True):
    K.set_image_dim_ordering('tf')
    env = gym.make('grid-v0')
    episodic_scores = deque(maxlen=out_freq)
    avg_scores = deque(maxlen=n_ep)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit([[0], [env.max_time]])

    # Create agent
    agent = DQNAgent(input_shape=env.observation_space.shape,
                     output_shape=env.action_space.n,
                     mem_len=1000000,
                     log_dir='log/cnn1')

    t_start = t.time()
    # Iterate the game
    for i in range(1, n_ep + 1):
        utils.print_progress(t_start, i, n_ep)
        score = 0

        # Init
        state = env.reset()
        state = scaler.transform(state)
        while True:
            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)
            next_state = scaler.transform(next_state)

            # Record experience
            agent.store(state, action, reward, next_state, done)

            # Train
            agent.train(32)
            state = next_state
            score += reward

            if done:
                print("episode: {}/{}, score: {}, epsilon: {}, memory: {}, max score: {} at {}"
                      .format(i, n_ep, score, agent.epsilon, agent.memory.len, max_score[0], max_score[1]))
                episodic_scores.append(score)
                break

        if (i % out_freq == 0):
            avg_scores.append(np.mean(episodic_scores))

    if plot:
        utils.plot_reward(avg_scores, n_ep, title='DQN CNN Policy')

    print(('Best Average Reward over %d Episodes: ' %
           out_freq), np.max(avg_scores))


if __name__ == "__main__":
    run()
