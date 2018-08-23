import gym
import gym_grid.envs.grid_env as g
import sys
import random
from random import randrange, sample
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
import utils


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
    def __init__(self, input_shape, output_shape, mem_len, lr=0.00001, gamma=0.99):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.memory = Memory(mem_len)
        self.gamma = gamma
        self.lr = lr
        self.epsilon = 1
        self.min_epsilon = 0.1
        self.epsilon_decay_steps = 500
        self.total_steps = 0
        self.target_model_update_count = 0
        self.target_model_update_freq = 10
        self.min_memory_to_train = 50
        self.epsilons = np.linspace(
            self.epsilon, self.min_epsilon, self.epsilon_decay_steps)
        self.model = self.__build_model()
        self.target_model = self.__build_model()
        self.update_target_model()

    def __build_model(self, hidden_size=[64, 64]):
        model = Sequential()
        model.add(
            Dense(units=hidden_size[0], input_dim=self.input_shape, activation='relu'))
        model.add(Dense(units=hidden_size[1], activation='relu'))
        model.add(Dense(units=self.output_shape, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def train(self, bath_size):
        assert self.min_memory_to_train >= bath_size
        if self.memory.len < self.min_memory_to_train:
            return

        samples = self.memory.remember(bath_size)
        input_states = np.zeros((bath_size, self.input_shape))
        target_states = np.zeros((bath_size, self.input_shape))
        actions, rewards, dones = [], [], []

        for i, (s, a, r, n_s, done) in enumerate(samples):
            input_states[i] = s
            target_states[i] = n_s
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

        self.model.fit(input_states, cummrewards,
                       batch_size=bath_size, epochs=1, verbose=0)

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
            #nb_res = sample(range(self.output_shape), 1)
            return sample(range(self.output_shape), 1)[0]

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])


def MaxMinScale(value, mi, ma):
    return (value - mi) / (ma - mi)


def prep_input(enc, state):
    res_state = state['resources']
    res_state = np.array(enc.transform(res_state)).ravel()
    job_res = MaxMinScale(state['job']['res'], 0, len(state['resources']))
    job_wall = MaxMinScale(state['job']['requested_time'], 0, 7200)
    job_wait = MaxMinScale(state['job']['waiting_time'], 0, 7200)
    res_state = np.append(
        res_state, [job_res, job_wall, job_wait]).reshape(1, -1)
    return res_state


def run(n_ep=1000, plot=True):
    env = gym.make('grid-v0')
    episodic_scores = deque(maxlen=100)
    avg_scores = deque(maxlen=(n_ep//2))
    encoder = LabelBinarizer(neg_label=0, pos_label=1)
    encoder.fit(['sleeping', 'idle', 'computing'])

    nb_resources = env.observation_space.spaces['resources'].shape[0]
    resource_space = 3
    job_space = 3
    max_score = [0, 0]
    input_space = nb_resources * resource_space + job_space
    action_space = len(env.action_space.spaces)

    agent = DQNAgent(input_space, action_space, 500)
    # Iterate the game
    for i in range(1, n_ep + 1):
        utils.print_progress(i, n_ep)
        score = 0
        state = env.reset()
        state = prep_input(encoder, state)
        while True:
            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)
            next_state = prep_input(encoder, next_state)

            agent.store(state, action, reward, next_state, done)
            agent.train(32)
            state = next_state
            score += reward

            if done:
                if score > max_score[0]:
                    max_score[0] = score
                    max_score[1] = i
                print("episode: {}/{}, score: {}, epsilon: {}, memory: {}, max score: {} at {}"
                      .format(i, n_ep, score, agent.epsilon, agent.memory.len, max_score[0], max_score[1]))
                episodic_scores.append(score)
                break

        if (i % 2 == 0):
            avg_scores.append(np.mean(episodic_scores))

    if plot:
        utils.plot_graph(avg_scores, n_ep)


# def plot(rewards):
#    def running_mean(x, N):
#        cumsum = np.cumsum(np.insert(x, 0, 0))
#        return (cumsum[N:] - cumsum[:-N]) / N
#
#    eps, rews = np.array(rewards).T
#    smoothed_rews = running_mean(rews, 10)
#    plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
#    plt.plot(eps, rews, color='grey', alpha=0.3)
#    plt.xlabel('Episode')
#    plt.ylabel('Total Reward')
#    plt.show()


if __name__ == "__main__":
    run()
