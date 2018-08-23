import gym
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam


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
    def __init__(self, input_shape, output_shape, mem_len, lr=0.001, gamma=0.99):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.memory = Memory(mem_len)
        self.gamma = gamma
        self.lr = lr
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.decay_rate = 0.9999
        self.model = self.__build_model()
        self.target_model = self.__build_model()
        self.update_target_model()

    def __build_model(self, hidden_size=[64, 64]):
        model = Sequential()
        model.add(Dense(units=hidden_size[0], input_shape=(self.input_shape,),
                        activation='relu'))
        model.add(Dense(units=hidden_size[1], activation='relu'))
        model.add(Dense(units=self.output_shape, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def train(self, bath_size):
        if self.memory.len < bath_size:
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

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay_rate

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.output_shape)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

def run():
    #env = gym.make('grid-v0')
    env = gym.make('CartPole-v1')
    rewards = []
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, 2000)
    # Iterate the game
    for e in range(300):
        state = env.reset()
        for score in range(500):
           # env.render()
            state = np.reshape(state, [1,4])
            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)
            reward = reward if not done or score == 499 else -100
            next_state = np.reshape(next_state, [1,4])

            agent.store(state, action, reward, next_state, done)
            agent.train(32)
            state = next_state

            if done:
                print("episode: {}/{}, score: {}, epsilon: {}, memory: {}".format(e,
                                                                                  200, score, agent.epsilon, agent.memory.len))
                rewards.append((e, score))
                agent.update_target_model()
                break

    plot(rewards)


def plot(rewards):
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / N

    eps, rews = np.array(rewards).T
    smoothed_rews = running_mean(rews, 10)
    plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
    plt.plot(eps, rews, color='grey', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


if __name__ == "__main__":
    run()
