import gym
import gym_grid.envs.grid_env as g
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque


def q_learning(env, num_episodes, alpha, gamma=1.0):
    def print_progress(i_episode, num_episodes):
        # if i_episode % (num_episodes//10) == 0:
        print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
        sys.stdout.flush()

    def epsilon_greedy(Q, state, nA, eps):
        if np.random.random() > eps:
            return np.argmax(Q[state])
        else:
            return np.random.choice(np.arange(nA))

    def plot_graph(avg_scores, num_episodes):
        # plot performance
        plt.plot(np.linspace(0, num_episodes, len(avg_scores),
                             endpoint=False), np.asarray(avg_scores))
        plt.xlabel('Episode Number')
        plt.ylabel('Average Reward (Over Next %d Episodes)' % num_episodes//10)
        plt.show()
        # print best 100-episode performance
        print(('Best Average Reward over %d Episodes: ' %
               num_episodes//10), np.max(avg_scores))

    nA = len(env.action_space.spaces)
    Q = defaultdict(lambda: np.zeros(nA))
    tmp_scores = deque(maxlen=100)
    avg_scores = deque(maxlen=num_episodes//10)

    for i in range(1, num_episodes + 1):
        print_progress(i, num_episodes)
        score = 0
        action = [ [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0] ]
        bb = 0
        state = env.reset()
        eps = 1.0 / i
        while True:
            # epsilon_greedy(Q, state, nA, eps)
            next_state, reward, done, _ = env.step(action[bb])
            score += reward

            #Q[state][action] += alpha * \
            #    (reward + gamma * Q[next_state]
            #     [np.argmax(Q[next_state])] - Q[state][action])
            state = next_state
            bb+=1
            if done:
                tmp_scores.append(score)
                break
        if (i % num_episodes//10 == 0):
            avg_scores.append(np.mean(tmp_scores))

    #plot_graph(avg_scores, num_episodes)
    return Q


def run():
    env = gym.make('grid-v0')

    Q = q_learning(env, 44, 0.1)


if __name__ == "__main__":
    run()
