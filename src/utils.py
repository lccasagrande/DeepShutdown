import numpy as np
import matplotlib.pyplot as plt
import sys

def print_progress(i_episode, num_episodes):
    # if i_episode % (num_episodes//10) == 0:
    print("\rEpisode {}/{}\n".format(i_episode, num_episodes), end="")
    sys.stdout.flush()


def plot_graph(avg_scores, num_episodes):
    # plot performance
    plt.plot(np.linspace(0, num_episodes, len(avg_scores),
                         endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % len(avg_scores))
    plt.show()
    # print best 100-episode performance
    print('Best Average Reward over {} Episodes: {}'.format(len(avg_scores), np.max(avg_scores)))
