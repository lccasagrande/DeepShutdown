import numpy as np
import matplotlib.pyplot as plt
import sys
import time as t


def print_progress(t_start, i_episode, num_episodes):
    runtime_per_ep = round(((t.time() - t_start) / i_episode), 2)
    total_runtime = round(runtime_per_ep * (num_episodes - i_episode), 2)
    print("\rEpisode {}/{}: in {}s/epi ({} s)\n".format(i_episode,
                                                      num_episodes,
                                                      runtime_per_ep,
                                                      total_runtime), end="")
    sys.stdout.flush()


def plot_graph(avg_scores, num_episodes):
    # plot performance
    plt.plot(np.linspace(0, num_episodes, len(avg_scores),
                         endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % len(avg_scores))
    plt.show()
    # print best 100-episode performance
    print('Best Average Reward over {} Episodes: {}'.format(
        len(avg_scores), np.max(avg_scores)))
