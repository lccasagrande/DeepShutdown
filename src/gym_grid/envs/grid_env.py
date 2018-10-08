import gym
from copy import deepcopy
from collections import defaultdict
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from .batsim import BatsimHandler, InsufficientResourcesError, UnavailableResourcesError, InvalidJobError
import numpy as np


class GridEnv(gym.Env):
    def __init__(self):
        self.job_slots = 11
        self.time_window = 20
        self.backlog_width = 4
        queue_size = self.job_slots + (self.backlog_width*self.time_window)

        self.simulator = BatsimHandler(job_slots=self.job_slots,
                                       time_window=self.time_window,
                                       queue_size=queue_size)
                                       
        self.action_space = spaces.Discrete(self.job_slots+1)
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=self.simulator.state_shape,
                                            dtype=np.uint8)

    @property
    def nb_resources(self):
        return self.simulator.nb_resources

    @property
    def max_time(self):
        return 15

    @property
    def max_speed(self):
        return self.simulator.max_resource_speed

    @property
    def max_energy_consumption(self):
        return self.simulator.max_resource_energy_cost

    def step(self, action):
        assert self.simulator.running_simulation, "Simulation is not running."
        slowdown_before = self.simulator.jobs_manager.runtime_slowdown

        try:
            self.simulator.schedule(action-1)
        except (UnavailableResourcesError, InvalidJobError):
            self.simulator.schedule(-1)

        done = not self.simulator.running_simulation
        slowdown_after = self.simulator.jobs_manager.runtime_slowdown - slowdown_before

        obs = self._get_obs()
        reward = -1 * slowdown_after
        info = self._get_info()

        return obs, reward, done, info

    def _get_info(self):
        info = {}
        if not self.simulator.running_simulation:
            info['makespan'] = self.simulator.metrics['makespan']
            info['mean_slowdown'] = self.simulator.metrics['mean_slowdown']
            info['energy_consumed'] = self.simulator.metrics['energy_consumed']
            info['total_slowdown'] = self.simulator.metrics['total_slowdown']
            info['total_turnaround_time'] = self.simulator.metrics['total_turnaround_time']
            info['total_waiting_time'] = self.simulator.metrics['total_waiting_time']

        return info

    def reset(self):
        self.simulator.close()
        self.simulator.start()
        return self._get_obs()

    def render(self, mode='image'):
        if mode == 'image':
            self._plot()
        elif mode == 'console':
            self._print()
        else:
            self._plot()

    def close(self):
        self.simulator.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self, type='image'):
        return self.simulator.get_state(type)

    def _print(self):
        stats = "\rSubmitted: {:5} Completed: {:5} | Running: {:5} In Queue: {:5}".format(
            self.simulator.nb_jobs_submitted,
            self.simulator.nb_jobs_completed,
            self.simulator.nb_jobs_running,
            self.simulator.nb_jobs_in_queue)
        print(stats, end="", flush=True)

    def _plot(self):
        obs = self._get_obs(type='image')
        obs = obs / 255.0
        def plot_resource_state():
            resource_state = obs[:, 0:self.simulator.nb_resources]
            plt.subplot(1, 1 + self.job_slots + 1, 1)
            plt.imshow(resource_state, interpolation='nearest',
                       vmax=1, aspect='auto')
            ax = plt.gca()
            ax.set_xticks(range(self.simulator.nb_resources))
            ax.set_yticks(range(self.time_window))
            ax.set_ylabel("Time Window")
            ax.set_xlabel("Id")
            ax.set_xticks(np.arange(.5, self.simulator.nb_resources, 1), minor=True)
            ax.set_yticks(np.arange(.5, self.time_window, 1), minor=True)
            ax.set_aspect('auto')
            ax.set_title("RES")
            ax.grid(which='minor', color='w',
                    linestyle='-', linewidth=1)

        def plot_job_state():
            end_idx = self.simulator.nb_resources + (self.simulator.nb_resources * self.job_slots)
            jobs = obs[:, self.simulator.nb_resources:end_idx]
            slot = 1
            for start_idx in range(0, self.job_slots*self.simulator.nb_resources, self.simulator.nb_resources):
                job_state = jobs[:, start_idx:start_idx+self.simulator.nb_resources]
                plt.subplot(1, 1 + self.job_slots + 1, slot + 1)
                plt.imshow(job_state, interpolation='nearest',
                           vmax=1, aspect='auto')
                ax = plt.gca()
                ax.set_xticks([], [])
                ax.set_yticks([], [])
                ax.set_xticks(np.arange(.5, self.simulator.nb_resources, 1), minor=True)
                ax.set_yticks(np.arange(.5, self.time_window, 1), minor=True)
                ax.set_title("Slot {}".format(slot))
                ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                slot += 1

        def plot_backlog():
            start_idx = self.simulator.nb_resources + (self.simulator.nb_resources * self.job_slots)
            backlog_state = obs[:, start_idx: start_idx+self.backlog_width]
            plt.subplot(1, 1 + self.job_slots + 1, self.job_slots + 2)

            plt.imshow(backlog_state, interpolation='nearest',
                       vmax=1,  aspect='auto')
            ax = plt.gca()
            ax.set_xticks(range(self.backlog_width))
            ax.set_yticks([], [])
            ax.set_xticks(np.arange(.5, self.backlog_width, 1), minor=True)
            ax.set_yticks(np.arange(.5, self.time_window, 1), minor=True)
            ax.set_title("Queue")
            ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

        plt.figure("screen", figsize=(20, 5))
        plot_resource_state()
        plot_job_state()
        plot_backlog()
        plt.tight_layout()
        plt.pause(0.01)
