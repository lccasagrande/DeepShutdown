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
        self.job_slots = 5
        self.time_window = 20
        self.backlog_width = 1
        self.simulator = BatsimHandler(job_slots=self.job_slots,
                                       time_window=self.time_window,
                                       backlog_width=self.backlog_width)
        self.action_space = spaces.Discrete(self.job_slots+1)
        
        state_shape = self._get_obs()
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=state_shape.shape,
                                            dtype=np.uint8)

    def step(self, action):
        assert self.simulator.running_simulation, "Simulation is not running."
        energy_before = self.simulator.resource_manager.energy_consumed
        slow_before = self.simulator.jobs_manager.runtime_slowdown

        try:
            self.simulator.schedule(action-1)
        except (UnavailableResourcesError, InvalidJobError):
            self.simulator.schedule(-1)

        #energy_after = self.simulator.resource_manager.energy_consumed - energy_before
        slow_after = self.simulator.jobs_manager.runtime_slowdown - slow_before

        obs = self._get_obs()
        reward = -1*slow_after
        #reward = -1*(energy_after / self.simulator.resource_manager.max_energy_usage)
        done = not self.simulator.running_simulation
        info = self._get_info()

        return obs, reward, done, info

    def reset(self):
        self.simulator.close()
        self.simulator.start()
        return self._get_obs()

    def render(self, mode='human'):
        if mode == 'human':
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

    def _get_info(self):
        return dict() if self.simulator.running_simulation else self.simulator.metrics

    def _get_obs(self, type='compact', reshape=False):
        state = self.simulator.get_state(type)
        if reshape:
            state = state.reshape(state.shape + (1,))
        return state

    def _print(self):
        stats = "\rSubmitted: {:5} Completed: {:5} | Running: {:5} Waiting: {:5}".format(
            self.simulator.nb_jobs_submitted,
            self.simulator.nb_jobs_completed,
            self.simulator.nb_jobs_running,
            self.simulator.nb_jobs_waiting)
        print(stats, end="", flush=True)

    def _plot(self):
        def plot_resource_state():
            resource_state = self.simulator.get_resource_state()
            plt.subplot(1, 1 + self.job_slots + 1, 1)
            plt.imshow(resource_state, interpolation='nearest', vmin=0,
                       vmax=255, aspect='auto')
            ax = plt.gca()
            ax.set_xticks(range(self.simulator.nb_resources))
            ax.set_yticks(range(self.time_window))
            ax.set_ylabel("Time Window")
            ax.set_xlabel("Id")
            ax.set_xticks(
                np.arange(.5, self.simulator.nb_resources, 1), minor=True)
            ax.set_yticks(np.arange(.5, self.time_window, 1), minor=True)
            ax.set_aspect('auto')
            ax.set_title("RES")
            ax.grid(which='minor', color='w',
                    linestyle='-', linewidth=1)

        def plot_job_state():
            jobs = self.simulator.get_job_slot_state()
            slot = 1
            for start_idx in range(0, jobs.shape[1], self.simulator.nb_resources):
                job_state = jobs[:, start_idx:start_idx +
                                 self.simulator.nb_resources]
                plt.subplot(1, 1 + self.job_slots + 1, slot + 1)
                plt.imshow(job_state, interpolation='nearest', vmin=0,
                           vmax=255, aspect='auto')
                ax = plt.gca()
                ax.set_xticks([], [])
                ax.set_yticks([], [])
                ax.set_xticks(
                    np.arange(.5, self.simulator.nb_resources, 1), minor=True)
                ax.set_yticks(np.arange(.5, self.time_window, 1), minor=True)
                ax.set_title("Slot {}".format(slot))
                ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                slot += 1

        def plot_backlog():
            backlog_state = self.simulator.get_backlog_state()
            plt.subplot(1, 1 + self.job_slots + 1, self.job_slots + 2)

            plt.imshow(backlog_state, interpolation='nearest', vmin=0,
                       vmax=255,  aspect='auto')
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
