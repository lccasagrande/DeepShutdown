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
        self.time_window = 32
        self.backlog = 8
        self.max_slowdown = 1
        queue_size = self.job_slots + (self.backlog*self.time_window)
        self.simulator = BatsimHandler(queue_slots=self.job_slots,
                                       time_window=1,
                                       queue_size=queue_size)
        self.action_space = spaces.Discrete(self.simulator.queue_slots+1)
        self.nb_res = self.simulator.nb_resources
        width = self.nb_res + (self.job_slots*self.nb_res) + self.backlog
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(self.time_window, width, 1),
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
        slowdown_before = self.simulator.sched_manager.runtime_slowdown

        try:
            self.simulator.schedule(action-1)
        except (UnavailableResourcesError, InvalidJobError):
            self.simulator.schedule(-1)

        slowdown_after = self.simulator.sched_manager.runtime_slowdown - slowdown_before

        obs = self._get_obs()
        reward = -1 * slowdown_after
        done = not self.simulator.running_simulation
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
        self.all_time = 0
        self.my_reward = 0
        self.simulator.close()
        self.simulator.start()
        return self._get_obs()

    def render(self, mode='console'):
        if mode == 'image':
            self._plot()
        elif mode == 'console':
            self._print()
        else:
            self._plot()
            self._print()

    def close(self):
        self.simulator.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self, colorized=False):
        state = np.zeros(shape=self.observation_space.shape,
                         dtype=self.observation_space.dtype)
        sim_state = self.simulator.current_state

        # Get Resource State
        for res_idx, resource in enumerate(sim_state['gantt']):
            job = resource['queue'][0]  # Get first job
            if job is not None:
                time_window = min(self.time_window, int(job.remaining_time))
                state[0:time_window, res_idx] = job.color if colorized else 1

        # Get Jobs State
        jobs = sim_state['job_queue']['jobs']
        for slot in range(self.job_slots):
            if slot < len(jobs):
                job = jobs[slot]
                time_window = min(self.time_window, job.requested_time)
                start_idx = (slot*self.nb_res) + self.nb_res
                end_idx = start_idx + job.requested_resources

                state[0:time_window,
                      start_idx:end_idx] = job.color if colorized else 1

        # Get backlog State
        nb_backlog_jobs = self.simulator.nb_jobs_in_queue + \
            self.simulator.nb_jobs_waiting - \
            len(sim_state['job_queue']['jobs'])

        index = self.nb_res + (self.nb_res * self.job_slots)
        time_window = 0
        for _ in range(min(nb_backlog_jobs, self.backlog*self.time_window)):
            state[time_window][index] = [255, 255, 255] if colorized else 1
            time_window += 1
            if time_window == self.time_window:
                time_window = 0
                index += 1

        return state

    def _print(self):
        stats = "\rSubmitted: {:5} Completed: {:5} | Running: {:5} In Queue: {:5}".format(
            self.simulator.nb_jobs_submitted,
            self.simulator.nb_jobs_completed,
            self.simulator.nb_jobs_running,
            self.simulator.nb_jobs_in_queue)
        print(stats, end="", flush=True)

    def _plot(self):
        obs = self._get_obs(colorized=True)

        def plot_resource_state():
            resource_state = obs[:, 0:self.nb_res]
            plt.subplot(1, 1 + self.job_slots + 1, 1)
            plt.imshow(resource_state, interpolation='nearest',
                       vmax=1, aspect='auto')
            ax = plt.gca()
            ax.set_xticks(range(self.nb_res))
            ax.set_yticks(range(self.time_window))
            ax.set_ylabel("Time Window")
            ax.set_xlabel("Id")
            ax.set_xticks(np.arange(-.5, self.nb_res, 1), minor=True)
            ax.set_yticks(np.arange(-.5, self.time_window, 1), minor=True)
            ax.set_aspect('auto')
            ax.set_title("RES")
            ax.grid(which='minor', color='w',
                    linestyle='-', linewidth=1)

        def plot_job_state():
            end_idx = self.nb_res + (self.nb_res * self.job_slots)
            jobs = obs[:, self.nb_res:end_idx]
            slot = 0
            for start_idx in range(0, self.job_slots*self.nb_res, self.nb_res):
                job_state = jobs[:, start_idx:start_idx+self.nb_res]
                plt.subplot(1, 1 + self.job_slots + 1, slot + 2)
                plt.imshow(job_state, interpolation='nearest',
                           vmax=1, aspect='auto')
                ax = plt.gca()
                ax.set_xticks([], [])
                ax.set_yticks([], [])
                ax.set_xticks(np.arange(-.5, self.nb_resources, 1), minor=True)
                ax.set_yticks(np.arange(-.5, self.time_window, 1), minor=True)
                ax.set_title("Slot".format(slot))
                ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                slot += 1

        def plot_backlog():
            start_idx = self.nb_res + (self.nb_res * self.job_slots)
            backlog_state = obs[:, start_idx: start_idx+self.backlog]
            plt.subplot(1, 1 + self.job_slots + 1, self.job_slots + 2)

            plt.imshow(backlog_state, interpolation='nearest',
                       vmax=1,  aspect='auto')
            ax = plt.gca()
            ax.set_xticks(range(self.backlog))
            ax.set_yticks([], [])
            ax.set_xticks(np.arange(.5, self.backlog, 1), minor=True)
            ax.set_yticks(np.arange(.5, self.time_window, 1), minor=True)
            ax.set_title("Queue")
            ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

        plt.figure("screen", figsize=(20, 5))
        plot_resource_state()
        plot_job_state()
        plot_backlog()
        plt.tight_layout()
        plt.pause(0.01)
