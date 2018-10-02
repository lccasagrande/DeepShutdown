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
        self.job_slots = 10
        self.time_window = 20
        self.backlog = 4
        self.max_slowdown = 1
        self.simulator = BatsimHandler(self.job_slots, time_window=1)
        self._update_state()
        self.action_space = self._get_action_space()

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
        time_after = self.simulator.current_time
        jobs_requested_time = self._get_jobs_requested_time()

        try:
            self.simulator.schedule(action)
        except (UnavailableResourcesError, InvalidJobError):
            self.simulator.schedule(-1)

        time_passed = self.simulator.current_time - time_after

        reward = self._get_reward(time_passed, jobs_requested_time)

        self._update_state()

        done = not self.simulator.running_simulation
        info = self._get_info()

        return self.state, reward, done, info

    def _get_info(self):
        if self.simulator.running_simulation:
            info = dict(
                nb_jobs_submitted=self.simulator.nb_jobs_submitted,
                nb_jobs_completed=self.simulator.nb_jobs_completed,
                nb_jobs_running=self.simulator.nb_jobs_running,
                nb_jobs_in_queue=self.simulator.nb_jobs_in_queue)
        else:
            info = dict(
                makespan=self.simulator.metrics['makespan'],
                mean_slowdown=self.simulator.metrics['mean_slowdown'],
                energy_consumed=self.simulator.metrics['energy_consumed'])
        return info

    def _get_jobs_requested_time(self):
        jobs = []
        for j in self.simulator.sched_manager.jobs_queue:
            jobs.append(j.requested_time)
        for j in self.simulator.sched_manager.jobs_waiting:
            jobs.append(j.requested_time)
        for j in self.simulator.sched_manager.jobs_running:
            jobs.append(j.requested_time)
        return jobs

    def _get_reward(self, time_passed, jobs_req_time):
        reward = 0
        if time_passed != 0:
            for time in jobs_req_time:
                reward += -1 * (time_passed / time)

        return reward

    def reset(self):
        self.all_time = 0
        self.my_reward = 0
        self.simulator.close()
        self.simulator.start()
        self._update_state()
        return self.state

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

    def _update_state(self):
        self.state = self.simulator.current_state

    def _get_action_space(self):
        return spaces.Discrete(self.simulator.queue_slots+1)

    def _get_observation_space(self):
        raise NotImplementedError()

    def _print(self):
        stats = "\rSubmitted: {:5} Completed: {:5} | Running: {:5} In Queue: {:5}".format(
            self.simulator.nb_jobs_submitted,
            self.simulator.nb_jobs_completed,
            self.simulator.nb_jobs_running,
            self.simulator.nb_jobs_in_queue)
        print(stats, end="", flush=True)

    def _plot(self):
        def plot_resource_state(offset=1):
            # plot the backlog at the end, +1 to avoid 0
            resource_state = np.zeros(
                shape=(self.time_window, self.simulator.nb_resources, 3), dtype=int)
            for res_idx, resource in enumerate(self.state['gantt']):
                job = resource['queue'][0]
                if job is not None:
                    time_window = min(self.time_window, job.remaining_time)
                    resource_state[0:time_window, res_idx] = job.color

            plt.subplot(1, 1 + self.simulator.queue_slots + 1, 1)
            plt.imshow(resource_state, interpolation='nearest',
                       vmax=1, aspect='auto')
            ax = plt.gca()
            ax.set_xticks(range(self.nb_resources))
            ax.set_yticks(range(self.time_window))
            ax.set_ylabel("Time Window")
            ax.set_xlabel("Id")
            ax.set_xticks(np.arange(-.5, self.nb_resources, 1), minor=True)
            ax.set_yticks(np.arange(-.5, self.time_window, 1), minor=True)
            ax.set_aspect('auto')
            ax.set_title("RES")
            ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

        def plot_queue_state(offset=1):
            nb_res = self.simulator.nb_resources
            jobs = self.state['job_queue']['jobs']
            for slot in range(self.simulator.queue_slots):
                job_state = np.zeros(
                    shape=(self.time_window, nb_res, 3), dtype=int)
                if slot < len(jobs):
                    job = jobs[slot]
                    time_window = min(self.time_window, job.requested_time)
                    job_state[0:time_window,
                              0:job.requested_resources] = job.color

                plt.subplot(1, 1 + self.simulator.queue_slots + 1, slot + 2)
                plt.imshow(job_state, interpolation='nearest',
                           vmax=1, aspect='auto')
                ax = plt.gca()
                ax.set_xticks([], [])
                ax.set_yticks([], [])
                ax.set_xticks(np.arange(-.5, self.nb_resources, 1), minor=True)
                ax.set_yticks(np.arange(-.5, self.time_window, 1), minor=True)
                ax.set_title("Slot {}".format(slot))
                ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

        def plot_backlog(offset=1):
            backlog_state = np.zeros(
                shape=(self.time_window, self.backlog, 3), dtype=int)
            nb_backlog_jobs = self.simulator.nb_jobs_in_queue + \
                self.simulator.nb_jobs_waiting - \
                len(self.state['job_queue']['jobs'])

            j = 0
            time_window = 0
            for _ in range(nb_backlog_jobs):
                if j == self.backlog:
                    break

                backlog_state[time_window][j] = [255, 255, 255]
                time_window += 1
                if time_window == self.time_window:
                    time_window = 0
                    j += 1

            plt.subplot(1, 1 + self.simulator.queue_slots +
                        1, self.simulator.queue_slots + 2)

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
        plot_queue_state()
        plot_backlog()
        plt.tight_layout()
        plt.pause(0.01)
