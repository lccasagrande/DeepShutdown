import itertools
import math
import numpy as np
import random
from sortedcontainers import SortedList
from collections import deque
from matplotlib.colors import XKCD_COLORS as allcolors
from enum import Enum
from itertools import count
import heapq


class Gantt():
    class Resource:
        def __init__(self, time_window):
            self.time_window = time_window
            self.queue = []

        @property
        def free_time(self):
            free_time = self.time_window
            for j in self.queue:
                free_time -= j.remaining_time
            return free_time

        def get_state(self):
            state = np.zeros(shape=(self.time_window, 3), dtype=np.uint8)
            for j in self.queue:
                end_idx = j.time_left_to_start + int(j.remaining_time)
                state[j.time_left_to_start:end_idx] = j.color
            return state

        def get_job(self):
            return min(self.queue, key=(lambda j: j.time_left_to_start)) if self.queue else None

        def reserve(self, job):
            self.queue.append(job)

        def release(self, job):
            self.queue.remove(job)

        def clear(self):
            self.queue.clear()

    def __init__(self, nb_resources, time_window):
        # Size: we can have N jobs of 1 lenght (e.g.: exec time is 1 second)
        self._state = [Gantt.Resource(time_window=time_window)
                       for _ in range(nb_resources)]
        self.shape = (time_window, nb_resources, 3)
        self.nb_resources = nb_resources
        self.time_window = time_window

    @property
    def free_time(self):
        return np.asarray([res.free_time for res in self._state])

    def get_state(self):
        state = np.zeros(shape=self.shape, dtype=np.uint8)
        for i, r in enumerate(self._state):
            state[:, i] = r.get_state()

        return state

    def get_jobs(self):
        for res in self._state:
            yield res.get_job()

    def clear(self):
        for res in self._state:
            res.clear()

    def reserve(self, job):
        for res in job.allocation:
            res_state = self._state[res].get_state()
            if np.any(res_state[job.time_left_to_start:job.requested_time] != 0):
                return False

        for res in job.allocation:
            self._state[res].reserve(job)

    def release(self, job):
        for res in job.allocation:
            self._state[res].release(job)


class SchedulerManager():
    def __init__(self, nb_resources, time_window, queue_size):
        self.gantt = Gantt(nb_resources, time_window)
        self._jobs_queue = []
        self._jobs_running = dict()
        self._jobs_allocated = dict()
        self.queue_size = queue_size
        self.reset()

    @property
    def nb_jobs_running(self):
        return len(self._jobs_running)

    @property
    def nb_jobs_in_queue(self):
        return len(self._jobs_queue)

    @property
    def jobs_running(self):
        return list(self._jobs_running.values())

    @property
    def jobs_queue(self):
        return list(self._jobs_queue)

    def lookup(self, idx):
        return self._jobs_queue[idx]

    def reset(self):
        self.gantt.clear()
        self._jobs_queue.clear()
        self._jobs_running.clear()
        self.first_job = None
        self.last_job = None
        self.nb_jobs_submitted = 0
        self.nb_jobs_completed = 0
        self.total_waiting_time = 0
        self.total_slowdown = 0
        self.total_turnaround_time = 0
        self.runtime_slowdown = 0
        self.runtime_waiting_time = 0

    def update_state(self, now):
        for _, job in self._jobs_allocated.items():
            slow_before = job.runtime_slowdown
            wait_before = job.waiting_time
            job.update_state(now)
            self.runtime_slowdown += job.runtime_slowdown - slow_before
            self.runtime_waiting_time += job.waiting_time - wait_before

        for _, job in self._jobs_running.items():
            slow_before = job.runtime_slowdown
            job.update_state(now)
            self.runtime_slowdown += job.runtime_slowdown - slow_before

        for job in self._jobs_queue:
            slow_before = job.runtime_slowdown
            wait_before = job.waiting_time
            job.update_state(now)
            self.runtime_slowdown += job.runtime_slowdown - slow_before
            self.runtime_waiting_time += job.waiting_time - wait_before

    def is_ready(self):
        free_time = self.gantt.free_time
        for j in self._jobs_queue:
            if (len(np.where(free_time >= j.requested_time)[0]) >= j.requested_resources):
                return True
        return False

    def allocate_job(self, job_idx):
        if job_idx >= len(self._jobs_queue):
            raise InvalidJobError(
                "There is no job {} to schedule".format(job_idx))

        job = self._jobs_queue[job_idx]

        resources, time_left_to_start = self._select_available_resources(
            job.requested_resources, job.requested_time)

        if resources is None:
            raise UnavailableResourcesError(
                "There is no resource available for this job.")

        job.allocation = resources
        job.time_left_to_start = time_left_to_start
        self.gantt.reserve(job)
        self._jobs_allocated[job.id] = job
        del self._jobs_queue[job_idx]
        return job.allocation

    def on_job_scheduled(self, job_id, time):
        job = self._jobs_allocated.pop(job_id)
        job.state = Job.State.RUNNING
        job.start_time = time
        job.time_left_to_start = 0
        self._jobs_running[job.id] = job
        if self.first_job == None:
            self.first_job = job

    def on_job_completed(self, time, data):
        job = self._jobs_running.pop(data['job_id'])
        self.gantt.release(job)

        job.finish_time = time
        job.runtime = job.finish_time - job.start_time
        job.turnaround_time = job.waiting_time + job.runtime
        job.slowdown = job.turnaround_time / job.runtime
        job.consumed_energy = data['job_consumed_energy']
        job.state = Job.State[data['job_state']]
        assert job.remaining_time == 0

        self._update_stats(job)
        self.last_job = job
        self.nb_jobs_completed += 1

    def on_job_submitted(self, time, data):
        if self.nb_jobs_in_queue == self.queue_size:
            return False

        if data['res'] > self.gantt.nb_resources:
            return False

        job = Job.from_json(data)
        job.state = Job.State.SUBMITTED
        self._jobs_queue.append(job)
        self.nb_jobs_submitted += 1
        return True

    def _select_available_resources(self, nb_res, req_time):
        gantt_state = self.gantt.get_state()
        # check all time window
        for time in range(0, self.gantt.time_window-req_time+1):
            # check all resources
            for r in range(0, self.gantt.nb_resources - nb_res+1):
                if not np.any(gantt_state[time:time+req_time, r:r+nb_res] != 0):
                    return list(range(r, r+nb_res)), time

        return None, -1


    def _update_stats(self, job):
        self.total_slowdown += job.slowdown
        self.total_waiting_time += job.waiting_time
        self.total_turnaround_time += job.turnaround_time


class Job(object):
    class State(Enum):
        UNKNOWN = -1
        NOT_SUBMITTED = 0
        SUBMITTED = 1
        RUNNING = 2
        COMPLETED_SUCCESSFULLY = 3
        COMPLETED_FAILED = 4
        COMPLETED_WALLTIME_REACHED = 5
        COMPLETED_KILLED = 6
        REJECTED = 7
        IN_KILLING = 8

    def __init__(
            self,
            id,
            subtime,
            walltime,
            res,
            profile,
            json_dict,
            profile_dict):
        self.id = id
        self.submit_time = subtime
        self.requested_time = walltime
        self.requested_resources = res
        self.profile = profile
        self.start_time = -1  # will be set on scheduling by batsim
        self.time_left_to_start = -1  # will be set on scheduling by batsim
        self.finish_time = -1  # will be set on completion by batsim
        self.turnaround_time = -1  # will be set on completion by batsim
        self.waiting_time = 0  # will be set on completion by batsim
        self.runtime = 0  # will be set on completion by batsim
        self.consumed_energy = -1  # will be set on completion by batsim
        self.slowdown = 0
        self.runtime_slowdown = 0
        self.state = Job.State.UNKNOWN
        self.return_code = None
        self.json_dict = json_dict
        self.profile_dict = profile_dict
        self.allocation = []
        self.color = list(np.random.choice(range(25, 225), size=3))
        self.metadata = None

    @property
    def remaining_time(self):
        return self.requested_time - self.runtime if self.finish_time == -1 else 0

    def update_state(self, now):
        time_passed = 0
        if self.state == Job.State.RUNNING:
            time_passed = now - self.runtime - self.start_time
            self.runtime += time_passed
        elif self.state == Job.State.SUBMITTED:
            if self.submit_time < now:
                time_passed = now - self.waiting_time - self.submit_time
                self.waiting_time += time_passed
        else:
            raise ValueError("Job cannot update its state.")

        if time_passed != 0 and self.time_left_to_start > 0:
            self.time_left_to_start = int(
                max(0, self.time_left_to_start - time_passed))

        self.runtime_slowdown = (
            self.waiting_time + self.runtime) / self.requested_time

    @staticmethod
    def from_json(json_dict, profile_dict=None):
        return Job(json_dict["id"],
                   json_dict["subtime"],
                   json_dict.get("walltime", -1),
                   json_dict["res"],
                   json_dict["profile"],
                   json_dict,
                   profile_dict)


class InsufficientResourcesError(Exception):
    pass


class UnavailableResourcesError(Exception):
    pass


class InvalidJobError(Exception):
    pass
