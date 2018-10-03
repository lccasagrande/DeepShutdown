import itertools
import math
import numpy as np
import random
from matplotlib.colors import XKCD_COLORS as allcolors
from enum import Enum
from collections import deque


class SchedulerManager():
    def __init__(self, nb_resources, time_window, queue_size, queue_slots):
        self.gantt_shape = (nb_resources, time_window)
        self.queue_size = queue_size
        self.queue_slots = queue_slots
        self.reset()

    @property
    def nb_jobs_running(self):
        return len(self._jobs_running)

    @property
    def nb_jobs_waiting(self):
        return len(self._jobs_waiting)

    @property
    def nb_jobs_in_queue(self):
        return len(self._jobs_queue)

    @property
    def jobs_waiting(self):
        return list(self._jobs_waiting)

    @property
    def jobs_running(self):
        return list(self._jobs_running.values())

    @property
    def jobs_queue(self):
        return list(self._jobs_queue)

    @property
    def is_ready(self):
        free_spaces = 0
        for res in self.gantt:
            if np.any(res == None):
                free_spaces += 1
        last_idx = min(self.nb_jobs_in_queue, self.queue_slots)
        return any(j.requested_resources <= free_spaces for j in self._jobs_queue[0:last_idx])

    def lookup(self, idx):
        return self._jobs_queue[idx]

    def reset(self):
        self.gantt = np.empty(shape=self.gantt_shape, dtype=object)
        self._jobs_queue = []
        self._jobs_running = dict()
        self._jobs_waiting = []
        self.first_job = None
        self.last_job = None
        self.nb_jobs_submitted = 0
        self.nb_jobs_completed = 0
        self.total_waiting_time = 0
        self.total_slowdown = 0
        self.total_turnaround_time = 0
        self.runtime_slowdown = 0
        self.runtime_waiting_time = 0

    def has_free_space(self, nb=1):
        free_spaces = 0
        for res in self.gantt:
            if np.any(res == None):
                free_spaces += 1
            if free_spaces == nb:
                return True

        return False

    def update_state(self, now):
        for _, job in self._jobs_running.items():
            slow_before = job.runtime_slowdown
            job.update_state(now)
            self.runtime_slowdown += job.runtime_slowdown - slow_before

        for job in self.jobs_queue:
            slow_before = job.runtime_slowdown
            wait_before = job.waiting_time
            job.update_state(now)
            self.runtime_slowdown += job.runtime_slowdown - slow_before
            self.runtime_waiting_time += job.waiting_time - wait_before

        for job in self.jobs_waiting:
            slow_before = job.runtime_slowdown
            wait_before = job.waiting_time
            job.update_state(now)
            self.runtime_slowdown += job.runtime_slowdown - slow_before
            self.runtime_waiting_time += job.waiting_time - wait_before

    def allocate_job(self, job_idx):
        if job_idx > self.nb_jobs_in_queue - 1:
            raise InvalidJobError(
                "There is no job {} to schedule".format(job_idx))

        if not self.has_free_space(self._jobs_queue[job_idx].requested_resources):
            raise UnavailableResourcesError(
                "There is no resource available for this job.")

        job = self._jobs_queue.pop(job_idx)
        res_idx = 0
        allocated = 0
        while allocated != job.requested_resources:
            resource_queue = self.gantt[res_idx]
            for i in range(len(resource_queue)):  # Find the first space available
                if resource_queue[i] == None:
                    resource_queue[i] = job
                    job.allocation.append(res_idx)
                    allocated += 1
                    break
            res_idx += 1

        return job.allocation

    def delay_first_job(self):
        self._jobs_waiting.append(self._jobs_queue.pop(0))

    def get_ready_jobs(self):
        ready_jobs = []
        jobs = self.gantt[:, 0][self.gantt[:, 0] != None]
        for job in jobs:
            if job.state != Job.State.RUNNING \
                    and job not in ready_jobs \
                    and len(self.gantt[:, 0][self.gantt[:, 0] == job]) == job.requested_resources:
                ready_jobs.append(job)
        return ready_jobs

    def enqueue_jobs_waiting(self, time):
        while self.nb_jobs_waiting > 0:
            job = self._jobs_waiting.pop(0)
            self._jobs_queue.append(job)

    def on_alarm_fired(self, time):
        self.enqueue_jobs_waiting(time)

    def on_resource_pstate_changed(self, time):
        self.enqueue_jobs_waiting(time)

    def on_job_scheduled(self, job, time):
        job.state = Job.State.RUNNING
        job.start_time = time
        self._jobs_running[job.id] = job
        if self.first_job == None:
            self.first_job = job

    def on_job_completed(self, time, resources, data):
        job = self._remove_from_gantt(resources, data['job_id'])

        job.finish_time = time
        job.runtime = job.finish_time - job.start_time
        job.turnaround_time = job.waiting_time + job.runtime
        job.slowdown = job.turnaround_time / job.runtime
        job.consumed_energy = data['job_consumed_energy']
        assert job.remaining_time == 0
        try:
            job.state = Job.State[ data['job_state']]
        except KeyError:
            job.state = Job.State.UNKNOWN

        self._update_stats(job)
        self.enqueue_jobs_waiting(time)
        self.last_job = job
        del self._jobs_running[job.id]
        self.nb_jobs_completed += 1

    def on_job_submitted(self, time, data):
        if (len(self.jobs_queue) + len(self.jobs_waiting)) == self.queue_size:
            return False

        if data['res'] > self.gantt_shape[0]:
            return False

        job = Job.from_json(data)
        job.state = Job.State.SUBMITTED
        self._jobs_queue.append(job)
        self.enqueue_jobs_waiting(time)
        self.nb_jobs_submitted += 1
        return True

    def _remove_from_gantt(self, resources, job_id):
        job = None
        for res_id in resources:
            resource_queue = self.gantt[res_id]
            assert resource_queue[0] != None, "There is no job using this resource"
            assert resource_queue[0].id == job_id, "This resource is being used by another job"

            job = resource_queue[0]
            for space in range(1, len(resource_queue)):
                resource_queue[space-1] = resource_queue[space]

            resource_queue[-1] = None
        assert job is not None
        return job

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
        self.color = list(np.random.choice(range(1, 255), size=3))
        self.metadata = None

    @property
    def remaining_time(self):
        return self.requested_time - self.runtime

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
            raise ValueError("Job cannot update state.")

        self.runtime_slowdown = (self.waiting_time + self.runtime) / self.requested_time

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
