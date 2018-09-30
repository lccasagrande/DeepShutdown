import itertools
import math
import numpy as np
from enum import Enum
from collections import deque


class SchedulerManager():
    def __init__(self, nb_resources, time_window):
        self.gantt_shape = (nb_resources, time_window)
        self.reset()

    @property
    def jobs_running(self):
        jobs = []
        for job in self.gantt[:, 0]:
            if job != None and job.state == Job.State.RUNNING:
                jobs.append(job)

        return jobs

    @property
    def nb_jobs_waiting(self):
        return len(self.jobs_waiting)

    @property
    def nb_jobs_in_queue(self):
        return len(self.jobs_queue)

    @property
    def nb_jobs_running(self):
        count = 0
        for job in self.gantt[:, 0]:
            if job != None and job.state == Job.State.RUNNING:
                count += 1

        return count

    def lookup(self, idx):
        return self.jobs_queue[idx]

    def lookup_jobs_queue(self, nb):
        nb_jobs_in_queue = self.nb_jobs_in_queue
        if nb_jobs_in_queue == 0:
            return np.array([None])

        jobs = list(itertools.islice(
            self.jobs_queue, 0, min(nb, nb_jobs_in_queue)))
        return np.array(jobs)

    def has_free_space(self, nb=1):
        free_spaces = 0
        for res in self.gantt:
            if np.any(res == None):
                free_spaces += 1
            if free_spaces == nb:
                return True

        return False

    def has_jobs_in_queue(self):
        return self.nb_jobs_in_queue > 0

    def has_jobs_waiting(self):
        return self.nb_jobs_waiting > 0

    def is_ready(self):
        return self.has_free_space() and self.has_jobs_in_queue()

    def reset(self):
        self.gantt = np.empty(shape=self.gantt_shape, dtype=object)
        self.jobs_queue = []
        self.jobs_waiting = deque()
        self.jobs_finished = dict()
        self.nb_jobs_submitted = 0
        self.nb_jobs_completed = 0

    def update_jobs_progress(self, time):
        jobs = self.gantt[:, 0]
        for job in jobs:
            if job != None and job.state == Job.State.RUNNING:
                job.update_remaining_time(time)

    def allocate_job(self, job_idx):
        try:
            if not self.has_free_space(self.jobs_queue[job_idx].requested_resources):
                raise UnavailableResourcesError(
                "There is no resource available for this job.")
        except IndexError as e:
            return None

        job = self.jobs_queue.pop(job_idx)
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
        self.jobs_waiting.append(self.jobs_queue.pop(0))

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
        while self.has_jobs_waiting():
            job = self.jobs_waiting.popleft()
            job.update_waiting_time(time)
            self.jobs_queue.append(job)

    def on_delay_expired(self, time):
        self.enqueue_jobs_waiting(time)

    def on_resource_pstate_changed(self, time):
        self.enqueue_jobs_waiting(time)

    def on_job_scheduled(self, job, time):
        job.state = Job.State.RUNNING
        job.start_time = time
        job.update_waiting_time(time)

    def on_job_completed(self, time, resources, data):
        self.nb_jobs_completed += 1
        job = self._remove_from_gantt(resources, data['job_id'])
        self._update_job_stats(job, time, data)
        self.jobs_finished[job.id] = job
        self.enqueue_jobs_waiting(time)

    def on_job_submitted(self, time, data):
        self.nb_jobs_submitted += 1
        job = Job.from_json(data)
        job.state = Job.State.SUBMITTED
        self.jobs_queue.append(job)
        self.enqueue_jobs_waiting(time)

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

    def _update_job_stats(self, job, timestamp, data):
        job.finish_time = timestamp
        job.waiting_time = data["job_waiting_time"]
        job.turnaround_time = data["job_turnaround_time"]
        job.runtime = data["job_runtime"]
        job.consumed_energy = data["job_consumed_energy"]
        job.remaining_time = 0
        job.return_code = data["return_code"]

        try:
            job.state = Job.State[data["job_state"]]
        except KeyError:
            job.state = Job.State.UNKNOWN


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
        self.waiting_time = 0  # will be set on completion by batsim
        self.turnaround_time = -1  # will be set on completion by batsim
        self.runtime = -1  # will be set on completion by batsim
        self.consumed_energy = -1  # will be set on completion by batsim
        self.remaining_time = self.requested_time  # will be updated by batsim
        self.state = Job.State.UNKNOWN
        self.return_code = None
        self.json_dict = json_dict
        self.profile_dict = profile_dict
        self.allocation = []
        self.metadata = None

    def update_waiting_time(self, t):
        self.waiting_time = t - self.submit_time

    def update_remaining_time(self, t):
        exec_time = int(t - self.start_time)
        self.remaining_time = self.requested_time - exec_time
        self.remaining_time = 1 if self.remaining_time == 0 else self.remaining_time

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
