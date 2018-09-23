import itertools
import numpy as np
from enum import Enum
from collections import deque


class SchedulerManager():
    def __init__(self, nb_resources, time_window):
        self.gantt_shape = (nb_resources, time_window)
        self.reset()

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

    def lookup_first_job(self):
        if self.nb_jobs_in_queue > 0:
            return self.jobs_queue[0]
        return None

    def lookup_jobs_queue(self, nb):
        nb_jobs_in_queue = self.nb_jobs_in_queue
        if nb_jobs_in_queue == 0:
            return np.array([None])

        jobs = list(itertools.islice(
            self.jobs_queue, 0, min(nb, nb_jobs_in_queue)))
        return np.array(jobs)

    def has_free_space(self):
        return np.any(self.gantt == None)

    def has_jobs_in_queue(self):
        return self.nb_jobs_in_queue > 0

    def has_jobs_waiting(self):
        return self.nb_jobs_waiting > 0

    def is_ready(self):
        return self.has_free_space() and self.has_jobs_in_queue()

    def reset(self):
        self.gantt = np.empty(shape=self.gantt_shape, dtype=object)
        self.jobs_queue = deque()
        self.jobs_waiting = deque()
        self.jobs_finished = dict()
        self.nb_jobs_submitted = 0
        self.nb_jobs_completed = 0

    def update_jobs_progress(self, time):
        jobs = self.gantt[:, 0]
        for job in jobs:
            if job != None and job.state == Job.State.RUNNING:
                job.update_remaining_time(time)

    def allocate_first_job(self, resources):
        assert self.has_jobs_in_queue()
        job = self.jobs_queue.popleft()
        try:
            assert job.requested_resources == len(
                resources), "The job requested more resources than it was allocated."

            for res in resources:
                success = False
                resource_queue = self.gantt[res]
                for i in range(len(resource_queue)):  # Find the first space available
                    if resource_queue[i] == None:
                        resource_queue[i] = job
                        success = True
                        break

                if not success:
                    raise UnavailableResourcesError(
                        "There is no space available on the resource selected.")
            job.allocation = resources
        except:
            self.jobs_queue.appendleft(job)
            raise

    def delay_first_job(self):
        job = self.jobs_queue.popleft()
        self.jobs_waiting.append(job)

    def get_ready_jobs(self):
        jobs = self.gantt[:, 0][self.gantt[:, 0] != None]
        return [job for job in jobs if job.state != Job.State.RUNNING]

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

    def on_job_completed(self, time, data):
        self.nb_jobs_completed += 1
        job = self._remove_from_gantt(
            list(map(int, data["alloc"].split(" "))), data['job_id'])
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
        self.allocation = None
        self.metadata = None

    def update_waiting_time(self, t):
        self.waiting_time = t - self.submit_time

    def update_remaining_time(self, t):
        exec_time = (t - self.start_time)
        self.remaining_time = float(self.requested_time) - exec_time

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
