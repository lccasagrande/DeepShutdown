import json
from sortedcontainers import SortedList
from .scheduler import Job
from .network import BatsimEvent


class GridSimulator:
    def __init__(self, workload_fn, jobs_manager):
        self.jobs_manager = jobs_manager
        self.workload = self._get_workload(workload_fn)
        self.workload_nb_jobs = len(self.workload)
        self.max_tracking_time_since_last_job = 10
        self.close()

    def close(self):
        self.curr_workload = None
        self.jobs_submmited = -1
        self.jobs_completed = -1
        self.running = False
        self.current_time = -1
        self.time_since_last_new_job = -1

    def _get_workload(self, workload_fn):
        with open(workload_fn, 'r') as f:
            data = json.load(f)
            jobs = SortedList(key=lambda f: f.submit_time)
            for j in data['jobs']:
                jobs.add(Job.from_json(j))

        return jobs

    def get_jobs_completed(self, time):
        jobs_running = self.jobs_manager.jobs_running
        for job in jobs_running:
            if job.remaining_time == 0:
                yield job

    def get_jobs_submmited(self, time):
        while len(self.curr_workload) > 0 and self.curr_workload[0].submit_time == time:
            yield self.curr_workload.pop(0)

    def reject_job(self, job_id):
        self.jobs_completed += 1

    def get_job_submitted_event(self, time, job):
        data = dict(job_id=job.id,
                    job=dict(profile=job.profile,
                             res=job.requested_resources,
                             id=job.id,
                             subtime=job.submit_time,
                             walltime=job.requested_time))
        return BatsimEvent(time, "JOB_SUBMITTED", data)

    def get_job_completed_event(self, time, job):
        data = dict(
            job_id=job.id,
            job_state=Job.State.COMPLETED,
            return_code=0,
            kill_reason="",
            alloc=job.allocation)
        return BatsimEvent(time, "JOB_COMPLETED", data)

    def get_simulation_ended_event(self, time):
        return BatsimEvent(time, "SIMULATION_ENDS", dict())

    def get_simulation_begins_event(self, time):
        return BatsimEvent(time, "SIMULATION_BEGINS", dict())

    @property
    def simulation_ended(self):
        return self.jobs_submmited == self.workload_nb_jobs and self.jobs_completed == self.workload_nb_jobs

    def proceed_time(self, t):
        self.current_time += t
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1

    def start(self):
        self.curr_workload = self.workload.copy()
        self.current_time = 0
        self.jobs_submmited = 0
        self.jobs_completed = 0
        self.time_since_last_new_job = 0
        self.running = True

    def read_events(self):
        assert self.running
        events = []

        for j in self.get_jobs_submmited(self.current_time):
            self.time_since_last_new_job = 0
            self.jobs_submmited += 1
            events.append(self.get_job_submitted_event(self.current_time, j))

        for j in self.get_jobs_completed(self.current_time):
            self.jobs_completed += 1
            events.append(self.get_job_completed_event(self.current_time, j))

        if self.simulation_ended:
            self.running = False
            events.append(self.get_simulation_ended_event(self.current_time))

        return events
