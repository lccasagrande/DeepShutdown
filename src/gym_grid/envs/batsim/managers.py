from collections import defaultdict, deque
from .models import Job, Resource


class JobsManager(object):
    def __init__(self):
        self.queue = deque()
        self.statistics = {'job_total': 0, 'job_timeout': 0, 'job_failed': 0,
                           'job_successful': 0, 'job_killed': 0,
                           'waiting_time': 0}

    def enqueue(self, job):
        assert(isinstance(job, Job))
        self.statistics['job_total'] += 1
        self.queue.append(job)

    def dequeue(self, curr_time):
        job = self.queue.popleft()
        self.statistics['waiting_time'] += (curr_time - job.subtime)
        return job

    def at(self, i):
        return self.queue[i]

    def onJobCompleted(self, event):
        if event.data['job_state'] == "COMPLETED_SUCCESSFULLY":
            self.statistics['job_successful'] += 1
        elif event.data['job_state'] == "COMPLETED_WALLTIME_REACHED":
            self.statistics['job_timeout'] += 1
        elif event.data['job_state'] == "COMPLETED_FAILED":
            self.statistics['job_failed'] += 1
        elif event.data['job_state'] == "COMPLETED_KILLED":
            self.statistics['job_killed'] += 1

    @property
    def is_empty(self):
        return len(self.queue) == 0


class ResourcesManager:
    def __init__(self, resources):
        self.resources = resources
        self.statistics = {'alloc_failed': 0}

    # {sleeping, idle, computing, switching_on, switching_off}
    def set_resource_state(self, id, state):
        self.resources[id].state = state

    def change_to_computing(self, id):
        self.resources[id].state = 'computing'

    def reset_states(self):
        for res in resources:
            res.state = 'idle'

    def has_available(self, n):
        counter = 0
        for _, res in self.resources.items():
            if res.is_available():
                counter += 1

        return n <= counter

    def is_available(self, id):
        return self.resources[id].is_available()

    def onJobCompleted(self, event):
        for interval in event.data['alloc'].split(" "):
            for node in interval.split("-"):
                self.set_resource_state(int(node), 'idle')

    def onResourceStateChanged(self, data):
        for resource in data["resources"].split(" "):
            hosts = resource.split("-")
            if len(hosts) == 1:
                self.set_resource_state(hosts[0], data['state'])
            elif len(hosts) == 2:
                for node in range(hosts[0], hosts[1]+1):
                    self.set_resource_state(node, data['state'])
            else:
                raise Exception("Multiple intervals are not supported")

    @staticmethod
    def import_from(data):
        res = defaultdict(object)
        for resource in data['resources_data']:
            res[resource['id']] = Resource(resource)
        return ResourcesManager(res)
