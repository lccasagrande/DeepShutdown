import shlex
import logging
import sys
import os
import json
import zmq
import time
import itertools
import pandas as pd
from xml.dom import minidom
from enum import Enum, unique
import subprocess
from copy import deepcopy
from collections import defaultdict, deque
import numpy as np


class BatsimHandler:
    WORKLOAD_JOB_SEPARATOR = "!"
    ATTEMPT_JOB_SEPARATOR = "#"
    WORKLOAD_JOB_SEPARATOR_REPLACEMENT = "%"
    PLATFORM = "platform.xml"
    WORKLOAD = "nantes_1.json"
    CONFIG = "config.json"
    SOCKET_ENDPOINT = "tcp://*:28000"
    OUTPUT_DIR = "results"

    def __init__(self, output_freq, verbose='quiet'):
        fullpath = os.path.join(os.path.dirname(__file__), "files")
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        if not os.path.exists(BatsimHandler.OUTPUT_DIR):
            raise IOError("Dir %s does not exist" % BatsimHandler.OUTPUT_DIR)

        self._output_freq = output_freq
        self._platform = os.path.join(fullpath, BatsimHandler.PLATFORM)
        self._workload = os.path.join(fullpath, BatsimHandler.WORKLOAD)
        self._config = os.path.join(fullpath, BatsimHandler.CONFIG)
        self._verbose = verbose
        self._simulator_process = None
        self.running_simulation = False
        self.nb_simulation = 0
        self.max_walltime = self.get_max_walltime(self._workload)
        self.resource_manager = ResourceManager.from_xml(self._platform)
        self.network = NetworkHandler(BatsimHandler.SOCKET_ENDPOINT)
        self.protocol_manager = BatsimProtocolHandler()
        self.sched_manager = SchedulerManager(
            1, self.resource_manager.nb_resources)
        self._initialize_vars()

    def get_max_walltime(self, workload):
        with open(workload) as f:
            data = json.load(f)
            max_walltime = max(data['jobs'], key=(
                lambda job: job['walltime']))['walltime']

        return max_walltime

    @property
    def nb_jobs_in_queue(self):
        return len(self.sched_manager.jobs_queue)

    @property
    def nb_jobs_running(self):
        return len(self.sched_manager.get_jobs_running())

    @property
    def nb_resources(self):
        return self.resource_manager.nb_resources

    @property
    def current_state(self):
        gantt = self.sched_manager.get_gantt(with_queue=True)
        resources = self.resource_manager.get_resources()
        resources = np.append(resources, None).reshape(gantt.shape)
        gantt = np.concatenate((resources, gantt), axis=1)

        return gantt

    @property
    def state_shape(self):
        gantt_shape = self.sched_manager.gantt_shape
        # Add job queue space along with resources
        return (gantt_shape[0]+1, gantt_shape[1]+1)

    def close(self):
        self.network.close()
        self.running_simulation = False
        if self._simulator_process is not None:
            time.sleep(0.5)
            self._simulator_process.kill()
            self._simulator_process.wait()
            time.sleep(0.5)
            self._simulator_process = None

    def start(self):
        self._initialize_vars()
        self.nb_simulation += 1
        self._simulator_process = self._start_simulator()
        self.network.bind()
        self._wait_state_change()
        if not self.running_simulation:
            raise ConnectionError(
                "An error ocurred during simulator starting.")

    def now(self):
        return self.protocol_manager.now()

    def schedule_job(self, resources):
        assert self.running_simulation, "Simulation is not running."
        assert resources is not None, "Allocation cannot be null."

        if len(resources) == 0:  # Handle VOID Action
            self.sched_manager.delay_first_job(self.now())
        else:
            job = self.sched_manager.allocate_first_job(resources, self.now())
            self.protocol_manager.start_job(job.id,  job.allocation)
            self.resource_manager.set_state(job.allocation, Resource.State.COMPUTING)
            self.sched_manager.on_job_scheduled(job, self.now())

        # All jobs in the queue has to be scheduled or delayed
        if self.sched_manager.has_work():
            return

        if self.sched_manager.has_jobs_waiting() and not self._alarm_is_set:
            self.protocol_manager.wake_me_up_at(self.now() + 10)
            self._alarm_is_set = True

        #self._schedule_gantt_jobs()

        self._wait_state_change()

    def _wait_state_change(self):
        self._update_state()
        while self.running_simulation and not self.sched_manager.has_work():
            self._update_state()

    def _start_jobs(self, jobs):
        for job in jobs:
            self.protocol_manager.start_job(job.id,  job.allocation)
            self.resource_manager.set_state(
                job.allocation, Resource.State.COMPUTING)
            self.sched_manager.on_job_scheduled(job, self.now())

    def _start_simulator(self):
        if self.nb_simulation % self._output_freq == 0:
            output_path = BatsimHandler.OUTPUT_DIR + \
                "/" + str(self.nb_simulation)
        else:
            output_path = BatsimHandler.OUTPUT_DIR + "/tmp."

        cmd = "batsim -p {} -w {} -v {} -E --config-file {} -e {}".format(self._platform,
                                                                          self._workload,
                                                                          self._verbose,
                                                                          self._config,
                                                                          output_path)

        return subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)

    def _initialize_vars(self):
        self.nb_jobs_completed = 0
        self.nb_jobs_submitted = 0
        self.metrics = {}
        self._alarm_is_set = False
        self.energy_consumed = 0
        self.sched_manager.reset()

    def _schedule_gantt_jobs(self):
        jobs_to_sched = self.sched_manager.get_jobs_to_schedule()
        if len(jobs_to_sched) != 0:
            self._start_jobs(jobs_to_sched)

    def _handle_resource_pstate_changed(self, data):
        self.sched_manager.enqueue_jobs_waiting(self.now())
        res_ids = list(map(int, data["resources"].split(" ")))
        self.resource_manager.set_pstate(
            res_ids, Resource.PowerState[int(data["state"])])

    def _handle_job_completed(self, timestamp, data):
        self.nb_jobs_completed += 1
        job = self.sched_manager.on_job_completed(timestamp, data)
        self.resource_manager.set_state(job.allocation, Resource.State.IDLE)
        self._schedule_gantt_jobs()

    def _handle_job_submitted(self, data):
        if data['job']['res'] > self.resource_manager.nb_resources:
            self.protocol_manager.reject_job(data['job_id'])
        else:
            self.sched_manager.on_job_submitted(self.now(), data['job'])
            self.nb_jobs_submitted += 1

    def _handle_simulation_begins(self, data):
        self.running_simulation = True
        self.batconf = data["config"]
        self.time_sharing = data["allow_time_sharing"]
        self.dynamic_job_submission_enabled = self.batconf[
            "job_submission"]["from_scheduler"]["enabled"]
        self.profiles = data["profiles"]
        self.workloads = data["workloads"]

    def _handle_simulation_ends(self, data):
        self.protocol_manager.acknowledge()
        self._send_events()
        self.running_simulation = False
        self.metrics["scheduling_time"] = float(
            data["scheduling_time"])
        self.metrics["nb_jobs"] = int(data["nb_jobs"])
        self.metrics["nb_jobs_finished"] = int(
            data["nb_jobs_finished"])
        self.metrics["nb_jobs_success"] = int(
            data["nb_jobs_success"])
        self.metrics["nb_jobs_killed"] = int(data["nb_jobs_killed"])
        self.metrics["success_rate"] = float(data["success_rate"])
        self.metrics["makespan"] = float(data["makespan"])
        self.metrics["mean_waiting_time"] = float(
            data["mean_waiting_time"])
        self.metrics["mean_turnaround_time"] = float(
            data["mean_turnaround_time"])
        self.metrics["mean_slowdown"] = float(data["mean_slowdown"])
        self.metrics["max_waiting_time"] = float(
            data["max_waiting_time"])
        self.metrics["max_turnaround_time"] = float(
            data["max_turnaround_time"])
        self.metrics["max_slowdown"] = float(data["max_slowdown"])
        self.metrics["energy_consumed"] = float(
            data["consumed_joules"])
        self._export_metrics()

    def _handle_requested_call(self):
        self._alarm_is_set = False
        self.sched_manager.enqueue_jobs_waiting(self.now())

    def _handle_batsim_events(self, event):
        if event.type == "SIMULATION_BEGINS":
            assert not self.running_simulation, "A simulation is already running (is more than one instance of Batsim active?!)"
            self._handle_simulation_begins(event.data)
        elif event.type == "SIMULATION_ENDS":
            assert self.running_simulation, "No simulation is currently running"
            self._handle_simulation_ends(event.data)
        elif event.type == "JOB_SUBMITTED":
            self._handle_job_submitted(event.data)
        elif event.type == "JOB_COMPLETED":
            self._handle_job_completed(event.timestamp, event.data)
        elif event.type == "RESOURCE_STATE_CHANGED":
            self._handle_resource_pstate_changed(event.data)
        elif event.type == "REQUESTED_CALL":
            self._handle_requested_call()
        elif event.type == "ANSWER":
            return
        elif event.type == "NOTIFY":
            if event.data['type'] == 'no_more_static_job_to_submit':
                return
            else:
                raise Exception(
                    "Unknown NOTIFY event type {}".format(event.type))
        else:
            raise Exception("Unknown event type {}".format(event.type))

    def _export_metrics(self):
        data = pd.DataFrame(self.metrics, index=[0])
        fn = "{}/{}_{}.csv".format(
            BatsimHandler.OUTPUT_DIR,
            self.nb_simulation,
            "schedule_metrics")
        data.to_csv(fn, index=False)

    def _send_events(self):
        msg = self.protocol_manager.flush()
        assert msg is not None, "Cannot send a message if no event ocurred."
        self.network.send(msg)

    def _read_events(self):
        def get_msg():
            msg = None
            while msg is None:
                msg = self.network.recv(blocking=not self.running_simulation)
                if msg is None:
                    raise ValueError(
                        "Batsim is not responding (maybe deadlocked)")
            return BatsimMessage.from_json(msg)

        msg = get_msg()

        self.protocol_manager.update_time(msg.now)

        for event in msg.events:
            self._handle_batsim_events(event)

    def _update_state(self):
        if self.protocol_manager.has_events():
            self._send_events()

        self._read_events()

        self.sched_manager.update_jobs_progress(self.now())

        # Remember to always ack
        if self.running_simulation:
            self.protocol_manager.acknowledge()


class BatsimEvent:
    def __init__(self, timestamp, type, data):
        self.timestamp = timestamp
        self.type = type
        self.data = data


class BatsimMessage:
    def __init__(self, now, events):
        self.now = now
        self.events = [BatsimEvent(
            event['timestamp'], event['type'], event['data']) for event in events]

    @staticmethod
    def from_json(data):
        return BatsimMessage(data['now'], data['events'])


class BatsimProtocolHandler:
    def __init__(self):
        self.events = []
        self.current_time = 0
        self._ack = False

    def flush(self):
        if not self.has_events():
            return None

        self._ack = False

        if len(self.events) > 0:
            self.events = sorted(
                self.events, key=lambda event: event['timestamp'])

        msg = {
            "now": self.now(),
            "events": self.events
        }

        self.events = []

        return msg

    def has_events(self):
        return len(self.events) > 0 or self._ack

    def update_time(self, time):
        self.current_time = time

    def now(self):
        return self.current_time

    def consume_time(self, t):
        self.current_time += float(t)
        return self.current_time

    def wake_me_up_at(self, time):
        self.events.append(
            {"timestamp": self.now(),
             "type": "CALL_ME_LATER",
             "data": {"timestamp": time}})

    def notify_submission_finished(self):
        self.events.append({
            "timestamp": self.now(),
            "type": "NOTIFY",
            "data": {
                    "type": "submission_finished",
            }
        })

    def notify_submission_continue(self):
        self.events.append({
            "timestamp": self.now(),
            "type": "NOTIFY",
            "data": {
                    "type": "continue_submission",
            }
        })

    def send_message_to_job(self, job, message):
        self.events.append({
            "timestamp": self.now(),
            "type": "TO_JOB_MSG",
            "data": {
                    "job_id": job.id,
                    "msg": message,
            }
        })

    def start_job(self, job_id, res):
        """ args:res: is list of int (resources ids) """
        self.events.append({
            "timestamp": self.now(),
            "type": "EXECUTE_JOB",
            "data": {
                    "job_id": job_id,
                    "alloc": " ".join(str(r) for r in res)
            }
        })

    def execute_jobs(self, jobs, io_jobs=None):
        """ args:jobs: list of jobs to execute (job.allocation MUST be set) """

        for job in jobs:
            assert job.allocation is not None

            message = {
                "timestamp": self.now(),
                "type": "EXECUTE_JOB",
                "data": {
                        "job_id": job.id,
                        "alloc": str(job.allocation)
                }
            }
            if io_jobs is not None and job.id in io_jobs:
                message["data"]["additional_io_job"] = io_jobs[job.id]

            self.events.append(message)

    def reject_job(self, job_id):
        """Reject the given jobs."""
        self.events.append({
            "timestamp": self.now(),
            "type": "REJECT_JOB",
            "data": {
                    "job_id": job_id,
            }
        })

    def change_state(self, job, state):
        """Change the state of a job."""
        self.events.append({
            "timestamp": self.now(),
            "type": "CHANGE_state",
            "data": {
                    "job_id": job.id,
                    "state": state.name,
            }
        })

    def kill_jobs(self, jobs):
        """Kill the given jobs."""
        assert len(jobs) > 0, "The list of jobs to kill is empty"
        for job in jobs:
            job.state = Job.State.IN_KILLING
        self.events.append({
            "timestamp": self.now(),
            "type": "KILL_JOB",
            "data": {
                    "job_ids": [job.id for job in jobs],
            }
        })

    def submit_profiles(self, workload_name, profiles):
        for profile_name, profile in profiles.items():
            msg = {
                "timestamp": self.now(),
                "type": "SUBMIT_PROFILE",
                "data": {
                    "workload_name": workload_name,
                    "profile_name": profile_name,
                    "profile": profile,
                }
            }
            self.events.append(msg)

    def submit_job(
            self,
            id,
            res,
            walltime,
            profile_name,
            subtime=None,
            profile=None):

        job_dict = {
            "profile": profile_name,
            "id": id,
            "res": res,
            "walltime": walltime,
            "subtime": self.now() if subtime is None else subtime,
        }
        msg = {
            "timestamp": self.now(),
            "type": "SUBMIT_JOB",
            "data": {
                "job_id": id,
                "job": job_dict,
            }
        }
        if profile is not None:
            assert isinstance(profile, dict)
            msg["data"]["profile"] = profile

        self.events.append(msg)

        return id

    def set_resource_pstate(self, resources, state):
        self.events.append({
            "timestamp": self.now(),
            "type": "SET_RESOURCE_STATE",
            "data": {
                "resources": " ".join([str(r) for r in resources]),
                "state": str(state.value)
            }
        })

    def request_consumed_energy(self):  # TODO CHANGE NAME
        self.events.append(
            {
                "timestamp": self.now(),
                "type": "QUERY",
                "data": {
                    "requests": {"consumed_energy": {}}
                }
            }
        )

    def request_air_temperature_all(self):
        self.events.append(
            {
                "timestamp": self.now(),
                "type": "QUERY",
                "data": {
                    "requests": {"air_temperature_all": {}}
                }
            }
        )

    def request_processor_temperature_all(self):
        self.events.append(
            {
                "timestamp": self.now(),
                "type": "QUERY",
                "data": {
                    "requests": {"processor_temperature_all": {}}
                }
            }
        )

    def notify_resources_added(self, resources):
        self.events.append(
            {
                "timestamp": self.now(),
                "type": "RESOURCES_ADDED",
                "data": {
                    "resources": resources
                }
            }
        )

    def notify_resources_removed(self, resources):
        self.events.append(
            {
                "timestamp": self.now(),
                "type": "RESOURCES_REMOVED",
                "data": {
                    "resources": resources
                }
            }
        )

    def set_job_metadata(self, job_id, metadata):
        # Consume some time to be sure that the job was created before the
        # metadata is set

        if self.events == None:
            self.events = []
        self.events.append(
            {
                "timestamp": self.now(),
                "type": "SET_JOB_METADATA",
                "data": {
                    "job_id": str(job_id),
                    "metadata": str(metadata)
                }
            }
        )

    def resubmit_job(self, job, delay=1):
        job_id = job.id
        if job.id.find(BatsimHandler.ATTEMPT_JOB_SEPARATOR) == -1:
            job_id += BatsimHandler.ATTEMPT_JOB_SEPARATOR + "1"
        else:
            job_id = job_id[:-1] + str(int(job_id[-1]) + 1)

        self.reject_job(job.id)

        self.consume_time(delay)

        self.submit_job(
            job_id,
            job.requested_resources,
            job.requested_time,
            job.profile,
            profile=job.profile_dict,
            subtime=job.submit_time)

    def acknowledge(self):
        self._ack = True


class ResourceHardware:
    def __init__(self, speed, watt_idle, watt_comp):
        self.speed = speed
        self.watt_idle = watt_idle
        self.watt_comp = watt_comp


class Resource:
    class State(Enum):
        IDLE = 'idle'
        COMPUTING = 'computing'

    class PowerState(Enum):
        SHUT_DOWN = 0
        NORMAL = 1

    def __init__(self, id, state, pstate, name, hw):
        assert isinstance(state, Resource.State)
        assert isinstance(pstate, Resource.PowerState)
        assert isinstance(hw, dict)
        assert isinstance(id, int)
        self.state = state
        self.pstate = pstate
        self.name = name
        self.id = id
        self.hw = hw
        self.max_watt = self.hw[Resource.PowerState.NORMAL]['watt_comp']
        self.min_watt = self.hw[Resource.PowerState.SHUT_DOWN]['watt_idle']
        self.max_speed = self.hw[Resource.PowerState.NORMAL]['speed']

    @staticmethod
    def from_xml(id, data):
        name = data.getAttribute('id')
        host_speed = data.getAttribute('speed').split(',')
        host_watts = data.getElementsByTagName(
            'prop')[0].getAttribute('value').split(',')
        assert "Mf" in host_speed[-1], "Speed is not in Mega Flops"

        hw = {}
        for i, speed in enumerate(host_speed):
            (idle, comp) = host_watts[i].split(":")
            hw[Resource.PowerState(i)] = {
                'speed': float(speed.replace("Mf", "")),
                'watt_idle': float(idle),
                'watt_comp': float(comp)
            }
        return Resource(id, Resource.State.IDLE, Resource.PowerState.NORMAL, name, hw)

    #@property
    #def max_cost_to_compute(self):
    #    speed = self.hw[Resource.PowerState.NORMAL]['speed']
    #    return (self.max_watt - self.min_watt)

    @property
    def cost_to_compute(self):
        speed, w_idle, w_computing = self.get_hw()
        return w_computing - w_idle

    @property
    def is_available(self):
        return True if self.state == Resource.State.IDLE else False

    @property
    def is_computing(self):
        return True if self.state == Resource.State.COMPUTING else False

    def set_state(self, state):
        assert isinstance(state, Resource.State)
        self.state = state

    def set_pstate(self, pstate):
        assert isinstance(pstate, Resource.PowerState)
        self.pstate = pstate

    def get_hw(self):
        hw = self.hw[self.pstate]
        return hw['speed'], hw['watt_idle'], hw['watt_comp']

    def get_speed(self):
        speed, _, _ = self.get_hw()
        return speed

    def get_energy_consumption(self):
        _, w_idle, w_computing = self.get_hw()
        return w_computing if self.state == Resource.State.COMPUTING else w_idle

    def get_energy_pstate_diff(self):
        _, w_idle, w_computing = self.get_hw()
        return w_computing - w_idle


class ResourceManager:
    def __init__(self, resources):
        assert isinstance(resources, dict)
        self.nb_resources = len(resources)
        self.resources = resources
        self.max_watt = max(self.resources.items(), key=(lambda item: item[1].max_watt))[1].max_watt
        self.max_speed = max(self.resources.items(), key=(lambda item: item[1].max_speed))[1].max_speed

    @property
    def nb_resources_unused(self):
        nb = len([k for k, res in self.resources.items() if not res.is_computing])
        return nb

    @staticmethod
    def from_xml(platform_fn):
        platform = minidom.parse(platform_fn)
        hosts = platform.getElementsByTagName('host')
        hosts.sort(key=lambda x: x.attributes['id'].value)
        resources = {}
        id = 0
        for host in hosts:
            if host.getAttribute('id') != 'master_host':
                resources[id] = Resource.from_xml(id, host)
                id += 1

        return ResourceManager(resources)

    def estimate_energy_consumption(self, res_ids):
        energy = 0
        for id in res_ids:
            energy += self.resources[id].cost_to_compute
        return energy

    def get_resources(self):
        resources = np.empty((self.nb_resources,), dtype=object)
        for k, value in self.resources.items():
            resources[int(k)] = value

        return resources

    def is_available(self, res_ids):
        for id in res_ids:
            try:
                resource = self.resources[id]
            except KeyError:
                raise KeyError("Could not find resource: {}".format(id))
            if not resource.is_available:
                return False

        return True

    def set_state(self, res_ids, state):
        for id in res_ids:
            try:
                self.resources[id].set_state(state)
            except KeyError:
                raise KeyError("Could not find resource: {}".format(id))

    def get_state(self, output_values=False):
        if output_values:
            return [res.state.value for _, res in self.resources.items()]

        return [res.state for _, res in self.resources.items()]

    def set_pstate(self, res_ids, pstate):
        for id in res_ids:
            try:
                self.resources[id].set_pstate(pstate)
            except KeyError:
                raise KeyError("Could not find resource: {}".format(id))

    def get_pstate(self):
        return [res.pstate for _, res in self.resources.items()]


class SchedulerManager():
    def __init__(self, space, nb_resources):
        self.gantt_shape = (nb_resources, space)
        self.reset()

    @property
    def nb_jobs_waiting(self):
        return len(self.jobs_queue) + len(self.jobs_waiting)

    def lookup_jobs_queue(self, nb):
        nb_jobs_in_queue = len(self.jobs_queue) 
        if nb_jobs_in_queue == 0: 
            return None

        jobs = list(itertools.islice(self.jobs_queue, 0, min(nb, nb_jobs_in_queue)))
        return jobs

    def get_jobs_running(self):
        jobs = self.gantt[:, 0][self.gantt[:, 0] != None]
        jobs_running = [job for job in jobs if job.state == Job.State.RUNNING]
        return jobs_running

    def has_work(self):
        ready = self.has_free_space() and self.has_jobs_in_queue()
        return ready

    def reset(self):
        self.gantt = np.empty(shape=self.gantt_shape, dtype=object)
        self.jobs_queue = deque()
        self.jobs_waiting = deque()
        self.jobs_finished = dict()

    def get_gantt(self, with_queue=True):
        if not with_queue:
            return self.gantt

        jobs_in_queue = np.empty(shape=self.gantt_shape[1], dtype=object)
        for i in range(min(self.gantt_shape[1], len(self.jobs_queue))):
            jobs_in_queue[i] = self.jobs_queue[i]

        return np.append(self.gantt, [jobs_in_queue], axis=0)

    def update_jobs_progress(self, time):
        jobs = self.gantt[:, 0]
        for job in jobs[jobs != None]:
            if job.state == Job.State.RUNNING:
                job.update_remaining_time(time)

    def allocate_first_job(self, resources, time):
        assert self.has_jobs_in_queue()
        job = self.jobs_queue.popleft()
        try:
            assert job.requested_resources == len(
                resources), "The job requested more resources than it was allocated."

            for res in resources:
                success = False
                resource_spaces = self.gantt[res]
                for i in range(self.gantt_shape[1]):
                    if resource_spaces[i] == None:
                        resource_spaces[i] = job
                        success = True
                        break

                if not success:
                    raise UnavailableResourcesError(
                        "There is no space available on the resource selected.")
            job.allocation = resources
        except:
            self.jobs_queue.appendleft(job)
            raise
        return job

    def get_first_job_walltime(self):
        if len(self.jobs_queue) == 0:
            return 0

        return self.jobs_queue[0].requested_time

    def get_first_job_wait_time(self):
        if len(self.jobs_queue) == 0:
            return 0

        return self.jobs_queue[0].waiting_time

    def get_total_wait_time(self):
        wait_time = 0
        # queued jobs
        for j in self.jobs_queue:
            wait_time += j.waiting_time

        for j in self.jobs_waiting:
            wait_time += j.waiting_time
        return wait_time

    def delay_first_job(self, time):
        job = self.jobs_queue.popleft()
        self.jobs_waiting.append(job)

    def get_jobs_to_schedule(self):
        jobs = self.gantt[:, 0][self.gantt[:, 0] != None]
        return [job for job in jobs if job.state != Job.State.RUNNING]

    def enqueue_jobs_waiting(self, time):
        while self.has_jobs_waiting():
            job = self.jobs_waiting.popleft()
            job.set_waiting_time(time - job.submit_time)
            self.jobs_queue.append(job)

    def has_free_space(self):
        return np.any(self.gantt == None)

    def has_jobs_in_queue(self):
        return len(self.jobs_queue) > 0

    def has_jobs_waiting(self):
        return len(self.jobs_waiting) > 0

    def on_job_scheduled(self, job, time):
        job.state = Job.State.RUNNING
        job.set_start_time(time)
        job.set_waiting_time(job.start_time - job.submit_time)

    def on_job_completed(self, time, data):
        job = self._remove_from_gantt(data['job_id'])
        self._update_job_stats(job, time, data)
        self.jobs_finished[job.id] = job
        self.enqueue_jobs_waiting(time)
        return job

    def on_job_submitted(self, time, data):
        job = Job.from_json(data)
        job.state = Job.State.SUBMITTED
        job.set_waiting_time(0)
        self.jobs_queue.append(job)
        self.enqueue_jobs_waiting(time)

    def _remove_from_gantt(self, job_id):
        job = None
        for res in range(self.gantt_shape[0]):
            res_job = self.gantt[res][0]
            if (res_job == None) or (res_job.id != job_id):
                continue

            job = res_job
            for space in range(1, self.gantt_shape[1]):
                self.gantt[res][space-1] = self.gantt[res][space]

            self.gantt[res][-1] = None
            break
        assert job is not None
        return job

    def _update_job_stats(self, job, timestamp, data):
        job.set_finish_time(timestamp)
        job.set_waiting_time(data["job_waiting_time"])
        job.set_turnaround_time(data["job_turnaround_time"])
        job.set_runtime(data["job_runtime"])
        job.set_consumed_energy(data["job_consumed_energy"])
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
        self.waiting_time = -1  # will be set on completion by batsim
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

    def set_start_time(self, t):
        self.start_time = t

    def set_waiting_time(self, t):
        self.waiting_time = t

    def set_finish_time(self, t):
        self.finish_time = t

    def set_turnaround_time(self, t):
        self.turnaround_time = t

    def set_runtime(self, t):
        self.runtime = t

    def set_consumed_energy(self, t):
        self.consumed_energy = t

    def update_remaining_time(self, t):
        exec_time = (t - self.start_time)
        self.remaining_time = float(self.requested_time) - exec_time

    @property
    def workload(self):
        return self.id.split(BatsimHandler.WORKLOAD_JOB_SEPARATOR)[0]

    @staticmethod
    def from_json(json_dict, profile_dict=None):
        return Job(json_dict["id"],
                   json_dict["subtime"],
                   json_dict.get("walltime", -1),
                   json_dict["res"],
                   json_dict["profile"],
                   json_dict,
                   profile_dict)


class NetworkHandler:

    def __init__(
            self,
            socket_endpoint,
            verbose=0,
            timeout=10000,
            type=zmq.REP):
        self.socket_endpoint = socket_endpoint
        self.verbose = verbose
        self.timeout = timeout
        self.context = zmq.Context()
        self.connection = None
        self.type = type

    def send(self, msg):
        self.send_string(json.dumps(msg))

    def send_string(self, msg):
        assert self.connection, "Connection not open"
        self.connection.send_string(msg)

    def recv(self, blocking=False):
        msg = self.recv_string(blocking=blocking)
        if msg is not None:
            msg = json.loads(msg)
        return msg

    def recv_string(self, blocking=False):
        assert self.connection, "Connection not open"
        if blocking or self.timeout is None or self.timeout <= 0:
            self.connection.RCVTIMEO = -1
        else:
            self.connection.RCVTIMEO = self.timeout
        try:
            msg = self.connection.recv_string()
        except zmq.error.Again:
            return None

        return msg

    def bind(self):
        assert not self.connection, "Connection already open"
        self.connection = self.context.socket(self.type)

        self.connection.bind(self.socket_endpoint)

    def connect(self):
        assert not self.connection, "Connection already open"
        self.connection = self.context.socket(self.type)
        self.connection.connect(self.socket_endpoint)

    def subscribe(self, pattern=b''):
        self.type = zmq.SUB
        self.connect()
        self.connection.setsockopt(zmq.SUBSCRIBE, pattern)

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None


class InsufficientResourcesError(Exception):
    pass


class UnavailableResourcesError(Exception):
    pass
