import shlex
import logging
import sys
import os
import json
import zmq
from xml.dom import minidom
from enum import Enum
import subprocess
from copy import deepcopy
from procset import ProcSet
from collections import defaultdict, deque


class BatsimHandler:
    WORKLOAD_JOB_SEPARATOR = "!"
    ATTEMPT_JOB_SEPARATOR = "#"
    WORKLOAD_JOB_SEPARATOR_REPLACEMENT = "%"
    PLATFORM = "platform.xml"
    WORKLOAD = "workload.json"
    CONFIG = "config.json"
    SOCKET_ENDPOINT = "tcp://*:28000"

    def __init__(self, verbose="information"):
        fullpath = os.path.join(os.path.dirname(__file__), "files")
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self._platform = os.path.join(fullpath, BatsimHandler.PLATFORM)
        self._workload = os.path.join(fullpath, BatsimHandler.WORKLOAD)
        self._config = os.path.join(fullpath, BatsimHandler.CONFIG)
        self._verbose = verbose
        self._simulator_process = None
        self.running_simulation = False
        self.network = NetworkHandler(BatsimHandler.SOCKET_ENDPOINT)
        self.nb_resources = self._count_resources(self._platform)
        self.nb_jobs = self._count_jobs(self._workload)
        self.protocol_manager = BatsimProtocolHandler()
        self._initialize_vars()

    def get_job_info(self):
        if len(self.jobs_waiting) == 0:
            return None
        job = self.jobs_waiting[0]
        job.waiting_time = int(self.now()) - job.submit_time
        return job

    def get_resources_info(self):
        return self.resource_manager.get_state()

    def close(self):
        if self._simulator_process is not None:
            self._simulator_process.kill()
            self._simulator_process.wait()
            self._simulator_process = None
        self.network.close()
        self.running_simulation = False

    def start(self):
        self._initialize_vars()
        self._simulator_process = self._start_simulator()
        self.network.bind()
        self.wait_until_next_event()
        if not self.running_simulation:
            raise ConnectionError(
                "An error ocurred during simulator starting.")

    def schedule_job(self, res):
        def handle_reject():
            self.protocol_manager.reject_job([job])
            self.nb_jobs_rejected += 1

        def handle_void():
            self.protocol_manager.resubmit_job(job)
            self.nb_jobs_resubmit += 1

        def handle_allocation():
            if job.requested_resources != len(res):
                raise InsufficientResourcesError(
                    "Job requested {} resources while {} resources were selected.".format(job.requested_resources, len(res)))

            if not self.resource_manager.is_available(res):
                raise UnavailableResourcesError(
                    "Cannot allocate unavailable resources for job {}.".format(job.id))

            self.protocol_manager.start_job(job, res)
            self.protocol_manager.set_resource_state(
                res, ResourceState.computing)
            job.allocation = res
            self.nb_jobs_scheduled += 1

        assert self.running_simulation, "Simulation is not running."
        job = self.jobs_waiting.popleft()
        job.waiting_time = int(self.now()) - job.submit_time

        if res == None:
            handle_reject()
        elif len(res) == 0:
            handle_void()
        else:
            handle_allocation()

        if len(self.jobs_waiting) == 0:
            self._send_events()
            self.wait_until_next_event()

    def wait_until_next_event(self):
        jobs_received = self.nb_jobs_received
        self._read_events()
        while self.nb_jobs_received == jobs_received and self.running_simulation and len(self.jobs_waiting) == 0:
            self._read_events()

    def now(self):
        assert self.running_simulation, "Simulation not running."
        return self.protocol_manager.now()

    def _count_resources(self, platform):
        return len(minidom.parse(platform).getElementsByTagName('host')) - 1

    def _count_jobs(self, workload):
        with open(workload, 'r') as f:
            data = json.load(f)
            nb_jobs = len(data['jobs'])
        return nb_jobs

    def _start_simulator(self, dev=True):
        if dev:
            path = "/home/d2/Projects/batsim/build/batsim"
        else:
            path = "batsim"
        cmd = path + " -p %s -w %s -v %s -E --config-file %s" % (
            self._platform, self._workload, self._verbose, self._config)
        # return Popen(shlex.split(cmd))
        return subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)

    def _initialize_vars(self):
        self.nb_jobs_received = 0
        self.nb_jobs_submitted = 0
        self.nb_jobs_killed = 0
        self.nb_jobs_rejected = 0
        self.nb_jobs_scheduled = 0
        self.nb_jobs_completed = 0
        self.nb_jobs_successful = 0
        self.nb_jobs_failed = 0
        self.nb_jobs_timeout = 0
        self.nb_jobs_resubmit = 0
        self.energy_consumed = 0
        self.proc_temp = 0
        self.air_temp = 0
        #self._ack = False
        self._job_not_scheduled = False
        #self._events_to_send = []
        self.jobs = dict()
        self.jobs_waiting = deque()

    def _handle_all_jobs_completed(self):
        if self.dynamic_job_submission_enabled:
            self.protocol_manager.notify_submission_finished()
            self.dynamic_job_submission_enabled = False

    def _handle_resource_state_changed(self, resources, state):
        self.resource_manager.set_state(resources, state)

    def _handle_job_completed(self, job):
        def update_job_stats(j):
            if j.job_state == Job.State.COMPLETED_WALLTIME_REACHED:
                self.nb_jobs_timeout += 1
            elif j.job_state == Job.State.COMPLETED_FAILED:
                self.nb_jobs_failed += 1
            elif j.job_state == Job.State.COMPLETED_SUCCESSFULLY:
                self.nb_jobs_successful += 1
            elif j.job_state == Job.State.COMPLETED_KILLED:
                self.nb_jobs_killed += 1
            self.nb_jobs_completed += 1

        self.nb_jobs_completed += 1
        self.protocol_manager.set_resource_state(
            job.allocation, ResourceState.idle)
        update_job_stats(job)

    def _handle_job_submitted(self, job):
        job.job_state = Job.State.SUBMITTED

        # don't override dynamic job
        if job.id not in self.jobs:
            self.jobs[job.id] = job

        self.jobs_waiting.append(job)
        self._job_not_scheduled = True
        self.nb_jobs_received += 1

    def _send_events(self):
        if self._job_not_scheduled:
            self._job_not_scheduled = False
            return

        msg = self.protocol_manager.flush()
        if msg == None:
            return

        self.network.send(msg)
        #self._events_to_send = []

    def _read_events(self):
        msg = None
        while msg is None:
            msg = self.network.recv(blocking=not self.running_simulation)
            if msg is None:
                raise ValueError("Batsim is not responding (maybe deadlocked)")

        self.protocol_manager.update_time(msg["now"])

        for event in msg["events"]:
            event_type = event["type"]
            event_data = event.get("data", {})
            if event_type == "SIMULATION_BEGINS":
                assert not self.running_simulation, "A simulation is already running (is more than one instance of Batsim active?!)"
                assert event_data["nb_resources"] == self.nb_resources, "Batsim platform and Simulator platform does not match."
                self.running_simulation = True
                self.batconf = event_data["config"]
                self.time_sharing = event_data["allow_time_sharing"]
                self.dynamic_job_submission_enabled = self.batconf[
                    "job_submission"]["from_scheduler"]["enabled"]
                self.resource_manager = ResourceManager(
                    {res["id"]: res for res in event_data["compute_resources"]})
                self.profiles = event_data["profiles"]
                self.workloads = event_data["workloads"]
            elif event_type == "SIMULATION_ENDS":
                assert self.running_simulation, "No simulation is currently running"
                assert self.nb_jobs_received == self.nb_jobs_completed, "There are some jobs unfinished"
                assert self.nb_jobs_received == (self.nb_jobs_resubmit+self.nb_jobs_scheduled)
                self.running_simulation = False
            elif event_type == "JOB_SUBMITTED":
                json_dict = event_data["job"]
                try:
                    profile_dict = event_data["profile"]
                except KeyError:
                    profile_dict = {}
                job = Job.from_json_dict(json_dict, profile_dict)

                self._handle_job_submitted(job)
            elif event_type == "JOB_COMPLETED":
                job_id = event_data["job_id"]
                j = self.jobs[job_id]
                j.finish_time = event["timestamp"]

                try:
                    j.job_state = Job.State[event["data"]["job_state"]]
                except KeyError:
                    j.job_state = Job.State.UNKNOWN
                j.return_code = event["data"]["return_code"]

                self._handle_job_completed(j)
            elif event_type == "RESOURCE_STATE_CHANGED":
                res = list(map(int, event_data["resources"].split(" ")))
                self._handle_resource_state_changed(
                    res, int(event_data["state"]))
            elif event_type == "ANSWER":
                if "consumed_energy" in event_data:
                    self.energy_consumed = event_data["consumed_energy"]
                elif "processor_temperature_all" in event_data:
                    self.proc_temp = event_data["processor_temperature_all"]
                elif "air_temperature_all" in event_data:
                    self.air_temp = event_data["air_temperature_all"]
            else:
                raise Exception("Unknown event type {}".format(event_type))

        self.protocol_manager.acknowledge()

        if self.nb_jobs_completed == self.nb_jobs:
            self._handle_all_jobs_completed()

        self._send_events()

        return not self.running_simulation


class BatsimProtocolHandler:
    def __init__(self):
        self.events = []
        self.current_time = 0
        self._ack = False

    def flush(self):
        if (len(self.events) == 0 and not self._ack):
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

    def start_job(self, job, res):
        """ args:res: is list of int (resources ids) """
        self.events.append({
            "timestamp": self.now(),
            "type": "EXECUTE_JOB",
            "data": {
                    "job_id": job.id,
                    "alloc": str(ProcSet(*res))
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

    def reject_job(self, job):
        """Reject the given jobs."""
        assert job is not None, "The job to reject is empty"
        self.events.append({
            "timestamp": self.now(),
            "type": "REJECT_JOB",
            "data": {
                    "job_id": job.id,
            }
        })

    def change_job_state(self, job, state):
        """Change the state of a job."""
        self.events.append({
            "timestamp": self.now(),
            "type": "CHANGE_JOB_STATE",
            "data": {
                    "job_id": job.id,
                    "job_state": state.name,
            }
        })

    def kill_jobs(self, jobs):
        """Kill the given jobs."""
        assert len(jobs) > 0, "The list of jobs to kill is empty"
        for job in jobs:
            job.job_state = Job.State.IN_KILLING
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

    def set_resource_state(self, resources, state):
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

        self.reject_job(job)

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


class ResourceManager:
    def __init__(self, resources):
        self.resources = resources

    def is_available(self, resources):
        for r in resources:
            try:
                resource = self.resources[r]
            except KeyError as e:
                raise KeyError("Could not find resource: {}".format(e))
            if not resource['state'] == ResourceState.idle.name:
                return False

        return True

    def set_state(self, resources, state):
        state_name = ResourceState.to_name(state)
        for res in resources:
            self.resources[res]['state'] = state_name

    def get_state(self):
        return [ResourceState.to_value(s['state']) for _, s in self.resources.items()]


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
        self.finish_time = None  # will be set on completion by batsim
        self.waiting_time = None  # will be set by batsim
        self.job_state = Job.State.UNKNOWN
        self.return_code = None
        self.progress = None
        self.json_dict = json_dict
        self.profile_dict = profile_dict
        self.allocation = None
        self.metadata = None

    def __repr__(self):
        return(
            ("<Job {0}; sub:{1} res:{2} reqtime:{3} prof:{4} "
                "state:{5} ret:{6} alloc:{7}>\n").format(
                self.id, self.submit_time, self.requested_resources,
                self.requested_time, self.profile,
                self.job_state,
                self.return_code, self.allocation))

    @property
    def workload(self):
        return self.id.split(BatsimHandler.WORKLOAD_JOB_SEPARATOR)[0]

    @staticmethod
    def from_json_string(json_str):
        json_dict = json.loads(json_str)
        return Job.from_json_dict(json_dict)

    @staticmethod
    def from_json_dict(json_dict, profile_dict=None):
        return Job(json_dict["id"],
                   json_dict["subtime"],
                   json_dict.get("walltime", -1),
                   json_dict["res"],
                   json_dict["profile"],
                   json_dict,
                   profile_dict)


class ResourceState(Enum):
    sleeping = 0
    idle = 1
    computing = 2

    @staticmethod
    def to_value(name):
        if name == ResourceState.sleeping.name:
            return ResourceState.sleeping.value
        elif name == ResourceState.idle.name:
            return ResourceState.idle.value
        elif name == ResourceState.computing.name:
            return ResourceState.computing.value
        else:
            raise Exception("Unknown resource state name {}".format(name))

    @staticmethod
    def to_name(value):
        if value == ResourceState.sleeping.value:
            return ResourceState.sleeping.name
        elif value == ResourceState.idle.value:
            return ResourceState.idle.name
        elif value == ResourceState.computing.value:
            return ResourceState.computing.name
        else:
            raise Exception("Unknown resource state value {}".format(value))


class NetworkHandler:

    def __init__(
            self,
            socket_endpoint,
            verbose=0,
            timeout=2000,
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
        if self.verbose > 0:
            print("[PYBATSIM]: SEND_MSG\n {}".format(msg))
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

        if self.verbose > 0:
            print('[PYBATSIM]: RECEIVED_MSG\n {}'.format(msg))

        return msg

    def bind(self):
        assert not self.connection, "Connection already open"
        self.connection = self.context.socket(self.type)

        if self.verbose > 0:
            print("[PYBATSIM]: binding to {addr}"
                  .format(addr=self.socket_endpoint))
        self.connection.bind(self.socket_endpoint)

    def connect(self):
        assert not self.connection, "Connection already open"
        self.connection = self.context.socket(self.type)

        if self.verbose > 0:
            print("[PYBATSIM]: connecting to {addr}"
                  .format(addr=self.socket_endpoint))
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
