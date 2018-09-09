import shlex
import logging
import sys
import os
import json
import zmq
from xml.dom import minidom
from enum import Enum, unique
import subprocess
from copy import deepcopy
from collections import defaultdict, deque


class BatsimHandler:
    WORKLOAD_JOB_SEPARATOR = "!"
    ATTEMPT_JOB_SEPARATOR = "#"
    WORKLOAD_JOB_SEPARATOR_REPLACEMENT = "%"
    PLATFORM = "platform.xml"
    WORKLOAD = "workload.json"
    CONFIG = "config.json"
    SOCKET_ENDPOINT = "tcp://*:28000"
    OUTPUT_DIR = "results"

    def __init__(self, output_freq, verbose):
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
        self.network = NetworkHandler(BatsimHandler.SOCKET_ENDPOINT)
        self.nb_resources = self._count_resources(self._platform)
        self.protocol_manager = BatsimProtocolHandler()
        self.nb_simulation = 0
        self._initialize_vars()

    def get_job_info(self):
        return self.job_manager.lookup()

    def get_resources_info(self):
        return self.resource_manager.get_state(output_values=True)

    def close(self):
        if self._simulator_process is not None:
            self._simulator_process.kill()
            self._simulator_process.wait()
            self._simulator_process = None
        self.network.close()
        self.running_simulation = False

    def start(self):
        self._initialize_vars()
        self.nb_simulation += 1
        self._simulator_process = self._start_simulator()
        self.network.bind()
        self.wait_until_next_event()
        if not self.running_simulation:
            raise ConnectionError(
                "An error ocurred during simulator starting.")

    def schedule_job(self, allocation):
        assert self.running_simulation, "Simulation is not running."
        assert allocation is not None, "Allocation cannot be null."

        job = self.get_job_info()
        assert job is not None

        if len(allocation) == 0:  # Handle VOID Action
            self.job_manager.on_job_waiting()
            self.protocol_manager.acknowledge()
        else:
            self._validate_allocation(job.requested_resources, allocation)
            self.job_manager.on_job_scheduling(allocation)
            self.protocol_manager.start_job(job.id,  allocation)
            self.resource_manager.set_state(
                allocation, Resource.State.COMPUTING)

        if self.job_manager.has_jobs_in_queue():
            return

        if self.job_manager.has_jobs_waiting() and not self._alarm_is_set:
            self.protocol_manager.wake_me_up_at(self.now() + 5)
            self._alarm_is_set = True 

        self._send_events()
        self.wait_until_next_event()
        # Enqueue jobs if another type of event has ocurred first.
        if self.running_simulation:
            self.job_manager.enqueue_jobs_waiting(self.now())

    def wait_until_next_event(self):
        self._read_events()
        while self.running_simulation and not self.job_manager.has_jobs_in_queue():
            self._read_events()

    def now(self):
        assert self.running_simulation, "Simulation not running."
        return self.protocol_manager.now()

    def _validate_allocation(self, req_res, allocation):
        if req_res != len(allocation):
            raise InsufficientResourcesError(
                "Job requested {} resources while {} resources were selected.".format(req_res, len(allocation)))

        if not self.resource_manager.is_available(allocation):
            raise UnavailableResourcesError(
                "Cannot allocate unavailable resources.")

    def _count_resources(self, platform):
        return len(minidom.parse(platform).getElementsByTagName('host')) - 1

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
        self._wait_for_scheduler_action = False
        self._alarm_is_set = False
        self.energy_consumed = 0
        self.job_manager = JobManager()

    def _handle_resource_pstate_changed(self, data):
        res_ids = list(map(int, data["resources"].split(" ")))
        self.resource_manager.set_pstate(
            res_ids, Resource.PowerState[int(data["state"])])

    def _handle_job_completed(self, timestamp, data):
        self.nb_jobs_completed += 1
        job = self.job_manager.on_job_completed(timestamp, data)
        self.resource_manager.set_state(job.allocation, Resource.State.IDLE)

    def _handle_job_submitted(self, data):
        if data['job']['res'] > self.resource_manager.nb_res:
            self.protocol_manager.reject_job(data['job_id'])
        else:
            self.job_manager.on_job_submitted(data['job'])
            self._wait_for_scheduler_action = True

    def _handle_simulation_begins(self, data):
        assert data["nb_resources"] == self.nb_resources, "Batsim platform and Simulator platform does not match."

        self.running_simulation = True
        self.batconf = data["config"]
        self.time_sharing = data["allow_time_sharing"]
        self.dynamic_job_submission_enabled = self.batconf[
            "job_submission"]["from_scheduler"]["enabled"]
        self.resource_manager = ResourceManager.from_json(
            data["compute_resources"])
        self.profiles = data["profiles"]
        self.workloads = data["workloads"]

    def _handle_simulation_ends(self, data):
        self.running_simulation = False

    def _handle_requested_call(self):
        self._alarm_is_set = False

        if not self.job_manager.has_jobs_waiting():
            return

        self.job_manager.enqueue_jobs_waiting(self.now())
        self._wait_for_scheduler_action = True

    def _send_events(self):
        if self._wait_for_scheduler_action:
            self._wait_for_scheduler_action = False
            return

        msg = self.protocol_manager.flush()
        if msg == None:
            return

        self.network.send(msg)

    def _get_msg(self):
        msg = None
        while msg is None:
            msg = self.network.recv(blocking=not self.running_simulation)
            if msg is None:
                raise ValueError("Batsim is not responding (maybe deadlocked)")
        return BatsimMessage.from_json(msg)

    def _read_events(self):
        msg = self._get_msg()
        self.protocol_manager.update_time(msg.now)

        for event in msg.events:
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
                if "consumed_energy" in event.data:
                    self.energy_consumed = event.data["consumed_energy"]
            elif event.type == "NOTIFY":
                if event.data['type'] == 'no_more_static_job_to_submit':
                    continue
                else:
                    raise Exception(
                        "Unknown NOTIFY event type {}".format(event.type))
            else:
                raise Exception("Unknown event type {}".format(event.type))

        self.protocol_manager.acknowledge()
        self._send_events()


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


class Resource:
    class State(Enum):
        SLEEPING = 'sleeping'
        IDLE = 'idle'
        COMPUTING = 'computing'

        @staticmethod
        def convert(str):
            if str == Resource.State.SLEEPING.value:
                return Resource.State.SLEEPING
            elif str == Resource.State.IDLE.value:
                return Resource.State.IDLE
            elif str == Resource.State.COMPUTING.value:
                return Resource.State.COMPUTING
            else:
                raise KeyError("Unknown resource state")

    class PowerState(Enum):
        SHUT_DOWN = 0
        NORMAL = 1

    def __init__(self, id, state, pstate, name):
        assert isinstance(state, Resource.State)
        assert isinstance(pstate, Resource.PowerState)
        self.state = state
        self.pstate = pstate
        self.name = name
        self.id = id

    @property
    def is_available(self):
        return True if self.state == Resource.State.IDLE else False

    def set_state(self, state):
        assert isinstance(state, Resource.State)
        self.state = state

    def set_pstate(self, pstate):
        assert isinstance(pstate, Resource.PowerState)
        self.pstate = pstate


class ResourceManager:
    def __init__(self, resources):
        assert isinstance(resources, dict)
        self.resources = resources

    @staticmethod
    def from_json(data):
        resources = {}
        for res in data:
            resources[res['id']] = Resource(res['id'],
                                            Resource.State[res['state'].upper()],
                                            Resource.PowerState.NORMAL,
                                            res['name'])

        return ResourceManager(resources)

    @property
    def nb_res(self):
        return len(self.resources)

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


class JobManager():
    def __init__(self):
        self.jobs_queue = deque()
        self.jobs_waiting = deque()
        self.jobs_finished = dict()
        self.jobs_running = dict()

    def on_job_waiting(self):
        job = self.jobs_queue.popleft()
        self.jobs_waiting.append(job)

    def lookup(self):
        if self.has_jobs_in_queue():
            return self.jobs_queue[0]
        return None

    def enqueue_jobs_waiting(self, time):
        while self.has_jobs_waiting():
            job = self.jobs_waiting.popleft()
            job.waiting_time = time - job.submit_time
            self.jobs_queue.append(job)

    def on_job_scheduling(self, allocation):
        job = self.jobs_queue.popleft()
        job.allocation = allocation
        job.state = Job.State.RUNNING
        self.jobs_running[job.id] = job

    def has_jobs_in_queue(self):
        return len(self.jobs_queue) > 0

    def has_jobs_waiting(self):
        return len(self.jobs_waiting) > 0

    def on_job_completed(self, timestamp, data):
        job = self.jobs_running.pop(data['job_id'])
        job.finish_time = timestamp
        job.waiting_time = round(data["job_waiting_time"], 4)
        job.turnaround_time = round(data["job_turnaround_time"], 4)
        job.runtime = round(data["job_runtime"], 4)
        job.consumed_energy = round(data["job_consumed_energy"], 4)
        job.return_code = round(data["return_code"], 4)

        try:
            job.state = Job.State[data["job_state"]]
        except KeyError:
            job.state = Job.State.UNKNOWN

        self.jobs_finished[job.id] = job
        return job

    def on_job_submitted(self, data):
        job = Job.from_json(data)
        job.state = Job.State.SUBMITTED
        self.jobs_queue.append(job)


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
        self.finish_time = -1  # will be set on completion by batsim
        self.waiting_time = -1  # will be set on completion by batsim
        self.turnaround_time = -1  # will be set on completion by batsim
        self.runtime = -1  # will be set on completion by batsim
        self.consumed_energy = -1  # will be set on completion by batsim
        self.state = Job.State.UNKNOWN
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
                self.state,
                self.return_code, self.allocation))

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
