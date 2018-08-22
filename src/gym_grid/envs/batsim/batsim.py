import shlex
import logging
import sys
import os
import json
from xml.dom import minidom
from enum import Enum
import subprocess
from batsim.batsim import BatsimScheduler, NetworkHandler
from copy import deepcopy
from procset import ProcSet
from collections import defaultdict, deque


class SchedulerManager(BatsimScheduler):
    def get_job_info(self):
        if len(self.jobs_waiting) == 0:
            return None
        else:
            return self.jobs_waiting[0]

    def get_resources_info(self):
        return [ResourceState.to_value(s['state']) for _, s in self.bs.resources.items()]

    def schedule_job(self, res):
        def handle_void():
            self.bs.resubmit_job(job)

        def handle_allocation():
            self._validate_allocation(job, res)
            self.bs.start_job(job, res)
            self.bs.set_resource_state(res, ResourceState.computing)
            self.jobs_running[job.id] = res

        assert self.bs.running_simulation, "Simulation is not running."
        job = self._get_job()
        assert job is not None, "Could not find job {}".format(id)

        if len(res) == 0:
            handle_void()
        else:
            handle_allocation()

        if len(self.jobs_waiting) != 0:
            return

        self.bs._send_bat_events()
        self.bs.wait_until_next_event()

    def onJobSubmission(self, job):
        self.bs.logger.info("[SchedulerManager] Job Submitted: {} - Jobs in Queue: {}".format(
            job.id, " ".join(j.id for j in self.jobs_waiting)))
        self.jobs_waiting.append(job)
        self.bs._block_event = True

    def onJobCompletion(self, job):
        res = self.jobs_running.pop(job.id)
        assert res is not None, "Could not find job's resource"
        self.nb_jobs_completed += 1
        self.bs.set_resource_state(res, ResourceState.idle)
        # self.bs.acknowledge()
        # self.bs.request_consumed_energy()

    def onMachinePStateChanged(self, resources, state):
        state_name = ResourceState.to_name(state)
        for res in resources:
            self.bs.resources[res]['state'] = state_name

        self.bs.acknowledge()

    def onSimulationBegins(self):
        self.jobs_running = defaultdict(lambda: None)
        self.jobs_finished = defaultdict(lambda: None)
        self.jobs_waiting = deque()
        self.nb_jobs_completed = 0
        self.bs.acknowledge()

    def onSimulationEnds(self):
        self.bs.acknowledge()

    def onDeadlock(self):
        raise ValueError("Batsim is not responding (maybe deadlocked)")

    def onReportEnergyConsumed(self, consumed_energy):
        raise NotImplementedError()

    def onAnswerProcessorTemperatureAll(self, proc_temperature_all):
        raise NotImplementedError()

    def onAnswerAirTemperatureAll(self, air_temperature_all):
        raise NotImplementedError()

    def onBeforeEvents(self):
        pass

    def onNoMoreEvents(self):
        pass

    def onAllJobsCompleted(self):
        if self.bs.dynamic_job_submission_enabled:
            self.bs.notify_submission_finished()
            self.bs.dynamic_job_submission_enabled = False

    def _get_job(self):
        if len(self.jobs_waiting) == 0:
            return None
        else:
            return self.jobs_waiting.popleft()

    def _validate_allocation(self, job, res):
        if job.requested_resources != len(res):
            raise InsufficientResourcesError(
                "Job requested {} resources while {} resources were selected.".format(job.requested_resources, len(res)))

        if not self._check_resources_availability(res):
            raise UnavailableResourcesError(
                "Cannot allocate unavailable resources for job {}.".format(job.id))

    def _check_resources_availability(self, res):
        for r in res:
            try:
                resource = self.bs.resources[r]
            except KeyError as e:
                raise KeyError("Could not find resource: {}".format(e))
            if not resource['state'] == ResourceState.idle.name:
                return False

        return True
    # def onReportEnergyConsumed(self, consumed_energy):
        # assert self.__last_job_finished != -1, "No job to report energy consumption"
        # job_stats = self.jobs_finished[self.__last_job_finished]
        # assert job_stats is not None, "No job to report energy consumption"
        # job_stats[0] = consumed_energy


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


class BatsimHandler:
    WORKLOAD_JOB_SEPARATOR = "!"
    ATTEMPT_JOB_SEPARATOR = "#"
    WORKLOAD_JOB_SEPARATOR_REPLACEMENT = "%"

    PLATFORM = "platform.xml"
    WORKLOAD = "workload.json"

    def __init__(self, socket_endpoint="tcp://*:28000", platform="platform.xml", workload="workload.json", verbose="information"):
        fullpath = os.path.join(os.path.dirname(__file__), "files")
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self._platform = os.path.join(fullpath, platform)
        self._workload = os.path.join(fullpath, workload)
        self._config = os.path.join(fullpath, 'config.json')
        self._verbose = verbose
        self._simulator_process = None
        FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.CRITICAL)
        self.running_simulation = False
        self.network = NetworkHandler(socket_endpoint)
        sys.setrecursionlimit(10000)
        self.nb_resources = self._count_resources(self._platform)
        self.nb_jobs = self._count_jobs(self._workload)
        self.manager = SchedulerManager(None)
        self.manager.bs = self
        self.manager.onAfterBatsimInit()
        self._initialize_vars()

    def _count_resources(self, platform):
        return len(minidom.parse(platform).getElementsByTagName('host')) - 1

    def _count_jobs(self, workload):
        with open(workload, 'r') as f:
            data = json.load(f)
            nb_jobs = len(data['jobs'])
        return nb_jobs

    def close(self):
        if self._simulator_process is not None:
            self._simulator_process.kill()
            self._simulator_process.wait()
            self._simulator_process = None
        self.network.close()
        self.running_simulation = False

    def start(self):
        self._initialize_vars()
        self._simulator_process = self._start_simulator(True)
        self.network.bind()
        self.wait_until_next_event()

        if not self.running_simulation:
            raise ConnectionError(
                "An error ocurred during simulator starting.")

    def _start_simulator(self, dev=False):
        if dev:
            path = "/home/d2/Projects/batsim/build/batsim"
        else:
            path = "batsim"
        cmd = path + " -p %s -w %s -v %s -E --config-file %s" % (
            self._platform, self._workload, self._verbose, self._config)
        # return Popen(shlex.split(cmd))
        return subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)

    def time(self):
        return self._current_time

    def consume_time(self, t):
        self._current_time += float(t)
        return self._current_time

    def wake_me_up_at(self, time):
        self._events_to_send.append(
            {"timestamp": self.time(),
             "type": "CALL_ME_LATER",
             "data": {"timestamp": time}})

    def notify_submission_finished(self):
        self._events_to_send.append({
            "timestamp": self.time(),
            "type": "NOTIFY",
            "data": {
                    "type": "submission_finished",
            }
        })

    def notify_submission_continue(self):
        self._events_to_send.append({
            "timestamp": self.time(),
            "type": "NOTIFY",
            "data": {
                    "type": "continue_submission",
            }
        })

    def send_message_to_job(self, job, message):
        self._events_to_send.append({
            "timestamp": self.time(),
            "type": "TO_JOB_MSG",
            "data": {
                    "job_id": job.id,
                    "msg": message,
            }
        })

    def start_jobs_continuous(self, allocs):
        """
        allocs should have the following format:
        [ (job, (first res, last res)), (job, (first res, last res)), ...]
        """

        if len(allocs) == 0:
            return

        for (job, (first_res, last_res)) in allocs:
            self._events_to_send.append({
                "timestamp": self.time(),
                "type": "EXECUTE_JOB",
                "data": {
                        "job_id": job.id,
                        "alloc": "{}-{}".format(first_res, last_res)
                }
            }
            )
            self.nb_jobs_scheduled += 1

    def start_job(self, job, res):
        """ args:res: is list of int (resources ids) """
        self._events_to_send.append({
            "timestamp": self.time(),
            "type": "EXECUTE_JOB",
            "data": {
                    "job_id": job.id,
                    "alloc": str(ProcSet(*res))
            }
        })
        self.nb_jobs_scheduled += 1

    def execute_jobs(self, jobs, io_jobs=None):
        """ args:jobs: list of jobs to execute (job.allocation MUST be set) """

        for job in jobs:
            assert job.allocation is not None

            message = {
                "timestamp": self.time(),
                "type": "EXECUTE_JOB",
                "data": {
                        "job_id": job.id,
                        "alloc": str(job.allocation)
                }
            }
            if io_jobs is not None and job.id in io_jobs:
                message["data"]["additional_io_job"] = io_jobs[job.id]

            self._events_to_send.append(message)
            self.nb_jobs_scheduled += 1

    def reject_jobs(self, jobs):
        """Reject the given jobs."""
        assert len(jobs) > 0, "The list of jobs to reject is empty"
        for job in jobs:
            self._events_to_send.append({
                "timestamp": self.time(),
                "type": "REJECT_JOB",
                "data": {
                        "job_id": job.id,
                }
            })
            self.nb_jobs_rejected += 1

    def change_job_state(self, job, state):
        """Change the state of a job."""
        self._events_to_send.append({
            "timestamp": self.time(),
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
        self._events_to_send.append({
            "timestamp": self.time(),
            "type": "KILL_JOB",
            "data": {
                    "job_ids": [job.id for job in jobs],
            }
        })

    def submit_profiles(self, workload_name, profiles):
        for profile_name, profile in profiles.items():
            msg = {
                "timestamp": self.time(),
                "type": "SUBMIT_PROFILE",
                "data": {
                    "workload_name": workload_name,
                    "profile_name": profile_name,
                    "profile": profile,
                }
            }
            self._events_to_send.append(msg)

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
            "subtime": self.time() if subtime is None else subtime,
        }
        msg = {
            "timestamp": self.time(),
            "type": "SUBMIT_JOB",
            "data": {
                "job_id": id,
                "job": job_dict,
            }
        }
        if profile is not None:
            assert isinstance(profile, dict)
            msg["data"]["profile"] = profile

        self._events_to_send.append(msg)
        self.nb_jobs_submitted += 1

        # Create the job here
        self.jobs[id] = Job.from_json_dict(job_dict, profile_dict=profile)
        self.jobs[id].job_state = Job.State.SUBMITTED

        return id

    def set_resource_state(self, resources, state):
        self._events_to_send.append({
            "timestamp": self.time(),
            "type": "SET_RESOURCE_STATE",
            "data": {
                "resources": " ".join([str(r) for r in resources]),
                "state": str(state.value)
            }
        })

    def get_job(self, event):
        json_dict = event["data"]["job"]
        try:
            profile_dict = event["data"]["profile"]
        except KeyError:
            profile_dict = {}
        job = Job.from_json_dict(json_dict, profile_dict)
        return job

    def request_consumed_energy(self):  # TODO CHANGE NAME
        self._events_to_send.append(
            {
                "timestamp": self.time(),
                "type": "QUERY",
                "data": {
                    "requests": {"consumed_energy": {}}
                }
            }
        )

    def request_air_temperature_all(self):
        self._events_to_send.append(
            {
                "timestamp": self.time(),
                "type": "QUERY",
                "data": {
                    "requests": {"air_temperature_all": {}}
                }
            }
        )

    def request_processor_temperature_all(self):
        self._events_to_send.append(
            {
                "timestamp": self.time(),
                "type": "QUERY",
                "data": {
                    "requests": {"processor_temperature_all": {}}
                }
            }
        )

    def notify_resources_added(self, resources):
        self._events_to_send.append(
            {
                "timestamp": self.time(),
                "type": "RESOURCES_ADDED",
                "data": {
                    "resources": resources
                }
            }
        )

    def notify_resources_removed(self, resources):
        self._events_to_send.append(
            {
                "timestamp": self.time(),
                "type": "RESOURCES_REMOVED",
                "data": {
                    "resources": resources
                }
            }
        )

    def set_job_metadata(self, job_id, metadata):
        # Consume some time to be sure that the job was created before the
        # metadata is set

        if self._events_to_send == None:
            self._events_to_send = []
        self.jobs[job_id].metadata = metadata
        self._events_to_send.append(
            {
                "timestamp": self.time(),
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

        self.reject_jobs([job])

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

    def wait_until_next_event(self):
        jobs_received = self.nb_jobs_received
        self._read_bat_msg()
        while self.nb_jobs_received == jobs_received and self.running_simulation and len(self.manager.jobs_waiting) == 0:
            self._read_bat_msg()

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
        self._ack = False
        self._block_event = False
        self._events_to_send = []
        self.jobs = dict()

    def _send_bat_events(self):
        if (len(self._events_to_send) == 0 and not self._ack) or self._block_event:
            self._block_event = False
            return

        self._ack = False

        if len(self._events_to_send) > 0:
            self._events_to_send = sorted(
                self._events_to_send, key=lambda event: event['timestamp'])

        new_msg = {
            "now": self._current_time,
            "events": self._events_to_send
        }
        self.network.send(new_msg)
        self.logger.info("Message Sent to Batsim: {}".format(new_msg))
        self._events_to_send = []

    def _read_bat_msg(self):
        msg = None
        while msg is None:
            msg = self.network.recv(blocking=not self.running_simulation)
            if msg is None:
                self.manager.onDeadlock()
                continue
        self.logger.info("Message Received from Batsim: {}".format(msg))

        self._current_time = msg["now"]

        if "air_temperatures" in msg:
            self.air_temperatures = msg["air_temperatures"]

        self.manager.onBeforeEvents()

        for event in msg["events"]:
            event_type = event["type"]
            event_data = event.get("data", {})
            if event_type == "SIMULATION_BEGINS":
                assert not self.running_simulation, "A simulation is already running (is more than one instance of Batsim active?!)"
                assert event_data["nb_resources"] == self.nb_resources, "Batsim platform and Simulator platform does not match."
                self.running_simulation = True
                self.nb_compute_resources = event_data["nb_compute_resources"]
                self.nb_storage_resources = event_data["nb_storage_resources"]
                self.machines = {"compute": event_data["compute_resources"],
                                 "storage": event_data["storage_resources"]}
                self.batconf = event_data["config"]
                self.time_sharing = event_data["allow_time_sharing"]
                self.dynamic_job_submission_enabled = self.batconf[
                    "job_submission"]["from_scheduler"]["enabled"]
                self.resources = {
                    res["id"]: res for res in event_data["compute_resources"]}
                self.storage_resources = {
                    res["id"]: res for res in event_data["storage_resources"]}
                self.profiles = event_data["profiles"]
                self.workloads = event_data["workloads"]
                self.manager.onSimulationBegins()
            elif event_type == "SIMULATION_ENDS":
                assert self.running_simulation, "No simulation is currently running"
                self.running_simulation = False
                self.logger.info("All jobs have been submitted and completed!")
                self.manager.onSimulationEnds()
            elif event_type == "JOB_SUBMITTED":
                # Received WORKLOAD_NAME!JOB_ID
                job_id = event_data["job_id"]
                job = self.get_job(event)
                job.job_state = Job.State.SUBMITTED

                # don't override dynamic job
                if job_id not in self.jobs:
                    self.jobs[job_id] = job

                self.manager.onJobSubmission(job)
                self.nb_jobs_received += 1
            elif event_type == "JOB_COMPLETED":
                job_id = event_data["job_id"]
                j = self.jobs[job_id]
                j.finish_time = event["timestamp"]

                try:
                    j.job_state = Job.State[event["data"]["job_state"]]
                except KeyError:
                    j.job_state = Job.State.UNKNOWN
                j.return_code = event["data"]["return_code"]

                self.manager.onJobCompletion(j)

                if j.job_state == Job.State.COMPLETED_WALLTIME_REACHED:
                    self.nb_jobs_timeout += 1
                elif j.job_state == Job.State.COMPLETED_FAILED:
                    self.nb_jobs_failed += 1
                elif j.job_state == Job.State.COMPLETED_SUCCESSFULLY:
                    self.nb_jobs_successful += 1
                elif j.job_state == Job.State.COMPLETED_KILLED:
                    self.nb_jobs_killed += 1
                self.nb_jobs_completed += 1
            elif event_type == "RESOURCE_STATE_CHANGED":
                res = list(map(int, event_data["resources"].split(" ")))
                self.manager.onMachinePStateChanged(
                    res, int(event_data["state"]))
            elif event_type == "ANSWER":
                if "consumed_energy" in event_data:
                    self.manager.onReportEnergyConsumed(
                        event_data["consumed_energy"])
                elif "processor_temperature_all" in event_data:
                    self.manager.onAnswerProcessorTemperatureAll(
                        event_data["processor_temperature_all"])
                elif "air_temperature_all" in event_data:
                    self.manager.onAnswerAirTemperatureAll(
                        event_data["air_temperature_all"])
            else:
                raise Exception("Unknown event type {}".format(event_type))

        self.manager.onNoMoreEvents()

        if self.nb_jobs_completed == self.nb_jobs:
            self.manager.onAllJobsCompleted()

        self._send_bat_events()

        return not self.running_simulation


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


class InsufficientResourcesError(Exception):
    pass


class UnavailableResourcesError(Exception):
    pass
