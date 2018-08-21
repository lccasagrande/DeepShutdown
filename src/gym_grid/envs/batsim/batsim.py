import shlex
import logging
import sys
import os
from xml.dom import minidom
from enum import Enum
from subprocess import Popen
from batsim.batsim import Job, BatsimScheduler, NetworkHandler
from copy import deepcopy
from procset import ProcSet
from collections import defaultdict, deque
# import json
# if __name__ == "__main__":
#    with open("src/gym_grid/envs/batsim/workloads/workload.json", 'r+') as f:
#        data = json.load(f)
#
#        for job in data['jobs']:
#            job["res"] = 1
#
#    with open("src/gym_grid/envs/batsim/workloads/workload2.json", 'w') as f:
#        json.dump(data, f)


class SchedulerManager(BatsimScheduler):
    # {sleeping, idle, computing, switching_on, switching_off}
    def onAfterBatsimInit(self):
        self.jobs_running = defaultdict(lambda: None)
        self.jobs_finished = defaultdict(lambda: None)
        self.jobs_waiting = deque()

    def get_job_info(self):
        if len(self.jobs_waiting) == 0:
            return None
        else:
            return self.jobs_waiting[0]

    def get_resources_info(self):
        return [ResourceState.to_value(s['state']) for _, s in self.bs.resources.items()]

    def schedule_job(self, res):
        job = self._get_job()
        assert job is not None, "Could not find job {}".format(id)

        if len(res) == 0:
            self.jobs_waiting.append(job)
            self.bs.acknowledge()
        else:
            if not self._check_resources_availability(res):
                return False
            if job.requested_resources != len(res):
                return False
            self.bs.start_jobs([job], {job.id: res})
            self.bs.set_resource_state(res, ResourceState.computing)
            self.jobs_running[job.id] = res

        self.bs._send_bat_events()
        self.bs.wait_until_next_event()
        return True

    def onJobSubmission(self, job):
        self.bs.logger.info("[SchedulerManager] Job Submitted: {} - Jobs in Queue: {}".format(
            job.id, " ".join(j.id for j in self.jobs_waiting)))
        self.jobs_waiting.append(job)

    def onJobCompletion(self, job):
        res = self.jobs_running.pop(job.id)
        assert res is not None, "Could not find job's resource"
        self.bs.set_resource_state(res, ResourceState.idle)
        self.bs.acknowledge()
        # self.bs.request_consumed_energy()

    def onMachinePStateChanged(self, nodeid, pstate):
        raise NotImplementedError()

    def onSimulationBegins(self):
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

    def _get_job(self):
        if len(self.jobs_waiting) == 0:
            return None
        else:
            return self.jobs_waiting.popleft()

    def _check_resources_availability(self, res):
        for r in res:
            try:
                resource = self.bs.resources[r]
            except KeyError as e:
                raise KeyError("Could find resource: {}".format(e))
            if (resource['state'] == ResourceState.computing.name) or \
                (resource['state'] == ResourceState.switching_off.name) or \
                    (resource['state'] == ResourceState.switching_on.name):
                return False

        return True
    # def onReportEnergyConsumed(self, consumed_energy):
        #assert self.__last_job_finished != -1, "No job to report energy consumption"
        #job_stats = self.jobs_finished[self.__last_job_finished]
        #assert job_stats is not None, "No job to report energy consumption"
        #job_stats[0] = consumed_energy


class ResourceState(Enum):
    sleeping = 0
    idle = 1
    computing = 2
    switching_on = 3
    switching_off = 4

    @staticmethod
    def to_value(name):
        if name == ResourceState.sleeping.name:
            return ResourceState.sleeping.value
        elif name == ResourceState.idle.name:
            return ResourceState.idle.value
        elif name == ResourceState.computing.name:
            return ResourceState.computing.value
        elif name == ResourceState.switching_on.name:
            return ResourceState.switching_on.value
        elif name == ResourceState.switching_off.name:
            return ResourceState.switching_off.value
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
        elif value == ResourceState.switching_on.value:
            return ResourceState.switching_on.name
        elif value == ResourceState.switching_off.value:
            return ResourceState.switching_off.name
        else:
            raise Exception("Unknown resource state value {}".format(value))


class BatsimHandler:

    WORKLOAD_JOB_SEPARATOR = "!"
    ATTEMPT_JOB_SEPARATOR = "#"
    WORKLOAD_JOB_SEPARATOR_REPLACEMENT = "%"

    def __init__(self,  socket_endpoint, platform, workload, verbose="information"):
        fullpath = os.path.join(os.path.dirname(__file__), "files")
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self._platform = os.path.join(fullpath, platform)
        self._workload = os.path.join(fullpath, workload)
        self._verbose = verbose
        self._simulator_process = None
        FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.running_simulation = False
        self.network = NetworkHandler(socket_endpoint)
        self.jobs = dict()
        sys.setrecursionlimit(10000)
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
        self._events_to_send = []
        self.nb_resources = len(minidom.parse(
            self._platform).getElementsByTagName('host')) - 1
        self.manager = SchedulerManager(None)
        self.manager.bs = self
        self.manager.onAfterBatsimInit()

    def close(self):
        if self._simulator_process is not None:
            self._simulator_process.kill()
            self._simulator_process.wait()
            self._simulator_process = None
        self.network.close()
        self.running_simulation = False

    def start(self):
        self.__simulator_process = self._start_simulator()
        self.network.bind()
        self.wait_until_next_event()

        if not self.running_simulation:
            raise ConnectionError(
                "An error ocurred during simulator starting.")

    def _start_simulator(self):
        cmd = "batsim -p %s -w %s -v %s -E" % (
            self._platform, self._workload, self._verbose)
        return Popen(shlex.split(cmd))

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

    def start_jobs(self, jobs, res):
        """ args:res: is list of int (resources ids) """
        for job in jobs:
            self._events_to_send.append({
                "timestamp": self.time(),
                "type": "EXECUTE_JOB",
                "data": {
                        "job_id": job.id,
                        "alloc": str(ProcSet(*res[job.id]))
                }
            }
            )
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

        if subtime is None:
            subtime = self.time()
        job_dict = {
            "profile": profile_name,
            "id": id,
            "res": res,
            "walltime": walltime,
            "subtime": subtime,
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
        """ args:resources: is a list of resource numbers.
            args:state: is a state identifier configured in the platform specification.
        """

        for res in resources:
            self.resources[res]['state'] = state.name
        # self._events_to_send.append({
        #    "timestamp": self.time(),
        #    "type": "SET_RESOURCE_STATE",
        #    "data": {
        #            "resources": " ".join([str(r) for r in resources]),
        #            "state": str(state.value)
        #    }
        # })

    def start_jobs_interval_set_strings(self, jobs, res):
        """ args:res: is a jobID:interval_set_string dict """
        for job in jobs:
            self._events_to_send.append({
                "timestamp": self.time(),
                "type": "EXECUTE_JOB",
                "data": {
                        "job_id": job.id,
                        "alloc": res[job.id]
                }
            }
            )
            self.nb_jobs_scheduled += 1

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

    def resubmit_job(self, job):
        """
        The given job is resubmited but in a dynamic workload. The name of this
        workload is "resubmit=N" where N is the number of resubmission.
        The job metadata is fill with a dict that contains the original job
        full id in "parent_job" and the number of resubmission in "nb_resumit".

        Warning: The profile_dict of the given job must be filled
        """

        if job.metadata is None:
            metadata = {"parent_job": job.id, "nb_resubmit": 1}
        else:
            metadata = deepcopy(job.metadata)
            if "parent_job" not in metadata:
                metadata["parent_job"] = job.id
            metadata["nb_resubmit"] = metadata["nb_resubmit"] + 1

        # Keep the current workload and add a resubmit number
        splitted_id = job.id.split(BatsimHandler.ATTEMPT_JOB_SEPARATOR)
        if len(splitted_id) == 0:
            new_job_name = job.id
        else:
            # This job as already an attempt number
            new_job_name = splitted_id[0]
        new_job_name = new_job_name + BatsimHandler.ATTEMPT_JOB_SEPARATOR + \
            str(metadata["nb_resubmit"])

        new_job_id = self.submit_job(
            new_job_name,
            job.requested_resources,
            job.requested_time,
            job.profile,
            profile=job.profile_dict)

        # log in job metadata parent job and nb resubmit
        self.set_job_metadata(new_job_id, metadata)

    def acknowledge(self):
        self._ack = True

    def wait_until_next_event(self):
        jobs_received = self.nb_jobs_received
        jobs_completed = self.nb_jobs_completed
        self._read_bat_msg()
        while self.nb_jobs_received == jobs_received and self.running_simulation and \
                not (self.nb_jobs_completed != jobs_completed and len(self.manager.jobs_waiting) != 0):
            self._read_bat_msg()

    def _send_bat_events(self):
        if len(self._events_to_send) == 0 and not self._ack:
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
                self.running_simulation = True
                self.nb_resources = event_data["nb_resources"]
                self.batconf = event_data["config"]
                self.time_sharing = event_data["allow_time_sharing"]
                self.dynamic_job_submission_enabled = self.batconf[
                    "job_submission"]["from_scheduler"]["enabled"]

                if self.dynamic_job_submission_enabled:
                    self.logger.warning(
                        "Dynamic submission of jobs is ENABLED. The scheduler must send a NOTIFY event of type 'submission_finished' to let Batsim end the simulation.")

                # Retro compatibility for old Batsim API > 1.0 < 3.0
                if "resources_data" in event_data:
                    res_key = "resources_data"
                else:
                    res_key = "compute_resources"

                self.resources = {
                    res["id"]: res for res in event_data[res_key]}

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
                if j.job_state == "COMPLETED_WALLTIME_REACHED":
                    self.nb_jobs_timeout += 1
                elif j.job_state == Job.State.COMPLETED_FAILED:
                    self.nb_jobs_failed += 1
                elif j.job_state == Job.State.COMPLETED_SUCCESSFULLY:
                    self.nb_jobs_successful += 1
                elif j.job_state == Job.State.COMPLETED_KILLED:
                    self.nb_jobs_killed += 1
                self.nb_jobs_completed += 1
            elif event_type == "RESOURCE_STATE_CHANGED":
                raise NotImplementedError()
                #self.manager.onMachinePStateChanged(resources, state)
            elif event_type == "ANSWER":
                if "consumed_energy" in event_data:
                    consumed_energy = event_data["consumed_energy"]
                    self.manager.onReportEnergyConsumed(consumed_energy)
                elif "processor_temperature_all" in event_data:
                    proc_temperature_all = event_data["processor_temperature_all"]
                    self.manager.onAnswerProcessorTemperatureAll(
                        proc_temperature_all)
                elif "air_temperature_all" in event_data:
                    air_temperature_all = event_data["air_temperature_all"]
                    self.manager.onAnswerAirTemperatureAll(
                        air_temperature_all)
            else:
                raise Exception("Unknown event type {}".format(event_type))

        self.manager.onNoMoreEvents()

        self._send_bat_events()

        return not self.running_simulation
