import os
import json
import zmq
import time
import itertools
import shutil
import pandas as pd
import subprocess
import numpy as np
from enum import Enum
from copy import deepcopy
from collections import deque
from .resource import Resource, ResourceManager
from .scheduler import SchedulerManager
from .network import BatsimProtocolHandler


class BatsimHandler:
    PLATFORM = "platform_hg_10.xml"
    WORKLOAD = "workloads"
    CONFIG = "config.json"
    SOCKET_ENDPOINT = "tcp://*:28000"
    OUTPUT_DIR = "results/batsim"

    def __init__(self, queue_slots, time_window, verbose='quiet'):
        fullpath = os.path.join(os.path.dirname(__file__), "files")
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        if os.path.exists(BatsimHandler.OUTPUT_DIR):
            shutil.rmtree(BatsimHandler.OUTPUT_DIR, ignore_errors=True)

        os.makedirs(BatsimHandler.OUTPUT_DIR)
        self._config = os.path.join(fullpath, BatsimHandler.CONFIG)
        self._platform = os.path.join(fullpath, BatsimHandler.PLATFORM)
        workloads_path = os.path.join(fullpath, BatsimHandler.WORKLOAD)
        self._workloads = [workloads_path + "/" +
                           w for w in os.listdir(workloads_path) if w.endswith('.json')]
        self._workload = None
        self._simulator_process = None
        self._verbose = verbose
        self.time_window = time_window
        self.running_simulation = False
        self.nb_simulation = 0
        self._workload_idx = 0
        self.queue_slots = queue_slots
        self.protocol_manager = BatsimProtocolHandler(
            BatsimHandler.SOCKET_ENDPOINT)
        self.resource_manager = ResourceManager.from_xml(self._platform)
        self.sched_manager = SchedulerManager(self.nb_resources, time_window)
        self._reset()

    def _load_workload(self):
        if len(self._workloads) == self._workload_idx:
            np.random.shuffle(self._workloads)
            self._workload_idx = 0

        self._workload = self._workloads[self._workload_idx]
        self._workload_idx += 1

    def _get_max_walltime(self, workload):
        with open(workload) as f:
            data = json.load(f)
            max_walltime = max(data['jobs'], key=(
                lambda job: job['walltime']))['walltime']

        return max_walltime

    @property
    def nb_jobs_in_queue(self):
        return self.sched_manager.nb_jobs_in_queue

    @property
    def max_resource_speed(self):
        return self.resource_manager.max_resource_speed

    @property
    def max_resource_energy_cost(self):
        return self.resource_manager.max_resource_energy_cost

    @property
    def nb_jobs_running(self):
        return self.sched_manager.nb_jobs_running

    @property
    def nb_jobs_submitted(self):
        return self.sched_manager.nb_jobs_submitted

    @property
    def nb_jobs_completed(self):
        return self.sched_manager.nb_jobs_completed

    @property
    def nb_jobs_waiting(self):
        return self.sched_manager.nb_jobs_waiting

    @property
    def nb_resources(self):
        return self.resource_manager.nb_resources

    @property
    def current_state(self):
        resources_states = []
        resources = self.resource_manager.get_resources()
        for i in range(self.nb_resources):
            resource_state = dict(
                resource=resources[i],
                queue=self.sched_manager.gantt[i]
            )
            resources_states.append(resource_state)

        jobs = list(self.sched_manager.jobs_queue)
        queue_sz = len(jobs)
        job_queue = dict(
            jobs=jobs[0:min(queue_sz, self.queue_slots)],
            nb_jobs_in_queue=self.sched_manager.nb_jobs_in_queue,
            nb_jobs_waiting=self.sched_manager.nb_jobs_waiting
        )

        state = dict(gantt=deepcopy(resources_states),
                     job_queue=deepcopy(job_queue))
        return state

    @property
    def current_time(self):
        return self.protocol_manager.current_time

    def lookup(self, idx):
        return deepcopy(self.sched_manager.lookup(idx))

    def estimate_energy_consumption(self, res_ids):
        return self.resource_manager.estimate_energy_consumption(res_ids)

    def close(self):
        self.protocol_manager.close()
        self.running_simulation = False
        if self._simulator_process is not None:
            self._simulator_process.terminate()
            self._simulator_process.wait()
            self._simulator_process = None

    def start(self):
        self._reset()
        self._load_workload()
        self.nb_simulation += 1
        self._simulator_process = self._start_simulator()
        self.protocol_manager.start()
        self._wait_state_change()
        if not self.running_simulation:
            raise ConnectionError(
                "An error ocurred during simulator starting.")

    def schedule(self, job):
        assert self.running_simulation, "Simulation is not running."
        assert job is not None, "Job cannot be null."

        if job == -1 and not self._alarm_is_set:  # Handle VOID Action
            self.protocol_manager.wake_me_up_at(self.current_time + 1)
            self._alarm_is_set = True
        else:
            resources = self.sched_manager.allocate_job(job)
            if resources == None:
                print("OPA")
            self.resource_manager.on_job_allocated(resources)
            if self.sched_manager.is_ready():
                return

        self._start_ready_jobs()

        self._wait_state_change()

    def _wait_state_change(self):
        self._update_state()
        while self.running_simulation and (self._alarm_is_set or not self.sched_manager.is_ready()):
            self._update_state()

    def _start_ready_jobs(self):
        ready_jobs = self.sched_manager.get_ready_jobs()
        for job in ready_jobs:
            self.protocol_manager.start_job(job.id,  job.allocation)
            self.resource_manager.on_job_allocated(job.allocation)
            self.sched_manager.on_job_scheduled(job, self.current_time)

    def _start_simulator(self):
        output_path = BatsimHandler.OUTPUT_DIR + "/" + str(self.nb_simulation)
        cmd = "batsim -p {} -w {} -v {} -E --config-file {} -e {}".format(self._platform,
                                                                          self._workload,
                                                                          self._verbose,
                                                                          self._config,
                                                                          output_path)

        return subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=False)

    def _reset(self):
        self.metrics = {}
        self._alarm_is_set = False
        self.sched_manager.reset()

    def _handle_resource_pstate_changed(self, timestamp, data):
        resources = self._get_resources_from_json(data["resources"])
        self.sched_manager.on_resource_pstate_changed(timestamp)
        self.resource_manager.on_resource_pstate_changed(
            resources, Resource.PowerState[int(data["state"])])

    def _get_resources_from_json(self, data):
        resources = []
        for alloc in data.split(" "):
            nodes = alloc.split("-")
            if len(nodes) == 2:
                resources.extend(range(int(nodes[0]), int(nodes[1]) + 1))
            else:
                resources.append(int(nodes[0]))
        return resources

    def _handle_job_completed(self, timestamp, data):
        resources = self._get_resources_from_json(data["alloc"])
        self.sched_manager.on_job_completed(timestamp, resources, data)
        self.resource_manager.on_job_completed(resources)
        self._start_ready_jobs()

    def _handle_job_submitted(self, timestamp, data):
        if data['job']['res'] > self.resource_manager.nb_resources:
            self.protocol_manager.reject_job(data['job_id'])
        else:
            self.sched_manager.on_job_submitted(timestamp, data['job'])

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
        self.protocol_manager.send_events()
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

    def _handle_requested_call(self, timestamp):
        self._alarm_is_set = False
        self.sched_manager.on_delay_expired(timestamp)

    def _handle_batsim_events(self, event):
        if event.type == "SIMULATION_BEGINS":
            assert not self.running_simulation, "A simulation is already running (is more than one instance of Batsim active?!)"
            self._handle_simulation_begins(event.data)
        elif event.type == "SIMULATION_ENDS":
            assert self.running_simulation, "No simulation is currently running"
            self._handle_simulation_ends(event.data)
        elif event.type == "JOB_SUBMITTED":
            self._handle_job_submitted(event.timestamp, event.data)
        elif event.type == "JOB_COMPLETED":
            self._handle_job_completed(event.timestamp, event.data)
        elif event.type == "RESOURCE_STATE_CHANGED":
            self._handle_resource_pstate_changed(event.timestamp, event.data)
        elif event.type == "REQUESTED_CALL":
            self._handle_requested_call(event.timestamp)
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

    def _update_state(self):
        self.protocol_manager.send_events()

        events = self.protocol_manager.read_events(
            blocking=not self.running_simulation)
        for event in events:
            self._handle_batsim_events(event)

        self.sched_manager.update_jobs_progress(self.current_time)

        # Remember to always ack
        if self.running_simulation:
            self.protocol_manager.acknowledge()
