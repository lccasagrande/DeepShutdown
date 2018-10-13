import os
import json
import zmq
import time
import itertools
import shutil
import pandas as pd
import subprocess
import numpy as np
import random
import math
from enum import Enum
from copy import deepcopy
from collections import deque
from .resource import Resource, ResourceManager
from .scheduler import SchedulerManager, Job
from .network import BatsimProtocolHandler


class BatsimHandler:
    PLATFORM = "platform_hg_10.xml"
    CONFIG = "cfg.json"
    WORKLOAD_DIR = "workload"
    OUTPUT_DIR = "results/batsim"

    def __init__(self, job_slots, time_window, backlog_width, verbose='quiet'):
        fullpath = os.path.join(os.path.dirname(__file__), "files")
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self._output_dir = self._make_random_dir(BatsimHandler.OUTPUT_DIR)
        self._config = os.path.join(fullpath, BatsimHandler.CONFIG)
        self._platform = os.path.join(fullpath, BatsimHandler.PLATFORM)
        workloads_path = os.path.join(fullpath, BatsimHandler.WORKLOAD_DIR)
        self._workloads = [workloads_path + "/" +
                           w for w in os.listdir(workloads_path) if w.endswith('.json')]

        self._workload_idx = -1
        self._simulator_process = None
        self.running_simulation = False
        self.nb_simulation = 0
        self._verbose = verbose
        self.time_window = time_window
        self.job_slots = job_slots
        self.protocol_manager = BatsimProtocolHandler()
        self.resource_manager = ResourceManager.from_xml(
            self._platform, time_window)
        self.jobs_manager = SchedulerManager(self.nb_resources, job_slots)
        self.backlog_width = backlog_width
        self.state_shape = (self.time_window, self.nb_resources +
                            (self.nb_resources*self.job_slots) + backlog_width)
        self._reset()

    @property
    def nb_jobs_waiting(self):
        return self.jobs_manager.nb_jobs_in_backlog + self.jobs_manager.job_slots.lenght

    @property
    def nb_jobs_running(self):
        return self.jobs_manager.nb_jobs_running

    @property
    def nb_jobs_submitted(self):
        return self.jobs_manager.nb_jobs_submitted

    @property
    def nb_jobs_completed(self):
        return self.jobs_manager.nb_jobs_completed

    @property
    def nb_resources(self):
        return self.resource_manager.nb_resources

    @property
    def current_time(self):
        return self.protocol_manager.current_time

    def get_state(self, type=''):
        if type == 'image':
            return self._get_image()
        else:
            return self._get_state()

    def lookup(self, idx):
        return deepcopy(self.jobs_manager.lookup(idx))

    def close(self):
        self.protocol_manager.close()
        self.running_simulation = False
        if self._simulator_process is not None:
            self._simulator_process.terminate()
            self._simulator_process.wait()
            self._simulator_process = None

    def start(self):
        assert not self.running_simulation, "A simulation is already running."
        self._reset()
        self._simulator_process = self._start_simulator()
        self.protocol_manager.start()
        self._update_state()
        self._wait_state_change()
        assert self.running_simulation, "An error ocurred during simulator starting."
        self.nb_simulation += 1

    def schedule(self, job_pos):
        assert self.running_simulation, "Simulation is not running."

        if job_pos != -1:  # Try to schedule job
            job = self.jobs_manager.get_job(job_pos)
            self.resource_manager.allocate(job)
            self.jobs_manager.on_job_allocated(job_pos)
        elif not self._alarm_is_set:  # Handle VOID Action
            self.protocol_manager.set_alarm(self.current_time + 1)
            self._alarm_is_set = True

        self._start_ready_jobs()

        self._wait_state_change()

    def _wait_state_change(self):
        while self.running_simulation and (self._alarm_is_set or self.jobs_manager.is_empty):
            self._update_state()

    def _get_ready_jobs(self):
        ready_jobs = []
        jobs = self.resource_manager.get_jobs()
        for job in jobs:
            if job is not None and \
                    job.state != Job.State.RUNNING and \
                    job.time_left_to_start == 0 and \
                    job not in ready_jobs and \
                    self.resource_manager.is_available(job.allocation):
                ready_jobs.append(job)
        return ready_jobs

    def _start_ready_jobs(self):
        jobs = self.resource_manager.get_jobs()
        for job in jobs:
            if job.state != Job.State.RUNNING and \
                    job.time_left_to_start == 0 and \
                    self.resource_manager.is_available(job.allocation):
               #self._wake_up_resources(job.allocation)
                self._start_job(job)

    def _start_job(self, job):
        self.protocol_manager.start_job(job.id,  job.allocation)
        self.resource_manager.start_job(job)
        self.jobs_manager.on_job_started(job.id, self.current_time)

    def _start_simulator(self):
        output_path = self._output_dir + "/" + str(self.nb_simulation)
        cmd = "batsim -s {} -p {} -w {} -v {} -E --config-file {} -e {}".format(self.protocol_manager.socket_endpoint,
                                                                                self._platform,
                                                                                self._get_workload(),
                                                                                self._verbose,
                                                                                self._config,
                                                                                output_path)

        return subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=False)

    def _reset(self):
        self.metrics = {}
        self._alarm_is_set = False
        self.jobs_manager.reset()
        self.protocol_manager.reset()
        self.resource_manager.reset()

    def _handle_requested_call(self, timestamp):
        self._alarm_is_set = False

    def _handle_resource_pstate_changed(self, timestamp, data):
        return

    def _handle_job_completed(self, timestamp, data):
        job = self.jobs_manager.on_job_completed(timestamp, data)
        self.resource_manager.release(job)
        self._start_ready_jobs()

    def _handle_job_submitted(self, timestamp, data):
        if data['job']['res'] > self.resource_manager.nb_resources:
            self.protocol_manager.reject_job(data['job_id'])
        else:
            self.jobs_manager.on_job_submitted(timestamp, data['job'])

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
        self.makespan = self.jobs_manager.last_job.finish_time - \
            self.jobs_manager.first_job.submit_time
        self.metrics["scheduling_time"] = float(data["scheduling_time"])
        self.metrics["nb_jobs"] = int(data["nb_jobs"])
        self.metrics["nb_jobs_finished"] = int(data["nb_jobs_finished"])
        self.metrics["nb_jobs_success"] = int(data["nb_jobs_success"])
        self.metrics["nb_jobs_killed"] = int(data["nb_jobs_killed"])
        self.metrics["success_rate"] = float(data["success_rate"])
        self.metrics["makespan"] = float(data["makespan"])
        self.metrics["mean_waiting_time"] = float(data["mean_waiting_time"])
        self.metrics["mean_turnaround_time"] = float(
            data["mean_turnaround_time"])
        self.metrics["mean_slowdown"] = float(data["mean_slowdown"])
        self.metrics["max_waiting_time"] = float(data["max_waiting_time"])
        self.metrics["max_turnaround_time"] = float(
            data["max_turnaround_time"])
        self.metrics["max_slowdown"] = float(data["max_slowdown"])
        self.metrics["energy_consumed"] = float(data["consumed_joules"])
        self.metrics['total_slowdown'] = self.jobs_manager.total_slowdown
        self.metrics['total_turnaround_time'] = self.jobs_manager.total_turnaround_time
        self.metrics['total_waiting_time'] = self.jobs_manager.total_waiting_time
        self._export_metrics()

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

    def _get_resources_from_json(self, data):
        resources = []
        for alloc in data.split(" "):
            nodes = alloc.split("-")
            if len(nodes) == 2:
                resources.extend(range(int(nodes[0]), int(nodes[1]) + 1))
            else:
                resources.append(int(nodes[0]))
        return resources

    def _export_metrics(self):
        data = pd.DataFrame(self.metrics, index=[0])
        fn = "{}/{}_{}.csv".format(
            self._output_dir,
            self.nb_simulation,
            "schedule_metrics")
        data.to_csv(fn, index=False)

    def _update_state(self):
        self.protocol_manager.send_events()

        old_time = self.current_time
        events = self.protocol_manager.read_events(
            blocking=not self.running_simulation)

        # New jobs does not need to be updated in this timestep
        # Update jobs if no time has passed does not make sense.
        time_passed = self.current_time - old_time
        if time_passed != 0:
            self.jobs_manager.update_state(time_passed)
            self.resource_manager.update_state(time_passed)
            #self._shut_down_unused_resources()

        for event in events:
            self._handle_batsim_events(event)

        # Remember to always ack
        if self.running_simulation:
            self.protocol_manager.acknowledge()

    def _wake_up_resources(self, resources):
        res = self.resource_manager.wake_up(resources)
        if len(res) != 0:
            self.protocol_manager.set_resource_pstate(
                res, Resource.PowerState.NORMAL)

    def _shut_down_unused_resources(self):
        unused = self.resource_manager.shut_down_unused()
        if len(unused) != 0:
            self.protocol_manager.set_resource_pstate(
                unused, Resource.PowerState.SHUT_DOWN)

    def _get_workload(self):
        if len(self._workloads) == self._workload_idx + 1:
            self._workload_idx = -1

        self._workload_idx += 1
        return self._workloads[self._workload_idx]

    def _get_max_walltime(self, workload):
        with open(workload) as f:
            data = json.load(f)
            max_walltime = max(data['jobs'], key=(
                lambda job: job['walltime']))['walltime']

        return max_walltime

    def _make_random_dir(self, path):
        num = 1
        while os.path.exists(path + str(num)):
            num += 1

        output_dir = path + str(num)
        os.makedirs(output_dir)
        return output_dir

    def get_resource_state(self):
        return self.resource_manager.get_state()

    def get_job_slot_state(self):
        s = np.zeros(
            shape=(self.time_window, self.nb_resources*self.job_slots), dtype=np.uint8)

        for i, job in enumerate(self.jobs_manager.job_slots):
            if job != None:
                start_idx = i * self.nb_resources
                end_idx = start_idx + job.requested_resources
                s[0:job.requested_time, start_idx:end_idx] = 255
        return s

    def get_backlog_state(self):
        s = np.zeros(shape=(self.time_window, self.backlog_width),
                     dtype=np.uint8)
        t, i = 0, 0
        nb_jobs = min(self.backlog_width*self.time_window,
                      self.jobs_manager.nb_jobs_in_backlog)
        for _ in range(nb_jobs):
            s[t, i] = 255
            i += 1
            if i == self.backlog_width:
                i = 0
                t += 1
        return s

    def _get_image(self):
        state = np.zeros(shape=self.state_shape, dtype=np.uint8)

        # RESOURCES
        resource_end = self.nb_resources
        state[:, 0:resource_end] = self.get_resource_state()

        # JOB SLOTS
        job_slot_end = self.nb_resources * self.job_slots + self.nb_resources
        state[:, resource_end:job_slot_end] = self.get_job_slot_state()

        # BACKLOG
        state[:, job_slot_end:self.state_shape[1]] = self.get_backlog_state()

        return state

    def _get_state(self):
        state = dict(
            resources_properties=self.resource_manager.get_resources(),
            resources_spaces=self.resource_manager.get_state(),
            queue=self.jobs_manager.job_slots,
            backlog=self.jobs_manager.nb_jobs_in_backlog)
        return state
