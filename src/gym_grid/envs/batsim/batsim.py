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
from .simulator import GridSimulator


class BatsimHandler:
    PLATFORM = "platform_hg_10.xml"
    CONFIG = "cfg.json"
    WORKLOAD_DIR = "workload"
    OUTPUT_DIR = "results/batsim"

    def __init__(self, job_slots, time_window, backlog_width):
        fullpath = os.path.join(os.path.dirname(__file__), "files")
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self._output_dir = self._make_random_dir(BatsimHandler.OUTPUT_DIR)
        self._platform = os.path.join(fullpath, BatsimHandler.PLATFORM)
        workloads_path = os.path.join(fullpath, BatsimHandler.WORKLOAD_DIR)
        self._workloads = [workloads_path + "/" + w for w in os.listdir(workloads_path) if w.endswith('.json')]
        self.nb_simulation = 0
        self.time_window = time_window
        self.job_slots = job_slots
        self.resource_manager = ResourceManager.from_xml(self._platform, time_window)
        self.jobs_manager = SchedulerManager(self.nb_resources, job_slots)
        self.simulator = GridSimulator(self._workloads[0], self.jobs_manager)
        self.backlog_width = backlog_width
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
        return self.simulator.current_time

    @property
    def running_simulation(self):
        return self.simulator.running

    def get_state(self, type):
        if type == 'image':
            return self._get_image()
        elif type == 'compact':
            return self._get_compact_state()
        else:
            return self._get_state()

    def close(self):
        self.simulator.close()

    def start(self):
        assert not self.running_simulation, "A simulation is already running."
        self.simulator.start()
        self._reset()
        self._wait_state_change()
        assert self.running_simulation, "An error ocurred during simulator starting."
        self.nb_simulation += 1

    def schedule(self, job_pos):
        assert self.running_simulation, "Simulation is not running."

        if job_pos != -1:  # Try to schedule job
            job = self.jobs_manager.get_job(job_pos)
            self.resource_manager.allocate(job)
            self.jobs_manager.on_job_allocated(job_pos)
        else:
            self._proceed_time()

        self._start_ready_jobs()

        self._update_state()

    def _wait_state_change(self):
        self._update_state()
        while self.running_simulation and (self.jobs_manager.is_empty or self.resource_manager.is_full()):
            self._proceed_time()
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
                self._start_job(job)

    def _start_job(self, job):
        self.resource_manager.start_job(job)
        self.jobs_manager.on_job_started(job.id, self.current_time)

    def _reset(self):
        self.metrics = {}
        self.jobs_manager.reset()
        self.resource_manager.reset()

    def _handle_job_completed(self, timestamp, data):
        job = self.jobs_manager.on_job_completed(timestamp, data)
        self.resource_manager.release(job)
        self._start_ready_jobs()

    def _handle_job_submitted(self, timestamp, data):
        if data['job']['res'] <= self.resource_manager.nb_resources:
            self.jobs_manager.on_job_submitted(timestamp, data['job'])
        else:
            self.simulator.reject_job(data['job_id'])

    def _handle_simulation_ends(self, data):
        self.metrics['energy_consumed'] = self.resource_manager.energy_consumption
        self.metrics['makespan'] = self.jobs_manager.last_job.finish_time - self.jobs_manager.first_job.submit_time
        self.metrics['total_slowdown'] = self.jobs_manager.total_slowdown
        self.metrics['mean_slowdown'] = self.jobs_manager.runtime_mean_slowdown
        self.metrics['total_turnaround_time'] = self.jobs_manager.total_turnaround_time
        self.metrics['total_waiting_time'] = self.jobs_manager.total_waiting_time
        self._export_metrics()

    def _handle_event(self, event):
        if event.type == "SIMULATION_ENDS":
            self._handle_simulation_ends(event.data)
        elif event.type == "JOB_SUBMITTED":
            self._handle_job_submitted(event.timestamp, event.data)
        elif event.type == "JOB_COMPLETED":
            self._handle_job_completed(event.timestamp, event.data)
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

    def _proceed_time(self):
        self.simulator.proceed_time(1)
        self.jobs_manager.update_state(1)
        self.resource_manager.update_state(1)
        #self.resource_manager.shut_down_unused()
        
    def _update_state(self):
        events = self.simulator.read_events()
        for event in events:
            self._handle_event(event)

    def _make_random_dir(self, path):
        num = 1
        while os.path.exists(path + str(num)):
            num += 1

        output_dir = path + str(num)
        os.makedirs(output_dir)
        return output_dir

    def get_job_slot_state(self):
        s = np.zeros(
            shape=(self.time_window, self.nb_resources*self.job_slots), dtype=np.uint8)

        for i, job in enumerate(self.jobs_manager.job_slots):
            if job != None:
                start_idx = i * self.nb_resources
                end_idx = start_idx + job.requested_resources
                s[0:job.requested_time, start_idx:end_idx] = 1
        return s

    def get_backlog_state(self):
        s = np.zeros(shape=(self.time_window, self.backlog_width), dtype=np.uint8)
        t, i = 0, 0
        nb_jobs = min(self.backlog_width*self.time_window,
                      self.jobs_manager.nb_jobs_in_backlog)
        for _ in range(nb_jobs):
            s[t, i] = 1
            i += 1
            if i == self.backlog_width:
                i = 0
                t += 1
        return s

    def get_time_state(self):
        v = self.simulator.time_since_last_new_job / float(self.simulator.max_tracking_time_since_last_job)
        return np.full(shape=self.time_window, fill_value= v, dtype=np.float)

    def _get_image(self):
        shape = (self.time_window, self.nb_resources + self.job_slots*self.nb_resources + self.backlog_width + 1)
        state = np.zeros(shape=shape, dtype=np.float)

        # RESOURCES
        resource_end = self.nb_resources
        state[:, 0:resource_end] = self.resource_manager.get_view()

        # JOB SLOTS
        job_slot_end = self.nb_resources * self.job_slots + self.nb_resources
        state[:, resource_end:job_slot_end] = self.get_job_slot_state()

        # BACKLOG
        state[:, job_slot_end:shape[1]] = self.get_backlog_state()

        state[:, -1] = self.get_time_state()

        return state

    def _get_compact_state(self):
        #nb_res = self.resource_manager.nb_resources
        state = np.zeros(shape=(self.nb_resources + self.job_slots*2 + 1), dtype=np.float)
        #_, idle, sleeping = self.resource_manager.nb_resources_state()


        index = 0
        for res in self.resource_manager.get_resources():
            #state[index] = int(not res.is_sleeping)
            state[index] = res.get_reserved_time()# / self.time_window
            index += 1

        #state[0] = idle / self.nb_resources
        #state[1] = sleeping / self.nb_resources

        jobs = self.jobs_manager.job_slots
        #print(self.simulator.time_since_last_new_job)
        #max_waiting_time = self.jobs_manager.get_max_waiting_time()
        #max_slowdown = self.jobs_manager.get_max_slowdown()
        for job in jobs:
            if job is not None:
                state[index] = job.requested_resources# / self.nb_resources
                state[index+1] = min(job.requested_time, self.time_window)# / self.time_window
                #state[index+2] = job.waiting_time / max_waiting_time if max_waiting_time != 0 else 0#job.estimate_slowdown() / max_slowdown if max_slowdown != 0 else 0
            index += 2
        
        state[-1] = self.simulator.time_since_last_new_job / float(self.simulator.max_tracking_time_since_last_job)

        return state

    def _get_state(self):
        state = dict(
            resources_properties=self.resource_manager.get_resources(),
            resources_spaces=self.resource_manager.get_view(),
            queue=self.jobs_manager.job_slots,
            backlog=self.jobs_manager.nb_jobs_in_backlog)
        return state
