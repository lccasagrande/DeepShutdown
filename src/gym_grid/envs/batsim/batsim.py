from .network import NetworkHandler
from .managers import JobsManager, ResourcesManager
from .models import Message, Event, Job
from xml.dom import minidom
from random import choice
from collections import namedtuple
import os
import shlex
from subprocess import Popen, PIPE
import signal


class BatsimHandler:
    def __init__(self, socket_endpoint, verbose="information"):
        self.__platform = "src/gym_grid/envs/batsim/platforms/energy_platform_homogeneous_no_net_128.xml"
        self.__workload = "src/gym_grid/envs/batsim/workloads/energy_small.json"
        self.__verbose = verbose
        self.__resource_manager = None
        self.__jobs_manager = JobsManager()
        self.__simulator_process = None
        self.running = False
        self.current_time = 0
        self.__network_handler = NetworkHandler(socket_endpoint, 0)
        self.statistics = None
        self.nb_resources = len(minidom.parse(
            self.__platform).getElementsByTagName('host')) - 1

    def close(self):
        self.statistics = self.__get_statistics()
        if self.__simulator_process is not None:
            # os.kill(self.__simulator_process.pid, signal.SIGINT)
            self.__simulator_process.kill()
            self.__simulator_process.wait()
            self.__simulator_process = None
        self.__network_handler.close()
        self.__resource_manager = None
        self.__jobs_manager = JobsManager()
        self.running = False
        self.current_time = 0

    def start(self):
        self.statistics = None
        self.__simulator_process = self.__start_simulator()
        self.__connect()
        self.__update_state()
        self.update_state()

        if not self.running:
            raise ConnectionError("Não foi possível inicial o simulador.")

    def alloc_job(self, resource):
        def handle_void():
            job = self.__jobs_manager.dequeue(0)
            self.__network_handler.ack(self.current_time)
            self.update_state()
            self.__jobs_manager.enqueue(job)
            return True

        def format_msg(job):
            msg = {"timestamp": self.current_time,
                   "type": "EXECUTE_JOB",
                   "data": {"job_id": job.id,
                            "alloc": ' '.join(str(h) for h in job.alloc_hosts)}}
            return msg

        if self.__jobs_manager.is_empty:
            raise RuntimeError("Não existe um job para ser alocado.")

        job = self.get_job()
        needed_resources = job.res - len(job.alloc_hosts)
        if not self.__resource_manager.has_available(needed_resources):
            if resource == -1:  # VOID
                return handle_void()
            return False
        else:
            if resource == -1:  # VOID
                return False

        if not self.__resource_manager.is_available(resource):
            return False

        self.__resource_manager.change_to_computing(resource)

        job.alloc_hosts.append(resource)
        if len(job.alloc_hosts) != job.res:
            return True

        job = self.__jobs_manager.dequeue(self.current_time)
        msg = format_msg(job)
        self.__send_msg(msg)
        self.update_state()
        return True

    def __get_statistics(self):
        if self.__resource_manager is None:
            return None

        stats = self.__resource_manager.statistics
        stats.update(self.__jobs_manager.statistics)
        stats['makespan'] = self.current_time

        return stats

    def get_resources(self):
        if self.__resource_manager is None:
            return None

        return self.__resource_manager.resources

    def get_job(self):
        if self.__jobs_manager.is_empty:
            return None
        return self.__jobs_manager.at(0)

    def update_state(self):
        jobs_len = len(self.__jobs_manager.queue)
        while len(self.__jobs_manager.queue) == jobs_len and self.running:
            self.__update_state()

    def __send_msg(self, event):
        msg = {"now": self.current_time, "events": [event]}
        self.__network_handler.send(msg)

    def __connect(self):
        self.__network_handler.bind()

    def __get_msg(self):
        msg = None
        while msg is None:
            msg = self.__network_handler.recv(not self.running)
        return msg

    def __start_simulator(self):
        cmd = "batsim -p %s -w %s -v %s" % (
            self.__platform, self.__workload, self.__verbose)
        # return Popen(cmd, stdout=subprocess.PIPE, shell=True)
        # return Popen(cmd, preexec_fn=os.setsid)
        return Popen(shlex.split(cmd))

    def __update_state(self):
        msg = self.__get_msg()
        self.current_time = msg.now
        ack = True
        close = False
        for event in msg.events:
            if event.type == "SIMULATION_BEGINS":
                self.__resource_manager = ResourcesManager.import_from(
                    event.data)
                self.running = True
            elif event.type == "JOB_SUBMITTED":
                if event.data["job_id"] == "63b90c!93":
                    print("JOB_SUBMITTED")
                self.__jobs_manager.enqueue(Job(event.data))
                ack = False
            elif event.type == "RESOURCE_STATE_CHANGED":
                self.__resource_manager.onResourceStateChanged(
                    event.data["resources"])
            elif event.type == "JOB_COMPLETED":
                if event.data["job_id"] == "63b90c!93":
                    print("JOB_COMPLETED")
                self.__jobs_manager.onJobCompleted(event)
                self.__resource_manager.onJobCompleted(event)
            elif event.type == "SIMULATION_ENDS":
                close = True

        if ack:
            self.__network_handler.ack(self.current_time)

        if close:
            self.__simulator_process.wait()
            self.__simulator_process = None
            self.close()
            print("SIMULATION CLOSED")
