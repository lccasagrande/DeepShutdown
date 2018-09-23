from enum import Enum, unique
from xml.dom import minidom
import numpy as np


class Resource:
    class State(Enum):
        SLEEPING = 'sleeping'
        IDLE = 'idle'
        COMPUTING = 'computing'

    class PowerState(Enum):
        SHUT_DOWN = 0
        NORMAL = 1

    def __init__(self, id, state, pstate, name, properties):
        assert isinstance(state, Resource.State)
        assert isinstance(pstate, Resource.PowerState)
        assert isinstance(properties, dict)
        assert isinstance(id, int)
        self.state = state
        self.pstate = pstate
        self.name = name
        self.id = id
        self.properties = properties
        self.max_watt = self.properties[Resource.PowerState.NORMAL]['watt_comp']
        self.max_speed = self.properties[Resource.PowerState.NORMAL]['speed']
        self.min_watt = self.properties[Resource.PowerState.SHUT_DOWN]['watt_idle']

    @staticmethod
    def from_xml(id, data):
        name = data.getAttribute('id')
        host_speed = data.getAttribute('speed').split(',')
        host_watts = data.getElementsByTagName(
            'prop')[0].getAttribute('value').split(',')
        assert "Mf" in host_speed[-1], "Speed is not in Mega Flops"

        properties = {}
        for i, speed in enumerate(host_speed):
            (idle, comp) = host_watts[i].split(":")
            properties[Resource.PowerState(i)] = {
                'speed': float(speed.replace("Mf", "")),
                'watt_idle': float(idle),
                'watt_comp': float(comp)
            }
        return Resource(id, Resource.State.IDLE, Resource.PowerState.NORMAL, name, properties)

    @property
    def is_sleeping(self):
        return self.state == Resource.State.SLEEPING

    @property
    def is_available(self):
        return not self.is_computing

    @property
    def is_computing(self):
        return self.state == Resource.State.COMPUTING

    @property
    def cost_to_compute(self):
        if self.pstate == Resource.PowerState.SHUT_DOWN:
            _, w_min, _ = self._get_properties(Resource.PowerState.SHUT_DOWN)
            speed, _, w_comp = self._get_properties(Resource.PowerState.NORMAL)
        else:
            speed, w_min, w_comp = self._get_properties(self.pstate)

        return (w_comp - w_min) / speed

    def set_state(self, state):
        assert isinstance(state, Resource.State)
        self.state = state

    def set_pstate(self, pstate):
        assert isinstance(pstate, Resource.PowerState)
        if self.is_computing:
            raise InvalidPowerStateError(
                "Cannot change resurce power state while it is being used by a job.")

        self.pstate = pstate
        if pstate == Resource.PowerState.SHUT_DOWN:
            self.set_state(Resource.State.SLEEPING)
        else:
            self.set_state(Resource.State.IDLE)

    def _get_properties(self, pstate):
        prop = self.properties[pstate]
        return prop['speed'], prop['watt_idle'], prop['watt_comp']


class ResourceManager:
    def __init__(self, resources):
        assert isinstance(resources, dict)
        self.nb_resources = len(resources)
        self.resources = resources

    @property
    def nb_resources_unused(self):
        nb = len([k for k, res in self.resources.items() if not res.is_computing])
        return nb

    @property
    def max_resource_energy_cost(self):
        return max(self.resources.items(), key=(lambda item: item[1].cost_to_compute))[1].cost_to_compute

    @property
    def max_resource_speed(self):
        return max(self.resources.items(), key=(lambda item: item[1].max_speed))[1].max_speed

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

    def on_resource_pstate_changed(self, resources, pstate):
        assert isinstance(pstate, Resource.PowerState)
        self.set_pstate(resources, pstate)

    def on_job_scheduled(self, resources):
        self.set_state(resources, Resource.State.COMPUTING)

    def on_job_allocated(self, resources):
        self.set_state(resources, Resource.State.COMPUTING)

    def on_job_completed(self, resources):
        self.set_state(resources, Resource.State.IDLE)

    def estimate_energy_consumption(self, res_ids):
        energy = 0
        for id in res_ids:
            try:
                energy += self.resources[id].cost_to_compute
            except KeyError:
                raise KeyError("Could not find resource: {}".format(id))
        return energy

    def get_resources(self):
        resources = np.empty((self.nb_resources,), dtype=object)
        for k, value in self.resources.items():
            resources[k] = value

        return resources

    def is_available(self, res_ids):
        for id in res_ids:
            try:
                resource = self.resources[id]
                if not resource.is_available:
                    return False
            except KeyError:
                raise KeyError("Could not find resource: {}".format(id))

        return True

    def set_state(self, res_ids, state):
        for id in res_ids:
            try:
                self.resources[id].set_state(state)
            except KeyError:
                raise KeyError("Could not find resource: {}".format(id))

    def set_pstate(self, res_ids, pstate):
        for id in res_ids:
            try:
                self.resources[id].set_pstate(pstate)
            except KeyError:
                raise KeyError("Could not find resource: {}".format(id))


class InvalidPowerStateError(Exception):
    pass
