class Event:
    def __init__(self, data):
        self.timestamp = data['timestamp']
        self.type = data['type']
        self.data = data['data']


class Message:
    def __init__(self, data):
        self.now = data['now']
        self.events = [Event(event) for event in data['events']]


class Job:
    def __init__(self, data):
        self.id = data['job']['id']
        self.res = data['job']['res']
        self.walltime = data['job']['walltime']
        self.subtime = data['job']['subtime']
        self.exectime = 0
        self.alloc_hosts = []


class Resource:
    states = ["sleeping", "idle", "computing", "switching_on", "switching_off"]

    def __init__(self, data):
        self.id = data['id']
        self.name = data['name']
        self.state = data['state']
        self.properties = data['properties']

    def is_available(self):
        return (self.state == 'sleeping') or (self.state == 'idle')

    @property
    def state_value(self):
        return Resource.states.index(self.state)
