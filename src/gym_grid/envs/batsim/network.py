"""
    batsim.network
    ~~~~~~~~~~~~~~

    Handle zmq network connections.
"""
import zmq
import json
from .models import Message


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
        self.__send_string(json.dumps(msg))

    def __send_string(self, msg):
        assert self.connection, "Connection not open"
        if self.verbose > 0:
            print("[PYBATSIM]: SEND_MSG\n {}".format(msg))
        self.connection.send_string(msg)

    def ack(self, time):
        msg = {"now": time, "events": []}
        self.send(msg)

    def recv(self, blocking=False):
        msg = self.__recv_string(blocking=blocking)
        if msg is not None:
            msg = Message(json.loads(msg))
        return msg

    def __recv_string(self, blocking=False):
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
