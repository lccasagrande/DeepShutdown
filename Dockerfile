FROM lccasagrande/batsim:latest

RUN apt-get install -y python3 python3-pip
RUN pip3 install zmq sklearn pyglet plotly pandas numpy matplotlib keras keras-rl gym tensorflow image

WORKDIR /deepscheduler

COPY . /deepscheduler

CMD []