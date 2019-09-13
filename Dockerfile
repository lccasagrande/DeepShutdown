FROM tensorflow/tensorflow:latest-gpu-py3

ARG WORKLOAD=lyon_4
ARG USE_PYTHON_3_NOT_2=True
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

RUN apt-get update && \
    apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip \
    python-opengl

RUN ${PIP} install --upgrade pip setuptools

#RUN git clone https://github.com/lccasagrande/baselines.git && cd baselines && pip install -e .

RUN apt install -y libsm6 libxrender1 libfontconfig1

WORKDIR /app/src

COPY . /app

RUN cd /app && \
    rm -r src/GridGym/gridgym/envs/simulator/files/workloads/* && \
    cp eval/${WORKLOAD}/workloads/train/* src/GridGym/gridgym/envs/simulator/files/workloads && \
    pip install -e . && \
    cd src/GridGym && \
    pip install -e .

CMD []