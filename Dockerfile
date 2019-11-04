FROM tensorflow/tensorflow:1.14.0-gpu-py3

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

RUN apt install -y libsm6 libxrender1 libfontconfig1

WORKDIR /app/eval

COPY . /app

RUN cd /app && \
    ${PIP} install -e .[tf_gpu] && \
    cd src/GridGym && \
    ${PIP} install -e . && \
    cd gridgym/envs/batsim-py && \
    ${PIP} install -e . 

#ENTRYPOINT ["python3", "run_experiments.py"]