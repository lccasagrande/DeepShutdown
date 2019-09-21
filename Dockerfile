FROM tensorflow/tensorflow:latest-gpu-py3

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
    pip install -e .[tf_gpu] && \
    cd src/GridGym && \
    pip install -e .

ENTRYPOINT ["python", "run_experiments.py"]