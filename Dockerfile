# Use python 3.6 
FROM python:3.6-slim

ENV KPFPIPE_TEST_DATA=/testdata
ENV KPFPIPE_DATA=/data
ENV KPFPIPE_TEST_OUTPUTS=/outputs

# install this way to fix paths in coverage report
ENV PYTHONPATH=$PYTHONPATH:/code/KPF-Pipeline
ENV PYTHONHASHSEED=0
ENV PYTHONUNBUFFERED=1

# turn off built-in Python multithreading
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# setup the working directory
RUN mkdir /code && \
    mkdir /code/KPF-Pipeline && \
    mkdir /data && \
    mkdir /outputs && \
    apt-get --yes update && \
    apt install build-essential -y --no-install-recommends && \
    apt-get install --yes git vim emacs nano && \
    /usr/local/bin/python -m pip install --upgrade pip && \
    cd /code/KPF-Pipeline && \
    mkdir -p logs && \
	mkdir -p outputs

# Set the working directory to KPF-Pipeline
WORKDIR /code/KPF-Pipeline
RUN git config --global --add safe.directory /code/KPF-Pipeline

ADD requirements.txt /code/KPF-Pipeline/
RUN pip3 install -r requirements.txt
