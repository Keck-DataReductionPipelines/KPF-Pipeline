# Use python 3.6 
FROM python:3.6-slim

ENV KPFPIPE_TEST_DATA=/data/KPF-Pipeline-TestData
ENV KPFPIPE_DATA=/data/kpf
ENV KPFPIPE_TEST_OUTPUTS=/outputs
ENV COVERALLS_REPO_TOKEN=YLrA2Q2Af7VGwyULXbs0KujYSjUBdn2jP

# install this way to fix paths in coverage report
ENV PYTHONPATH=$PYTHONPATH:/code/KPF-Pipeline
ENV PYTHONHASHSEED=0

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
    apt-get install --yes git vim emacs && \
    /usr/local/bin/python -m pip install --upgrade pip && \
    cd /code/KPF-Pipeline && \
    mkdir -p logs && \
	mkdir -p outputs

# Set the working directory to KPF-Pipeline
WORKDIR /code/KPF-Pipeline

ADD requirements.txt /code/KPF-Pipeline/
RUN pip3 install -r requirements.txt
