# Use python 3.6 
FROM python:3.6-slim

ENV KPFPIPE_TEST_DATA=/data
ENV KPFPIPE_TEST_OUTPUTS=/outputs
ENV COVERALLS_REPO_TOKEN=VDoVzb4ly0tzpBlgpp3oXsrZd39BZk30D

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
    apt-get install --yes git && \
    ulimit -n 4096 && \
    ulimit -u 1048576 && \
    ulimit -s unlimited && \
    cd /code
    # # Clone the KeckDRPFramework repository 
    # git clone https://github.com/Keck-DataReductionPipelines/KeckDRPFramework.git && \
    # # Current branch only run on develop branch of KeckDRPFramewke
    # cd KeckDRPFramework && \
    # git checkout develop

# Set the working directory to KPF-Pipeline
WORKDIR /code/KPF-Pipeline
ADD . /code/KPF-Pipeline

# Install the package
RUN pip3 install -r requirements.txt
