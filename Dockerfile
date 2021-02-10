# Use python 3.6 
FROM python:3.6-slim

ENV KPFPIPE_TEST_DATA=/data
ENV KPFPIPE_TEST_OUTPUTS=/outputs
ENV COVERALLS_REPO_TOKEN=VDoVzb4ly0tzpBlgpp3oXsrZd39BZk30D

# install this way to fix paths in coverage report
ENV PYTHONPATH=$PYTHONPATH:/code/KPF-Pipeline
ENV PYTHONHASHSEED=0

# setup the working directory
RUN mkdir /code && \
    mkdir /code/KPF-Pipeline && \
    mkdir /data && \
    mkdir /outputs && \
    apt-get --yes update && \
    apt install build-essential -y --no-install-recommends && \
    apt-get install --yes git && \
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
