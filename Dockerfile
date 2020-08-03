# Use python 3.6 
FROM python:3.6-slim

ENV KPFPIPE_TEST_DATA=/data
ENV KPFPIPE_TEST_OUTPUTS=/outputs
ENV COVERALLS_REPO_TOKEN=mX4VKMQS2HHM0jGm7qkBUNRtueI8omokr

# install this way to fix paths in coverage report
ENV PYTHONPATH=$PYTHONPATH:/code/KPF-Pipeline

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

# Run testswhen the container launches
CMD make init && \
    make test
