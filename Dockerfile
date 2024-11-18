# Use python 3.11 
FROM python:3.11-bullseye

# Set environment variables
ENV KPFPIPE_TEST_DATA=/testdata \
    KPFPIPE_DATA=/data \
    KPFPIPE_TEST_OUTPUTS=/outputs \
    PYTHONPATH=/code/KPF-Pipeline \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Create necessary directories and set working directory
WORKDIR /code/KPF-Pipeline
RUN mkdir -p /code/KPF-Pipeline /data /outputs /code/KPF-Pipeline/logs /code/KPF-Pipeline/outputs

# Install dependencies, fixing any broken installs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        git \
        vim \
        emacs \
        nano \
        gnupg \
        python3-distutils \
        dirmngr && \
    apt-get install -y --fix-missing sysstat || apt-get install -y --fix-broken && \
    # Reconfigure packages if necessary
    dpkg --configure -a && \
    apt-get update && \
    apt-get install -y parallel && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and set up the project structure
RUN /usr/local/bin/python -m pip install --upgrade pip

# Configure git to allow operations in this directory
RUN git config --global --add safe.directory /code/KPF-Pipeline

# Copy requirements and install Python dependencies
COPY requirements.txt /code/KPF-Pipeline/
RUN pip3 install --no-cache-dir -r requirements.txt
