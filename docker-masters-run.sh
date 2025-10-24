#!/bin/bash

# CAUTION: The following setting prints out the docker-run command with passwords as arguments,
# as a debugging tool.  Ideally, passwords should not be passed as environment variables, but
# instead sourced from an environment file with user-only read permission inside the container.
set -x

##############################################################
# Required environment variables and examples:
# KPFCRONJOB_CODE=/data/user/rlaher/git/KPF-Pipeline
# KPFCRONJOB_DOCKER_IMAGE=russkpfmasters:latest
# KPFPIPE_MASTERS_BASE_DIR=/data/kpf/masters
# KPFPIPE_L0_BASE_DIR=/data/kpf/L0
# KPFCRONJOB_SBX=/data/user/rlaher/sbx
# KPFPIPE_DB_PORT=6125
# KPFPIPE_DB_NAME=kpfopsdb
# KPFPIPE_DB_USER=kpfporuss
# KPFPIPE_DB_PASS=?????
# KPFPIPE_TSDB_PORT=6127
# KPFPIPE_TSDB_NAME=timeseriesopsdb
# KPFPIPE_TSDB_USER=timeseriesopsuser
# KPFPIPE_TSDB_USER=????

# Optional environment variable and example:
# KPFPIPE_PORT=6107
##############################################################

# Suppress Docker warnings and run commands quietly
export DOCKER_CLI_EXPERIMENTAL=enabled
export DOCKER_BUILDKIT=1


# Function to run docker with suppressed warnings
run_docker() {
    local port="$1"
    local message="$2"

    echo "$message"

    if [ -n "$port" ]; then
        # Run with port mapping
        docker run -it \
            -p "$port:$port" \
            -v "${KPFCRONJOB_CODE}:/code/KPF-Pipeline" \
            -v "${KPFCRONJOB_SBX}:/data/" \
            -v "${KPFPIPE_L0_BASE_DIR:-/data/kpf/L0}:/data/L0:ro" \
            -v "${KPFPIPE_MASTERS_BASE_DIR:-/data/kpf/masters}:/masters" \
            -e KPFPIPE_PORT="$KPFPIPE_PORT" \
            --network=host \
            -e DBPORT=${KPFPIPE_DB_PORT:-} \
            -e DBNAME=$KPFPIPE_DB_NAME \
            -e DBUSER="${KPFPIPE_DB_USER:-}" \
            -e DBPASS="${KPFPIPE_DB_PASS:-}" \
            -e DBSERVER=127.0.0.1 \
            -e TSDBPORT=$KPFPIPE_TSDB_PORT \
            -e TSDBNAME=$KPFPIPE_TSDB_NAME \
            -e TSDBUSER="${KPFPIPE_TSDB_USER:-}" \
            -e TSDBPASS="${KPFPIPE_TSDB_PASS:-}" \
            -e TSDBSERVER=127.0.0.1 \
            -e PYTHONUNBUFFERED=1 \
            -e PYTHONPATH=/code/KPF-Pipeline:/code/KPF-Pipeline/polly/src \
            $KPFCRONJOB_DOCKER_IMAGE bash 2>/dev/null
    else
        # Run without port mapping
        docker run -it \
            -v "${KPFCRONJOB_CODE}:/code/KPF-Pipeline" \
            -v "${KPFCRONJOB_SBX}:/data/" \
            -v "${KPFPIPE_L0_BASE_DIR:-/data/kpf/L0}:/data/L0:ro" \
            -v "${KPFPIPE_MASTERS_BASE_DIR:-/data/kpf/masters}:/masters" \
            --network=host \
            -e DBPORT=${KPFPIPE_DB_PORT:-} \
            -e DBNAME=$KPFPIPE_DB_NAME \
            -e DBUSER="${KPFPIPE_DB_USER:-}" \
            -e DBPASS="${KPFPIPE_DB_PASS:-}" \
            -e DBSERVER=127.0.0.1 \
            -e TSDBPORT=$KPFPIPE_TSDB_PORT \
            -e TSDBNAME=$KPFPIPE_TSDB_NAME \
            -e TSDBUSER="${KPFPIPE_TSDB_USER:-}" \
            -e TSDBPASS="${KPFPIPE_TSDB_PASS:-}" \
            -e TSDBSERVER=127.0.0.1 \
            -e PYTHONUNBUFFERED=1 \
            -e PYTHONPATH=/code/KPF-Pipeline:/code/KPF-Pipeline/polly/src \
            $KPFCRONJOB_DOCKER_IMAGE bash 2>/dev/null
    fi
}


# Main execution.

if [ -n "$KPFPIPE_PORT" ]; then
    run_docker "$KPFPIPE_PORT" "Starting Docker container on port $KPFPIPE_PORT..."
else
    run_docker "" "Starting Docker container (no port specified)..."
fi
