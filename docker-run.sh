#!/bin/bash

# Suppress Docker warnings and run commands quietly
export DOCKER_CLI_EXPERIMENTAL=enabled
export DOCKER_BUILDKIT=1

# Function to run docker with suppressed warnings
run_docker() {
    local port="$1"
    local message="$2"

    echo "$message"
    echo "System: $(free -g | awk '/^Mem:/{print $2}')GB total RAM, $(free -g | awk '/^Mem:/{print $7}')GB available"

    if [ -n "$port" ]; then
        # Run with port mapping
        docker run -it \
            -p "$port:$port" \
            --network=host \
            -e KPFPIPE_PORT="$port" \
            -e DBPORT=6125 \
            -e DBNAME=kpfopsdb \
            -e DBUSER="${KPFPIPE_DB_USER:-}" \
            -e DBPASS="${KPFPIPE_DB_PASS:-}" \
            -e DBSERVER=127.0.0.1 \
            -e TSDBPORT=6127 \
            -e TSDBNAME=timeseriesopsdb \
            -e TSDBUSER="${KPFPIPE_TSDB_USER:-}" \
            -e TSDBPASS="${KPFPIPE_TSDB_PASS:-}" \
            -e TSDBSERVER=127.0.0.1 \
            -v "${PWD}:/code/KPF-Pipeline" \
            -v "${KPFPIPE_TEST_DATA:-/data/KPF-Pipeline-TestData}:/testdata" \
            -v "${KPFPIPE_DATA:-/data/kpf}:/data" \
            -v "${KPFPIPE_DATA:-/data/kpf}/masters:/masters" \
            kpf-drp:latest bash 2>/dev/null
    else
        # Run without port mapping
        docker run -it \
            --network=host \
            -e DBPORT=6125 \
            -e DBNAME=kpfopsdb \
            -e DBUSER="${KPFPIPE_DB_USER:-}" \
            -e DBPASS="${KPFPIPE_DB_PASS:-}" \
            -e TSDBSERVER=127.0.0.1 \
            -e TSDBPORT=6127 \
            -e TSDBNAME=timeseriesopsdb \
            -e TSDBUSER="${KPFPIPE_TSDB_USER:-}" \
            -e TSDBPASS="${KPFPIPE_TSDB_PASS:-}" \
            -e DBSERVER_TSDB=127.0.0.1 \
            -v "${PWD}:/code/KPF-Pipeline" \
            -v "${KPFPIPE_TEST_DATA:-/data/KPF-Pipeline-TestData}:/testdata" \
            -v "${KPFPIPE_DATA:-/data/kpf}:/data" \
            -v "${KPFPIPE_DATA:-/data/kpf}/masters:/masters" \
            kpf-drp:latest bash 2>/dev/null
    fi
}

# Main execution
if [ -n "$KPFPIPE_PORT" ]; then
    run_docker "$KPFPIPE_PORT" "Starting Docker container on port $KPFPIPE_PORT..."
else
    run_docker "" "Starting Docker container (no port specified)..."
fi
