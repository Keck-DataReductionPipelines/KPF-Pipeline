#!/bin/bash

# If a processing date (yyyymmdd) is given on the command line, then the masters pipeline will be executed
# inside a detached container.  This can be executed as a cronjob as well.  Here is an example line for the crontab:
# 15 06 * * * source $HOME/.bash_profile; ${KPFCRONJOB_CODE}/docker-masters-run.sh 20251030 >& ${KPFCRONJOB_CODE}/docker-masters-run_20251030.out

# CAUTION: The following setting, which is normally commented out, prints out the docker-run command with
# passwords as arguments, as a debugging tool.  Ideally, passwords should not be passed as environment variables,
# but instead sourced from an environment file with user-only read permission inside the container.
# set -x


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
# KPFCRONJOB_DOCKER_NAME_MASTERS=kpfmastersdrp

# Optional environment variable and example:
# KPFPIPE_PORT=6107
##############################################################

# Suppress Docker warnings and run commands quietly
export DOCKER_CLI_EXPERIMENTAL=enabled
export DOCKER_BUILDKIT=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_ENV_FILE="${KPFPIPE_ENV_FILE:-${SCRIPT_DIR}/.env}"

load_env_file() {
	if [ ! -f "$1" ]; then
		return
	fi
	if ! command -v python3 >/dev/null 2>&1; then
		echo "Warning: python3 not available; skipping env file $1" >&2
		return
	fi
	local exports
	exports="$(python3 - "$1" <<'PY'
import pathlib, shlex, sys
path = pathlib.Path(sys.argv[1])
try:
    text = path.read_text()
except FileNotFoundError:
    sys.exit(0)
for raw in text.splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("export "):
        line = line[7:].strip()
    if "=" not in line:
        continue
    key, _, value = line.partition("=")
    key = key.strip()
    value = value.strip()
    if value and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    print(f"export {key}={shlex.quote(value)}")
PY
)"
	if [ -n "$exports" ]; then
		eval "$exports"
	fi
}

load_env_file "$ROOT_ENV_FILE"

# Verify required environment variables before proceeding
required_vars=(
	KPFCRONJOB_CODE
	KPFCRONJOB_DOCKER_IMAGE
	KPFPIPE_MASTERS_BASE_DIR
	KPFPIPE_L0_BASE_DIR
	KPFCRONJOB_SBX
	KPFPIPE_DB_PORT
	KPFPIPE_DB_NAME
	KPFPIPE_DB_USER
	KPFPIPE_DB_PASS
	KPFPIPE_TSDB_PORT
	KPFPIPE_TSDB_NAME
	KPFPIPE_TSDB_USER
	KPFPIPE_TSDB_PASS
	KPFCRONJOB_DOCKER_NAME_MASTERS
)
missing_vars=()
for var in "${required_vars[@]}"; do
	if [ -z "${!var}" ]; then
		missing_vars+=("$var")
	fi
done
if [ "${#missing_vars[@]}" -ne 0 ]; then
	echo "Error: Missing required environment variables:" >&2
	for mv in "${missing_vars[@]}"; do
		printf '  - %s\n' "$mv" >&2
	done
	exit 1
fi


# Function to run KPF masters pipeline in detached docker container.
run_kpf_masters_pipeline() {

    overridescript="/code/KPF-Pipeline/cronjobs/kpf_masters_pipeline.sh"

    if [ "${KPFCRONJOB_DOCKER_NAME_MASTERS:+x}" ]; then
        echo "KPFCRONJOB_DOCKER_NAME_MASTERS is set and not empty."
    else
        echo "KPFCRONJOB_DOCKER_NAME_MASTERS is not set or is empty; quitting..."
        exit 64
    fi

    docker rm $KPFCRONJOB_DOCKER_NAME_MASTERS

    docker run -d --name ${KPFCRONJOB_DOCKER_NAME_MASTERS} \
        --entrypoint $overridescript \
        -v "${KPFCRONJOB_CODE}:/code/KPF-Pipeline" \
        -v "${KPFCRONJOB_SBX}:/data/" \
        -v "${KPFPIPE_L0_BASE_DIR:-/data/kpf/L0}:/data/L0:ro" \
        -v "${KPFPIPE_MASTERS_BASE_DIR:-/data/kpf/masters}:/masters" \
        --network=host \
        -e PROCDATE=${PROCDATE} \
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
        $KPFCRONJOB_DOCKER_IMAGE

}


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

if [ -z "$1" ]; then
    if [ -n "$KPFPIPE_PORT" ]; then
        run_docker "$KPFPIPE_PORT" "Starting Docker container on port $KPFPIPE_PORT..."
    else
        run_docker "" "Starting Docker container (no port specified)..."
    fi
else
    procdate=$1
    echo "Running the masters pipeline for $procdate"
    export PROCDATE=$procdate
    run_kpf_masters_pipeline
fi
