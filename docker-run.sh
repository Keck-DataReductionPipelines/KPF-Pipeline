#!/bin/bash

set -euo pipefail

# Suppress Docker warnings and run commands quietly
export DOCKER_CLI_EXPERIMENTAL=enabled
export DOCKER_BUILDKIT=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_ENV_FILE="${KPFPIPE_ENV_FILE:-${SCRIPT_DIR}/.env}"

load_env_file() {
    local file="$1"
    if [ ! -f "$file" ]; then
        return
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "Warning: python3 not available; skipping env file ${file}" >&2
        return
    fi
    local exports
    exports="$(python3 - "$file" <<'PY'
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

DBSERVER_VALUE="${DBSERVER:-${KPFPIPE_DB_SERVER:-127.0.0.1}}"
DBPORT_VALUE="${DBPORT:-${KPFPIPE_DB_PORT:-6125}}"
DBNAME_VALUE="${DBNAME:-${KPFPIPE_DB_NAME:-kpfopsdb}}"
TSDBSERVER_VALUE="${TSDBSERVER:-${KPFPIPE_TSDB_SERVER:-127.0.0.1}}"
TSDBPORT_VALUE="${TSDBPORT:-${KPFPIPE_TSDB_PORT:-6127}}"
TSDBNAME_VALUE="${TSDBNAME:-${KPFPIPE_TSDB_NAME:-timeseriesopsdb}}"

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
            -e DBPORT="$DBPORT_VALUE" \
            -e DBNAME="$DBNAME_VALUE" \
            -e DBUSER="${KPFPIPE_DB_USER:-}" \
            -e DBPASS="${KPFPIPE_DB_PASS:-}" \
            -e DBSERVER="$DBSERVER_VALUE" \
            -e TSDBPORT="$TSDBPORT_VALUE" \
            -e TSDBNAME="$TSDBNAME_VALUE" \
            -e TSDBUSER="${KPFPIPE_TSDB_USER:-}" \
            -e TSDBPASS="${KPFPIPE_TSDB_PASS:-}" \
            -e TSDBSERVER="$TSDBSERVER_VALUE" \
            -v "${PWD}:/code/KPF-Pipeline" \
            -v "${KPFPIPE_TEST_DATA:-/data/KPF-Pipeline-TestData}:/testdata" \
            -v "${KPFPIPE_DATA:-/data/kpf}:/data" \
            -v "${KPFPIPE_DATA:-/data/kpf}/masters:/masters" \
            kpf-drp:latest bash 2>/dev/null
    else
        # Run without port mapping
        docker run -it \
            --network=host \
            -e DBPORT="$DBPORT_VALUE" \
            -e DBNAME="$DBNAME_VALUE" \
            -e DBUSER="${KPFPIPE_DB_USER:-}" \
            -e DBPASS="${KPFPIPE_DB_PASS:-}" \
            -e DBSERVER="$DBSERVER_VALUE" \
            -e TSDBSERVER="$TSDBSERVER_VALUE" \
            -e TSDBPORT="$TSDBPORT_VALUE" \
            -e TSDBNAME="$TSDBNAME_VALUE" \
            -e TSDBUSER="${KPFPIPE_TSDB_USER:-}" \
            -e TSDBPASS="${KPFPIPE_TSDB_PASS:-}" \
            -v "${PWD}:/code/KPF-Pipeline" \
            -v "${KPFPIPE_TEST_DATA:-/data/KPF-Pipeline-TestData}:/testdata" \
            -v "${KPFPIPE_DATA:-/data/kpf}:/data" \
            -v "${KPFPIPE_DATA:-/data/kpf}/masters:/masters" \
            kpf-drp:latest bash 2>/dev/null
    fi
}

# Main execution
if [ -n "${KPFPIPE_PORT:-}" ]; then
    run_docker "$KPFPIPE_PORT" "Starting Docker container on port $KPFPIPE_PORT..."
else
    run_docker "" "Starting Docker container (no port specified)..."
fi
