#!/usr/bin/env python3
"""
Helper script to manage the local A/B-test Postgres stack via Docker Compose.

Usage:
    python scripts/abtest_pg.py start   # launch postgres container
    python scripts/abtest_pg.py stop    # stop containers and remove network
    python scripts/abtest_pg.py status  # show compose service state
    python scripts/abtest_pg.py shell   # open an interactive DRP shell (equivalent to make docker)
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "docker-compose.abtest.yml"
COMPOSE_ENV_FILE = REPO_ROOT / "docker/abtest-postgres/compose.env"
DEFAULT_HOST_PORT = 6125


def run_compose(args: list[str], env: dict[str, str]) -> int:
    cmd = [
        "docker",
        "compose",
        "--env-file",
        str(COMPOSE_ENV_FILE),
        "-f",
        str(COMPOSE_FILE),
        *args,
    ]
    return subprocess.run(cmd, env=env, check=False).returncode


@lru_cache()
def compose_settings() -> dict[str, str]:
    settings: dict[str, str] = {}
    if not COMPOSE_ENV_FILE.exists():
        return settings
    for raw_line in COMPOSE_ENV_FILE.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        settings[key.strip()] = value.strip()
    return settings


def ensure_compose_file() -> None:
    if not COMPOSE_FILE.exists():
        print(f"Compose file not found at {COMPOSE_FILE}", file=sys.stderr)
        sys.exit(2)
    if not COMPOSE_ENV_FILE.exists():
        print(f"Compose env file not found at {COMPOSE_ENV_FILE}", file=sys.stderr)
        sys.exit(2)


def make_env() -> dict[str, str]:
    env = os.environ.copy()
    # Compose resolves ${PWD} at load time; ensure it's set to the repo root.
    env.setdefault("PWD", str(REPO_ROOT))
    return env


def resolve_host_port() -> int:
    raw = os.environ.get("ABTEST_PG_HOST_PORT")
    if raw is None:
        raw = compose_settings().get("ABTEST_PG_HOST_PORT")
    if raw is None or not raw:
        return DEFAULT_HOST_PORT
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid ABTEST_PG_HOST_PORT value '{raw}'") from exc


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def do_start(env: dict[str, str]) -> None:
    print("Starting local Postgres for A/B testing...")
    host_port = resolve_host_port()
    if not port_available(host_port):
        print(
            f"Error: host port {host_port} is already in use. "
            "Stop the conflicting service or override the helper port via "
            "`ABTEST_PG_HOST_PORT=<port> python scripts/abtest_pg.py start` "
            "or by editing docker/abtest-postgres/compose.env.",
            file=sys.stderr,
        )
        sys.exit(1)
    rc = run_compose(["up", "-d", "abtest-postgres"], env)
    if rc != 0:
        sys.exit(rc)

    # Ensure helper roles have SUPERUSER/CREATEDB inside the helper container.
    grant_cmd = [
        "docker",
        "compose",
        "--env-file",
        str(COMPOSE_ENV_FILE),
        "-f",
        str(COMPOSE_FILE),
        "exec",
        "-T",
        "abtest-postgres",
        "psql",
        "-U",
        os.environ.get("ABTEST_DB_SUPERUSER", "postgres"),
        "-d",
        "postgres",
        "-c",
        "ALTER ROLE kpfopsuser SUPERUSER CREATEDB; "
        "ALTER ROLE timeseriesopsuser SUPERUSER CREATEDB;",
    ]
    grant_rc = subprocess.run(grant_cmd, env=env).returncode
    if grant_rc == 0:
        print("Helper roles updated with SUPERUSER/CREATEDB.")
    else:
        print("Warning: failed to elevate helper roles; check container logs.")
    print(
        "\nPostgres is up. Default connection info:\n"
        "  host: abtest-postgres (inside compose network) or localhost\n"
        f"  port: 5432 inside compose (host port {host_port})\n"
        "  DBNAME: kpfopsdb | TSDBNAME: timeseriesopsdb\n"
        "  DBUSER: kpfopsuser | TSDBUSER: timeseriesopsuser\n"
    )
    print("Use `python scripts/abtest_pg.py shell` for an interactive DRP shell.")


def do_stop(env: dict[str, str]) -> None:
    print("Stopping compose stack…")
    rc = run_compose(["down"], env)
    if rc != 0:
        sys.exit(rc)


def do_status(env: dict[str, str]) -> None:
    rc = run_compose(["ps"], env)
    if rc != 0:
        sys.exit(rc)


def do_shell(env: dict[str, str], shell_args: list[str]) -> None:
    cmd = [
        "docker",
        "compose",
        "--env-file",
        str(COMPOSE_ENV_FILE),
        "-f",
        str(COMPOSE_FILE),
        "run",
        "--rm",
        "--service-ports",
        "drp",
    ]
    if shell_args:
        cmd.extend(shell_args)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        sys.exit(rc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("start", help="Start the Postgres container")
    sub.add_parser("stop", help="Stop all compose services")
    sub.add_parser("status", help="Show compose service status")
    shell = sub.add_parser("shell", help="Launch an interactive DRP shell")
    shell.add_argument("shell_args", nargs=argparse.REMAINDER, help="Optional command to run inside the DRP container")
    return parser.parse_args()


def main() -> None:
    ensure_compose_file()
    env = make_env()
    args = parse_args()

    if args.command == "start":
        do_start(env)
    elif args.command == "stop":
        do_stop(env)
    elif args.command == "status":
        do_status(env)
    elif args.command == "shell":
        do_shell(env, args.shell_args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

