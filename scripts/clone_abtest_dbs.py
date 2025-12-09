#!/usr/bin/env python3
"""
Clone the operational Postgres databases for A/B testing.

The script duplicates the primary KPF pipeline database and the time-series
database, tagging each clone with the current git hash. It also emits a
dotenv-compatible file (e.g., `.env.abtest.<hash>`) that can be sourced prior
to running `make docker` so containers point at the cloned databases.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_ENV_FILE = REPO_ROOT / ".env"

DEFAULTS = {
    "primary": {"server": "127.0.0.1", "port": 6125, "name": "kpfopsdb"},
    "timeseries": {"server": "127.0.0.1", "port": 6127, "name": "timeseriesopsdb"},
}

REQUIRED_BINARIES = ("psql", "pg_dump")


ABTEST_COMPOSE_ENV = REPO_ROOT / "docker/abtest-postgres/compose.env"
ABTEST_HOST_FALLBACK = "127.0.0.1"


@dataclass
class DbConfig:
    label: str
    server: str
    port: int
    name: str
    user: str
    password: str

    def clone_name(self, tag: str) -> str:
        return f"{self.name}_abtest_{tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone the two Postgres databases for A/B testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-env-file",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional KEY=VALUE env file(s) that load before the main --env-file list. "
            "Useful for injecting supplemental credentials such as superuser accounts."
        ),
    )
    parser.add_argument(
        "--env-file",
        action="append",
        type=Path,
        default=[],
        help="Optional file(s) with KEY=VALUE lines to preload into the environment.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Override the output env filename (default uses the git hash tag).",
    )
    parser.add_argument(
        "--tag",
        help="Override the git hash tag embedded in the cloned DB names.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop any existing clone databases before recreating them.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Drop the cloned databases corresponding to the resolved tag and exit.",
    )
    parser.add_argument(
        "--show-privileges",
        action="store_true",
        help="Print table/sequence privileges for the source DB roles and exit.",
    )
    return parser.parse_args()


def ensure_binaries_present() -> None:
    missing = [binary for binary in REQUIRED_BINARIES if shutil.which(binary) is None]
    if missing:
        print(
            "Error: missing required Postgres client binaries: "
            f"{', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)


def load_env_file(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def load_compose_defaults(environ: Dict[str, str]) -> None:
    if not ABTEST_COMPOSE_ENV.exists():
        return
    try:
        defaults = load_env_file(ABTEST_COMPOSE_ENV)
    except FileNotFoundError:
        return
    for key, value in defaults.items():
        environ.setdefault(key, value)

    # Ensure helper hostnames resolve from the host by default.
    helper_host = environ.get("ABTEST_HOST_OVERRIDE", ABTEST_HOST_FALLBACK)
    for key in ("ABTEST_DB_SERVER", "ABTEST_TSDB_SERVER"):
        if not environ.get(key) or environ[key] == "abtest-postgres":
            environ[key] = helper_host


def load_root_env(environ: Dict[str, str]) -> None:
    if not ROOT_ENV_FILE.exists():
        return
    try:
        defaults = load_env_file(ROOT_ENV_FILE)
    except FileNotFoundError:
        return
    for key, value in defaults.items():
        environ.setdefault(key, value)


def get_short_git_hash(tag_override: str | None = None) -> str:
    if tag_override:
        return tag_override
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        from datetime import datetime

        fallback = datetime.utcnow().strftime("manual_%Y%m%d%H%M%S")
        print(
            "Warning: git hash unavailable, using fallback tag "
            f"'{fallback}'.",
            file=sys.stderr,
        )
        return fallback


def select_env_value(
    environ: Dict[str, str], candidates: Iterable[str], default: str | None = None
) -> str | None:
    for key in candidates:
        if environ.get(key):
            return environ[key]
    return default


def read_db_config(
    environ: Dict[str, str],
    label: str,
    prefixes: Tuple[str, str],
    defaults: Dict[str, str],
    require_credentials: bool = True,
) -> DbConfig:
    prefixed, generic = prefixes
    server = select_env_value(
        environ,
        (f"{prefixed}SERVER", f"{generic}SERVER"),
        defaults["server"],
    )
    port_value = select_env_value(
        environ,
        (f"{prefixed}PORT", f"{generic}PORT"),
        str(defaults["port"]),
    )
    name = select_env_value(
        environ,
        (f"{prefixed}NAME", f"{generic}NAME"),
        defaults["name"],
    )
    user = select_env_value(environ, (f"{prefixed}USER", f"{generic}USER"))
    password = select_env_value(environ, (f"{prefixed}PASS", f"{generic}PASS"))

    if require_credentials:
        missing = []
        if not user:
            missing.append(f"{prefixed}USER/{generic}USER")
        if not password:
            missing.append(f"{prefixed}PASS/{generic}PASS")
        if missing:
            raise RuntimeError(
                f"Missing required environment variables for {label} database: "
                + ", ".join(missing)
            )

    try:
        port_int = int(port_value) if port_value is not None else defaults["port"]
    except ValueError as exc:
        raise RuntimeError(f"Invalid port value '{port_value}' for {label} database.") from exc

    return DbConfig(
        label=label,
        server=server or defaults["server"],
        port=port_int,
        name=name or defaults["name"],
        user=user,
        password=password,
    )


def read_dest_config(
    environ: Dict[str, str],
    label: str,
    prefixes: Tuple[str, str],
    fallback: DbConfig,
) -> DbConfig:
    prefixed, generic = prefixes
    defaults = {
        "server": fallback.server,
        "port": fallback.port,
        "name": fallback.name,
    }
    # Destination user/pass fall back to source credentials.
    dest = read_db_config(
        environ,
        label,
        (prefixed, generic),
        defaults,
        require_credentials=False,
    )
    if not dest.user:
        dest.user = fallback.user
    if not dest.password:
        dest.password = fallback.password
    return dest


def pg_env(password: str) -> Dict[str, str]:
    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password
    return env


def base_psql_cmd(config: DbConfig, database: str) -> List[str]:
    return [
        "psql",
        "--host",
        config.server,
        "--port",
        str(config.port),
        "--username",
        config.user,
        "--dbname",
        database,
        "-v",
        "ON_ERROR_STOP=1",
    ]


def quote_ident(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def escape_literal(value: str) -> str:
    return value.replace("'", "''")


TABLE_PRIV_SQL = """
    SELECT table_schema, table_name, privilege_type
    FROM information_schema.table_privileges
    WHERE grantee = current_user
    ORDER BY table_schema, table_name, privilege_type
"""

SEQUENCE_PRIV_SQL = """
    WITH seqs AS (
        SELECT n.nspname AS sequence_schema,
               c.relname AS sequence_name,
               format('%I.%I', n.nspname, c.relname) AS qualified_name
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind = 'S'
    ),
    perms AS (
        SELECT sequence_schema,
               sequence_name,
               privilege
        FROM seqs,
        LATERAL (
            VALUES
                ('SELECT', has_sequence_privilege(current_user, qualified_name, 'SELECT')),
                ('USAGE',  has_sequence_privilege(current_user, qualified_name, 'USAGE')),
                ('UPDATE', has_sequence_privilege(current_user, qualified_name, 'UPDATE'))
        ) AS p(privilege, granted)
        WHERE granted
    )
    SELECT sequence_schema, sequence_name, privilege
    FROM perms
    ORDER BY sequence_schema, sequence_name, privilege
"""


def print_privileges(config: DbConfig) -> None:
    print(f"\n[{config.label}] privileges for user '{config.user}' on database '{config.name}':")
    sections = (
        ("tables/views", TABLE_PRIV_SQL),
        ("sequences", SEQUENCE_PRIV_SQL),
    )
    for section, sql in sections:
        cmd = base_psql_cmd(config, config.name) + ["-F", "|", "-tA", "-c", sql]
        completed = subprocess.run(
            cmd,
            env=pg_env(config.password),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            print(
                f"  (failed to enumerate {section}: {completed.stderr.strip() or 'unknown error'})"
            )
            continue
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            print(f"  (no explicit {section} privileges granted)")
            continue
        for line in lines:
            parts = line.split("|")
            if len(parts) == 3:
                schema, name, privilege = parts
                print(f"  {section}: {schema}.{name} -> {privilege}")
            else:
                print(f"  {section}: {line}")


def database_exists(config: DbConfig, dbname: str) -> bool:
    cmd = base_psql_cmd(config, "postgres") + [
        "-tA",
        "-c",
        f"SELECT 1 FROM pg_database WHERE datname = '{escape_literal(dbname)}';",
    ]
    completed = subprocess.run(
        cmd,
        env=pg_env(config.password),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Failed to check for existing database '{dbname}': {completed.stderr.strip()}"
        )
    return completed.stdout.strip() == "1"


def drop_database(config: DbConfig, dbname: str) -> None:
    cmd = base_psql_cmd(config, "postgres") + [
        "-c",
        f"DROP DATABASE IF EXISTS {quote_ident(dbname)};",
    ]
    subprocess.run(cmd, env=pg_env(config.password), check=True)


def create_database(config: DbConfig, dbname: str) -> None:
    sql = (
        f"CREATE DATABASE {quote_ident(dbname)} "
        f"WITH OWNER {quote_ident(config.user)} ENCODING 'UTF8';"
    )
    cmd = base_psql_cmd(config, "postgres") + ["-c", sql]
    try:
        subprocess.run(cmd, env=pg_env(config.password), check=True)
    except subprocess.CalledProcessError as exc:
        hint = (
            "Creating databases requires superuser privileges. If the production "
            "server refuses CREATE DATABASE, launch the local helper first:\n"
            "  python scripts/abtest_pg.py start\n"
            "Then point DB* variables at the helper (host abtest-postgres, port 5432) "
            "and rerun this script."
        )
        raise RuntimeError(f"Failed to create database '{dbname}': permission denied?\n{hint}") from exc


def restore_database(source_config: DbConfig, dest_config: DbConfig, source: str, target: str) -> None:
    dump_cmd = [
        "pg_dump",
        "--host",
        source_config.server,
        "--port",
        str(source_config.port),
        "--username",
        source_config.user,
        "--dbname",
        source,
        "--no-owner",
        "--no-privileges",
        "--no-tablespaces",
    ]
    restore_cmd = base_psql_cmd(dest_config, target)

    with subprocess.Popen(
        dump_cmd,
        env=pg_env(source_config.password),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    ) as dump_proc:
        assert dump_proc.stdout is not None
        restore = subprocess.run(
            restore_cmd,
            env=pg_env(dest_config.password),
            stdin=dump_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _, dump_stderr = dump_proc.communicate()
        if dump_proc.returncode != 0:
            print("\npg_dump failed; current privileges for the source role:")
            print_privileges(source_config)
            raise RuntimeError(
                "pg_dump failed for "
                f"{source}: {dump_stderr.decode(errors='ignore').strip() if dump_stderr else 'unknown error'}"
            )
        if restore.returncode != 0:
            raise RuntimeError(
                f"psql restore failed for {target}: {restore.stderr.strip()}"
            )


def clone_database(source: DbConfig, dest: DbConfig, tag: str, force: bool) -> str:
    new_name = dest.clone_name(tag)
    print(f"[{source.label}] cloning '{source.name}' -> '{new_name}' (dest host {dest.server}:{dest.port})")
    exists = database_exists(dest, new_name)
    if exists and not force:
        raise RuntimeError(
            f"Database '{new_name}' already exists on {dest.server}:{dest.port}. "
            "Use --force to drop and recreate it."
        )
    if exists and force:
        print(f"[{source.label}] dropping existing '{new_name}' (force enabled)")
        drop_database(dest, new_name)

    create_database(dest, new_name)
    restore_database(source, dest, source.name, new_name)
    print(f"[{source.label}] clone complete")
    return new_name


def cleanup_databases(configs: Iterable[DbConfig], tag: str) -> None:
    for config in configs:
        clone_name = config.clone_name(tag)
        print(f"[{config.label}] removing clone '{clone_name}' (tag {tag})")
        if database_exists(config, clone_name):
            drop_database(config, clone_name)
            print(f"[{config.label}] dropped '{clone_name}'")
        else:
            print(f"[{config.label}] clone '{clone_name}' not found; nothing to do")


def write_env_file(
    path: Path,
    tag: str,
    primary_source: DbConfig,
    primary_dest: DbConfig,
    primary_clone: str,
    ts_source: DbConfig,
    ts_dest: DbConfig,
    ts_clone: str,
) -> None:
    entries = [
        ("# Generated by scripts/clone_abtest_dbs.py", None),
        ("# Tag: " + tag, None),
        (f"# Source primary: {primary_source.server}:{primary_source.port} (user={primary_source.user})", None),
        (f"# Source timeseries: {ts_source.server}:{ts_source.port} (user={ts_source.user})", None),
        ("ABTEST_DB_TAG", tag),
        ("KPFPIPE_DB_SERVER", primary_dest.server),
        ("DBSERVER", primary_dest.server),
        ("KPFPIPE_DB_PORT", str(primary_dest.port)),
        ("DBPORT", str(primary_dest.port)),
        ("KPFPIPE_DB_NAME", primary_clone),
        ("DBNAME", primary_clone),
        ("KPFPIPE_DB_USER", primary_dest.user),
        ("DBUSER", primary_dest.user),
        ("KPFPIPE_DB_PASS", primary_dest.password),
        ("DBPASS", primary_dest.password),
        ("KPFPIPE_TSDB_SERVER", ts_dest.server),
        ("TSDBSERVER", ts_dest.server),
        ("KPFPIPE_TSDB_PORT", str(ts_dest.port)),
        ("TSDBPORT", str(ts_dest.port)),
        ("KPFPIPE_TSDB_NAME", ts_clone),
        ("TSDBNAME", ts_clone),
        ("KPFPIPE_TSDB_USER", ts_dest.user),
        ("TSDBUSER", ts_dest.user),
        ("KPFPIPE_TSDB_PASS", ts_dest.password),
        ("TSDBPASS", ts_dest.password),
    ]
    lines: List[str] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    for key, value in entries:
        if value is None:
            lines.append(key)
        else:
            lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote environment overrides to {path}")


def main() -> None:
    args = parse_args()
    ensure_binaries_present()

    environ = os.environ.copy()
    load_root_env(environ)
    load_compose_defaults(environ)
    for env_file in args.source_env_file:
        environ.update(load_env_file(env_file))
    for env_file in args.env_file:
        environ.update(load_env_file(env_file))

    tag = get_short_git_hash(args.tag)
    primary_source = read_db_config(environ, "primary", ("KPFPIPE_DB_", "DB"), DEFAULTS["primary"])
    ts_source = read_db_config(environ, "timeseries", ("KPFPIPE_TSDB_", "TSDB"), DEFAULTS["timeseries"])
    primary_dest = read_dest_config(
        environ,
        "primary (destination)",
        ("ABTEST_DB_", "ABTEST_DB_"),
        primary_source,
    )
    ts_dest = read_dest_config(
        environ,
        "timeseries (destination)",
        ("ABTEST_TSDB_", "ABTEST_TSDB_"),
        ts_source,
    )

    if args.cleanup:
        try:
            cleanup_databases((primary_dest, ts_dest), tag)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        env_path = args.out or Path.cwd() / f".env.abtest.{tag}"
        if env_path.exists():
            env_path.unlink()
            print(f"Removed env file {env_path}")
        return

    if args.show_privileges:
        print_privileges(primary_source)
        print_privileges(ts_source)

    try:
        primary_clone = clone_database(primary_source, primary_dest, tag, args.force)
        ts_clone = clone_database(ts_source, ts_dest, tag, args.force)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    env_path = args.out or Path.cwd() / f".env.abtest.{tag}"
    write_env_file(
        env_path,
        tag,
        primary_source,
        primary_dest,
        primary_clone,
        ts_source,
        ts_dest,
        ts_clone,
    )

    print("\nNext steps:")
    print(f"  1. source {env_path}")
    print("  2. make docker  # or make docker_masters")
    print("  3. docker containers will point at the cloned databases above")


if __name__ == "__main__":
    main()

