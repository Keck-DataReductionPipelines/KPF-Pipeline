"""Shared helpers for the scripts/CLI-layer tests (not a test module).

Everything here serves ``@pytest.mark.cli`` modules: launching a script in a child
process under the suite's warning policy, standing in for ConfigHandler, and
staging a synthetic night tree on disk. Builders only -- no assertions.
"""

import os
import subprocess
import sys
from pathlib import Path

from astropy.io import fits

# Repo root: tests/regression/_scripts.py -> tests/regression -> tests -> repo.
REPO_ROOT = str(Path(__file__).resolve().parents[2])

# pytest's filterwarnings does not reach a child process, so mirror the pyproject
# rules in: a child warning then becomes the non-zero exit the tests check for.
# PYTHONWARNINGS resolves categories at interpreter startup, before astropy is
# importable, so the astropy rule is carried by its message prefix alone.
# TestChildWarningParity in test_qc_script.py asserts this stays in sync.
CHILD_WARNINGS = ",".join(
    (
        "error",
        "ignore:Card is too long",
        "default::ResourceWarning",
    )
)

# A child that reaches the network (AstroQuery, astropy's IERS auto-download) can
# block forever, so cap every run: a hang becomes a loud TimeoutExpired instead of
# a stalled suite. Generous enough for a slow-but-working run.
CHILD_TIMEOUT = 120


def run_script(script, *argv, timeout=CHILD_TIMEOUT):
    """Run ``script`` (a repo-relative path) in a child process with ``argv``.

    Takes arbitrary argv rather than named flags so the no-args and bad-config
    cases use the same launcher as the happy path. Returns the CompletedProcess;
    the caller asserts on returncode/stdout/stderr.
    """
    env = {**os.environ, "PYTHONPATH": REPO_ROOT, "PYTHONWARNINGS": CHILD_WARNINGS}
    return subprocess.run(
        [sys.executable, script, *map(str, argv)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class _FakeConfig:
    """Stand-in for ConfigHandler in the exit-code tests (no real file read)."""

    def __init__(self, path):
        pass

    def get_params(self, keys):
        return {"KPF_DATA_INPUT": "/in", "log_dir": "/l"}


class _NoLogDirConfig(_FakeConfig):
    """A config with a resolvable data input but no configured log_dir."""

    def get_params(self, keys):
        return {"KPF_DATA_INPUT": "/in"}


def write_config(path, data_dirs):
    """Write a minimal ``[DATA_DIRS]`` TOML config at ``path``."""
    lines = ["[DATA_DIRS]"]
    lines += [f'{key} = "{value}"' for key, value in data_dirs.items()]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_l0_tree(
    data_input,
    datecode,
    seconds,
    *,
    obj="10700",
    imtype="Object",
    targname=None,
    exptime=60.0,
    elapsed=60.0,
):
    """Write one PRIMARY-only L0 frame under ``{data_input}/L0/{datecode}``.

    Enough header for a mini-db header scan and nothing more; returns the obs_id.
    ``targname`` defaults to ``obj`` (calibration frames set them independently).
    """
    l0_dir = Path(data_input) / "L0" / datecode
    l0_dir.mkdir(parents=True, exist_ok=True)
    obs_id = f"KP.{datecode}.{seconds:05d}.00"
    header = fits.Header(
        {
            "OBJECT": obj,
            "IMTYPE": imtype,
            "TARGNAME": obj if targname is None else targname,
            "EXPTIME": exptime,
            "ELAPSED": elapsed,
        }
    )
    fits.PrimaryHDU(header=header).writeto(l0_dir / f"{obs_id}.fits")
    return obs_id
