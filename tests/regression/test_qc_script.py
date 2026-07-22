"""Tests for scripts/quality_control/qc.py: the standalone QC runner.

Subprocess smoke tests that drive the CLI end-to-end on synthetic L0 fixtures --
exit codes and the ISGOOD summary (TestQCScript) -- plus a fail-loud check that a
config missing a required DATA_DIRS key errors instead of substituting a default
(TestQCScriptConfig).
"""

import os
import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Self-consistent raw exposure times so DATTIMOK passes (END-BEG == ELAPSED).
_GOOD_DATES = {
    "DATE-BEG": "2024-09-23T09:12:09.484",
    "DATE-MID": "2024-09-23T09:12:15.519",
    "DATE-END": "2024-09-23T09:12:21.554",
    "ELAPSED": 12.07,
}


def _write_l0_fixture(path, *, passing=True, imtype="Object"):
    """Write a minimal L0 FITS fixture at path.

    passing=True  → all QCL0 checks pass (valid header keywords, EXPTIME finite
                    and consistent with ELAPSED, amps present).
    passing=False → inject a failure (negative EXPTIME so EXPTIMOK fails).
    imtype        → PRIMARY IMTYPE; a non-'Object' (calibration) frame carries no
                    pointing/DCS target block, so qc.py skips AstroQuery for it.
    """
    primary = fits.PrimaryHDU()
    primary.header["DATE-OBS"] = "2024-04-05T01:00:37"
    primary.header["MJD-OBS"] = 60405.04
    # Requested EXPTIME within tolerance of the elapsed time (_GOOD_DATES
    # ELAPSED = 12.07) so EXPTIMOK's ELAPSED-consistency check passes.
    primary.header["EXPTIME"] = 12.0 if passing else -1.0
    primary.header["OBJECT"] = "synthetic"
    primary.header["OFNAME"] = os.path.basename(path)
    primary.header["IMTYPE"] = imtype
    for k, v in _GOOD_DATES.items():
        primary.header[k] = v

    if imtype == "Object":
        # Pointing + DCS target (identical -> TARGOFF ~ 0) so AstroQuery resolves the
        # wmko record and DiagL0's required TARGOFF is available offline; the external
        # Gaia/SIMBAD lookups are disabled via config (see _write_astro_config).
        primary.header["RA"] = "12:00:00.00"
        primary.header["DEC"] = "+40:00:00.0"
        primary.header["TARGRA"] = "12:00:00.00"
        primary.header["TARGDEC"] = "+40:00:00.0"
        primary.header["TARGFRAM"] = "FK5"
        primary.header["TARGEQUI"] = 2000.0
        primary.header["TARGPMRA"] = 0.0
        primary.header["TARGPMDC"] = 0.0
        primary.header["TARGPLAX"] = 100.0
        primary.header["TARGEPOC"] = 2000.0

    hdus = [primary]
    for chip in ["GREEN", "RED"]:
        for amp in range(1, 5):
            data = np.ones((10, 10), dtype=np.float32)
            hdus.append(fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}"))

    fits.HDUList(hdus).writeto(path, overwrite=True)


def _write_config(path, data_dirs):
    """Write a TOML config with a [DATA_DIRS] section from ``data_dirs``."""
    lines = ["[DATA_DIRS]"]
    lines += [f'{key} = "{value}"' for key, value in data_dirs.items()]
    path.write_text("\n".join(lines) + "\n")


def _write_astro_config(path):
    """A minimal config that disables AstroQuery's network lookups.

    qc.py runs AstroQuery for L0; disabling Gaia/SIMBAD keeps these smoke tests
    offline while the header-native wmko record still yields TARGOFF. get_params
    requires the three sections to exist (empty is fine)."""
    path.write_text(
        "[DATA_DIRS]\n[TRACES]\n"
        "[MODULE_ASTRO_QUERY]\ndo_gaia_query = false\ndo_simbad_query = false\n"
    )
    return path


def _run_qc_script(fixture_path, level="L0", extra_args=None):
    """Run scripts/quality_control/qc.py via subprocess, return the CompletedProcess."""
    cmd = [
        sys.executable,
        "scripts/quality_control/qc.py",
        "--input",
        str(fixture_path),
        "--level",
        level,
    ]
    if extra_args:
        cmd.extend(extra_args)
    env = {**os.environ, "PYTHONPATH": _REPO_ROOT}
    return subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True)


class TestQCScript:
    """Smoke tests for scripts/quality_control/qc.py via subprocess."""

    def test_all_passing_exit_0_isgood_pass(self, tmp_path):
        """All-good L0 → exit code 0, stdout contains 'ISGOOD: PASS'."""
        fixture = tmp_path / "KP.20240405.00001.00.fits"
        _write_l0_fixture(str(fixture), passing=True)
        cfg = _write_astro_config(tmp_path / "astro.toml")

        result = _run_qc_script(fixture, level="L0", extra_args=["--config", str(cfg)])

        assert result.returncode == 0, (
            f"Expected exit 0, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "ISGOOD: PASS" in result.stdout, (
            f"Expected 'ISGOOD: PASS' in stdout:\n{result.stdout}"
        )

    def test_failure_injected_exit_1_isgood_fail(self, tmp_path):
        """L0 with negative EXPTIME → exit code 1, stdout contains 'ISGOOD: FAIL'."""
        fixture = tmp_path / "KP.20240405.00002.00.fits"
        _write_l0_fixture(str(fixture), passing=False)
        cfg = _write_astro_config(tmp_path / "astro.toml")

        result = _run_qc_script(fixture, level="L0", extra_args=["--config", str(cfg)])

        assert result.returncode == 1, (
            f"Expected exit 1, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "ISGOOD: FAIL" in result.stdout, (
            f"Expected 'ISGOOD: FAIL' in stdout:\n{result.stdout}"
        )

    def test_calibration_frame_skips_astroquery_no_exit_2(self, tmp_path):
        """A calibration L0 (IMTYPE != 'Object') stays inspectable: qc.py skips
        AstroQuery rather than erroring, so it never exits 2 on a cal frame."""
        fixture = tmp_path / "KP.20240405.00005.00.fits"
        _write_l0_fixture(str(fixture), passing=True, imtype="Bias")
        cfg = _write_astro_config(tmp_path / "astro.toml")

        result = _run_qc_script(fixture, level="L0", extra_args=["--config", str(cfg)])

        assert result.returncode != 2, (
            f"Calibration L0 should not exit 2\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "Skipping AstroQuery" in result.stdout, (
            f"Expected AstroQuery-skip note in stdout:\n{result.stdout}"
        )

    def test_missing_file_exit_2(self, tmp_path):
        """Non-existent file → exit code 2."""
        missing = tmp_path / "does_not_exist.fits"
        result = _run_qc_script(missing, level="L0")
        assert result.returncode == 2

    def test_no_args_exit_nonzero(self):
        """No args → argparse error → non-zero exit."""
        env = {**os.environ, "PYTHONPATH": _REPO_ROOT}
        result = subprocess.run(
            [sys.executable, "scripts/quality_control/qc.py"],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


class TestQCScriptConfig:
    """The config-driven input path fails loud on a missing DATA_DIRS key."""

    def test_missing_data_input_key_fails_loud(self, tmp_path):
        """--obs_id + a config without KPF_DATA_INPUT → nonzero exit naming the key.

        The runner reads params["KPF_DATA_INPUT"] directly (no default), so an
        absent key surfaces as an error rather than silently using a hardcoded path.
        """
        cfg = tmp_path / "cfg.toml"
        _write_config(cfg, {"KPF_SCIENCE_OUTPUT": str(tmp_path)})  # no KPF_DATA_INPUT
        env = {**os.environ, "PYTHONPATH": _REPO_ROOT}
        result = subprocess.run(
            [
                sys.executable,
                "scripts/quality_control/qc.py",
                "--obs_id",
                "KP.20240405.00001.00",
                "--level",
                "L0",
                "--config",
                str(cfg),
            ],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "KPF_DATA_INPUT" in result.stderr, (
            f"Expected 'KPF_DATA_INPUT' in stderr:\n{result.stderr}"
        )
