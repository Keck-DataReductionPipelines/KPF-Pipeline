"""Tests for scripts/quality_control/qc.py: the standalone QC runner.

Subprocess smoke tests driving the CLI on synthetic L0 fixtures -- exit codes and
the ISGOOD summary -- plus a fail-loud check that a config missing a required
DATA_DIRS key errors instead of substituting a default.
"""

import tomllib
from pathlib import Path

import pytest

from ._data_models import GOOD_DATES, write_amp_l0
from ._scripts import CHILD_WARNINGS, REPO_ROOT, run_script, write_config

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

_SCRIPT = "scripts/quality_control/qc.py"

# Pointing + DCS target (identical -> TARGOFF ~ 0) so the header-native wmko
# record supplies DiagL0's required TARGOFF with Gaia/SIMBAD disabled.
_TARGET_CARDS = {
    "RA": "12:00:00.00",
    "DEC": "+40:00:00.0",
    "TARGRA": "12:00:00.00",
    "TARGDEC": "+40:00:00.0",
    "TARGFRAM": "FK5",
    "TARGEQUI": 2000.0,
    "TARGPMRA": 0.0,
    "TARGPMDC": 0.0,
    "TARGPLAX": 100.0,
    "TARGEPOC": 2000.0,
    # The wmko record's colour is G-J, so COLOROK needs both magnitudes.
    "GAIAMAG": 6.0,
    "2MASSMAG": 5.0,
}


def _write_l0_fixture(path, *, passing=True, imtype="Object"):
    """Write a minimal L0 FITS fixture at path.

    passing=False injects a negative EXPTIME so EXPTIMOK fails. A non-'Object'
    imtype carries no pointing/DCS target block, so qc.py skips AstroQuery for it.
    """
    cards = {
        "DATE-OBS": "2024-04-05T01:00:37",
        "MJD-OBS": 60405.04,
        # Within tolerance of GOOD_DATES ELAPSED (12.07) so EXPTIMOK passes.
        "EXPTIME": 12.0 if passing else -1.0,
        "OBJECT": "synthetic",
        "IMTYPE": imtype,
        "PROGNAME": None,  # not a QC input here
        **GOOD_DATES,
    }
    if imtype == "Object":
        cards.update(_TARGET_CARDS)
    write_amp_l0(path, namps=4, shape=(10, 10), primary_cards=cards)


def _write_astro_config(path):
    """A minimal config that disables AstroQuery's network lookups.

    Disabling Gaia/SIMBAD keeps these smoke tests offline, so wmko -- the
    header-native row, the only one left -- must also be the permitted astrometric
    base for the merge to succeed and TARGOFF to exist.
    """
    path.write_text(
        "[DATA_DIRS]\n[TRACES]\n"
        "[MODULE_ASTRO_QUERY]\ndo_gaia_query = false\ndo_simbad_query = false\n"
        'astrometry_priority = ["wmko"]\n'
    )
    return path


def _run_qc_script(fixture_path, level="L0", extra_args=None):
    return run_script(
        _SCRIPT, "--input", fixture_path, "--level", level, *(extra_args or ())
    )


class TestQCScript:
    def test_all_passing_exit_0_isgood_pass(self, tmp_path):
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
        # A cal frame stays inspectable: qc.py skips AstroQuery rather than erroring.
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
        missing = tmp_path / "does_not_exist.fits"
        result = _run_qc_script(missing, level="L0")
        assert result.returncode == 2

    def test_no_args_exit_nonzero(self):
        result = run_script(_SCRIPT)
        assert result.returncode != 0


class TestQCScriptConfig:
    """The config-driven input path fails loud on a missing DATA_DIRS key."""

    def test_missing_data_input_key_fails_loud(self, tmp_path):
        # The runner reads KPF_DATA_INPUT with no default, so an absent key must
        # surface as an error rather than silently using a hardcoded path.
        cfg = tmp_path / "cfg.toml"
        write_config(cfg, {"KPF_SCIENCE_OUTPUT": str(tmp_path)})  # no KPF_DATA_INPUT

        result = run_script(
            _SCRIPT,
            "--obs_id",
            "KP.20240405.00001.00",
            "--level",
            "L0",
            "--config",
            cfg,
        )

        assert result.returncode != 0
        assert "KPF_DATA_INPUT" in result.stderr, (
            f"Expected 'KPF_DATA_INPUT' in stderr:\n{result.stderr}"
        )


class TestChildWarningParity:
    """``_scripts.CHILD_WARNINGS`` mirrors pyproject's filterwarnings.

    pytest's filterwarnings never crosses a subprocess boundary, so a rule added
    to pyproject.toml silently stops applying to every CLI test unless it is
    hand-mirrored into PYTHONWARNINGS. Nothing else asserts the mirror is complete.
    """

    @staticmethod
    def _parent_rules():
        pyproject = Path(REPO_ROOT) / "pyproject.toml"
        config = tomllib.loads(pyproject.read_text())
        return config["tool"]["pytest"]["ini_options"]["filterwarnings"]

    def test_pyproject_rules_are_all_mirrored(self):
        child = CHILD_WARNINGS.split(",")
        for rule in self._parent_rules():
            action, _, rest = rule.partition(":")
            message = rest.split(":")[0]
            # PYTHONWARNINGS resolves categories at interpreter startup, before
            # astropy is importable, so a mirrored rule may truncate the message
            # and drop the category. Require only that some child rule shares the
            # action and carries a message prefix of the parent's.
            assert any(
                c.partition(":")[0] == action
                and message.startswith(c.partition(":")[2].split(":")[0])
                for c in child
            ), f"pyproject filterwarnings rule {rule!r} is not in CHILD_WARNINGS"
