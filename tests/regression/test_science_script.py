"""Tests for scripts/processing/science.py: the science-reduction driver.

Cover the driver's own surface: arg parsing and obs_id validation, the
``_cli_task`` argv it fans out, and the ``main`` exit-code contract (nonzero iff
at least one frame failed). The shared fan-out engine (``run_stage``, the
cores-based job sizing) lives in ``_fanout`` and is tested in test_fanout.py.

Unit tests use trivial subprocess stubs -- no real testdata needed.
"""

import sys

import pytest

from scripts.processing import science as _science

_OID1 = "KP.20240405.40113.57"
_OID2 = "KP.20240405.40237.36"


class _FakeConfig:
    """Stand-in for ConfigHandler in the exit-code tests (no real file read)."""

    def __init__(self, path):
        pass

    def get_params(self, keys):
        return {"log_dir": "/l"}


@pytest.fixture(scope="module")
def s():
    """The science driver module (a normal package import)."""
    return _science


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_obs_id_list(self, s):
        ns = s.parse_args(["--obs_id_list", _OID2, _OID1])
        assert ns.obs_id_list == [_OID1, _OID2]  # sorted + deduped

    def test_dedupes_obs_ids(self, s):
        ns = s.parse_args(["--obs_id_list", _OID1, _OID1])
        assert ns.obs_id_list == [_OID1]

    def test_obs_id_list_required(self, s):
        with pytest.raises(SystemExit):
            s.parse_args([])

    @pytest.mark.parametrize(
        "argv",
        [
            ["--obs_id_list", "not-an-obs-id"],  # malformed obs_id
            ["--obs_id_list", "20240405"],  # a datecode is not an obs_id
            ["--obs_id_list", _OID1, "--jobs", "0"],  # jobs below 1
            ["--obs_id_list", _OID1, "--job_timeout", "0"],  # timeout below 1
        ],
    )
    def test_invalid_args_exit(self, s, argv):
        with pytest.raises(SystemExit):
            s.parse_args(argv)

    def test_config_defaults_to_none(self, s):
        assert s.parse_args(["--obs_id_list", _OID1]).config is None

    def test_jobs_unset_resolves_to_cores_default(self, s):
        ns = s.parse_args(["--obs_id_list", _OID1])
        assert ns.jobs == s._default_jobs()

    def test_jobs_override(self, s):
        assert s.parse_args(["--obs_id_list", _OID1, "--jobs", "3"]).jobs == 3

    def test_job_timeout_default_and_override(self, s):
        assert s.parse_args(["--obs_id_list", _OID1]).job_timeout == 600
        assert (
            s.parse_args(["--obs_id_list", _OID1, "--job_timeout", "120"]).job_timeout
            == 120
        )


# ---------------------------------------------------------------------------
# _cli_task
# ---------------------------------------------------------------------------


class TestCliTask:
    def test_builds_science_argv(self, s):
        tag, argv = s._cli_task(_OID1, ["--log_level", "DEBUG"])
        assert tag == _OID1
        assert argv == [
            sys.executable, "-m", "scripts.processing.reduce",
            "--science", "-o", _OID1, "--log_level", "DEBUG",
        ]  # fmt: skip

    def test_config_override_inserts_dash_c(self, s):
        _, argv = s._cli_task(_OID1, [], config="/custom.toml")
        assert argv == [
            sys.executable, "-m", "scripts.processing.reduce",
            "--science", "-c", "/custom.toml", "-o", _OID1,
        ]  # fmt: skip

    def test_no_config_omits_dash_c(self, s):
        _, argv = s._cli_task(_OID1, [])
        assert "-c" not in argv


# ---------------------------------------------------------------------------
# main exit-code contract
# ---------------------------------------------------------------------------


class TestMainExitCode:
    def _patch(self, s, monkeypatch, failed):
        # Skip the runtime setup and the real dir/config resolution + subprocess
        # fan-out; assert only the exit-code contract from run_stage's failure set.
        monkeypatch.setattr(s, "configure_runtime", lambda: None)
        monkeypatch.setattr(s, "shortcut_paths", lambda kind: ("/r.py", "/c.toml"))
        monkeypatch.setattr(s, "ConfigHandler", _FakeConfig)
        monkeypatch.setattr(s, "run_stage", lambda *a, **k: set(failed))

    def test_exits_zero_when_all_reduced(self, s, monkeypatch):
        self._patch(s, monkeypatch, failed=[])
        s.main(["--obs_id_list", _OID1])  # no SystemExit

    def test_exits_nonzero_when_any_failed(self, s, monkeypatch):
        self._patch(s, monkeypatch, failed=[_OID1])
        with pytest.raises(SystemExit) as exc:
            s.main(["--obs_id_list", _OID1])
        assert exc.value.code == 1
