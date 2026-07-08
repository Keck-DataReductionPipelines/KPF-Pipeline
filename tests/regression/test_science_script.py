"""Tests for scripts/processing/science.py: the science-reduction driver.

science.py is a standalone script (not an importable package module), so it is
loaded by path like the recipe/masters tests. These cover the pure helpers (arg
parsing and obs_id validation, the task/job-sizing helpers), the fail-soft
``run_stage`` dispatch, and the ``main`` exit-code contract (nonzero iff at least
one frame failed).

Unit tests use trivial subprocess stubs -- no real testdata needed.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCIENCE_PATH = (
    Path(__file__).parent.parent.parent / "scripts" / "processing" / "science.py"
)


@pytest.fixture(scope="module")
def s():
    """Load science.py by path (it is a script, not a package module)."""
    spec = importlib.util.spec_from_file_location("science", _SCIENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OK = [sys.executable, "-c", "pass"]
_FAIL = [sys.executable, "-c", "import sys; sys.exit(1)"]
_OID1 = "KP.20240405.40113.57"
_OID2 = "KP.20240405.40237.36"


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
# small helpers
# ---------------------------------------------------------------------------


class TestCliTask:
    def test_builds_science_argv(self, s):
        tag, argv = s._cli_task(_OID1, ["--log_level", "DEBUG"])
        assert tag == _OID1
        assert argv == [
            sys.executable, "-m", "tools.cli",
            "--science", "-o", _OID1, "--log_level", "DEBUG",
        ]  # fmt: skip

    def test_config_override_inserts_dash_c(self, s):
        _, argv = s._cli_task(_OID1, [], config="/custom.toml")
        assert argv == [
            sys.executable, "-m", "tools.cli",
            "--science", "-c", "/custom.toml", "-o", _OID1,
        ]  # fmt: skip

    def test_no_config_omits_dash_c(self, s):
        _, argv = s._cli_task(_OID1, [])
        assert "-c" not in argv


class TestDefaultJobs:
    @pytest.mark.parametrize(
        "cpus,expected",
        [(None, 1), (1, 1), (8, 8), (64, 16), (100, 25)],
    )
    def test_cap(self, s, monkeypatch, cpus, expected):
        monkeypatch.setattr(s.os, "cpu_count", lambda: cpus)
        assert s._default_jobs() == expected


# ---------------------------------------------------------------------------
# run_stage (fail-soft)
# ---------------------------------------------------------------------------


class TestRunStage:
    @pytest.fixture(autouse=True)
    def _no_throttle(self, s, monkeypatch):
        # The 1s launch throttle is not what these dispatch tests verify; zero it
        # so they stay fast and free of thread-timing flakiness.
        monkeypatch.setattr(s, "_LAUNCH_INTERVAL", 0.0)

    def test_empty_returns_empty_set(self, s, tmp_path):
        assert s.run_stage("science", [], 2, str(tmp_path)) == set()

    def test_all_succeed_returns_empty_set(self, s, tmp_path):
        tasks = [(_OID1, _OK), (_OID2, _OK)]
        assert s.run_stage("science", tasks, 2, str(tmp_path)) == set()

    def test_failed_canary_still_fans_out_and_is_reported(self, s, tmp_path, capsys):
        # Fail-soft: a bad canary frame does not stop the rest; it is collected.
        tasks = [(_OID1, _FAIL), (_OID2, _OK)]
        failed = s.run_stage("science", tasks, 2, str(tmp_path))
        assert failed == {_OID1}
        assert "continuing" in capsys.readouterr().out

    def test_collects_all_failures(self, s, tmp_path):
        tasks = [(_OID1, _OK), (_OID2, _FAIL), ("KP.20240405.50000.01", _FAIL)]
        assert s.run_stage("science", tasks, 2, str(tmp_path)) == {
            _OID2,
            "KP.20240405.50000.01",
        }


# ---------------------------------------------------------------------------
# main exit-code contract
# ---------------------------------------------------------------------------


class TestMainExitCode:
    def _patch(self, s, monkeypatch, failed):
        # Skip the real dir/config resolution and subprocess fan-out; assert only
        # the exit-code contract from run_stage's returned failure set.
        monkeypatch.setattr(s, "shortcut_paths", lambda kind: ("/r.py", "/c.toml"))
        monkeypatch.setattr(s, "_dir_params", lambda cfg, section: {"log_dir": "/l"})
        monkeypatch.setattr(s, "run_stage", lambda *a, **k: set(failed))

    def test_exits_zero_when_all_reduced(self, s, monkeypatch):
        self._patch(s, monkeypatch, failed=[])
        s.main(["--obs_id_list", _OID1])  # no SystemExit

    def test_exits_nonzero_when_any_failed(self, s, monkeypatch):
        self._patch(s, monkeypatch, failed=[_OID1])
        with pytest.raises(SystemExit) as exc:
            s.main(["--obs_id_list", _OID1])
        assert exc.value.code == 1
