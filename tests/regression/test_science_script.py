"""Tests for scripts/processing/science.py: the science-reduction driver.

Cover the driver's own surface: arg parsing and obs_id validation, the
``_cli_task`` argv it fans out, and the ``main`` exit-code contract (nonzero iff
at least one frame failed). The shared fan-out engine (``run_stage``, the
cores-based job sizing) lives in ``_dispatch`` and is tested in test_dispatch.py.

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


class _NoLogDirConfig(_FakeConfig):
    """A config with no configured log_dir."""

    def get_params(self, keys):
        return {}  # no log_dir


@pytest.fixture(scope="module")
def s():
    """The science driver module (a normal package import)."""
    return _science


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_obs_ids(self, s):
        ns = s.parse_args(["--obs_ids", _OID2, _OID1])
        assert ns.obs_ids == [_OID1, _OID2]  # sorted + deduped

    def test_dedupes_obs_ids(self, s):
        ns = s.parse_args(["--obs_ids", _OID1, _OID1])
        assert ns.obs_ids == [_OID1]

    def test_obs_ids_from_file(self, s, tmp_path):
        f = tmp_path / "frames.txt"
        f.write_text(f"{_OID2}\n\n{_OID1}\n")  # blank line skipped
        ns = s.parse_args(["--obs_ids", str(f)])
        assert ns.obs_ids == [_OID1, _OID2]  # sorted

    def test_obs_ids_mixes_inline_and_file(self, s, tmp_path):
        f = tmp_path / "frames.txt"
        f.write_text(f"{_OID2}\n{_OID1}\n")  # _OID1 also given inline
        ns = s.parse_args(["--obs_ids", _OID1, str(f)])
        assert ns.obs_ids == [_OID1, _OID2]  # merged + deduped

    def test_obs_ids_file_with_bad_obs_id_errors(self, s, tmp_path):
        f = tmp_path / "frames.txt"
        f.write_text(f"{_OID1}\nnot-an-obs-id\n")
        with pytest.raises(SystemExit):
            s.parse_args(["--obs_ids", str(f)])

    def test_obs_ids_empty_file_errors(self, s, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("\n  \n")
        with pytest.raises(SystemExit):
            s.parse_args(["--obs_ids", str(f)])

    def test_obs_id_entry_neither_obs_id_nor_file_errors(self, s):
        with pytest.raises(SystemExit):
            s.parse_args(["--obs_ids", "/no/such/file.txt"])

    def test_obs_ids_required(self, s):
        with pytest.raises(SystemExit):
            s.parse_args([])

    @pytest.mark.parametrize(
        "argv",
        [
            ["--obs_ids", "not-an-obs-id"],  # malformed obs_id
            ["--obs_ids", "20240405"],  # a datecode is not an obs_id
            ["--obs_ids", _OID1, "--jobs", "0"],  # jobs below 1
            ["--obs_ids", _OID1, "--job_timeout", "0"],  # timeout below 1
        ],
    )
    def test_invalid_args_exit(self, s, argv):
        with pytest.raises(SystemExit):
            s.parse_args(argv)

    def test_config_defaults_to_none(self, s):
        assert s.parse_args(["--obs_ids", _OID1]).config is None

    def test_recipe_defaults_to_none(self, s):
        assert s.parse_args(["--obs_ids", _OID1]).recipe is None

    def test_recipe_override_parses(self, s):
        ns = s.parse_args(["--obs_ids", _OID1, "-r", "/x.py"])
        assert ns.recipe == "/x.py"

    def test_jobs_unset_resolves_to_cores_default(self, s):
        ns = s.parse_args(["--obs_ids", _OID1])
        assert ns.jobs == s._default_science_jobs()

    def test_jobs_override(self, s):
        assert s.parse_args(["--obs_ids", _OID1, "--jobs", "3"]).jobs == 3

    def test_job_timeout_default_and_override(self, s):
        assert s.parse_args(["--obs_ids", _OID1]).job_timeout == 600
        assert (
            s.parse_args(["--obs_ids", _OID1, "--job_timeout", "120"]).job_timeout
            == 120
        )


# ---------------------------------------------------------------------------
# _cli_task
# ---------------------------------------------------------------------------


class TestCliTask:
    def test_builds_science_argv_with_defaults(self, s):
        # No -r/-c override: the science default recipe/config are passed explicitly.
        tag, argv = s._cli_task(_OID1, ["--log_level", "DEBUG"])
        assert tag == _OID1
        assert argv == [
            sys.executable, "-m", "scripts.processing.reduce",
            "-r", s.DEFAULT_SCIENCE_RECIPE, "-c", s.DEFAULT_SCIENCE_CONFIG,
            "-o", _OID1, "--log_level", "DEBUG",
        ]  # fmt: skip

    def test_recipe_and_config_overrides(self, s):
        _, argv = s._cli_task(_OID1, [], config="/c.toml", recipe="/x.py")
        assert argv == [
            sys.executable, "-m", "scripts.processing.reduce",
            "-r", "/x.py", "-c", "/c.toml", "-o", _OID1,
        ]  # fmt: skip

    def test_recipe_override_keeps_default_config(self, s):
        _, argv = s._cli_task(_OID1, [], recipe="/x.py")
        assert argv == [
            sys.executable, "-m", "scripts.processing.reduce",
            "-r", "/x.py", "-c", s.DEFAULT_SCIENCE_CONFIG, "-o", _OID1,
        ]  # fmt: skip

    def test_config_override_keeps_default_recipe(self, s):
        _, argv = s._cli_task(_OID1, [], config="/c.toml")
        assert argv == [
            sys.executable, "-m", "scripts.processing.reduce",
            "-r", s.DEFAULT_SCIENCE_RECIPE, "-c", "/c.toml", "-o", _OID1,
        ]  # fmt: skip


# ---------------------------------------------------------------------------
# main exit-code contract
# ---------------------------------------------------------------------------


class TestMainExitCode:
    def _patch(self, s, monkeypatch, failed):
        # Skip the runtime setup and the real dir/config resolution + subprocess
        # fan-out; assert only the exit-code contract from run_stage's failure set.
        # setup_batch_logging is stubbed so main() writes no real batch log file.
        monkeypatch.setattr(s, "configure_runtime", lambda: None)
        monkeypatch.setattr(s, "ConfigHandler", _FakeConfig)
        monkeypatch.setattr(s, "setup_batch_logging", lambda *a, **k: "/l/x.log")
        monkeypatch.setattr(s, "run_stage", lambda *a, **k: set(failed))

    def test_exits_zero_when_all_reduced(self, s, monkeypatch):
        self._patch(s, monkeypatch, failed=[])
        s.main(["--obs_ids", _OID1])  # no SystemExit

    def test_exits_nonzero_when_any_failed(self, s, monkeypatch):
        self._patch(s, monkeypatch, failed=[_OID1])
        with pytest.raises(SystemExit) as exc:
            s.main(["--obs_ids", _OID1])
        assert exc.value.code == 1

    def test_errors_when_log_dir_unset(self, s, monkeypatch):
        # A missing log_dir is fatal before any fan-out (DRP-RUN-07).
        monkeypatch.setattr(s, "configure_runtime", lambda: None)
        monkeypatch.setattr(s, "ConfigHandler", _NoLogDirConfig)
        with pytest.raises(SystemExit) as exc:
            s.main(["--obs_ids", _OID1])
        assert "log directory" in str(exc.value)
