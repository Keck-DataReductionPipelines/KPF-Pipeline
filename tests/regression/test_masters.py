"""Tests for scripts/processing/masters.py: the nightly-masters build driver.

masters.py is a standalone script (not an importable package module), so it is
loaded by path like the recipe/rv_timeseries tests. These cover the pure helpers
(arg parsing and the two input forms, datecode/task helpers, masters job sizing),
the fail-soft ``run_stage`` dispatch, and the ``main`` exit-code contract
(nonzero iff at least one night failed).

Unit tests use synthetic dir trees in tmp_path -- no real testdata needed.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_MASTERS_PATH = (
    Path(__file__).parent.parent.parent / "scripts" / "processing" / "masters.py"
)


@pytest.fixture(scope="module")
def m():
    """Load masters.py by path (it is a script, not a package module)."""
    spec = importlib.util.spec_from_file_location("masters", _MASTERS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OK = [sys.executable, "-c", "pass"]
_FAIL = [sys.executable, "-c", "import sys; sys.exit(1)"]


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_explicit_datecodes(self, m):
        ns = m.parse_args(["--datecode_list", "20240712", "20240405"])
        assert ns.datecode_list == ["20240405", "20240712"]  # sorted + deduped
        assert ns.date_range is None

    def test_dedupes_datecodes(self, m):
        ns = m.parse_args(["--datecode_list", "20240405", "20240405"])
        assert ns.datecode_list == ["20240405"]

    def test_date_range(self, m):
        ns = m.parse_args(["--date_range", "20240101", "20240131"])
        assert ns.date_range == ["20240101", "20240131"]
        assert ns.datecode_list is None

    def test_neither_form_errors(self, m):
        with pytest.raises(SystemExit):
            m.parse_args([])

    def test_both_forms_error(self, m):
        with pytest.raises(SystemExit):
            m.parse_args(
                ["--datecode_list", "20240405", "--date_range", "20240101", "20240131"]
            )

    @pytest.mark.parametrize(
        "argv",
        [
            ["--datecode_list", "2024"],  # malformed datecode (list form)
            ["--date_range", "2024", "20240131"],  # malformed datecode (range form)
            ["--date_range", "20240201", "20240101"],  # start > end
            ["--datecode_list", "20240405", "--jobs", "0"],  # jobs below 1
            ["--datecode_list", "20240405", "--job_timeout", "0"],  # timeout below 1
        ],
    )
    def test_invalid_args_exit(self, m, argv):
        with pytest.raises(SystemExit):
            m.parse_args(argv)

    def test_config_defaults_to_none(self, m):
        assert m.parse_args(["--datecode_list", "20240405"]).config is None

    def test_jobs_unset_resolves_to_masters_default(self, m):
        ns = m.parse_args(["--datecode_list", "20240405"])
        assert ns.jobs == m._default_masters_jobs()

    def test_jobs_override(self, m):
        assert m.parse_args(["--datecode_list", "20240405", "--jobs", "3"]).jobs == 3

    def test_job_timeout_default_and_override(self, m):
        assert m.parse_args(["--datecode_list", "20240405"]).job_timeout == 600
        assert (
            m.parse_args(
                ["--datecode_list", "20240405", "--job_timeout", "120"]
            ).job_timeout
            == 120
        )


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


class TestCliTask:
    def test_builds_masters_argv(self, m):
        tag, argv = m._cli_task("20240405", ["--log_level", "DEBUG"])
        assert tag == "20240405"
        assert argv == [
            sys.executable, "-m", "tools.cli",
            "--masters", "-d", "20240405", "--log_level", "DEBUG",
        ]  # fmt: skip

    def test_config_override_inserts_dash_c(self, m):
        _, argv = m._cli_task("20240405", [], config="/custom.toml")
        assert argv == [
            sys.executable, "-m", "tools.cli",
            "--masters", "-c", "/custom.toml", "-d", "20240405",
        ]  # fmt: skip

    def test_no_config_omits_dash_c(self, m):
        _, argv = m._cli_task("20240405", [])
        assert "-c" not in argv


class TestDatecodeDirs:
    def test_filters_by_range_and_sorts(self, m, tmp_path):
        for name in ["20240101", "20240115", "20240201", "notadate", "20231231"]:
            (tmp_path / name).mkdir()
        (tmp_path / "20240110_file").write_text("x")  # datecode-ish, but not a dir
        got = m._datecode_dirs(str(tmp_path), "20240101", "20240131")
        assert got == ["20240101", "20240115"]


class TestResolveDatecodes:
    def test_explicit_list_passes_through(self, m, tmp_path):
        args = m.parse_args(["--datecode_list", "20240405", "20240712"])
        # data_input is ignored for the explicit-list form.
        assert m.resolve_datecodes(args, str(tmp_path)) == ["20240405", "20240712"]

    def test_range_scans_l0_tree(self, m, tmp_path):
        l0 = tmp_path / "L0"
        for name in ["20240101", "20240115", "20240201"]:
            (l0 / name).mkdir(parents=True)
        args = m.parse_args(["--date_range", "20240101", "20240131"])
        assert m.resolve_datecodes(args, str(tmp_path)) == ["20240101", "20240115"]

    def test_range_missing_l0_root_exits(self, m, tmp_path):
        args = m.parse_args(["--date_range", "20240101", "20240131"])
        with pytest.raises(SystemExit):
            m.resolve_datecodes(args, str(tmp_path))  # no L0/ dir

    def test_range_no_nights_in_range_exits(self, m, tmp_path):
        (tmp_path / "L0" / "20250101").mkdir(parents=True)
        args = m.parse_args(["--date_range", "20240101", "20240131"])
        with pytest.raises(SystemExit):
            m.resolve_datecodes(args, str(tmp_path))


class TestDefaultMastersJobs:
    @staticmethod
    def _fake_sysconf(ram_gib):
        page = 4096  # SC_PHYS_PAGES * SC_PAGE_SIZE = total bytes; 4 KiB page.
        pages = int(ram_gib * 2**30) // page
        return lambda name: {"SC_PHYS_PAGES": pages, "SC_PAGE_SIZE": page}[name]

    def test_big_host_gets_fixed_cap(self, m, monkeypatch):
        monkeypatch.setattr(m.os, "cpu_count", lambda: 256)
        monkeypatch.setattr(m.os, "sysconf", self._fake_sysconf(2048))
        assert m._default_masters_jobs() == m._MASTERS_JOBS

    def test_ram_floors_below_fixed_cap(self, m, monkeypatch):
        monkeypatch.setattr(m.os, "cpu_count", lambda: 256)
        monkeypatch.setattr(m.os, "sysconf", self._fake_sysconf(24))
        assert m._default_masters_jobs() == 24 // m._MASTERS_JOB_GIB
        assert m._default_masters_jobs() < m._MASTERS_JOBS

    def test_cores_floor_below_fixed_cap(self, m, monkeypatch):
        monkeypatch.setattr(m.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(m.os, "sysconf", self._fake_sysconf(256))
        assert m._default_masters_jobs() == m._default_jobs() == 8

    def test_unknown_ram_uses_cores_floor_only(self, m, monkeypatch):
        monkeypatch.setattr(m.os, "cpu_count", lambda: 256)

        def _raise(_name):
            raise ValueError("SC_PHYS_PAGES unavailable")

        monkeypatch.setattr(m.os, "sysconf", _raise)
        assert m._default_masters_jobs() == m._MASTERS_JOBS


# ---------------------------------------------------------------------------
# run_stage (fail-soft)
# ---------------------------------------------------------------------------


class TestRunStage:
    def test_empty_returns_empty_set(self, m, tmp_path):
        assert m.run_stage("masters", [], 2, str(tmp_path)) == set()

    def test_all_succeed_returns_empty_set(self, m, tmp_path):
        tasks = [("20240101", _OK), ("20240102", _OK), ("20240103", _OK)]
        assert m.run_stage("masters", tasks, 2, str(tmp_path)) == set()

    def test_failed_canary_still_fans_out_and_is_reported(self, m, tmp_path, capsys):
        # Fail-soft: a bad canary night does not stop the rest; it is collected.
        tasks = [("20240101", _FAIL), ("20240102", _OK), ("20240103", _OK)]
        failed = m.run_stage("masters", tasks, 2, str(tmp_path))
        assert failed == {"20240101"}
        assert "continuing" in capsys.readouterr().out

    def test_collects_all_failures(self, m, tmp_path):
        tasks = [("20240101", _OK), ("20240102", _FAIL), ("20240103", _FAIL)]
        assert m.run_stage("masters", tasks, 2, str(tmp_path)) == {
            "20240102",
            "20240103",
        }


# ---------------------------------------------------------------------------
# main exit-code contract
# ---------------------------------------------------------------------------


class TestMainExitCode:
    def _patch(self, m, monkeypatch, failed):
        # Skip the real dir/config resolution and subprocess fan-out; assert only
        # the exit-code contract from run_stage's returned failure set.
        monkeypatch.setattr(m, "shortcut_paths", lambda kind: ("/r.py", "/c.toml"))
        monkeypatch.setattr(
            m,
            "_dir_params",
            lambda cfg, section: {"KPF_DATA_INPUT": "/in", "log_dir": "/l"},
        )
        monkeypatch.setattr(m, "resolve_datecodes", lambda args, di: ["20240405"])
        monkeypatch.setattr(m, "run_stage", lambda *a, **k: set(failed))

    def test_exits_zero_when_all_built(self, m, monkeypatch):
        self._patch(m, monkeypatch, failed=[])
        m.main(["--datecode_list", "20240405"])  # no SystemExit

    def test_exits_nonzero_when_any_failed(self, m, monkeypatch):
        self._patch(m, monkeypatch, failed=["20240405"])
        with pytest.raises(SystemExit) as exc:
            m.main(["--datecode_list", "20240405"])
        assert exc.value.code == 1
