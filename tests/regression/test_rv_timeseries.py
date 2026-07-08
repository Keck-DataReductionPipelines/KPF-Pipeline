"""Tests for scripts/processing/rv_timeseries.py: the RV-timeseries orchestrator.

rv_timeseries is a standalone script (not an importable package module), so it is
loaded by path like the recipe tests. These cover the pure helpers (arg parsing,
job sizing, datecode/task/path helpers, burst grouping, L4 reading), the
fail-fast/fail-soft ``run_stage`` dispatch and its per-night masters gate, and --
the regression that motivated this file -- that the threaded L0 discovery does
not cross-contaminate nights (each pooled scan uses its own FileHandler, so a
shared ``self._mini_db`` can no longer race).

Unit tests use synthetic FITS frames in temp trees -- no real testdata needed.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from kpfpipe.utils.io import kpf_filepath

_RV_TS_PATH = (
    Path(__file__).parent.parent.parent / "scripts" / "processing" / "rv_timeseries.py"
)


@pytest.fixture(scope="module")
def rv():
    """Load rv_timeseries.py by path (it is a script, not a package module)."""
    spec = importlib.util.spec_from_file_location("rv_timeseries", _RV_TS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Synthetic-frame helpers
# ---------------------------------------------------------------------------


def _write_l0(data_input, datecode, seconds, obj, imtype="Object", junk=False):
    """Write one L0 frame under {data_input}/L0/{datecode}; return its obs_id."""
    l0_dir = Path(data_input) / "L0" / datecode
    l0_dir.mkdir(parents=True, exist_ok=True)
    obs_id = f"KP.{datecode}.{seconds:05d}.00"
    header = fits.Header(
        {
            "OBJECT": obj,
            "IMTYPE": imtype,
            "TARGNAME": obj,
            "EXPTIME": 60.0,
            "ELAPSED": 60.0,
        }
    )
    fits.PrimaryHDU(header=header).writeto(l0_dir / f"{obs_id}.fits")
    if junk:
        _add_junk(data_input, obs_id)
    return obs_id


def _add_junk(data_input, obs_id):
    """Append obs_id to the WMKO junk list under {data_input}/vNext/reference/."""
    ref = Path(data_input) / "vNext" / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    junk_csv = ref / "junk_obs.csv"
    if not junk_csv.exists():
        junk_csv.write_text("Junk Observations for KPF\nobservation_id\n")
    with junk_csv.open("a") as fh:
        fh.write(f"{obs_id}\n")


def _write_l4(science_output, datecode, name, *, bjd, rv_val, rverr, obj="10700"):
    """Write one L4 product under {science_output}/L4/{datecode}; return its path.

    A None keyword value is omitted from the header (FITS headers cannot store
    NaN), which is how _read_l4_rv sees an absent RV/RVERR/BJDTDB and skips it.
    """
    l4_dir = Path(science_output) / "L4" / datecode
    l4_dir.mkdir(parents=True, exist_ok=True)
    header = fits.Header({"OBJECT": obj})
    for key, val in (("BJDTDB", bjd), ("RV", rv_val), ("RVERR", rverr)):
        if val is not None:
            header[key] = val
    path = l4_dir / f"kpf_SL4_{name}.fits"
    fits.PrimaryHDU(header=header).writeto(path)
    return str(path)


def _write_master(masters_output, datecode, cal_type, level):
    """Write a stand-in master matching FileHandler.find_masters' glob."""
    d = Path(masters_output) / "masters" / datecode
    d.mkdir(parents=True, exist_ok=True)
    (d / f"KP.{datecode}.00001.00_master_{cal_type}_{level}.fits").write_text("x")


_OK = [sys.executable, "-c", "pass"]
_FAIL = [sys.executable, "-c", "import sys; sys.exit(1)"]
_SLEEP = [sys.executable, "-c", "import time; time.sleep(30)"]  # a wedged job
_BASE_ARGS = ["--target", "10700", "--date_range", "20240101", "20240131"]


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_minimal_valid(self, rv):
        ns = rv.parse_args(_BASE_ARGS)
        assert ns.target == "10700"
        assert ns.date_range == ["20240101", "20240131"]
        assert ns.jobs >= 1

    @pytest.mark.parametrize(
        "extra",
        [
            ["--date_range", "2024", "20240131"],  # malformed datecode
            ["--date_range", "20240201", "20240101"],  # start > end
            ["--file_limit", "0"],  # below 1
            ["--jobs", "0"],  # below 1
            ["--job_timeout", "0"],  # below 1
        ],
    )
    def test_invalid_args_exit(self, rv, extra):
        argv = ["--target", "10700"]
        if "--date_range" not in extra:
            argv += ["--date_range", "20240101", "20240131"]
        with pytest.raises(SystemExit):
            rv.parse_args(argv + extra)

    def test_input_dir_shorthand_populates_data_input(self, rv):
        ns = rv.parse_args(_BASE_ARGS + ["--input_dir", "/in"])
        assert ns.kpf_data_input == "/in"

    def test_input_dir_conflicts_with_long_form(self, rv):
        with pytest.raises(SystemExit):
            rv.parse_args(_BASE_ARGS + ["--input_dir", "/a", "--kpf_data_input", "/b"])

    def test_output_dir_routes_all_outputs(self, rv):
        ns = rv.parse_args(_BASE_ARGS + ["--output_dir", "/out"])
        assert ns.kpf_masters_output == "/out"
        assert ns.kpf_science_output == "/out"
        assert ns.log_dir == "/out/logs"
        assert ns.plot_dir == "/out/QLP/timeseries"

    def test_output_dir_conflicts_with_explicit_override(self, rv):
        with pytest.raises(SystemExit):
            rv.parse_args(
                _BASE_ARGS + ["--output_dir", "/o", "--kpf_masters_output", "/m"]
            )

    def test_plots_only_requires_plot_destination(self, rv):
        with pytest.raises(SystemExit):
            rv.parse_args(_BASE_ARGS + ["--plots_only"])

    def test_plots_only_with_output_dir_ok(self, rv):
        ns = rv.parse_args(_BASE_ARGS + ["--plots_only", "--output_dir", "/out"])
        assert ns.plots_only and ns.plot_dir == "/out/QLP/timeseries"

    def test_plot_dir_and_log_level_flags(self, rv):
        # --plot_dir (renamed from --plot_directory) and --log_level parse and are
        # available for forwarding to the CLI subprocesses.
        ns = rv.parse_args(_BASE_ARGS + ["--plot_dir", "/p", "--log_level", "DEBUG"])
        assert ns.plot_dir == "/p"
        assert ns.log_level == "DEBUG"

    def test_config_overrides_default_to_none(self, rv):
        # Unset --masters_config/--science_config default to None so the CLI's
        # --masters/--science shortcut supplies the config (single source).
        ns = rv.parse_args(_BASE_ARGS)
        assert ns.masters_config is None
        assert ns.science_config is None

    def test_job_timeout_default_and_override(self, rv):
        assert rv.parse_args(_BASE_ARGS).job_timeout == 600
        assert rv.parse_args(_BASE_ARGS + ["--job_timeout", "120"]).job_timeout == 120

    def test_jobs_unset_resolves_per_stage(self, rv):
        # Unset --jobs: science gets the cores default, masters the RAM-capped
        # one; masters never exceeds science.
        ns = rv.parse_args(_BASE_ARGS)
        assert ns.jobs == rv._default_jobs()
        assert ns.masters_jobs == rv._default_masters_jobs()
        assert ns.masters_jobs <= ns.jobs

    def test_jobs_override_drives_both_stages(self, rv):
        ns = rv.parse_args(_BASE_ARGS + ["--jobs", "3"])
        assert ns.jobs == 3 and ns.masters_jobs == 3


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


class TestDefaultJobs:
    @pytest.mark.parametrize(
        "cpus,expected",
        [(None, 1), (1, 1), (8, 8), (64, 16), (100, 25)],
    )
    def test_cap(self, rv, monkeypatch, cpus, expected):
        monkeypatch.setattr(rv.os, "cpu_count", lambda: cpus)
        assert rv._default_jobs() == expected


class TestDefaultMastersJobs:
    @staticmethod
    def _fake_sysconf(ram_gib):
        # SC_PHYS_PAGES * SC_PAGE_SIZE = total bytes; use a 4 KiB page.
        page = 4096
        pages = int(ram_gib * 2**30) // page
        return lambda name: {"SC_PHYS_PAGES": pages, "SC_PAGE_SIZE": page}[name]

    def test_big_host_gets_fixed_cap(self, rv, monkeypatch):
        # Many cores + ample RAM: neither floor binds, so the fixed cap wins.
        monkeypatch.setattr(rv.os, "cpu_count", lambda: 256)
        monkeypatch.setattr(rv.os, "sysconf", self._fake_sysconf(2048))
        assert rv._default_masters_jobs() == rv._MASTERS_JOBS

    def test_ram_floors_below_fixed_cap(self, rv, monkeypatch):
        # Many cores but modest RAM: the RAM floor drops it below the fixed cap.
        monkeypatch.setattr(rv.os, "cpu_count", lambda: 256)
        monkeypatch.setattr(rv.os, "sysconf", self._fake_sysconf(24))
        assert rv._default_masters_jobs() == 24 // rv._MASTERS_JOB_GIB
        assert rv._default_masters_jobs() < rv._MASTERS_JOBS

    def test_cores_floor_below_fixed_cap(self, rv, monkeypatch):
        # Few cores: the cores floor drops it below the fixed cap.
        monkeypatch.setattr(rv.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(rv.os, "sysconf", self._fake_sysconf(256))
        assert rv._default_masters_jobs() == rv._default_jobs() == 8

    def test_unknown_ram_uses_cores_floor_only(self, rv, monkeypatch):
        # sysconf unavailable -> only the cores floor applies (RAM floor skipped).
        monkeypatch.setattr(rv.os, "cpu_count", lambda: 256)

        def _raise(_name):
            raise ValueError("SC_PHYS_PAGES unavailable")

        monkeypatch.setattr(rv.os, "sysconf", _raise)
        assert rv._default_masters_jobs() == rv._MASTERS_JOBS

    def test_never_below_one(self, rv, monkeypatch):
        # Tiny RAM would floor the cap to 0; the helper clamps to 1.
        monkeypatch.setattr(rv.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(rv.os, "sysconf", self._fake_sysconf(1))
        assert rv._default_masters_jobs() == 1


class TestDatecodeDirs:
    def test_filters_by_range_and_sorts(self, rv, tmp_path):
        for name in ["20240101", "20240115", "20240201", "notadate", "20231231"]:
            (tmp_path / name).mkdir()
        (tmp_path / "20240110_file").write_text("x")  # datecode-ish, but not a dir
        got = rv._datecode_dirs(str(tmp_path), "20240101", "20240131")
        assert got == ["20240101", "20240115"]


class TestCliTask:
    def test_masters_uses_shortcut_and_datecode(self, rv):
        tag, argv = rv._cli_task("20240405", "masters", ["--log_dir", "/l"])
        assert tag == "20240405"
        assert argv == [
            sys.executable, "-m", "tools.cli",
            "--masters", "-d", "20240405", "--log_dir", "/l",
        ]  # fmt: skip

    def test_science_uses_shortcut_and_obs_id(self, rv):
        tag, argv = rv._cli_task(
            "KP.20240405.40113.57", "science", ["--log_level", "DEBUG"]
        )
        assert tag == "KP.20240405.40113.57"
        assert argv == [
            sys.executable, "-m", "tools.cli",
            "--science", "-o", "KP.20240405.40113.57", "--log_level", "DEBUG",
        ]  # fmt: skip

    def test_config_override_inserts_dash_c(self, rv):
        # A custom config is forwarded as -c, overriding the shortcut default
        # (tools.cli relaxes --science/-c to override rather than error).
        _, argv = rv._cli_task("KP.x", "science", [], config="/custom.toml")
        assert argv == [
            sys.executable, "-m", "tools.cli",
            "--science", "-c", "/custom.toml", "-o", "KP.x",
        ]  # fmt: skip

    def test_no_config_omits_dash_c(self, rv):
        _, argv = rv._cli_task("20240405", "masters", [])
        assert "-c" not in argv


class TestScienceComplete:
    def test_true_only_when_l4_exists(self, rv, tmp_path):
        obs_id = "KP.20240405.03600.00"
        sci = str(tmp_path)
        assert not rv._science_complete(obs_id, sci)
        l4 = kpf_filepath(obs_id, "L4", data_root=sci)
        Path(l4).parent.mkdir(parents=True, exist_ok=True)
        Path(l4).write_text("x")
        assert rv._science_complete(obs_id, sci)


class TestMissingMasters:
    def test_all_missing_on_empty_tree(self, rv, tmp_path):
        assert rv._missing_masters(str(tmp_path), "20240101") == rv._REQUIRED_MASTERS

    def test_none_missing_when_all_present(self, rv, tmp_path):
        for cal_type, level in rv._REQUIRED_MASTERS:
            _write_master(str(tmp_path), "20240101", cal_type, level)
        assert rv._missing_masters(str(tmp_path), "20240101") == []

    def test_reports_only_the_absent_one(self, rv, tmp_path):
        _write_master(str(tmp_path), "20240101", "bias", "L1")
        _write_master(str(tmp_path), "20240101", "dark", "L1")
        assert rv._missing_masters(str(tmp_path), "20240101") == [("thar", "L2")]


# ---------------------------------------------------------------------------
# run_stage: fail-fast (science) vs fail-soft (masters)
# ---------------------------------------------------------------------------


class TestRunStage:
    def test_empty_returns_empty_set(self, rv, tmp_path):
        assert rv.run_stage("masters", [], 2, str(tmp_path)) == set()

    def test_all_succeed_returns_empty_set(self, rv, tmp_path):
        failed = rv.run_stage("science", [("a", _OK), ("b", _OK)], 2, str(tmp_path))
        assert failed == set()

    def test_fail_fast_canary_exits(self, rv, tmp_path):
        with pytest.raises(SystemExit) as exc:
            rv.run_stage("science", [("a", _FAIL)], 2, str(tmp_path))
        assert exc.value.code == 1

    def test_fail_fast_pool_job_exits(self, rv, tmp_path):
        with pytest.raises(SystemExit) as exc:
            rv.run_stage("science", [("a", _OK), ("b", _FAIL)], 2, str(tmp_path))
        assert exc.value.code == 1

    def test_fail_soft_canary_continues(self, rv, tmp_path, capsys):
        # A failed canary must still fan out the rest and merely report the failure.
        failed = rv.run_stage(
            "masters",
            [("a", _FAIL), ("b", _OK)],
            2,
            str(tmp_path),
            abort_on_failure=False,
        )
        assert failed == {"a"}
        assert "ok: b" in capsys.readouterr().out

    def test_fail_soft_collects_all_failures(self, rv, tmp_path):
        failed = rv.run_stage(
            "masters",
            [("a", _FAIL), ("b", _FAIL)],
            2,
            str(tmp_path),
            abort_on_failure=False,
        )
        assert failed == {"a", "b"}

    def test_slow_fanout_job_is_killed_and_counts_as_failure(self, rv, tmp_path):
        # A fanned-out job that overruns job_timeout is a wedged subprocess: it is
        # killed and reported as a failure, so one stuck unit can't hang the batch.
        # The canary is fast; the 30s sleeper is the fan-out job, bounded to 1s.
        start = time.monotonic()
        failed = rv.run_stage(
            "masters",
            [("canary", _OK), ("slow", _SLEEP)],
            2,
            str(tmp_path),
            job_timeout=1,
            abort_on_failure=False,
        )
        assert failed == {"slow"}
        assert time.monotonic() - start < 15  # killed at ~1s, not the full 30s

    def test_canary_uses_canary_timeout_not_job_timeout(self, rv, tmp_path):
        # job_timeout bounds only the fan-out; the canary keeps its own, larger
        # limit, so a canary slower than job_timeout is not killed by it.
        slow_canary = [sys.executable, "-c", "import time; time.sleep(2)"]
        failed = rv.run_stage(
            "masters",
            [("canary", slow_canary), ("b", _OK)],
            2,
            str(tmp_path),
            job_timeout=1,
            canary_timeout=30,
            abort_on_failure=False,
        )
        assert failed == set()  # the 2s canary survived a 1s job_timeout


class TestRunOneInterrupt:
    def test_interrupt_in_launch_window_kills_child(self, rv, monkeypatch):
        # Reproduce the launch-vs-track race: the interrupt lands after Popen but
        # before the child is tracked. Wrapping Popen to set _interrupted models
        # exactly that -- the top-of-function guard is clear, so launch proceeds,
        # and the post-track re-check must catch it, kill the child, and return 130
        # (a missed child would otherwise run its full 30s sleep untracked).
        rv._interrupted.clear()  # top-level guard must pass
        real_popen = rv.subprocess.Popen
        launched = []

        def popen_then_interrupt(*a, **k):
            proc = real_popen(*a, **k)
            launched.append(proc)
            rv._interrupted.set()  # interrupt arrives in the launch/track window
            return proc

        monkeypatch.setattr(rv.subprocess, "Popen", popen_then_interrupt)
        try:
            rc, _ = rv._run_one(_SLEEP, timeout=None)
        finally:
            rv._interrupted.clear()  # module-global; don't leak to other tests
        assert rc == 130
        assert launched and launched[0].poll() is not None  # child killed + reaped


class TestReportFailures:
    def test_header_and_sentinels(self, rv, capsys):
        rv._report_failures(
            [("masters", "20240101", 1, "boom\ntraceback line")],
            "/logs",
            header="WARNING: 1 masters job(s) failed",
        )
        err = capsys.readouterr().err
        assert "WARNING: 1 masters job(s) failed" in err
        assert "FAILED [masters] 20240101 (exit 1)" in err
        assert "/logs" in err and "kpf_masters_20240101_" in err
        assert "boom" in err  # stderr tail echoed


# ---------------------------------------------------------------------------
# discover_science_obs_ids: correctness + threaded no-contamination regression
# ---------------------------------------------------------------------------


class TestDiscoverScienceObsIds:
    def _build_tree(self, data_input, nights, per_night, target="10700"):
        """One target burst per night (plus a decoy non-target frame); return the
        expected sorted obs_id set."""
        expected = set()
        for i, dc in enumerate(nights):
            for j in range(per_night):
                expected.add(_write_l0(data_input, dc, 3600 + j * 100, target))
            _write_l0(data_input, dc, 8000 + i, "99999")  # non-target decoy
        return expected

    def test_threaded_discovery_matches_expected(self, rv, tmp_path):
        # Six nights scanned by a pool: with a shared FileHandler this raced and
        # collapsed nights via duplicate obs_ids; per-thread handlers keep it exact.
        nights = [
            "20240101", "20240102", "20240103", "20240104", "20240105", "20240106",
        ]  # fmt: skip
        expected = self._build_tree(str(tmp_path), nights, per_night=4)
        got = rv.discover_science_obs_ids(
            str(tmp_path), "10700", "20240101", "20240131", file_limit=500, jobs=8
        )
        assert got == sorted(expected)
        assert len(got) == len(set(got)) == 24  # no duplicates, none dropped

    def test_matches_serial_result(self, rv, tmp_path):
        # The fixed discovery is deterministic: jobs=8 equals jobs=1.
        nights = ["20240101", "20240102", "20240103", "20240104"]
        self._build_tree(str(tmp_path), nights, per_night=3)
        common = dict(target="10700", start="20240101", end="20240131", file_limit=500)
        serial = rv.discover_science_obs_ids(str(tmp_path), jobs=1, **common)
        parallel = rv.discover_science_obs_ids(str(tmp_path), jobs=8, **common)
        assert serial == parallel

    def test_excludes_junk(self, rv, tmp_path):
        good = _write_l0(str(tmp_path), "20240101", 3600, "10700")
        junk = _write_l0(str(tmp_path), "20240101", 3700, "10700", junk=True)
        got = rv.discover_science_obs_ids(
            str(tmp_path), "10700", "20240101", "20240131", file_limit=500, jobs=2
        )
        assert got == [good] and junk not in got

    def test_no_matches_exits(self, rv, tmp_path):
        _write_l0(str(tmp_path), "20240101", 3600, "99999")  # no target frames
        with pytest.raises(SystemExit):
            rv.discover_science_obs_ids(
                str(tmp_path), "10700", "20240101", "20240131", file_limit=500, jobs=2
            )

    def test_file_limit_exceeded_exits(self, rv, tmp_path):
        for j in range(3):
            _write_l0(str(tmp_path), "20240101", 3600 + j * 100, "10700")
        with pytest.raises(SystemExit):
            rv.discover_science_obs_ids(
                str(tmp_path), "10700", "20240101", "20240131", file_limit=2, jobs=2
            )


# ---------------------------------------------------------------------------
# discover_l4_files / _read_l4_rv / _group_bursts (plot inputs)
# ---------------------------------------------------------------------------


class TestDiscoverL4Files:
    def test_keeps_only_target_frames(self, rv, tmp_path):
        sci = str(tmp_path)
        keep = _write_l4(sci, "20240101", "a", bjd=1.0, rv_val=0.0, rverr=1.0)
        _write_l4(sci, "20240101", "b", bjd=1.0, rv_val=0.0, rverr=1.0, obj="55555")
        got = rv.discover_l4_files(sci, "10700", "20240101", "20240131", jobs=2)
        assert got == [keep]


class TestReadL4Rv:
    def test_reads_finite_skips_nonfinite(self, rv, tmp_path, capsys):
        sci = str(tmp_path)
        _write_l4(sci, "20240101", "a", bjd=2451545.0, rv_val=0.01, rverr=0.001)
        _write_l4(sci, "20240102", "b", bjd=2451546.0, rv_val=0.02, rverr=0.002)
        # A frame missing RV -> _read_l4_rv skips it (FITS headers can't hold NaN).
        bad = _write_l4(sci, "20240103", "c", bjd=2451547.0, rv_val=None, rverr=0.003)

        times, rvs, errs, nights = rv._read_l4_rv(
            rv.discover_l4_files(sci, "10700", "20240101", "20240131", jobs=2)
        )
        assert times.size == 2
        assert set(nights) == {"20240101", "20240102"}
        assert "no finite RV" in capsys.readouterr().out
        assert bad  # the nan frame existed but was dropped


class TestGroupBursts:
    def test_splits_on_gap_and_weights(self, rv):
        # Two frames 1 min apart, then one a day later -> two bursts.
        times = np.array([0.0, 1.0 / 1440.0, 1.0])
        rvs = np.array([10.0, 20.0, 30.0])
        errs = np.array([1.0, 1.0, 1.0])
        g_times, g_rvs, g_errs = rv._group_bursts(times, rvs, errs)

        assert g_times.size == 2
        np.testing.assert_allclose(g_rvs, [15.0, 30.0])
        np.testing.assert_allclose(g_errs, [1.0 / np.sqrt(2), 1.0])
        np.testing.assert_allclose(g_times, [(0.0 + 1.0 / 1440.0) / 2, 1.0])
