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
        assert ns.log_directory == "/out/logs"
        assert ns.plot_directory == "/out/QLP/timeseries"

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
        assert ns.plots_only and ns.plot_directory == "/out/QLP/timeseries"


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


class TestDatecodeDirs:
    def test_filters_by_range_and_sorts(self, rv, tmp_path):
        for name in ["20240101", "20240115", "20240201", "notadate", "20231231"]:
            (tmp_path / name).mkdir()
        (tmp_path / "20240110_file").write_text("x")  # datecode-ish, but not a dir
        got = rv._datecode_dirs(str(tmp_path), "20240101", "20240131")
        assert got == ["20240101", "20240115"]


class TestCliTask:
    def test_builds_argv(self, rv):
        tag, argv = rv._cli_task(
            "20240405", "/r.py", "/c.toml", "-d", ["--log_dir", "/l"]
        )
        assert tag == "20240405"
        assert argv == [
            sys.executable, "-m", "tools.cli",
            "-r", "/r.py", "-c", "/c.toml",
            "-d", "20240405", "--log_dir", "/l",
        ]  # fmt: skip


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
