"""Tests for scripts/processing/timeseries.py: the RV-timeseries wrapper.

timeseries is a thin discovery + dispatch wrapper: it combs the L0 tree for a
target's science frames over a datecode range (steps 1-2), then runs one masters
and one science orchestrator subprocess (steps 3-4). The robust fan-out engine
lives in _dispatch (tested in test_dispatch.py) and the orchestrators own their
own arg parsing (test_masters_script.py / test_science_script.py). These cover
only what timeseries owns: arg parsing, the threaded L0 discovery, the
orchestrator argv it builds, and the main() dispatch order.

Unit tests use synthetic FITS frames in temp trees -- no real testdata needed.
"""

import sys
from pathlib import Path

import pytest
from astropy.io import fits

from scripts.processing import timeseries as _ts

_OID1 = "KP.20240101.03600.00"
_BASE_ARGS = ["--target", "10700", "--date_range", "20240101", "20240131"]


@pytest.fixture(scope="module")
def ts():
    """The timeseries wrapper module (a normal package import)."""
    return _ts


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


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_minimal_valid(self, ts):
        ns = ts.parse_args(_BASE_ARGS)
        assert ns.target == "10700"
        assert ns.date_range == ["20240101", "20240131"]

    def test_recipe_config_overrides_default_to_none(self, ts):
        ns = ts.parse_args(_BASE_ARGS)
        assert ns.masters_recipe is None
        assert ns.masters_config is None
        assert ns.science_recipe is None
        assert ns.science_config is None

    def test_recipe_config_overrides_parse(self, ts):
        ns = ts.parse_args(
            _BASE_ARGS
            + [
                "--masters_recipe",
                "/m.py",
                "--masters_config",
                "/m.toml",
                "--science_recipe",
                "/s.py",
                "--science_config",
                "/s.toml",
            ]  # fmt: skip
        )
        assert ns.masters_recipe == "/m.py" and ns.masters_config == "/m.toml"
        assert ns.science_recipe == "/s.py" and ns.science_config == "/s.toml"

    def test_stage_toggles_default_on(self, ts):
        # All three stages run unless explicitly opted out.
        ns = ts.parse_args(_BASE_ARGS)
        assert ns.masters is True and ns.science is True and ns.plots is True

    def test_stage_toggles_opt_out(self, ts):
        ns = ts.parse_args(_BASE_ARGS + ["--no-masters", "--no-science", "--no-plots"])
        assert ns.masters is False and ns.science is False and ns.plots is False

    def test_input_dir_aliases_data_input(self, ts):
        ns = ts.parse_args(_BASE_ARGS + ["--input_dir", "/in"])
        assert ns.kpf_data_input == "/in"

    def test_output_dir_fans_out_including_plot_dir(self, ts):
        # timeseries has all four output slots: masters/science output, log, plot.
        ns = ts.parse_args(_BASE_ARGS + ["--output_dir", "/out"])
        assert ns.kpf_masters_output == "/out"
        assert ns.kpf_science_output == "/out"
        assert ns.log_dir == "/out/logs"
        assert ns.plot_dir == "/out/QLP/timeseries"

    def test_explicit_plot_dir_overrides_output_dir(self, ts):
        ns = ts.parse_args(_BASE_ARGS + ["--output_dir", "/out", "--plot_dir", "/p"])
        assert ns.plot_dir == "/p" and ns.kpf_science_output == "/out"

    def test_jobs_unset_stays_none(self, ts):
        # Unset --jobs is left None so each stage picks its own default; the
        # discovery scan falls back to _default_science_jobs() in main().
        assert ts.parse_args(_BASE_ARGS).jobs is None

    def test_job_timeout_default_and_override(self, ts):
        assert ts.parse_args(_BASE_ARGS).job_timeout == 600
        assert ts.parse_args(_BASE_ARGS + ["--job_timeout", "120"]).job_timeout == 120

    @pytest.mark.parametrize(
        "extra",
        [
            ["--date_range", "2024", "20240131"],  # malformed datecode
            ["--date_range", "20240201", "20240101"],  # start > end
            ["--jobs", "0"],  # below 1
            ["--job_timeout", "0"],  # below 1
        ],
    )
    def test_invalid_args_exit(self, ts, extra):
        argv = ["--target", "10700"]
        if "--date_range" not in extra:
            argv += ["--date_range", "20240101", "20240131"]
        with pytest.raises(SystemExit):
            ts.parse_args(argv + extra)

    def test_target_required(self, ts):
        with pytest.raises(SystemExit):
            ts.parse_args(["--date_range", "20240101", "20240131"])

    @pytest.mark.parametrize(
        "flag",
        [
            "--file_limit",
            "--plots_only",
            "--group_bursts",
            "--skip_existing_masters",
            "--skip_existing_science",
        ],  # fmt: skip
    )
    def test_removed_flags_rejected(self, ts, flag):
        # Every flag dropped in the rewrite is no longer accepted.
        with pytest.raises(SystemExit):
            ts.parse_args(_BASE_ARGS + [flag, "x"])


# ---------------------------------------------------------------------------
# _orchestrator_argv
# ---------------------------------------------------------------------------


class TestOrchestratorArgv:
    def test_masters_defaults(self, ts):
        argv = ts._orchestrator_argv(
            "masters", "--dates", ["20240101", "20240102"], ["--log_dir", "/l"]
        )
        assert argv == [
            sys.executable, "-m", "scripts.processing.masters",
            "--dates", "20240101", "20240102", "--log_dir", "/l",
        ]  # fmt: skip

    def test_science_with_recipe_and_config(self, ts):
        argv = ts._orchestrator_argv(
            "science", "--obs_ids", [_OID1], ["--jobs", "3"],
            recipe="/s.py", config="/s.toml",
        )  # fmt: skip
        assert argv == [
            sys.executable, "-m", "scripts.processing.science",
            "-r", "/s.py", "-c", "/s.toml", "--obs_ids", _OID1, "--jobs", "3",
        ]  # fmt: skip

    def test_units_precede_forward(self, ts):
        # The unit list sits before the forwarded flags so the --dates nargs stops
        # at the first flag rather than swallowing it.
        argv = ts._orchestrator_argv(
            "masters", "--dates", ["20240101"], ["--jobs", "2"]
        )
        assert argv.index("20240101") < argv.index("--jobs")


# ---------------------------------------------------------------------------
# discover_science_obs_ids: threaded no-contamination + filtering
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

    def test_threaded_discovery_matches_expected(self, ts, tmp_path):
        # Six nights scanned by a pool: with a shared FileHandler this raced and
        # collapsed nights via duplicate obs_ids; per-thread handlers keep it exact.
        nights = [
            "20240101", "20240102", "20240103", "20240104", "20240105", "20240106",
        ]  # fmt: skip
        expected = self._build_tree(str(tmp_path), nights, per_night=4)
        got = ts.discover_science_obs_ids(
            str(tmp_path), "10700", "20240101", "20240131", jobs=8
        )
        assert got == sorted(expected)
        assert len(got) == len(set(got)) == 24  # no duplicates, none dropped

    def test_discovery_writes_mini_db_cache(self, ts, tmp_path):
        # Discovery scans cache="rw" through the shared primitive, so it still
        # writes each night's on-disk CSV as a side effect.
        self._build_tree(str(tmp_path), ["20240101"], per_night=2)
        ts.discover_science_obs_ids(
            str(tmp_path), "10700", "20240101", "20240131", jobs=2
        )
        cache = Path(tmp_path) / "vNext" / "mini_db" / "20240101_L0.csv"
        assert cache.is_file()

    def test_matches_serial_result(self, ts, tmp_path):
        # The discovery is deterministic: jobs=8 equals jobs=1.
        nights = ["20240101", "20240102", "20240103", "20240104"]
        self._build_tree(str(tmp_path), nights, per_night=3)
        common = dict(target="10700", start="20240101", end="20240131")
        serial = ts.discover_science_obs_ids(str(tmp_path), jobs=1, **common)
        parallel = ts.discover_science_obs_ids(str(tmp_path), jobs=8, **common)
        assert serial == parallel

    def test_excludes_junk(self, ts, tmp_path):
        good = _write_l0(str(tmp_path), "20240101", 3600, "10700")
        junk = _write_l0(str(tmp_path), "20240101", 3700, "10700", junk=True)
        got = ts.discover_science_obs_ids(
            str(tmp_path), "10700", "20240101", "20240131", jobs=2
        )
        assert got == [good] and junk not in got

    def test_excludes_non_object_imtype(self, ts, tmp_path):
        obj = _write_l0(str(tmp_path), "20240101", 3600, "10700")
        _write_l0(str(tmp_path), "20240101", 3700, "10700", imtype="Bias")
        got = ts.discover_science_obs_ids(
            str(tmp_path), "10700", "20240101", "20240131", jobs=2
        )
        assert got == [obj]

    def test_no_matches_exits(self, ts, tmp_path):
        _write_l0(str(tmp_path), "20240101", 3600, "99999")  # no target frames
        with pytest.raises(SystemExit):
            ts.discover_science_obs_ids(
                str(tmp_path), "10700", "20240101", "20240131", jobs=2
            )

    def test_missing_l0_root_exits(self, ts, tmp_path):
        with pytest.raises(SystemExit):
            ts.discover_science_obs_ids(
                str(tmp_path), "10700", "20240101", "20240131", jobs=2
            )


# ---------------------------------------------------------------------------
# main: dispatch order (masters -> science -> plots) and the stage toggles
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Stand-in for ConfigHandler: yields the dir/log params main() reads."""

    def __init__(self, path, data_input, log_dir):
        self._data_input = data_input
        self._log_dir = log_dir

    def get_params(self, keys):
        (section,) = keys
        if section == "DATA_DIRS":
            return {
                "KPF_DATA_INPUT": self._data_input,
                "KPF_MASTERS_OUTPUT": "/m",
                "KPF_SCIENCE_OUTPUT": "/sci",
            }
        return {"log_dir": self._log_dir, "log_level": "INFO"}


class TestMainDispatch:
    def _patch(self, ts, monkeypatch, tmp_path):
        """Wire main()'s side effects to captured fakes; return the run-call log."""
        data_input = str(tmp_path)
        monkeypatch.setattr(
            ts,
            "ConfigHandler",
            lambda path: _FakeConfig(path, data_input, str(tmp_path)),
        )
        monkeypatch.setattr(ts, "setup_batch_logging", lambda *a, **k: "/logs/x.log")
        calls = []

        class _Result:
            returncode = 0

        def _run(argv, **kwargs):
            calls.append(argv)
            return _Result()

        monkeypatch.setattr(ts.subprocess, "run", _run)
        return calls

    def test_masters_science_plots_dispatch(self, ts, monkeypatch, tmp_path):
        # All three stages run in order: masters, then science with every discovered
        # frame (no gating), then plots handed the same frames (no rescan).
        a = _write_l0(str(tmp_path), "20240101", 3600, "10700")
        b = _write_l0(str(tmp_path), "20240102", 3600, "10700")
        calls = self._patch(ts, monkeypatch, tmp_path)

        ts.main(_BASE_ARGS)

        assert len(calls) == 3
        assert "scripts.processing.masters" in calls[0]
        assert "20240101" in calls[0] and "20240102" in calls[0]  # both nights
        assert "scripts.processing.science" in calls[1]
        assert a in calls[1] and b in calls[1]  # every frame reduced
        # timeseries owns what runs: it routes each stage's default recipe+config
        # explicitly (not left to the orchestrator's own default).
        assert ts.DEFAULT_MASTERS_RECIPE in calls[0]
        assert ts.DEFAULT_MASTERS_CONFIG in calls[0]
        assert ts.DEFAULT_SCIENCE_RECIPE in calls[1]
        assert ts.DEFAULT_SCIENCE_CONFIG in calls[1]
        # Discovery is the sole mini-db writer; both launched stages are forced
        # read-only (--cache r) so they don't redundantly re-warm the caches.
        for stage in (calls[0], calls[1]):
            assert stage[stage.index("--cache") + 1] == "r"
        # Plot stage: the discovered frames (no rescan), reading from the science
        # output root, writing to its default {KPF_SCIENCE_OUTPUT}/QLP/timeseries.
        assert "scripts.plots.plot_timeseries" in calls[2]
        assert a in calls[2] and b in calls[2]
        assert "/sci" in calls[2]
        assert "/sci/QLP/timeseries" in calls[2]

    def test_plot_dir_override(self, ts, monkeypatch, tmp_path):
        _write_l0(str(tmp_path), "20240101", 3600, "10700")
        calls = self._patch(ts, monkeypatch, tmp_path)

        ts.main(_BASE_ARGS + ["--plot_dir", "/custom/plots"])

        assert "scripts.plots.plot_timeseries" in calls[2]
        assert "/custom/plots" in calls[2]

    def test_jobs_rides_science_not_masters(self, ts, monkeypatch, tmp_path):
        # --jobs sizes only the science fan-out; forwarding it to masters would
        # defeat the fixed _MASTERS_JOBS concurrency cap, so it must not reach the
        # masters argv.
        _write_l0(str(tmp_path), "20240101", 3600, "10700")
        calls = self._patch(ts, monkeypatch, tmp_path)

        ts.main(_BASE_ARGS + ["--jobs", "8"])

        assert "scripts.processing.masters" in calls[0]
        assert "--jobs" not in calls[0]  # masters keeps its own capped sizing
        assert "scripts.processing.science" in calls[1]
        assert "--jobs" in calls[1] and "8" in calls[1]  # science gets it

    def test_no_masters_skips_masters_stage(self, ts, monkeypatch, tmp_path):
        # --no-masters runs the science and plots stages only.
        a = _write_l0(str(tmp_path), "20240101", 3600, "10700")
        calls = self._patch(ts, monkeypatch, tmp_path)

        ts.main(_BASE_ARGS + ["--no-masters"])

        assert len(calls) == 2
        assert "scripts.processing.science" in calls[0] and a in calls[0]
        assert "scripts.plots.plot_timeseries" in calls[1]
        assert not any("scripts.processing.masters" in c for c in calls)

    def test_no_science_still_plots(self, ts, monkeypatch, tmp_path):
        # --no-science runs masters, then plots the L4 already on disk.
        _write_l0(str(tmp_path), "20240101", 3600, "10700")
        calls = self._patch(ts, monkeypatch, tmp_path)

        ts.main(_BASE_ARGS + ["--no-science"])

        assert len(calls) == 2
        assert "scripts.processing.masters" in calls[0]
        assert "scripts.plots.plot_timeseries" in calls[1]
        assert not any("scripts.processing.science" in c for c in calls)

    def test_no_plots_skips_plot_stage(self, ts, monkeypatch, tmp_path):
        # --no-plots runs masters + science but no plotter.
        _write_l0(str(tmp_path), "20240101", 3600, "10700")
        calls = self._patch(ts, monkeypatch, tmp_path)

        ts.main(_BASE_ARGS + ["--no-plots"])

        assert len(calls) == 2
        assert not any("scripts.plots.plot_timeseries" in c for c in calls)

    def test_all_stages_skipped_runs_nothing(self, ts, monkeypatch, tmp_path):
        # Skipping every stage dispatches no subprocess and exits cleanly.
        _write_l0(str(tmp_path), "20240101", 3600, "10700")
        calls = self._patch(ts, monkeypatch, tmp_path)

        ts.main(_BASE_ARGS + ["--no-masters", "--no-science", "--no-plots"])

        assert calls == []

    def test_no_skip_logs_written(self, ts, monkeypatch, tmp_path):
        # The gate is gone: no stand-in per-frame skip logs are ever written.
        _write_l0(str(tmp_path), "20240101", 3600, "10700")
        self._patch(ts, monkeypatch, tmp_path)

        ts.main(_BASE_ARGS)

        assert list(tmp_path.rglob("kpf_science_*.log")) == []

    def test_nonzero_when_a_stage_fails(self, ts, monkeypatch, tmp_path):
        _write_l0(str(tmp_path), "20240101", 3600, "10700")
        self._patch(ts, monkeypatch, tmp_path)

        class _Fail:
            returncode = 1

        monkeypatch.setattr(ts.subprocess, "run", lambda *a, **k: _Fail())
        with pytest.raises(SystemExit) as exc:
            ts.main(_BASE_ARGS)
        assert exc.value.code == 1
