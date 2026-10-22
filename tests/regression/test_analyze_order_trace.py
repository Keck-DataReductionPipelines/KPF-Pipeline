"""Tests for scripts/analyze/order_trace.py: the order-trace stability analysis.

order_trace reads master order-trace CSVs from a masters tree, groups them by
science era, and measures how far each trace moves between epochs. These cover
what it owns -- arg parsing, era assignment, the tree walk, the deviation and jump
measurements, and main() end to end -- plus its ``kpfpipe analyze`` registration.

Unit tests use synthetic trace CSVs in temp trees -- no real testdata needed.
"""

import numpy as np
import pandas as pd
import pytest

import tools.cli as cli
from scripts.analyze import order_trace as _order_trace

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

_BASE_ARGS = ["--date_range", "20230101", "20230131", "--input_dir", "/in"]


@pytest.fixture(scope="module")
def order_trace():
    return _order_trace


def _write_trace(input_dir, datecode, offset=0.0, seconds=3600):
    """Write one epoch's trace CSV: two orders of SCI1 on each chip, every
    centerline raised by `offset` rows."""
    rows = [
        {
            "Chip": chip,
            "Fiber": "SCI1",
            "Order": order,
            "Coeff0": 100.0 + 50.0 * order + offset,
            "Coeff1": 1e-3,
            "Coeff2": 1e-7,
            "X1": 0.0,
            "X2": 4079.0,
            "Status": "full",
        }
        for chip in ("GREEN", "RED")
        for order in (0, 1)
    ]
    night = input_dir / "masters" / datecode
    night.mkdir(parents=True, exist_ok=True)
    path = night / f"KP.{datecode}.{seconds:05d}.00_master_order_trace.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _datecodes(start, count):
    return [d.strftime("%Y%m%d") for d in pd.date_range(start, periods=count)]


def _write_stepped_era(input_dir, before=10, after=20, step=0.05):
    """Era-1.0 epochs whose traces step by `step` rows after `before` epochs,
    over a little scatter; returns the datecode of the first stepped epoch."""
    rng = np.random.default_rng(0)
    dates = _datecodes("2023-01-01", before + after)
    for i, dc in enumerate(dates):
        level = step if i >= before else 0.0
        _write_trace(input_dir, dc, offset=level + rng.normal(0, 0.002))
    return dates[before]


def _load(order_trace, input_dir, start="20221109", end="20261231"):
    return order_trace.load_traces(
        str(input_dir), start, end, order_trace.load_science_eras()
    )


def _coeff_cols(traces):
    return [c for c in traces.columns if c.startswith("Coeff")]


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_output_dir_fans_out_to_analysis_and_log(self, order_trace):
        args = order_trace.parse_args([*_BASE_ARGS, "--output_dir", "/out"])
        assert args.analysis_dir == "/out/analysis"
        assert args.log_dir == "/out/logs"
        assert args.input_dir == "/in"

    def test_date_range_defaults_to_whole_archive(self, order_trace):
        args = order_trace.parse_args(["--input_dir", "/in", "--output_dir", "/o"])
        assert args.date_range[0] == "20221109"

    def test_input_dir_is_required(self, order_trace):
        with pytest.raises(SystemExit):
            order_trace.parse_args(["--output_dir", "/out"])

    def test_no_output_dirs_exits(self, order_trace, capsys):
        with pytest.raises(SystemExit):
            order_trace.parse_args(_BASE_ARGS)
        assert "pass --output_dir" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Eras
# ---------------------------------------------------------------------------


class TestEras:
    def test_engineering_eras_are_excluded(self, order_trace):
        eras = [era for era, _, _ in order_trace.load_science_eras()]
        assert "1.0" in eras and "2.0" in eras
        assert not {"0.5", "1.5", "2.5", "2.7", "3.5"} & set(eras)

    @pytest.mark.parametrize(
        ("datecode", "era"),
        [("20230101", "1.0"), ("20240301", "2.0"), ("20240205", None)],
    )
    def test_era_of(self, order_trace, datecode, era):
        assert order_trace.era_of(datecode, order_trace.load_science_eras()) == era


# ---------------------------------------------------------------------------
# load_traces
# ---------------------------------------------------------------------------


class TestLoadTraces:
    def test_tags_era_night_and_epoch(self, order_trace, tmp_path):
        _write_trace(tmp_path, "20230101")
        traces = _load(order_trace, tmp_path)
        assert set(traces["Era"]) == {"1.0"}
        assert set(traces["Night"]) == {"20230101"}
        assert set(traces["Epoch"]) == {"KP.20230101.03600.00"}

    def test_engineering_night_is_skipped(self, order_trace, tmp_path):
        _write_trace(tmp_path, "20230101")
        _write_trace(tmp_path, "20240205")  # Service Mission #1
        assert set(_load(order_trace, tmp_path)["Night"]) == {"20230101"}

    def test_honours_date_range_bounds(self, order_trace, tmp_path):
        for dc in ("20230101", "20230102", "20230103"):
            _write_trace(tmp_path, dc)
        traces = _load(order_trace, tmp_path, "20230102", "20230102")
        assert set(traces["Night"]) == {"20230102"}

    def test_unrecognised_filename_is_skipped(self, order_trace, tmp_path):
        _write_trace(tmp_path, "20230101")
        stray = tmp_path / "masters" / "20230101" / "bogus_master_order_trace.csv"
        stray.write_text("Chip\n")
        assert len(_load(order_trace, tmp_path)) == 4

    def test_missing_masters_dir_exits(self, order_trace, tmp_path):
        with pytest.raises(SystemExit, match="masters directory not found"):
            _load(order_trace, tmp_path)

    def test_no_traces_in_range_exits(self, order_trace, tmp_path):
        _write_trace(tmp_path, "20230101")
        with pytest.raises(SystemExit, match="no science-era order-trace CSVs"):
            _load(order_trace, tmp_path, "20240101", "20240131")


# ---------------------------------------------------------------------------
# Deviations and jumps
# ---------------------------------------------------------------------------


class TestPixelDeviations:
    def test_identical_epochs_have_zero_deviation(self, order_trace, tmp_path):
        for dc in _datecodes("2023-01-01", 5):
            _write_trace(tmp_path, dc)
        traces = _load(order_trace, tmp_path)
        pixel, epochs = order_trace.pixel_deviations(traces, _coeff_cols(traces))
        assert len(pixel) == 4
        assert (pixel["Epochs"] == 5).all()
        assert np.allclose(pixel[["Typical", "Max"]], 0.0)
        assert set(epochs["Chip"]) == {"GREEN", "RED", "BOTH"}

    def test_offset_epoch_is_measured_in_rows(self, order_trace, tmp_path):
        dates = _datecodes("2023-01-01", 5)
        for dc in dates[:-1]:
            _write_trace(tmp_path, dc)
        _write_trace(tmp_path, dates[-1], offset=0.3)
        traces = _load(order_trace, tmp_path)
        pixel, _ = order_trace.pixel_deviations(traces, _coeff_cols(traces))
        assert np.allclose(pixel["Max"], 0.3)

    def test_too_few_epochs_yields_empty_tables(self, order_trace, tmp_path):
        for dc in _datecodes("2023-01-01", order_trace.MIN_EPOCHS - 1):
            _write_trace(tmp_path, dc)
        traces = _load(order_trace, tmp_path)
        pixel, epochs = order_trace.pixel_deviations(traces, _coeff_cols(traces))
        assert pixel.empty and list(pixel.columns) == order_trace._PIXEL_COLUMNS
        assert epochs.empty and list(epochs.columns) == order_trace._EPOCH_COLUMNS


class TestFindJumps:
    def test_planted_step_is_found_once_near_its_date(self, order_trace, tmp_path):
        # The median windows locate a clean step only to within a few epochs:
        # boundaries next to it score nearly the same, and the scatter picks one.
        first_after = pd.Timestamp(_write_stepped_era(tmp_path))
        traces = _load(order_trace, tmp_path)
        _, epochs = order_trace.pixel_deviations(traces, _coeff_cols(traces))
        jumps = order_trace.find_jumps(epochs)
        assert sorted(jumps["Chip"]) == ["GREEN", "RED"]
        (date,) = order_trace.jump_dates(jumps, "1.0")
        assert abs((date - first_after).days) <= order_trace.JUMP_MERGE_DAYS

    def test_step_below_threshold_is_not_a_jump(self, order_trace, tmp_path):
        _write_stepped_era(tmp_path, step=0.01)
        traces = _load(order_trace, tmp_path)
        _, epochs = order_trace.pixel_deviations(traces, _coeff_cols(traces))
        jumps = order_trace.find_jumps(epochs)
        assert jumps.empty and list(jumps.columns) == order_trace._JUMP_COLUMNS


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    def test_writes_reports_under_output_dir(self, order_trace, tmp_path, monkeypatch):
        # setup_batch_logging installs root handlers; stub it so the run's logging
        # cannot leak into the rest of the suite.
        monkeypatch.setattr(
            order_trace,
            "setup_batch_logging",
            lambda *a, **k: ("analyze_order_trace_x", "/logs/x.log"),
        )
        input_dir, out = tmp_path / "in", tmp_path / "out"
        _write_stepped_era(input_dir)

        order_trace.main(
            [
                "--date_range",
                "20230101",
                "20230131",
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(out),
            ]  # fmt: skip
        )

        out_dir = out / "analysis" / "order_trace"
        for name in ("pixel", "epochs", "coefficients", "jumps"):
            assert (out_dir / f"order_trace_{name}_20230101_20230131.csv").is_file()
        assert len(list(out_dir.glob("*.png"))) == 6
        jumps = pd.read_csv(out_dir / "order_trace_jumps_20230101_20230131.csv")
        assert list(jumps.columns) == order_trace._JUMP_COLUMNS
        assert len(jumps) == 2

    def test_era_without_enough_epochs_exits(self, order_trace, tmp_path, monkeypatch):
        monkeypatch.setattr(
            order_trace, "setup_batch_logging", lambda *a, **k: ("x", "/logs/x.log")
        )
        _write_trace(tmp_path / "in", "20230101")
        with pytest.raises(SystemExit, match="trace epochs"):
            order_trace.main(
                ["--input_dir", str(tmp_path / "in"), "--output_dir", str(tmp_path)]
            )


def test_registered_with_analyze_dispatcher():
    assert cli._ANALYSES["order_trace"] is _order_trace.main
