"""Tests for scripts/analyze/order_trace.py: the order-trace stability analysis.

order_trace reads master order-trace CSVs from a masters tree, groups them by
science era, and measures how far each trace moves between epochs. These cover
what it owns -- arg parsing, era assignment, the tree walk, the deviation and jump
measurements, and main() end to end -- plus its ``kpfpipe analyze`` registration.

Unit tests use synthetic trace CSVs in temp trees -- no real testdata needed.
"""

import logging

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


def _write_trace(input_dir, datecode, offset=0.0, seconds=3600, **columns):
    """Write one epoch's trace CSV: two orders of SCI1 on each chip, every
    centerline raised by `offset` rows; `columns` override (or add) a column on
    every row."""
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
            **columns,
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


def _write_levels(input_dir, levels, start="2023-01-01"):
    """One epoch per night from `start`, each at its level in `levels` over a
    little scatter; returns the datecodes."""
    rng = np.random.default_rng(0)
    dates = _datecodes(start, len(levels))
    for dc, level in zip(dates, levels, strict=True):
        _write_trace(input_dir, dc, offset=level + rng.normal(0, 0.002))
    return dates


def _write_stepped_era(input_dir, before=10, after=20, step=0.05):
    """Era-1.0 epochs whose traces step by `step` rows after `before` epochs;
    returns the datecode of the first stepped epoch."""
    return _write_levels(input_dir, [0.0] * before + [step] * after)[before]


def _load(order_trace, input_dir, start="20221109", end="20261231"):
    return order_trace.load_traces(
        str(input_dir), start, end, order_trace.load_science_eras()
    )


def _coeff_cols(traces):
    return [c for c in traces.columns if c.startswith("Coeff")]


def _pixel(order_trace, input_dir):
    traces = _load(order_trace, input_dir)
    return order_trace.pixel_deviations(traces, _coeff_cols(traces))


def _jumps(order_trace, input_dir):
    return order_trace.find_jumps(_pixel(order_trace, input_dir)[1])


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
        ("obs_time", "era"),
        [
            ("2023-01-01 12:00", "1.0"),
            ("2024-03-01 12:00", "2.0"),
            ("2024-02-05 12:00", None),
            # 2024-02-23 is split: era 1.5 until 12:00 UT, era 2.0 after
            ("2024-02-23 11:30", None),
            ("2024-02-23 15:30", "2.0"),
        ],
    )
    def test_era_of(self, order_trace, obs_time, era):
        eras = order_trace.load_science_eras()
        assert order_trace.era_of(pd.Timestamp(obs_time), eras) == era


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

    @pytest.mark.parametrize("content", ["", "Chip,Fiber,Order\n"])
    def test_empty_csv_exits_naming_it(self, order_trace, tmp_path, content):
        path = _write_trace(tmp_path, "20230101")
        path.write_text(content)
        with pytest.raises(SystemExit, match=f"empty order-trace CSV: {path}"):
            _load(order_trace, tmp_path)

    def test_absent_higher_coefficient_is_zero(self, order_trace, tmp_path):
        # A degree change between runs: two epochs carry a Coeff3, three do not.
        dates = _datecodes("2023-01-01", 5)
        for dc in dates[:2]:
            _write_trace(tmp_path, dc, Coeff3=0.0)
        for dc in dates[2:]:
            _write_trace(tmp_path, dc)
        traces = _load(order_trace, tmp_path)
        assert (traces["Coeff3"] == 0.0).all()
        pixel, epochs = order_trace.pixel_deviations(traces, _coeff_cols(traces))
        assert (pixel["Epochs"] == 5).all()
        assert epochs["Epoch"].nunique() == 5


# ---------------------------------------------------------------------------
# Deviations
# ---------------------------------------------------------------------------


class TestPixelDeviations:
    def test_identical_epochs_have_zero_deviation(self, order_trace, tmp_path):
        for dc in _datecodes("2023-01-01", 5):
            _write_trace(tmp_path, dc)
        pixel, epochs = _pixel(order_trace, tmp_path)
        assert len(pixel) == 4
        assert (pixel["Epochs"] == 5).all()
        assert np.allclose(pixel[["Typical", "Max"]], 0.0)
        assert set(epochs["Chip"]) == {"GREEN", "RED", "BOTH"}

    def test_offset_epoch_is_measured_in_rows(self, order_trace, tmp_path):
        dates = _datecodes("2023-01-01", 5)
        for dc in dates[:-1]:
            _write_trace(tmp_path, dc)
        _write_trace(tmp_path, dates[-1], offset=0.3)
        pixel, _ = _pixel(order_trace, tmp_path)
        assert np.allclose(pixel["Max"], 0.3)

    def test_missing_rows_are_ignored(self, order_trace, tmp_path):
        dates = _datecodes("2023-01-01", 5)
        for dc in dates[:-1]:
            _write_trace(tmp_path, dc)
        _write_trace(tmp_path, dates[-1], offset=50.0, Status="missing")
        pixel, _ = _pixel(order_trace, tmp_path)
        assert (pixel["Epochs"] == 4).all()
        assert np.allclose(pixel["Max"], 0.0)

    def test_deviation_outside_fitted_span_is_masked(self, order_trace, tmp_path):
        # The tilted epoch departs from the others everywhere but column 0, the
        # only column its span covers.
        dates = _datecodes("2023-01-01", 5)
        for dc in dates[:-1]:
            _write_trace(tmp_path, dc)
        _write_trace(tmp_path, dates[-1], Coeff1=2e-3, X1=0.0, X2=0.0, Status="partial")
        pixel, _ = _pixel(order_trace, tmp_path)
        assert np.allclose(pixel["Max"], 0.0)

    def test_all_missing_trace_is_dropped(self, order_trace, tmp_path):
        for dc in _datecodes("2023-01-01", 5):
            _write_trace(tmp_path, dc, Status="missing")
        pixel, _ = _pixel(order_trace, tmp_path)
        assert pixel.empty

    def test_too_few_epochs_yields_empty_tables(self, order_trace, tmp_path):
        for dc in _datecodes("2023-01-01", order_trace.MIN_EPOCHS - 1):
            _write_trace(tmp_path, dc)
        pixel, epochs = _pixel(order_trace, tmp_path)
        assert pixel.empty and list(pixel.columns) == order_trace._PIXEL_COLUMNS
        assert epochs.empty and list(epochs.columns) == order_trace._EPOCH_COLUMNS


class TestCoefficientDeviations:
    def _coeffs(self, order_trace, input_dir):
        traces = _load(order_trace, input_dir)
        return order_trace.coefficient_deviations(traces, _coeff_cols(traces))

    def test_offset_is_recovered_in_its_coefficient_only(self, order_trace, tmp_path):
        dates = _datecodes("2023-01-01", 5)
        for dc in dates[:-1]:
            _write_trace(tmp_path, dc)
        _write_trace(tmp_path, dates[-1], Coeff1=1e-3 + 5e-5)
        coeffs = self._coeffs(order_trace, tmp_path).set_index("Coeff")
        assert np.allclose(coeffs.loc["Coeff1", "Max"], 5e-5)
        assert np.allclose(coeffs.loc[["Coeff0", "Coeff2"], "Max"], 0.0)

    def test_partial_traces_are_excluded(self, order_trace, tmp_path):
        dates = _datecodes("2023-01-01", 5)
        for dc in dates[:3]:
            _write_trace(tmp_path, dc)
        for dc in dates[3:]:
            _write_trace(tmp_path, dc, Coeff1=5e-3, Status="partial")
        assert np.allclose(self._coeffs(order_trace, tmp_path)["Max"], 0.0)

    def test_too_few_full_epochs_yields_empty_table(self, order_trace, tmp_path):
        dates = _datecodes("2023-01-01", 5)
        for dc in dates[:2]:
            _write_trace(tmp_path, dc)
        for dc in dates[2:]:
            _write_trace(tmp_path, dc, Status="partial")
        coeffs = self._coeffs(order_trace, tmp_path)
        assert coeffs.empty and list(coeffs.columns) == order_trace._COEFF_COLUMNS


# ---------------------------------------------------------------------------
# Jumps
# ---------------------------------------------------------------------------


class TestFindJumps:
    def test_planted_step_is_found_once_near_its_date(self, order_trace, tmp_path):
        # The median windows locate a clean step only to within a few epochs:
        # boundaries next to it score nearly the same, and the scatter picks one.
        # Both chips step, and their two detections merge into one date.
        first_after = pd.Timestamp(_write_stepped_era(tmp_path))
        jumps = _jumps(order_trace, tmp_path)
        assert sorted(jumps["Chip"]) == ["GREEN", "RED"]
        (date,) = order_trace.jump_dates(jumps, "1.0")
        assert abs((date - first_after).days) <= order_trace.JUMP_MERGE_DAYS

    def test_two_separated_steps_are_two_jumps(self, order_trace, tmp_path):
        _write_levels(tmp_path, [0.0] * 20 + [0.05] * 20 + [0.1] * 20)
        jumps = _jumps(order_trace, tmp_path)
        assert len(order_trace.jump_dates(jumps, "1.0")) == 2

    def test_step_below_threshold_is_not_a_jump(self, order_trace, tmp_path):
        _write_stepped_era(tmp_path, step=0.01)
        jumps = _jumps(order_trace, tmp_path)
        assert jumps.empty and list(jumps.columns) == order_trace._JUMP_COLUMNS

    def test_single_bad_flat_is_not_a_jump(self, order_trace, tmp_path):
        _write_levels(tmp_path, [0.0] * 15 + [0.1] + [0.0] * 14)
        assert _jumps(order_trace, tmp_path).empty

    def test_too_few_epochs_warns_instead(self, order_trace, tmp_path, caplog):
        _write_levels(tmp_path, [0.0] * (2 * order_trace.JUMP_WINDOW - 1))
        jumps = _jumps(order_trace, tmp_path)
        assert jumps.empty
        assert "too few to detect jumps" in caplog.text


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _run_main(order_trace, monkeypatch, input_dir, out, date_range):
    # setup_batch_logging installs root handlers; stub it so the run's logging
    # cannot leak into the rest of the suite.
    monkeypatch.setattr(
        order_trace,
        "setup_batch_logging",
        lambda *a, **k: ("analyze_order_trace_x", "/logs/x.log"),
    )
    order_trace.main(
        [
            "--date_range",
            *date_range,
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(out),
        ]  # fmt: skip
    )
    return out / "analysis" / "order_trace"


class TestMain:
    def test_writes_reports_under_output_dir(self, order_trace, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        _write_stepped_era(input_dir)
        out_dir = _run_main(
            order_trace,
            monkeypatch,
            input_dir,
            tmp_path / "out",
            ("20230101", "20230131"),
        )

        schemas = {
            "pixel": order_trace._PIXEL_COLUMNS,
            "epochs": order_trace._EPOCH_COLUMNS,
            "coefficients": order_trace._COEFF_COLUMNS,
            "jumps": order_trace._JUMP_COLUMNS,
        }
        tables = {
            name: pd.read_csv(out_dir / f"order_trace_{name}_20230101_20230131.csv")
            for name in schemas
        }
        for name, columns in schemas.items():
            assert list(tables[name].columns) == columns
        assert len(list(out_dir.glob("*.png"))) == 6
        assert len(tables["jumps"]) == 2
        assert (tables["epochs"].groupby("Epoch")["Chip"].nunique() == 3).all()
        assert pd.to_datetime(tables["epochs"]["Date"]).notna().all()
        assert np.allclose(tables["pixel"]["Max"], 0.05, atol=0.01)

    def test_multi_era_run(self, order_trace, tmp_path, monkeypatch, caplog):
        caplog.set_level(logging.INFO)
        input_dir = tmp_path / "in"
        _write_levels(input_dir, [0.0] * 10 + [0.05] * 20, start="2023-01-01")
        _write_levels(input_dir, [0.0] * 20, start="2024-03-01")  # era 2.0
        _write_levels(input_dir, [0.0] * 2, start="2025-01-10")  # era 2.6
        _write_levels(input_dir, [0.0] * 5, start="2025-05-01")  # era 3.0
        out_dir = _run_main(
            order_trace,
            monkeypatch,
            input_dir,
            tmp_path / "out",
            ("20221109", "20251231"),
        )

        pixel = pd.read_csv(
            out_dir / "order_trace_pixel_20221109_20251231.csv", dtype={"Era": str}
        )
        assert set(pixel["Era"]) == {"1.0", "2.0", "3.0"}
        assert len(list(out_dir.glob("*.png"))) == 6
        assert (
            "era 2.6: no trace has 3+ epochs (2 in range); not analyzed" in caplog.text
        )
        assert "era 1.0: 1 jumps" in caplog.text
        assert "era 2.0: 0 jumps" in caplog.text
        assert "era 3.0: jump detection skipped (5 epochs < 16)" in caplog.text
        assert "worst 10 traces:" in caplog.text

    def test_era_without_enough_epochs_exits(self, order_trace, tmp_path, monkeypatch):
        _write_trace(tmp_path / "in", "20230101")
        with pytest.raises(SystemExit, match="trace epochs"):
            _run_main(
                order_trace,
                monkeypatch,
                tmp_path / "in",
                tmp_path,
                ("20221109", "20261231"),
            )


# ---------------------------------------------------------------------------
# kpfpipe analyze: the second-level dispatcher that reaches this script
# ---------------------------------------------------------------------------


class TestAnalyzeDispatch:
    def test_order_trace_is_registered(self):
        assert cli._ANALYSES["order_trace"] is _order_trace.main

    def test_usage_lists_order_trace(self, capsys):
        assert cli.main(["analyze"]) == 0
        assert "order_trace" in capsys.readouterr().out
