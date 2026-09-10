"""Tests for scripts/analysis/thar.py: the ThAr lamp-serial analysis.

thar is a read-only scan: it picks each night's ThAr frames out of the L0 mini
database, reopens them for their lamp cards, and writes one CSV row per frame.
These cover what it owns -- arg parsing, the per-frame row, the threaded scan's
filtering and ordering, the CSV schema, and main() end to end -- plus the
``kpfpipe analyze`` dispatch that reaches it.

Unit tests use synthetic FITS frames in temp trees -- no real testdata needed.
"""

import csv
from pathlib import Path

import pytest

import tools.cli as cli
from scripts.analysis import thar as _thar

from ._scripts import add_junk_obs_id, write_l0_tree

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

_BASE_ARGS = ["--date_range", "20240101", "20240131", "--input_dir", "/in"]

# The lamp cards a real ThAr frame carries; see notes/thar_lamp_health.md.
_LAMP_CARDS = {
    "OCTAGON": "Th_daily",
    "HCLSN": "L74828",
    "THDAYON": "20240101T15:00:09 HST",
    "THDAYTON": 3475.3,
    "THAUON": "20230503T12:39:28 HST",
    "THAUTON": 1465.7,
}


@pytest.fixture(scope="module")
def thar():
    return _thar


def _write_thar(data_input, datecode, seconds, *, obj="autocal-thar-all-eve", **cards):
    """Write one ThAr L0 frame; `cards` override the default lamp cards."""
    return write_l0_tree(
        data_input,
        datecode,
        seconds,
        obj=obj,
        imtype="Arclamp",
        extra_cards={**_LAMP_CARDS, **cards},
    )


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_minimal_valid(self, thar):
        ns = thar.parse_args(_BASE_ARGS + ["--output_dir", "/out"])
        assert ns.date_range == ["20240101", "20240131"]
        assert ns.kpf_data_input == "/in"

    def test_output_dir_fans_out_to_analysis_and_log(self, thar):
        ns = thar.parse_args(_BASE_ARGS + ["--output_dir", "/out"])
        assert ns.analysis_dir == "/out/analysis"
        assert ns.log_dir == "/out/logs"

    def test_explicit_dirs_override_output_dir(self, thar):
        ns = thar.parse_args(
            _BASE_ARGS + ["--output_dir", "/out", "--analysis_dir", "/a"]
        )
        assert ns.analysis_dir == "/a" and ns.log_dir == "/out/logs"

    def test_explicit_dirs_without_output_dir(self, thar):
        ns = thar.parse_args(_BASE_ARGS + ["--analysis_dir", "/a", "--log_dir", "/l"])
        assert ns.analysis_dir == "/a" and ns.log_dir == "/l"

    def test_cache_defaults_to_rw(self, thar):
        # Analysis is often the first thing to read a night, so it warms the cache.
        assert thar.parse_args(_BASE_ARGS + ["--output_dir", "/out"]).cache == "rw"

    def test_jobs_unset_stays_none(self, thar):
        # Left None so main() falls back to _default_science_jobs().
        assert thar.parse_args(_BASE_ARGS + ["--output_dir", "/out"]).jobs is None

    @pytest.mark.parametrize(
        "extra",
        [
            ["--date_range", "2024", "20240131"],  # malformed datecode
            ["--date_range", "20240201", "20240101"],  # start > end
            ["--jobs", "0"],  # below 1
        ],
    )
    def test_invalid_args_exit(self, thar, capsys, extra):
        argv = ["--input_dir", "/in", "--output_dir", "/out"]
        if "--date_range" not in extra:
            argv += ["--date_range", "20240101", "20240131"]
        with pytest.raises(SystemExit) as exc:
            thar.parse_args(argv + extra)
        assert exc.value.code == 2
        assert extra[0] in capsys.readouterr().err

    @pytest.mark.parametrize(
        "argv, missing",
        [
            (["--input_dir", "/in", "--output_dir", "/out"], "--date_range"),
            (
                ["--date_range", "20240101", "20240131", "--output_dir", "/out"],
                "--kpf_data_input",
            ),
        ],
    )
    def test_required_flags(self, thar, capsys, argv, missing):
        with pytest.raises(SystemExit) as exc:
            thar.parse_args(argv)
        assert exc.value.code == 2
        assert missing in capsys.readouterr().err

    def test_no_output_dirs_at_all_exits(self, thar, capsys):
        # Analysis reads no recipe TOML, so nothing else can supply these.
        with pytest.raises(SystemExit) as exc:
            thar.parse_args(_BASE_ARGS)
        assert exc.value.code == 2
        assert "--output_dir" in capsys.readouterr().err

    def test_partial_output_dirs_exit(self, thar, capsys):
        # --analysis_dir alone leaves log_dir unset; the run would have no log.
        with pytest.raises(SystemExit) as exc:
            thar.parse_args(_BASE_ARGS + ["--analysis_dir", "/a"])
        assert exc.value.code == 2
        assert "--output_dir" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _frame_row
# ---------------------------------------------------------------------------


class TestFrameRow:
    def _record(self, tmp_path, **cards):
        obs_id = _write_thar(str(tmp_path), "20240101", 3600, **cards)
        path = Path(tmp_path) / "L0" / "20240101" / f"{obs_id}.fits"
        return {
            "FILENAME": str(path),
            "OBJECT": "autocal-thar-all-eve",
            "EXPTIME": 20.0,
            "ELAPSED": 20.0,
        }, obs_id

    def test_carries_mini_db_and_lamp_columns(self, thar, tmp_path):
        record, obs_id = self._record(tmp_path)
        row = thar._frame_row(record)
        assert row["OBS_ID"] == obs_id and row["DATECODE"] == "20240101"
        assert row["OBJECT"] == "autocal-thar-all-eve" and row["EXPTIME"] == 20.0
        assert row["HCLSN"] == "L74828" and row["OCTAGON"] == "Th_daily"
        assert set(row) == set(thar.CSV_COLUMNS)

    def test_absent_card_becomes_empty_field(self, thar, tmp_path):
        # HCLSN is genuinely absent on some early nights; the row must still be
        # written, with the gap visible rather than the frame dropped.
        record, _ = self._record(tmp_path, HCLSN="")
        assert thar._frame_row(record)["HCLSN"] == ""


# ---------------------------------------------------------------------------
# scan_lamp_serial_numbers
# ---------------------------------------------------------------------------


class TestScanLampSerialNumbers:
    def _scan(self, thar, tmp_path, start="20240101", end="20240131", jobs=4):
        return thar.scan_lamp_serial_numbers(str(tmp_path), start, end, jobs)

    def test_selects_every_thar_object_variant(self, thar, tmp_path):
        # Wider than the masters cal_type map on purpose: the fiber-specific cals
        # light the same lamp, and early nights carry only those.
        expected = {
            _write_thar(str(tmp_path), "20240101", 3600 + i * 100, obj=obj)
            for i, obj in enumerate(
                [
                    "autocal-thar-all-eve",
                    "autocal-thar-all-morn",
                    "autocal-thar-all",
                    "autocal-thar-sci",
                    "autocal-thar-sky",
                    "autocal-thar-cal",
                    "autocal-thar-hk",
                ]
            )
        }
        rows = self._scan(thar, tmp_path)
        assert {r["OBS_ID"] for r in rows} == expected

    def test_excludes_non_thar_frames(self, thar, tmp_path):
        thar_id = _write_thar(str(tmp_path), "20240101", 3600)
        write_l0_tree(str(tmp_path), "20240101", 3700, obj="autocal-bias")
        write_l0_tree(str(tmp_path), "20240101", 3800, obj="10700")
        assert [r["OBS_ID"] for r in self._scan(thar, tmp_path)] == [thar_id]

    def test_excludes_junk(self, thar, tmp_path):
        good = _write_thar(str(tmp_path), "20240101", 3600)
        junk = _write_thar(str(tmp_path), "20240101", 3700)
        add_junk_obs_id(str(tmp_path), junk)
        assert [r["OBS_ID"] for r in self._scan(thar, tmp_path)] == [good]

    def test_rows_are_chronological_across_nights(self, thar, tmp_path):
        # Nights complete out of pool order; the report must still read forward.
        for dc in ["20240103", "20240101", "20240102"]:
            _write_thar(str(tmp_path), dc, 3600)
        rows = self._scan(thar, tmp_path, jobs=4)
        assert [r["DATECODE"] for r in rows] == ["20240101", "20240102", "20240103"]

    def test_matches_serial_result(self, thar, tmp_path):
        for dc in ["20240101", "20240102", "20240103", "20240104"]:
            for i in range(3):
                _write_thar(str(tmp_path), dc, 3600 + i * 100)
        assert self._scan(thar, tmp_path, jobs=1) == self._scan(thar, tmp_path, jobs=8)

    def test_honours_date_range_bounds(self, thar, tmp_path):
        _write_thar(str(tmp_path), "20231231", 3600)
        inside = _write_thar(str(tmp_path), "20240101", 3600)
        _write_thar(str(tmp_path), "20240201", 3600)
        rows = self._scan(thar, tmp_path, start="20240101", end="20240131")
        assert [r["OBS_ID"] for r in rows] == [inside]

    def test_night_without_thar_is_skipped_not_fatal(self, thar, tmp_path):
        write_l0_tree(str(tmp_path), "20240101", 3600, obj="autocal-bias")
        good = _write_thar(str(tmp_path), "20240102", 3600)
        assert [r["OBS_ID"] for r in self._scan(thar, tmp_path)] == [good]

    def test_missing_l0_root_exits(self, thar, tmp_path):
        with pytest.raises(SystemExit, match="L0 input directory not found"):
            self._scan(thar, tmp_path)

    def test_no_nights_in_range_exits(self, thar, tmp_path):
        _write_thar(str(tmp_path), "20230311", 3600)
        with pytest.raises(SystemExit, match="no datecode dirs"):
            self._scan(thar, tmp_path)

    def test_no_thar_frames_at_all_exits(self, thar, tmp_path):
        # An empty CSV would read as "no lamp changes" rather than "nothing found".
        write_l0_tree(str(tmp_path), "20240101", 3600, obj="autocal-bias")
        with pytest.raises(SystemExit, match="no ThAr frames"):
            self._scan(thar, tmp_path)


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------


class TestWriteCsv:
    def test_schema_and_contents_round_trip(self, thar, tmp_path):
        row = dict.fromkeys(thar.CSV_COLUMNS, "")
        row.update({"OBS_ID": "KP.20240101.03600.00", "HCLSN": "L74828"})
        path = thar.write_csv([row], str(tmp_path / "thar"), "20240101", "20240131")

        assert Path(path).name == "thar_lamps_20240101_20240131.csv"
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            # Pins the report schema: a reader of last year's CSV must still parse.
            assert reader.fieldnames == thar.CSV_COLUMNS
            assert [r["HCLSN"] for r in reader] == ["L74828"]

    def test_creates_missing_output_directory(self, thar, tmp_path):
        out_dir = tmp_path / "analysis" / "thar"
        thar.write_csv([], str(out_dir), "20240101", "20240131")
        assert out_dir.is_dir()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    def test_writes_report_under_output_dir(self, thar, tmp_path, monkeypatch):
        # setup_batch_logging installs root handlers; stub it so the run's logging
        # cannot leak into the rest of the suite.
        monkeypatch.setattr(
            thar,
            "setup_batch_logging",
            lambda *a, **k: ("analyze_thar_x", "/logs/x.log"),
        )
        data_input, out = tmp_path / "in", tmp_path / "out"
        _write_thar(str(data_input), "20240101", 3600)
        _write_thar(str(data_input), "20240102", 3600, HCLSN="L82906")

        thar.main(
            [
                "--date_range",
                "20240101",
                "20240131",
                "--input_dir",
                str(data_input),
                "--output_dir",
                str(out),
            ]  # fmt: skip
        )

        report = out / "analysis" / "thar" / "thar_lamps_20240101_20240131.csv"
        with open(report, newline="") as f:
            rows = list(csv.DictReader(f))
        assert [r["HCLSN"] for r in rows] == ["L74828", "L82906"]


# ---------------------------------------------------------------------------
# kpfpipe analyze: the second-level dispatcher that reaches this script
# ---------------------------------------------------------------------------


class TestAnalyzeDispatch:
    def test_routes_to_subject_with_forwarded_argv(self, monkeypatch):
        seen = []
        monkeypatch.setitem(cli._ANALYSES, "thar", lambda rest: seen.append(rest))
        cli.main(["analyze", "thar", "--date_range", "20240101", "20240131"])
        assert seen == [["--date_range", "20240101", "20240131"]]

    def test_return_value_propagates(self, monkeypatch):
        monkeypatch.setitem(cli._ANALYSES, "thar", lambda rest: 7)
        assert cli.main(["analyze", "thar"]) == 7

    def test_thar_is_registered(self):
        assert cli._ANALYSES["thar"] is _thar.main

    @pytest.mark.parametrize("argv", [["analyze"], ["analyze", "-h"]])
    def test_usage_returns_zero(self, capsys, argv):
        assert cli.main(argv) == 0
        assert "usage: kpfpipe analyze <subject>" in capsys.readouterr().out

    def test_unknown_subject_exits_two(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["analyze", "nonsense"])
        assert exc.value.code == 2
        assert "unknown subject 'nonsense'" in capsys.readouterr().err
