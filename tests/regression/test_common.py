"""Tests for scripts/processing/_common.py: the tools-free datecode helper.

Covers ``_datecode_dirs`` -- used by the masters orchestrator to expand a datecode
range against the L0 tree. Synthetic tmp trees only -- no real testdata needed.
"""

from scripts.processing import _common


class TestDatecodeDirs:
    def test_filters_by_range_and_sorts(self, tmp_path):
        for name in ["20240101", "20240115", "20240201", "notadate", "20231231"]:
            (tmp_path / name).mkdir()
        (tmp_path / "20240110_file").write_text("x")  # datecode-ish, but not a dir
        got = _common._datecode_dirs(str(tmp_path), "20240101", "20240131")
        assert got == ["20240101", "20240115"]
