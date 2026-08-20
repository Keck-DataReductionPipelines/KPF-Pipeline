"""Tests for scripts/processing/_scan.py: up-front L0 mini-db cache warming.

_scan owns the parallel-by-datecode header scan the masters and science
orchestrators run before their fan-out, so the fanned-out reduces read the mini-db
cache read-only. Frames are synthetic FITS in temp trees; no testdata is needed.
"""

from pathlib import Path

import pytest

from scripts.processing import _scan

from ._scripts import write_l0_tree

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli


def _cache_path(data_input, datecode):
    return Path(data_input) / "vNext" / "mini_db" / f"{datecode}_L0.csv"


# ---------------------------------------------------------------------------
# scan_night_to_cache
# ---------------------------------------------------------------------------


class TestScanNightToCache:
    def test_scans_and_writes_cache(self, tmp_path):
        write_l0_tree(str(tmp_path), "20240101", 3600)
        write_l0_tree(str(tmp_path), "20240101", 3700)

        df = _scan.scan_night_to_cache(str(tmp_path), "20240101")

        assert df is not None
        assert len(df) == 2
        assert _cache_path(str(tmp_path), "20240101").is_file()  # default cache="rw"

    def test_read_only_mode_does_not_write(self, tmp_path):
        # Recipes read the cache; only the scripts layer writes it.
        write_l0_tree(str(tmp_path), "20240101", 3600)

        df = _scan.scan_night_to_cache(str(tmp_path), "20240101", cache="r")

        assert df is not None and len(df) == 1
        assert not _cache_path(str(tmp_path), "20240101").exists()

    def test_empty_night_returns_none(self, tmp_path):
        # An empty datecode dir raises ValueError inside build_mini_database; _scan
        # swallows it and returns None rather than aborting the batch.
        (Path(tmp_path) / "L0" / "20240101").mkdir(parents=True)
        assert _scan.scan_night_to_cache(str(tmp_path), "20240101") is None
        assert not _cache_path(str(tmp_path), "20240101").exists()

    def test_absent_night_returns_none(self, tmp_path):
        assert _scan.scan_night_to_cache(str(tmp_path), "20240101") is None


# ---------------------------------------------------------------------------
# scan_datecodes: generic parallel dispatcher
# ---------------------------------------------------------------------------


class TestScanDatecodes:
    def test_returns_per_night_results(self, tmp_path):
        datecodes = ["20240101", "20240102", "20240103"]
        results = _scan.scan_datecodes(datecodes, jobs=3, worker=lambda dc: (dc, ""))
        assert sorted(results) == datecodes  # order-agnostic: the pool is unordered

    def test_tolerates_jobs_exceeding_datecodes(self, tmp_path):
        results = _scan.scan_datecodes(["20240101"], jobs=8, worker=lambda dc: (dc, ""))
        assert results == ["20240101"]

    def test_empty_datecodes(self, tmp_path):
        assert _scan.scan_datecodes([], jobs=4, worker=lambda dc: (dc, "")) == []

    def test_threaded_scan_no_contamination(self, tmp_path):
        # Each night gets its own FileHandler inside scan_night_to_cache, so a pooled
        # scan never collapses nights via a shared self._mini_db.
        nights = [f"202401{d:02d}" for d in range(1, 7)]
        expected = {}
        for dc in nights:
            ids = {write_l0_tree(str(tmp_path), dc, 3600 + j * 100) for j in range(4)}
            expected[dc] = ids

        def _worker(dc):
            df = _scan.scan_night_to_cache(str(tmp_path), dc)
            obs_ids = {fn.split("/")[-1][:-5] for fn in df["FILENAME"]}
            return (frozenset(obs_ids), "")

        results = _scan.scan_datecodes(nights, jobs=8, worker=_worker)
        assert set(results) == {frozenset(v) for v in expected.values()}


# ---------------------------------------------------------------------------
# warm_mini_db_caches: side-effect entry point, fail-soft
# ---------------------------------------------------------------------------


class TestWarmMiniDbCaches:
    def test_writes_all_and_counts(self, tmp_path):
        nights = ["20240101", "20240102"]
        for dc in nights:
            write_l0_tree(str(tmp_path), dc, 3600)

        written, skipped = _scan.warm_mini_db_caches(str(tmp_path), nights, jobs=2)

        assert (written, skipped) == (2, 0)
        for dc in nights:
            assert _cache_path(str(tmp_path), dc).is_file()

    def test_read_only_mode_skips_prescan(self, tmp_path):
        # A read-only mode warms nothing: every night is reported skipped, unscanned.
        nights = ["20240101", "20240102"]
        for dc in nights:
            write_l0_tree(str(tmp_path), dc, 3600)

        written, skipped = _scan.warm_mini_db_caches(
            str(tmp_path), nights, jobs=2, cache="r"
        )

        assert (written, skipped) == (0, 2)
        for dc in nights:
            assert not _cache_path(str(tmp_path), dc).exists()

    def test_empty_night_counted_skipped(self, tmp_path):
        write_l0_tree(str(tmp_path), "20240101", 3600)  # good
        (Path(tmp_path) / "L0" / "20240102").mkdir(parents=True)  # empty

        written, skipped = _scan.warm_mini_db_caches(
            str(tmp_path), ["20240101", "20240102"], jobs=2
        )

        assert (written, skipped) == (1, 1)

    def test_fail_soft_on_pool_error(self, tmp_path, monkeypatch):
        # A pool-level failure must never abort the batch: warm reports every night
        # skipped so the reduces fall back to in-process scans.
        def _boom(datecodes, jobs, worker, *, label="scanning"):
            raise RuntimeError("pool exploded")

        monkeypatch.setattr(_scan, "scan_datecodes", _boom)
        written, skipped = _scan.warm_mini_db_caches(
            str(tmp_path), ["20240101", "20240102"], jobs=2
        )
        assert (written, skipped) == (0, 2)
