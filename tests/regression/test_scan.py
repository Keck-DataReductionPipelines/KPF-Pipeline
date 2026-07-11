"""Tests for scripts/processing/_scan.py: up-front L0 mini-db cache warming.

_scan owns the in-process, parallel-by-datecode header scan the masters and
science orchestrators run before their fan-out, so the fanned-out reduces read the
mini-db cache read-only. These cover the per-night primitive, the generic parallel
dispatcher (including the per-thread-FileHandler no-contamination invariant), and
the fail-soft warm entry point.

Unit tests use synthetic FITS frames in temp trees -- no real testdata needed.
"""

from pathlib import Path

from astropy.io import fits

from scripts.processing import _scan


def _write_l0(data_input, datecode, seconds, obj="10700", imtype="Object"):
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
    return obs_id


def _cache_path(data_input, datecode):
    return Path(data_input) / "vNext" / "mini_db" / f"{datecode}_L0.csv"


# ---------------------------------------------------------------------------
# scan_night_to_cache
# ---------------------------------------------------------------------------


class TestScanNightToCache:
    def test_scans_and_writes_cache(self, tmp_path):
        _write_l0(str(tmp_path), "20240101", 3600)
        _write_l0(str(tmp_path), "20240101", 3700)

        df = _scan.scan_night_to_cache(str(tmp_path), "20240101")

        assert df is not None
        assert len(df) == 2
        assert _cache_path(str(tmp_path), "20240101").is_file()  # cache="rw" wrote it

    def test_empty_night_returns_none(self, tmp_path):
        # A datecode dir with no FITS files -> ValueError inside build_mini_database,
        # swallowed here: returns None, no cache written, no raise.
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
        assert sorted(results) == datecodes  # one result per night, order-agnostic

    def test_tolerates_jobs_exceeding_datecodes(self, tmp_path):
        results = _scan.scan_datecodes(["20240101"], jobs=8, worker=lambda dc: (dc, ""))
        assert results == ["20240101"]

    def test_empty_datecodes(self, tmp_path):
        assert _scan.scan_datecodes([], jobs=4, worker=lambda dc: (dc, "")) == []

    def test_threaded_scan_no_contamination(self, tmp_path):
        # Each night gets its own FileHandler inside scan_night_to_cache, so a pooled
        # scan never collapses nights via a shared self._mini_db. Worker returns each
        # night's obs_id set; assert every night is exact under an 8-wide pool.
        nights = [f"202401{d:02d}" for d in range(1, 7)]  # six nights
        expected = {}
        for dc in nights:
            ids = {_write_l0(str(tmp_path), dc, 3600 + j * 100) for j in range(4)}
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
            _write_l0(str(tmp_path), dc, 3600)

        written, skipped = _scan.warm_mini_db_caches(str(tmp_path), nights, jobs=2)

        assert (written, skipped) == (2, 0)
        for dc in nights:
            assert _cache_path(str(tmp_path), dc).is_file()

    def test_empty_night_counted_skipped(self, tmp_path):
        _write_l0(str(tmp_path), "20240101", 3600)  # good
        (Path(tmp_path) / "L0" / "20240102").mkdir(parents=True)  # empty

        written, skipped = _scan.warm_mini_db_caches(
            str(tmp_path), ["20240101", "20240102"], jobs=2
        )

        assert (written, skipped) == (1, 1)

    def test_fail_soft_on_pool_error(self, tmp_path, monkeypatch):
        # A pool-level surprise must never abort the batch: warm swallows it and
        # reports every night as skipped so the reduces fall back to in-process scans.
        def _boom(datecodes, jobs, worker, *, label="scanning"):
            raise RuntimeError("pool exploded")

        monkeypatch.setattr(_scan, "scan_datecodes", _boom)
        written, skipped = _scan.warm_mini_db_caches(
            str(tmp_path), ["20240101", "20240102"], jobs=2
        )
        assert (written, skipped) == (0, 2)
