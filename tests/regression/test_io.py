"""Tests for kpfpipe.utils.io: the FileHandler (mini-database, calibration-stack
clustering, masters finder), the product-path builders (kpf_directory /
kpf_filename / kpf_filepath), and junk-frame exclusion.

Unit tests use synthetic DataFrames and temp directories — no real data needed.
Integration tests (slow) use real L0 data from tests/testdata/L0/20240405/.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.utils.io import (
    FileHandler,
    kpf_directory,
    kpf_filename,
    kpf_filepath,
    load_junk_obs_ids,
)
from kpfpipe.utils.kpf_utils import get_timestamp, utc_to_hst

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"


# ---------------------------------------------------------------------------
# Synthetic test data setup (unit tests only)
# ---------------------------------------------------------------------------

# Synthetic filenames: KP.YYYYMMDD.SSSSS.FF.fits
# Two bias clusters separated by a >2hr gap; one dark cluster; two ThAr clusters
# (morning and evening) with different OBJECT suffixes; science frames.
_BIAS_A = [
    f"/data/L0/20240405/KP.20240405.0{3600 + i * 100:04d}.00.fits" for i in range(5)
]  # 03600–04000
_BIAS_B = [
    f"/data/L0/20240405/KP.20240405.{14000 + i * 100:05d}.00.fits" for i in range(5)
]  # 14000–14400
_DARK_A = [
    f"/data/L0/20240405/KP.20240405.{18000 + i * 100:05d}.00.fits" for i in range(3)
]  # 18000–18200
_THAR_MORN = [
    f"/data/L0/20240405/KP.20240405.{60000 + i * 100:05d}.00.fits" for i in range(5)
]  # 60000–60400
_THAR_EVE = [
    f"/data/L0/20240405/KP.20240405.{75000 + i * 100:05d}.00.fits" for i in range(5)
]  # 75000–75400
_SCI_A = [
    f"/data/L0/20240405/KP.20240405.{50000 + i * 100:05d}.00.fits" for i in range(2)
]  # 50000–50100


def _rows(files, obj, imtype, targname=None):
    """Row dicts for a group of frames sharing OBJECT/IMTYPE/TARGNAME."""
    return [
        {"FILENAME": f, "IMTYPE": imtype, "OBJECT": obj, "TARGNAME": targname}
        for f in files
    ]


def _mini_db(rows, *, exptime=60.0, junk=()):
    """Assemble a synthetic mini database from row dicts.

    Adds the derived columns build_calibration_stacks needs: EXPTIME/ELAPSED, the
    UTC/HST timestamps (parsed from each FILENAME), and ISJUNK (the FILENAMEs in
    `junk` are flagged True). This is the single assembly path for every layout
    fixture below.
    """
    df = pd.DataFrame(rows)
    df["EXPTIME"] = exptime
    df["ELAPSED"] = exptime
    df["UTC"] = [get_timestamp(f) for f in df["FILENAME"]]
    df["HST"] = [utc_to_hst(t) for t in df["UTC"]]
    df["ISJUNK"] = df["FILENAME"].isin(set(junk))
    return df


def _make_mini_db():
    return _mini_db(
        _rows(_BIAS_A, "autocal-bias", "Bias")
        + _rows(_BIAS_B, "autocal-bias", "Bias")
        + _rows(_DARK_A, "autocal-dark", "Dark")
        + _rows(_THAR_MORN, "autocal-thar-all-morn", "Arclamp")
        + _rows(_THAR_EVE, "autocal-thar-all-eve", "Arclamp")
        + _rows(_SCI_A, "185144", "Object", "185144")
    )


_BIAS_SMALL = [
    f"/data/L0/20240405/KP.20240405.{30000 + i * 100:05d}.00.fits" for i in range(2)
]  # 30000–30100, a >2hr gap after _BIAS_A → a separate 2-file cluster


def _mixed_bias_db():
    """Mini_db with one 5-file bias cluster (_BIAS_A) and a later 2-file one."""
    return _mini_db(_rows(_BIAS_A + _BIAS_SMALL, "autocal-bias", "Bias")), _BIAS_SMALL


def _midnight_bias_db(n_before=5, n_after=5):
    """Bias frames straddling HST midnight (UTC 36000) within one UTC directory.

    The before/after groups are <cluster_gap_seconds apart, so only the HST-day
    boundary can separate them. Returns (df, before_files, after_files).
    """
    before = [
        f"/data/L0/20240405/KP.20240405.{35400 + i * 100:05d}.00.fits"  # HST 20240404
        for i in range(n_before)
    ]
    after = [
        f"/data/L0/20240405/KP.20240405.{36100 + i * 100:05d}.00.fits"  # HST 20240405
        for i in range(n_after)
    ]
    df = _mini_db(_rows(before + after, "autocal-bias", "Bias"))
    return df, before, after


def _cross_midnight_gap_db(n_before=2, n_after=2):
    """Sparse dark clusters on opposite HST days, split by a >2 h gap.

    Mirrors a real sparse-dark night (e.g. 20240806): a pre-midnight group and a
    post-midnight group, each below the dark min_file_count and separated by both
    the gap and HST midnight. Returns (df, before_files, after_files).
    """
    before = [
        f"/data/L0/20240405/KP.20240405.{34000 + i * 100:05d}.00.fits"  # HST 20240404
        for i in range(n_before)
    ]
    after = [
        f"/data/L0/20240405/KP.20240405.{50000 + i * 100:05d}.00.fits"  # HST 20240405
        for i in range(n_after)
    ]
    df = _mini_db(_rows(before + after, "autocal-dark", "Dark"), exptime=1200.0)
    return df, before, after


# ---------------------------------------------------------------------------
# FileHandler.build_calibration_stacks (synthetic mini databases)
# ---------------------------------------------------------------------------


def _cluster(cal_type, mini_db, **kwargs):
    """Cluster a synthetic mini_db through the (instance-method) API.

    build_calibration_stacks reads the handler's carried mini database by default;
    these logic tests pass a synthetic one via ``mini_db=`` on a bare handler.
    """
    return FileHandler().build_calibration_stacks(cal_type, mini_db=mini_db, **kwargs)


class TestSecondsSinceJ2000:
    """FileHandler._seconds_since_j2000 is the monotonic sort/gap scalar the
    clustering builds on; exercised directly via a config-free handler."""

    def test_basic(self):
        # J2000.0 itself: 2000-01-01 12:00 UTC = '20000101.43200.00'
        assert FileHandler()._seconds_since_j2000("20000101.43200.00") == 0

    def test_monotonic_across_year_boundary(self):
        # Dec 31 23:59:00 -> Jan 1 00:00:00 should differ by 60s exactly.
        fh = FileHandler()
        end = fh._seconds_since_j2000("20231231.86340.00")
        start_next_year = fh._seconds_since_j2000("20240101.00000.00")
        assert start_next_year - end == 60

    def test_raises_on_invalid_timestamp(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            FileHandler()._seconds_since_j2000("KP.20240405.99999.57.fits")

    def test_raises_when_no_timestamp_found(self):
        with pytest.raises(ValueError, match="No KPF timestamp found"):
            FileHandler()._seconds_since_j2000("notimestamp.fits")


class TestBuildCalibrationStacks:
    """Clustering depends only on the mini database, so these exercise it with
    synthetic DataFrames (no files on disk) via the ``mini_db=`` override."""

    def test_two_bias_clusters_returned_separately(self):
        lists = _cluster("bias", _make_mini_db())
        assert len(lists) == 2

    def test_bias_cluster_a_files(self):
        lists = _cluster("bias", _make_mini_db())
        assert lists[0] == sorted(_BIAS_A)

    def test_bias_cluster_b_files(self):
        lists = _cluster("bias", _make_mini_db())
        assert lists[1] == sorted(_BIAS_B)

    def test_files_are_sorted(self):
        for lst in _cluster("bias", _make_mini_db()):
            assert lst == sorted(lst)

    def test_raises_when_no_cluster_meets_min(self):
        # min_file_count=6: both bias clusters (5 files each) fall below and are
        # dropped, leaving nothing → raises.
        with pytest.raises(ValueError, match="no cluster with at least"):
            _cluster("bias", _make_mini_db(), min_file_count=6)

    def test_raises_when_no_frames_found(self):
        with pytest.raises(ValueError, match="No 'flat' calibration frames found"):
            _cluster("flat", _make_mini_db())

    def test_raises_when_only_cluster_below_default_min(self):
        # dark cluster has only 3 files; default min_file_count=5 → dropped →
        # raises.
        with pytest.raises(ValueError, match="no cluster with at least"):
            _cluster("dark", _make_mini_db())

    def test_drops_small_cluster_keeps_large(self):
        db, _ = _mixed_bias_db()
        lists = _cluster("bias", db)
        assert len(lists) == 1
        assert lists[0] == sorted(_BIAS_A)

    def test_merge_folds_small_into_neighbor(self):
        db, small = _mixed_bias_db()
        lists = _cluster("bias", db, merge_small_clusters=True)
        assert len(lists) == 1
        assert lists[0] == sorted(_BIAS_A + small)

    def test_merge_combines_two_small_clusters(self):
        # Two 5-file bias clusters, each below min=6; merged into one of 10.
        lists = _cluster(
            "bias", _make_mini_db(), min_file_count=6, merge_small_clusters=True
        )
        assert len(lists) == 1
        assert len(lists[0]) == 10

    def test_merge_raises_when_total_below_min(self):
        # 7 bias files total (5 + 2); merging cannot reach min=8 → raises.
        db, _ = _mixed_bias_db()
        with pytest.raises(ValueError, match="no cluster with at least"):
            _cluster("bias", db, min_file_count=8, merge_small_clusters=True)

    def test_hst_midnight_splits_cluster(self):
        # Same-OBJECT frames <gap apart but on opposite sides of HST midnight
        # (UTC 36000) must not share a cluster.
        db, before, after = _midnight_bias_db()
        lists = _cluster("bias", db, min_file_count=5)
        assert len(lists) == 2
        assert lists[0] == sorted(before)
        assert lists[1] == sorted(after)

    def test_hst_midnight_blocks_merge(self):
        # A small post-midnight cluster has no same-HST-day neighbor, so it is
        # dropped rather than merged across midnight into the pre-midnight one.
        db, before, _ = _midnight_bias_db(n_before=5, n_after=2)
        lists = _cluster("bias", db, min_file_count=5, merge_small_clusters=True)
        assert len(lists) == 1
        assert lists[0] == sorted(before)

    def test_no_boundary_keeps_cross_midnight_cluster(self):
        # With enforce_hst_midnight_boundary=False, frames <gap apart across HST
        # midnight stay in one cluster (only cluster_gap_seconds can split them).
        db, before, after = _midnight_bias_db()
        lists = _cluster(
            "bias",
            db,
            min_file_count=5,
            enforce_hst_midnight_boundary=False,
        )
        assert len(lists) == 1
        assert lists[0] == sorted(before + after)

    def test_no_boundary_merges_across_midnight(self):
        # Two sparse dark clusters on opposite HST days (the 20240806 case): with
        # the boundary enforced they cannot merge and both drop; with it lifted
        # they merge into one cluster that meets the threshold.
        db, before, after = _cross_midnight_gap_db()
        with pytest.raises(ValueError, match="no cluster with at least"):
            _cluster("dark", db, min_file_count=3, merge_small_clusters=True)
        lists = _cluster(
            "dark",
            db,
            min_file_count=3,
            merge_small_clusters=True,
            enforce_hst_midnight_boundary=False,
        )
        assert len(lists) == 1
        assert lists[0] == sorted(before + after)

    def test_invalid_imtype_raises(self):
        with pytest.raises(ValueError, match="cal_type must be one of"):
            _cluster("bogus", _make_mini_db())

    def test_thar_returns_two_clusters(self):
        # Morning and evening ThArs have different OBJECT suffixes and are >2hr
        # apart; each forms its own cluster.
        lists = _cluster("thar", _make_mini_db())
        assert len(lists) == 2

    def test_thar_morn_cluster(self):
        lists = _cluster("thar", _make_mini_db())
        assert lists[0] == sorted(_THAR_MORN)

    def test_thar_eve_cluster(self):
        lists = _cluster("thar", _make_mini_db())
        assert lists[1] == sorted(_THAR_EVE)


# ---------------------------------------------------------------------------
# build_calibration_stacks (real data)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBuildCalibrationStacksRealData:
    @pytest.fixture(scope="class")
    def fh(self):
        # A handler with the night loaded — the recipe's usage pattern.
        handler = FileHandler({"KPF_DATA_INPUT": str(TESTDATA_DIR)})
        handler.build_mini_database("20240405")
        return handler

    def test_bias_returns_single_cluster(self, fh):
        lists = fh.build_calibration_stacks("bias")
        assert len(lists) == 1
        assert len(lists[0]) == 5
        assert lists[0] == sorted(lists[0])

    def test_flat_returns_single_cluster(self, fh):
        lists = fh.build_calibration_stacks("flat")
        assert len(lists) == 1
        assert len(lists[0]) == 5
        assert lists[0] == sorted(lists[0])

    def test_dark_raises_on_undersized_clusters(self, fh):
        # The testdata has two dark clusters of 2 and 3 frames — both below
        # the default min_file_count=5 and dropped, leaving nothing → raises.
        with pytest.raises(ValueError, match="no cluster with at least"):
            fh.build_calibration_stacks("dark")

    def test_dark_merge_respects_hst_boundary(self, fh):
        # The 5 dark frames span two HST days (2 on one, 3 on the next), so they
        # can never merge into a single >=5 master without crossing HST midnight.
        # With the default min=5, no same-HST-day cluster reaches the threshold →
        # raises even with merge_small_clusters.
        with pytest.raises(ValueError, match="no cluster with at least"):
            fh.build_calibration_stacks("dark", merge_small_clusters=True)

    def test_dark_merges_within_hst_day(self, fh):
        # Lowering min to 3 lets the 3 same-HST-day darks merge into one cluster;
        # the 2 frames on the other HST day are dropped (no same-day neighbor).
        lists = fh.build_calibration_stacks(
            "dark", min_file_count=3, merge_small_clusters=True
        )
        assert len(lists) == 1
        assert len(lists[0]) == 3
        assert lists[0] == sorted(lists[0])


# ---------------------------------------------------------------------------
# kpf_filepath
# ---------------------------------------------------------------------------


class TestKpfFilepath:
    def test_master_bias_with_obs_id(self):
        path = kpf_filepath(
            "KP.20240405.03600.00", "L1", data_root="/data", master="bias"
        )
        assert path == "/data/masters/20240405/KP.20240405.03600.00_master_bias_L1.fits"

    def test_master_flat_with_obs_id(self):
        path = kpf_filepath(
            "KP.20240405.14000.00", "L1", data_root="/data", master="flat"
        )
        assert path == "/data/masters/20240405/KP.20240405.14000.00_master_flat_L1.fits"

    def test_master_bare_filename(self):
        name = kpf_filepath("KP.20240405.03600.00", "L1", master="bias")
        assert name == "KP.20240405.03600.00_master_bias_L1.fits"

    def test_science_l0(self):
        path = kpf_filepath("KP.20240405.49597.71", "L0", data_root="/data")
        assert path == "/data/L0/20240405/KP.20240405.49597.71.fits"

    def test_science_l1(self):
        # Science L1 keeps the KPF "kpf_L1" prefix (no EPRV "S": no L1 standard).
        # KP.20240405.49597.71 → 49597s = 13:46:37
        path = kpf_filepath("KP.20240405.49597.71", "L1", data_root="/data")
        assert path == "/data/L1/20240405/kpf_L1_20240405T134637.fits"

    def test_science_bare_filename_l0(self):
        name = kpf_filepath("KP.20240405.49597.71", "L0")
        assert name == "KP.20240405.49597.71.fits"

    def test_science_bare_filename_l1(self):
        # 49597s = 13:46:37; L1 keeps the "kpf_L1" prefix (no EPRV "S")
        name = kpf_filepath("KP.20240405.49597.71", "L1")
        assert name == "kpf_L1_20240405T134637.fits"

    def test_science_l2(self):
        # KP.20240405.40113.57 → 40113s = 11:08:33
        path = kpf_filepath("KP.20240405.40113.57", "L2", data_root="/data")
        assert path == "/data/L2/20240405/kpf_SL2_20240405T110833.fits"

    def test_science_l4(self):
        path = kpf_filepath("KP.20240405.40113.57", "L4", data_root="/data")
        assert path == "/data/L4/20240405/kpf_SL4_20240405T110833.fits"

    def test_science_l2_midnight_boundary(self):
        # 3600s = 01:00:00
        path = kpf_filepath("KP.20240405.03600.00", "L2", data_root="/data")
        assert path == "/data/L2/20240405/kpf_SL2_20240405T010000.fits"

    def test_science_l2_zero_seconds(self):
        # 0s = 00:00:00
        path = kpf_filepath("KP.20240405.00000.00", "L2", data_root="/data")
        assert path == "/data/L2/20240405/kpf_SL2_20240405T000000.fits"

    def test_science_bare_filename_l2(self):
        name = kpf_filepath("KP.20240405.40113.57", "L2")
        assert name == "kpf_SL2_20240405T110833.fits"

    def test_master_thar_with_obs_id(self):
        path = kpf_filepath(
            "KP.20240405.03600.00", "L2", data_root="/data", master="thar"
        )
        assert path == "/data/masters/20240405/KP.20240405.03600.00_master_thar_L2.fits"

    def test_invalid_obs_id_raises(self):
        with pytest.raises(ValueError, match="valid observation ID"):
            kpf_filepath("20240405", "L1")

    def test_invalid_data_root_empty_string_raises(self):
        with pytest.raises(
            ValueError, match="data_root must be None or a non-empty string"
        ):
            kpf_filepath("KP.20240405.40113.57", "L2", data_root="")

    def test_invalid_data_root_non_string_raises(self):
        with pytest.raises(
            ValueError, match="data_root must be None or a non-empty string"
        ):
            kpf_filepath("KP.20240405.40113.57", "L2", data_root=12345)

    def test_composes_directory_and_filename(self):
        # kpf_filepath is exactly kpf_directory joined with kpf_filename.
        obs_id = "KP.20240405.40113.57"
        assert kpf_filepath(obs_id, "L2", data_root="/data") == os.path.join(
            kpf_directory(obs_id, level="L2", data_root="/data", kind="science"),
            kpf_filename(obs_id, "L2"),
        )


# ---------------------------------------------------------------------------
# kpf_filename
# ---------------------------------------------------------------------------


class TestKpfFilename:
    def test_science_l0(self):
        assert kpf_filename("KP.20240405.49597.71", "L0") == "KP.20240405.49597.71.fits"

    def test_science_l1(self):
        # 49597 s = 13:46:37; L1 keeps the "kpf_L1" prefix (no EPRV "S").
        assert (
            kpf_filename("KP.20240405.49597.71", "L1") == "kpf_L1_20240405T134637.fits"
        )

    def test_science_l2(self):
        # 40113 s = 11:08:33; L2 uses the EPRV "kpf_SL2" prefix.
        assert (
            kpf_filename("KP.20240405.40113.57", "L2") == "kpf_SL2_20240405T110833.fits"
        )

    def test_science_l4(self):
        assert (
            kpf_filename("KP.20240405.40113.57", "L4") == "kpf_SL4_20240405T110833.fits"
        )

    def test_master(self):
        assert (
            kpf_filename("KP.20240405.03600.00", "L1", master="bias")
            == "KP.20240405.03600.00_master_bias_L1.fits"
        )

    def test_invalid_obs_id_raises(self):
        with pytest.raises(ValueError, match="valid observation ID"):
            kpf_filename("20240405", "L1")

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="'level' must be"):
            kpf_filename("KP.20240405.49597.71", "L9")

    def test_invalid_master_type_raises(self):
        with pytest.raises(ValueError, match="'master' must be"):
            kpf_filename("KP.20240405.03600.00", "L1", master="wls")

    def test_invalid_master_level_raises(self):
        with pytest.raises(ValueError, match="'level' for master products"):
            kpf_filename("KP.20240405.03600.00", "L0", master="bias")


# ---------------------------------------------------------------------------
# FileHandler.find_masters
# ---------------------------------------------------------------------------


class TestFindMasters:
    """`FileHandler.find_masters` (the masters finder) and `kpf_filepath` (the
    masters writer) build the same path independently. These guard that the two
    inline f-strings can't drift — same directory and `_master_{type}_{level}`
    filename, with the KOAID wildcarded in the finder."""

    def test_returns_empty_when_no_masters(self, tmp_path):
        fh = FileHandler({"KPF_MASTERS_OUTPUT": str(tmp_path)})
        assert fh.find_masters("bias", "L1", "20240405") == []

    def test_raises_without_masters_root(self):
        with pytest.raises(ValueError, match="KPF_MASTERS_OUTPUT"):
            FileHandler().find_masters("bias", "L1", "20240405")

    @pytest.mark.parametrize(
        "cal_type,level",
        [("bias", "L1"), ("dark", "L1"), ("flat", "L1"), ("thar", "L2")],
    )
    def test_finds_kpf_filepath_output(self, tmp_path, cal_type, level):
        # The finder must locate a master written at the kpf_filepath path with
        # only the KOAID wildcarded — same directory, same filename convention.
        obs_id = "KP.20240405.03600.00"
        root = str(tmp_path)
        written = kpf_filepath(obs_id, level, data_root=root, master=cal_type)
        os.makedirs(os.path.dirname(written), exist_ok=True)
        open(written, "w").close()
        fh = FileHandler({"KPF_MASTERS_OUTPUT": root})
        assert fh.find_masters(cal_type, level, "20240405") == [written]


# ---------------------------------------------------------------------------
# Filename-convention consistency contract
# ---------------------------------------------------------------------------


class TestFilenameConsistency:
    """`kpf_filepath` (the pipeline's path builder, from an obs_id string) and a
    data model's `generate_standard_filename` (the to_fits fallback) build the same
    product basename. Every level now routes both through `kpf_filename` -- the
    single source for the naming rule -- so this contract guards that each model
    carries its `obs_id` and delegates at the right level, and that the string and
    object builders can never silently diverge (the kind of bug behind the old
    `kpf_SL1` mix-up). All four levels are exercised.
    """

    OBS_ID = "KP.20240405.49597.71"  # 49597 s of day = 13:46:37 UT

    def _make(self, level):
        obj = {"L0": KPF0, "L1": KPF1, "L2": KPF2, "L4": KPF4}[level]()
        obj.obs_id = self.OBS_ID
        return obj

    @pytest.mark.parametrize("level", ["L0", "L1", "L2", "L4"])
    def test_generate_standard_filename_matches_kpf_filepath(self, level):
        obj = self._make(level)
        expected = os.path.basename(kpf_filepath(self.OBS_ID, level))
        assert obj.generate_standard_filename() == expected


# ---------------------------------------------------------------------------
# kpf_directory
# ---------------------------------------------------------------------------


class TestKpfDirectory:
    OBS_ID = "KP.20240405.49597.71"

    def test_science(self):
        path = kpf_directory(self.OBS_ID, level="L2", data_root="/data", kind="science")
        assert path == "/data/L2/20240405"

    def test_masters_ignores_level(self):
        path = kpf_directory(self.OBS_ID, data_root="/data", kind="masters")
        assert path == "/data/masters/20240405"

    def test_qlp(self):
        path = kpf_directory(self.OBS_ID, level="L0", data_root="/data", kind="QLP")
        assert path == "/data/QLP/20240405/KP.20240405.49597.71/L0"

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="kind must be one of"):
            kpf_directory(self.OBS_ID, level="L0", data_root="/data", kind="logs")

    def test_invalid_obs_id_raises(self):
        with pytest.raises(ValueError, match="valid observation ID"):
            kpf_directory("20240405", level="L0", data_root="/data", kind="QLP")

    def test_missing_level_raises_for_science(self):
        with pytest.raises(ValueError, match="'level' must be"):
            kpf_directory(self.OBS_ID, data_root="/data", kind="science")

    def test_invalid_data_root_none_raises(self):
        with pytest.raises(ValueError, match="data_root must be a non-empty string"):
            kpf_directory(self.OBS_ID, level="L0", data_root=None, kind="QLP")

    def test_invalid_data_root_empty_string_raises(self):
        with pytest.raises(ValueError, match="data_root must be a non-empty string"):
            kpf_directory(self.OBS_ID, level="L0", data_root="", kind="QLP")


# ---------------------------------------------------------------------------
# FileHandler.build_mini_database (real L0 data from tests/testdata/)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBuildMiniDatabase:
    @pytest.fixture(scope="class")
    def mini_db(self):
        return FileHandler({"KPF_DATA_INPUT": str(TESTDATA_DIR)}).build_mini_database(
            "20240405"
        )

    def test_has_required_columns(self, mini_db):
        for col in ("FILENAME", "TARGNAME", "IMTYPE", "OBJECT", "EXPTIME", "ELAPSED"):
            assert col in mini_db.columns

    def test_no_cluster_columns(self, mini_db):
        # CAL_START/CAL_END were moved out of the mini_db; cluster detection
        # now happens at build_calibration_stacks time.
        assert "CAL_START" not in mini_db.columns
        assert "CAL_END" not in mini_db.columns

    def test_all_files_are_fits(self, mini_db):
        assert mini_db["FILENAME"].str.endswith(".fits").all()

    def test_isjunk_column_present_and_false(self, mini_db):
        # No junk list ships in tests/testdata, so every frame is ISJUNK=False.
        assert "ISJUNK" in mini_db.columns
        assert not mini_db["ISJUNK"].any()

    def test_empty_directory_raises(self, tmp_path):
        (tmp_path / "L0" / "20240405").mkdir(parents=True)
        fh = FileHandler({"KPF_DATA_INPUT": str(tmp_path)})
        with pytest.raises(ValueError, match="No FITS files found"):
            fh.build_mini_database("20240405")

    def test_missing_data_input_raises(self):
        with pytest.raises(ValueError, match="KPF_DATA_INPUT"):
            FileHandler().build_mini_database("20240405")


# ---------------------------------------------------------------------------
# Junk exclusion: load_junk_obs_ids + build_calibration_stacks(exclude_junk=...)
# (synthetic; the only junk files written go to isolated tmp_path/reference/)
# ---------------------------------------------------------------------------


_JUNK_BIAS = [
    f"/data/L0/20240405/KP.20240405.{s:05d}.00.fits" for s in (1000, 1010, 1020, 1030)
]  # four bias frames 10 s apart on one HST day


class TestJunkExclusion:
    def test_load_junk_absent_file(self, tmp_path):
        assert load_junk_obs_ids(str(tmp_path)) == set()

    def test_load_junk_parses_wmko_format(self, tmp_path):
        # WMKO layout: a title line, an 'observation_id' header, one obs_id/row.
        ref = tmp_path / "reference"
        ref.mkdir()
        (ref / "Junk_Observations_for_KPF.csv").write_text(
            "Junk Observations for KPF\nobservation_id\n"
            "KP.20240405.00001.00\nKP.20240405.00002.00\n"
        )
        assert load_junk_obs_ids(str(tmp_path)) == {
            "KP.20240405.00001.00",
            "KP.20240405.00002.00",
        }

    def test_exclude_junk_default_drops_flagged_frame(self):
        junk_fn = _JUNK_BIAS[1]
        db = _mini_db(_rows(_JUNK_BIAS, "autocal-bias", "Bias"), junk=[junk_fn])
        lists = _cluster("bias", db, min_file_count=1)
        assert junk_fn not in [fn for cluster in lists for fn in cluster]

    def test_exclude_junk_false_keeps_flagged_frame(self):
        junk_fn = _JUNK_BIAS[1]
        db = _mini_db(_rows(_JUNK_BIAS, "autocal-bias", "Bias"), junk=[junk_fn])
        lists = _cluster("bias", db, min_file_count=1, exclude_junk=False)
        assert junk_fn in [fn for cluster in lists for fn in cluster]

    def test_exclude_junk_without_column_raises(self):
        # A mini database built before ISJUNK existed must fail loudly.
        db = _mini_db(_rows(_JUNK_BIAS, "autocal-bias", "Bias")).drop(columns="ISJUNK")
        with pytest.raises(KeyError):
            _cluster("bias", db, min_file_count=1)
