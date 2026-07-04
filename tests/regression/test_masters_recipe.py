"""
Tests for pipeline utils and the kpf_drp_masters recipe.

Unit tests use synthetic DataFrames and temp directories — no real data needed.
Integration tests use real L0 data from tests/testdata/L0/20240405/.
"""

import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.data_models.masters.level1 import KPFMasterL1
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import (
    build_filepath,
    build_l0_file_lists,
    build_mini_database,
    build_qlp_dir,
    glob_masters,
)
from kpfpipe.utils.kpf import get_timestamp, utc_to_hst

# ---------------------------------------------------------------------------
# Test data paths
# ---------------------------------------------------------------------------

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"
TESTDATA_L0_DIR = TESTDATA_DIR / "L0" / "20240405"
MASTERS_CONFIG_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "kpf_drp_masters.toml"
)


def _load_masters_recipe():
    spec = importlib.util.spec_from_file_location(
        "kpf_drp_masters",
        Path(__file__).parent.parent.parent / "recipes" / "kpf_drp_masters.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _make_mini_db():
    rows = (
        [
            {
                "FILENAME": f,
                "IMTYPE": "Bias",
                "OBJECT": "autocal-bias",
                "TARGNAME": None,
            }
            for f in _BIAS_A
        ]
        + [
            {
                "FILENAME": f,
                "IMTYPE": "Bias",
                "OBJECT": "autocal-bias",
                "TARGNAME": None,
            }
            for f in _BIAS_B
        ]
        + [
            {
                "FILENAME": f,
                "IMTYPE": "Dark",
                "OBJECT": "autocal-dark",
                "TARGNAME": None,
            }
            for f in _DARK_A
        ]
        + [
            {
                "FILENAME": f,
                "IMTYPE": "Arclamp",
                "OBJECT": "autocal-thar-all-morn",
                "TARGNAME": None,
            }
            for f in _THAR_MORN
        ]
        + [
            {
                "FILENAME": f,
                "IMTYPE": "Arclamp",
                "OBJECT": "autocal-thar-all-eve",
                "TARGNAME": None,
            }
            for f in _THAR_EVE
        ]
        + [
            {
                "FILENAME": f,
                "IMTYPE": "Object",
                "OBJECT": "185144",
                "TARGNAME": "185144",
            }
            for f in _SCI_A
        ]
    )
    df = pd.DataFrame(rows)
    df["EXPTIME"] = 60.0
    df["ELAPSED"] = 60.0
    df["UTC"] = [get_timestamp(f) for f in df["FILENAME"]]
    df["HST"] = [utc_to_hst(t) for t in df["UTC"]]
    return df


_BIAS_SMALL = [
    f"/data/L0/20240405/KP.20240405.{30000 + i * 100:05d}.00.fits" for i in range(2)
]  # 30000–30100, a >2hr gap after _BIAS_A → a separate 2-file cluster


def _mixed_bias_db():
    """Mini_db with one 5-file bias cluster (_BIAS_A) and a later 2-file one."""
    rows = [
        {"FILENAME": f, "IMTYPE": "Bias", "OBJECT": "autocal-bias", "TARGNAME": None}
        for f in _BIAS_A + _BIAS_SMALL
    ]
    df = pd.DataFrame(rows)
    df["EXPTIME"] = 60.0
    df["ELAPSED"] = 60.0
    df["UTC"] = [get_timestamp(f) for f in df["FILENAME"]]
    df["HST"] = [utc_to_hst(t) for t in df["UTC"]]
    return df, _BIAS_SMALL


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
    rows = [
        {"FILENAME": f, "IMTYPE": "Bias", "OBJECT": "autocal-bias", "TARGNAME": None}
        for f in before + after
    ]
    df = pd.DataFrame(rows)
    df["EXPTIME"] = 60.0
    df["ELAPSED"] = 60.0
    df["UTC"] = [get_timestamp(f) for f in df["FILENAME"]]
    df["HST"] = [utc_to_hst(t) for t in df["UTC"]]
    return df, before, after


def _write_test_csv(tmp_path, db):
    """
    Materialize a synthetic mini_db on disk: touch a stub .fits for every
    FILENAME row, rewrite the CSV's FILENAME column to point to the stubs,
    and write the CSV alongside them.
    """
    data_dir = tmp_path / "L0" / "20240405"
    data_dir.mkdir(parents=True)
    db = db.copy()
    new_filenames = []
    for original in db["FILENAME"]:
        new_path = str(data_dir / os.path.basename(original))
        open(new_path, "w").close()
        new_filenames.append(new_path)
    db["FILENAME"] = new_filenames
    csv_path = data_dir / "KP.20240405_L0.csv"
    db.to_csv(csv_path, index=False)
    return str(data_dir)


def _at(data_dir, synthetic_paths):
    """Translate /data/L0/... synthetic paths to absolute paths in data_dir."""
    return [os.path.join(data_dir, os.path.basename(p)) for p in synthetic_paths]


# ---------------------------------------------------------------------------
# build_l0_file_lists
# ---------------------------------------------------------------------------


class TestBuildL0FileLists:
    @pytest.fixture(scope="class")
    def data_dir(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("l0data")
        db = _make_mini_db()
        return _write_test_csv(tmp_path, db)

    def test_two_bias_clusters_returned_separately(self, data_dir):
        lists = build_l0_file_lists("bias", data_dir=data_dir)
        assert len(lists) == 2

    def test_bias_cluster_a_files(self, data_dir):
        lists = build_l0_file_lists("bias", data_dir=data_dir)
        assert lists[0] == sorted(_at(data_dir, _BIAS_A))

    def test_bias_cluster_b_files(self, data_dir):
        lists = build_l0_file_lists("bias", data_dir=data_dir)
        assert lists[1] == sorted(_at(data_dir, _BIAS_B))

    def test_files_are_sorted(self, data_dir):
        for lst in build_l0_file_lists("bias", data_dir=data_dir):
            assert lst == sorted(lst)

    def test_raises_when_no_cluster_meets_min(self, data_dir):
        # min_file_count=6: both bias clusters (5 files each) fall below and are
        # dropped, leaving nothing → raises.
        with pytest.raises(ValueError, match="no cluster with at least"):
            build_l0_file_lists("bias", min_file_count=6, data_dir=data_dir)

    def test_raises_when_no_frames_found(self, data_dir):
        with pytest.raises(ValueError, match="No 'flat' calibration frames found"):
            build_l0_file_lists("flat", data_dir=data_dir)

    def test_raises_when_only_cluster_below_default_min(self, data_dir):
        # dark cluster has only 3 files; default min_file_count=5 → dropped →
        # raises.
        with pytest.raises(ValueError, match="no cluster with at least"):
            build_l0_file_lists("dark", data_dir=data_dir)

    def test_drops_small_cluster_keeps_large(self):
        db, _ = _mixed_bias_db()
        lists = build_l0_file_lists("bias", mini_db=db)
        assert len(lists) == 1
        assert lists[0] == sorted(_BIAS_A)

    def test_merge_folds_small_into_neighbor(self):
        db, small = _mixed_bias_db()
        lists = build_l0_file_lists("bias", mini_db=db, merge_small_clusters=True)
        assert len(lists) == 1
        assert lists[0] == sorted(_BIAS_A + small)

    def test_merge_combines_two_small_clusters(self, data_dir):
        # Two 5-file bias clusters, each below min=6; merged into one of 10.
        lists = build_l0_file_lists(
            "bias", min_file_count=6, data_dir=data_dir, merge_small_clusters=True
        )
        assert len(lists) == 1
        assert len(lists[0]) == 10

    def test_merge_raises_when_total_below_min(self):
        # 7 bias files total (5 + 2); merging cannot reach min=8 → raises.
        db, _ = _mixed_bias_db()
        with pytest.raises(ValueError, match="no cluster with at least"):
            build_l0_file_lists(
                "bias", min_file_count=8, mini_db=db, merge_small_clusters=True
            )

    def test_hst_midnight_splits_cluster(self):
        # Same-OBJECT frames <gap apart but on opposite sides of HST midnight
        # (UTC 36000) must not share a cluster.
        db, before, after = _midnight_bias_db()
        lists = build_l0_file_lists("bias", mini_db=db, min_file_count=5)
        assert len(lists) == 2
        assert lists[0] == sorted(before)
        assert lists[1] == sorted(after)

    def test_hst_midnight_blocks_merge(self):
        # A small post-midnight cluster has no same-HST-day neighbor, so it is
        # dropped rather than merged across midnight into the pre-midnight one.
        db, before, _ = _midnight_bias_db(n_before=5, n_after=2)
        lists = build_l0_file_lists(
            "bias", mini_db=db, min_file_count=5, merge_small_clusters=True
        )
        assert len(lists) == 1
        assert lists[0] == sorted(before)

    def test_invalid_imtype_raises(self, data_dir):
        with pytest.raises(ValueError, match="cal_type must be one of"):
            build_l0_file_lists("bogus", data_dir=data_dir)

    def test_thar_returns_two_clusters(self, data_dir):
        # Morning and evening ThArs have different OBJECT suffixes and are >2hr
        # apart; each forms its own cluster.
        lists = build_l0_file_lists("thar", data_dir=data_dir)
        assert len(lists) == 2

    def test_thar_morn_cluster(self, data_dir):
        lists = build_l0_file_lists("thar", data_dir=data_dir)
        assert lists[0] == sorted(_at(data_dir, _THAR_MORN))

    def test_thar_eve_cluster(self, data_dir):
        lists = build_l0_file_lists("thar", data_dir=data_dir)
        assert lists[1] == sorted(_at(data_dir, _THAR_EVE))

    def test_raises_when_neither_source_provided(self):
        with pytest.raises(ValueError, match="Exactly one of"):
            build_l0_file_lists("bias")

    def test_raises_when_both_sources_provided(self, data_dir):
        with pytest.raises(ValueError, match="Exactly one of"):
            build_l0_file_lists("bias", data_dir=data_dir, mini_db=_make_mini_db())

    def test_accepts_mini_db_directly(self):
        lists = build_l0_file_lists("bias", mini_db=_make_mini_db())
        assert len(lists) == 2
        assert lists[0] == sorted(_BIAS_A)

    def test_rebuilds_db_if_csv_missing(self, tmp_path):
        data_dir = str(tmp_path / "L0" / "20240405")
        os.makedirs(data_dir)
        with patch("kpfpipe.utils.io.build_mini_database") as mock_bmd:
            mock_bmd.return_value = _make_mini_db()
            lists = build_l0_file_lists("bias", data_dir=data_dir)
        mock_bmd.assert_called_once_with(data_dir)
        assert len(lists) == 2

    def test_rebuilds_db_if_files_added_on_disk(self, tmp_path):
        # Materialize a consistent CSV + stubs, then plant an extra .fits
        # file the CSV does not know about.
        data_dir = _write_test_csv(tmp_path, _make_mini_db())
        open(os.path.join(data_dir, "KP.20240405.99999.99.fits"), "w").close()

        with patch("kpfpipe.utils.io.build_mini_database") as mock_bmd:
            mock_bmd.return_value = _make_mini_db()
            with pytest.warns(UserWarning, match=r"stale.*\+1 added"):
                build_l0_file_lists("bias", data_dir=data_dir)
        mock_bmd.assert_called_once_with(data_dir)

    def test_rebuilds_db_if_files_removed_from_disk(self, tmp_path):
        # Materialize a consistent CSV + stubs, then unlink one .fits
        # the CSV still references.
        data_dir = _write_test_csv(tmp_path, _make_mini_db())
        os.unlink(os.path.join(data_dir, os.path.basename(_BIAS_A[0])))

        with patch("kpfpipe.utils.io.build_mini_database") as mock_bmd:
            mock_bmd.return_value = _make_mini_db()
            with pytest.warns(UserWarning, match=r"stale.*-1 removed"):
                build_l0_file_lists("bias", data_dir=data_dir)
        mock_bmd.assert_called_once_with(data_dir)


# ---------------------------------------------------------------------------
# build_l0_file_lists (real data)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBuildL0FileListsRealData:
    @pytest.fixture(scope="class")
    def l0_dir(self):
        return str(TESTDATA_L0_DIR)

    def test_bias_returns_single_cluster(self, l0_dir):
        lists = build_l0_file_lists("bias", data_dir=l0_dir)
        assert len(lists) == 1
        assert len(lists[0]) == 5
        assert lists[0] == sorted(lists[0])

    def test_flat_returns_single_cluster(self, l0_dir):
        lists = build_l0_file_lists("flat", data_dir=l0_dir)
        assert len(lists) == 1
        assert len(lists[0]) == 5
        assert lists[0] == sorted(lists[0])

    def test_dark_raises_on_undersized_clusters(self, l0_dir):
        # The testdata has two dark clusters of 2 and 3 frames — both below
        # the default min_file_count=5 and dropped, leaving nothing → raises.
        with pytest.raises(ValueError, match="no cluster with at least"):
            build_l0_file_lists("dark", data_dir=l0_dir)

    def test_dark_merge_respects_hst_boundary(self, l0_dir):
        # The 5 dark frames span two HST days (2 on one, 3 on the next), so they
        # can never merge into a single >=5 master without crossing HST midnight.
        # With the default min=5, no same-HST-day cluster reaches the threshold →
        # raises even with merge_small_clusters.
        with pytest.raises(ValueError, match="no cluster with at least"):
            build_l0_file_lists("dark", data_dir=l0_dir, merge_small_clusters=True)

    def test_dark_merges_within_hst_day(self, l0_dir):
        # Lowering min to 3 lets the 3 same-HST-day darks merge into one cluster;
        # the 2 frames on the other HST day are dropped (no same-day neighbor).
        lists = build_l0_file_lists(
            "dark", data_dir=l0_dir, min_file_count=3, merge_small_clusters=True
        )
        assert len(lists) == 1
        assert len(lists[0]) == 3
        assert lists[0] == sorted(lists[0])


# ---------------------------------------------------------------------------
# build_filepath
# ---------------------------------------------------------------------------


class TestBuildFilepath:
    def test_master_bias_with_obs_id(self):
        path = build_filepath(
            "KP.20240405.03600.00", "L1", data_root="/data", master="bias"
        )
        assert path == "/data/masters/20240405/KP.20240405.03600.00_master_bias_L1.fits"

    def test_master_flat_with_obs_id(self):
        path = build_filepath(
            "KP.20240405.14000.00", "L1", data_root="/data", master="flat"
        )
        assert path == "/data/masters/20240405/KP.20240405.14000.00_master_flat_L1.fits"

    def test_master_bare_filename(self):
        name = build_filepath("KP.20240405.03600.00", "L1", master="bias")
        assert name == "KP.20240405.03600.00_master_bias_L1.fits"

    def test_science_l0(self):
        path = build_filepath("KP.20240405.49597.71", "L0", data_root="/data")
        assert path == "/data/L0/20240405/KP.20240405.49597.71.fits"

    def test_science_l1(self):
        # Science L1 keeps the KPF "kpf_L1" prefix (no EPRV "S": no L1 standard).
        # KP.20240405.49597.71 → 49597s = 13:46:37
        path = build_filepath("KP.20240405.49597.71", "L1", data_root="/data")
        assert path == "/data/L1/20240405/kpf_L1_20240405T134637.fits"

    def test_science_bare_filename_l0(self):
        name = build_filepath("KP.20240405.49597.71", "L0")
        assert name == "KP.20240405.49597.71.fits"

    def test_science_bare_filename_l1(self):
        # 49597s = 13:46:37; L1 keeps the "kpf_L1" prefix (no EPRV "S")
        name = build_filepath("KP.20240405.49597.71", "L1")
        assert name == "kpf_L1_20240405T134637.fits"

    def test_invalid_master_type_raises(self):
        with pytest.raises(ValueError, match="'master' must be"):
            build_filepath("KP.20240405.03600.00", "L1", master="wls")

    def test_invalid_master_level_raises(self):
        with pytest.raises(ValueError, match="'level' for master products"):
            build_filepath("KP.20240405.03600.00", "L0", master="bias")

    def test_science_l2(self):
        # KP.20240405.40113.57 → 40113s = 11:08:33
        path = build_filepath("KP.20240405.40113.57", "L2", data_root="/data")
        assert path == "/data/L2/20240405/kpf_SL2_20240405T110833.fits"

    def test_science_l4(self):
        path = build_filepath("KP.20240405.40113.57", "L4", data_root="/data")
        assert path == "/data/L4/20240405/kpf_SL4_20240405T110833.fits"

    def test_science_l2_midnight_boundary(self):
        # 3600s = 01:00:00
        path = build_filepath("KP.20240405.03600.00", "L2", data_root="/data")
        assert path == "/data/L2/20240405/kpf_SL2_20240405T010000.fits"

    def test_science_l2_zero_seconds(self):
        # 0s = 00:00:00
        path = build_filepath("KP.20240405.00000.00", "L2", data_root="/data")
        assert path == "/data/L2/20240405/kpf_SL2_20240405T000000.fits"

    def test_science_bare_filename_l2(self):
        name = build_filepath("KP.20240405.40113.57", "L2")
        assert name == "kpf_SL2_20240405T110833.fits"

    def test_master_thar_with_obs_id(self):
        path = build_filepath(
            "KP.20240405.03600.00", "L2", data_root="/data", master="thar"
        )
        assert path == "/data/masters/20240405/KP.20240405.03600.00_master_thar_L2.fits"

    def test_invalid_obs_id_raises(self):
        with pytest.raises(ValueError, match="valid observation ID"):
            build_filepath("20240405", "L1")

    def test_invalid_data_root_empty_string_raises(self):
        with pytest.raises(
            ValueError, match="data_root must be None or a non-empty string"
        ):
            build_filepath("KP.20240405.40113.57", "L2", data_root="")

    def test_invalid_data_root_non_string_raises(self):
        with pytest.raises(
            ValueError, match="data_root must be None or a non-empty string"
        ):
            build_filepath("KP.20240405.40113.57", "L2", data_root=12345)


# ---------------------------------------------------------------------------
# glob_masters
# ---------------------------------------------------------------------------


class TestGlobMasters:
    """`glob_masters` (the masters finder pattern) and `build_filepath` (the
    masters writer) build the same path independently. These guard that the two
    inline f-strings can't drift — same directory and `_master_{type}_{level}`
    filename, with the KOAID wildcarded in the finder."""

    def test_glob_masters(self):
        assert (
            glob_masters("/data", "bias", "L1", "20240405")
            == "/data/masters/20240405/*_master_bias_L1.fits"
        )

    @pytest.mark.parametrize(
        "cal_type,level",
        [("bias", "L1"), ("dark", "L1"), ("flat", "L1"), ("thar", "L2")],
    )
    def test_glob_masters_matches_build_filepath(self, cal_type, level):
        # The finder pattern must equal a written master path with only the
        # KOAID wildcarded — same directory, same filename convention.
        obs_id = "KP.20240405.03600.00"
        written = build_filepath(obs_id, level, data_root="/data", master=cal_type)
        assert glob_masters("/data", cal_type, level, "20240405") == written.replace(
            obs_id, "*"
        )


# ---------------------------------------------------------------------------
# Filename-convention consistency contract
# ---------------------------------------------------------------------------


class TestFilenameConsistency:
    """`build_filepath` (the pipeline's path builder, from an obs_id) and a data
    model's `generate_standard_filename` (the to_fits fallback, from headers) are
    two independent encodings of the same naming rule. This contract asserts they
    agree on the basename for every level, so the two can never silently drift
    (the kind of divergence behind the old `kpf_SL1` bug).

    L0/L1 use KPF overrides (no EPRV name for raw/assembled frames); L2/L4 use the
    EPRV-standard name inherited from rvdata. Both paths are exercised here.
    """

    OBS_ID = "KP.20240405.49597.71"  # 49597 s of day = 13:46:37 UT
    DATE_OBS = "2024-04-05T13:46:37"

    def _make(self, level):
        if level == "L0":
            obj = KPF0()
            obj.obs_id = self.OBS_ID
            return obj
        obj = {"L1": KPF1, "L2": KPF2, "L4": KPF4}[level]()
        obj.headers["PRIMARY"]["INSTRUME"] = "KPF"
        obj.headers["PRIMARY"]["DATE-OBS"] = self.DATE_OBS
        return obj

    @pytest.mark.parametrize("level", ["L0", "L1", "L2", "L4"])
    def test_generate_standard_filename_matches_build_filepath(self, level):
        obj = self._make(level)
        expected = os.path.basename(build_filepath(self.OBS_ID, level))
        assert obj.generate_standard_filename() == expected


# ---------------------------------------------------------------------------
# build_qlp_dir
# ---------------------------------------------------------------------------


class TestBuildQlpDir:
    def test_l0(self):
        path = build_qlp_dir("KP.20240405.49597.71", "L0", data_root="/data")
        assert path == "/data/QLP/20240405/KP.20240405.49597.71/L0"

    def test_l1(self):
        path = build_qlp_dir("KP.20240405.49597.71", "L1", data_root="/data")
        assert path == "/data/QLP/20240405/KP.20240405.49597.71/L1"

    def test_invalid_obs_id_raises(self):
        with pytest.raises(ValueError, match="valid observation ID"):
            build_qlp_dir("20240405", "L0", data_root="/data")

    def test_invalid_data_root_none_raises(self):
        with pytest.raises(ValueError, match="data_root must be a non-empty string"):
            build_qlp_dir("KP.20240405.40113.57", "L0", data_root=None)

    def test_invalid_data_root_empty_string_raises(self):
        with pytest.raises(ValueError, match="data_root must be a non-empty string"):
            build_qlp_dir("KP.20240405.40113.57", "L0", data_root="")


# ---------------------------------------------------------------------------
# build_mini_database (real L0 data from tests/testdata/)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBuildMiniDatabase:
    @pytest.fixture(scope="class")
    def mini_db(self):
        return build_mini_database(str(TESTDATA_L0_DIR), write=False)

    def test_has_required_columns(self, mini_db):
        for col in ("FILENAME", "TARGNAME", "IMTYPE", "OBJECT", "EXPTIME", "ELAPSED"):
            assert col in mini_db.columns

    def test_no_cluster_columns(self, mini_db):
        # CAL_START/CAL_END were moved out of the mini_db; cluster detection
        # now happens at build_l0_file_lists time.
        assert "CAL_START" not in mini_db.columns
        assert "CAL_END" not in mini_db.columns

    def test_all_files_are_fits(self, mini_db):
        assert mini_db["FILENAME"].str.endswith(".fits").all()

    def test_write_false_does_not_write_csv(self):
        csv_path = TESTDATA_L0_DIR / "KP.20240405_L0.csv"
        was_present = csv_path.exists()
        build_mini_database(str(TESTDATA_L0_DIR), write=False)
        assert csv_path.exists() == was_present

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No FITS files found"):
            build_mini_database(str(tmp_path))


# ---------------------------------------------------------------------------
# Masters recipe integration (real L0 data from tests/testdata/)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMastersRecipe:
    """End-to-end recipe test: build_l0_file_lists → Bias.make_master_l1 → to_fits."""

    @pytest.fixture(scope="class")
    def recipe_output(self, tmp_path_factory):
        from kpfpipe.modules.masters.bias import Bias
        from kpfpipe.utils.kpf import get_obs_id

        tmp_path = tmp_path_factory.mktemp("recipe_out")
        data_root_out = str(tmp_path)

        output_paths = []
        for files in build_l0_file_lists("bias", data_dir=str(TESTDATA_L0_DIR)):
            bias_handler = Bias(files)
            bias_l1 = bias_handler.make_master_l1()
            out_path = build_filepath(
                get_obs_id(files[0]), "L1", data_root=data_root_out, master="bias"
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            bias_l1.to_fits(out_path)
            output_paths.append(out_path)

        return output_paths

    def test_at_least_one_master_produced(self, recipe_output):
        assert len(recipe_output) >= 1

    def test_output_files_exist(self, recipe_output):
        for path in recipe_output:
            assert os.path.isfile(path), f"Expected output not found: {path}"

    def test_output_filename_format(self, recipe_output):
        for path in recipe_output:
            fname = os.path.basename(path)
            assert "_master_bias_L1.fits" in fname

    def test_output_is_valid_fits(self, recipe_output):
        for path in recipe_output:
            ml1 = KPFMasterL1.from_fits(path)
            assert ml1.data["GREEN_IMG"] is not None
            assert ml1.data["RED_IMG"] is not None

    def test_input_files_extension_present(self, recipe_output):
        for path in recipe_output:
            ml1 = KPFMasterL1.from_fits(path)
            assert "INPUT_FILES" in ml1.extensions

    def test_input_files_extension_has_correct_count(self, recipe_output):
        for path in recipe_output:
            ml1 = KPFMasterL1.from_fits(path)
            assert len(ml1.data["INPUT_FILES"]) == 5

    def test_input_files_all_fits(self, recipe_output):
        for path in recipe_output:
            ml1 = KPFMasterL1.from_fits(path)
            filenames = ml1.data["INPUT_FILES"]["FILENAME"].tolist()
            assert all(f.endswith(".fits") for f in filenames)


# ---------------------------------------------------------------------------
# Masters recipe error paths
# ---------------------------------------------------------------------------


class TestMastersRecipeErrors:
    def _make_config(self, data_input, data_masters):
        return ConfigHandler(
            str(MASTERS_CONFIG_PATH),
            overrides={
                "DATA_DIRS": {
                    "KPF_DATA_INPUT": str(data_input),
                    "KPF_MASTERS_OUTPUT": str(data_masters),
                }
            },
        )

    def test_nonexistent_l0_dir_raises(self, tmp_path):
        import argparse

        config = self._make_config(tmp_path, tmp_path)
        args = argparse.Namespace(datecode="20240405", obs_id=None)
        recipe = _load_masters_recipe()
        with pytest.raises(SystemExit, match="L0 data directory not found"):
            recipe.main(config, args)

    def test_missing_datecode_raises(self, tmp_path):
        import argparse

        config = self._make_config(tmp_path, tmp_path)
        args = argparse.Namespace(datecode=None, obs_id=None)
        recipe = _load_masters_recipe()
        with pytest.raises(SystemExit, match="--datecode is required"):
            recipe.main(config, args)
