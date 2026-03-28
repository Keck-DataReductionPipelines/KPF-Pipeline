"""
Tests for pipeline utils and the kpf_drp_masters recipe.

Unit tests use synthetic DataFrames and temp directories — no real data needed.
Integration tests use real L0 data from tests/testdata/L0/20240405/.
"""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from kpfpipe.data_models.masters.level1 import KPFMasterL1
from kpfpipe.utils.pipeline import (
    _utc_to_hst,
    _detect_calibration_stack_clusters,
    build_l0_file_lists,
    build_filepath,
    build_mini_database,
)


# ---------------------------------------------------------------------------
# Test data paths
# ---------------------------------------------------------------------------

TESTDATA_DIR    = Path(__file__).parent / 'testdata'
TESTDATA_L0_DIR = TESTDATA_DIR / 'L0' / '20240405'


# ---------------------------------------------------------------------------
# Session-scoped cleanup: remove CSV written into testdata by build_mini_database
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session', autouse=True)
def cleanup_testdata_csv():
    yield
    csv_path = TESTDATA_L0_DIR / 'KP.20240405_L0.csv'
    if csv_path.exists():
        csv_path.unlink()


# ---------------------------------------------------------------------------
# Synthetic test data setup (unit tests only)
# ---------------------------------------------------------------------------

# Synthetic filenames: KP.YYYYMMDD.SSSSS.FF.fits
# Two bias clusters separated by a >2hr gap; one dark cluster; science frames.
_BIAS_A = [f"/data/L0/20240405/KP.20240405.0{3600 + i*100:04d}.00.fits" for i in range(5)]  # 03600–04000
_BIAS_B = [f"/data/L0/20240405/KP.20240405.{14000 + i*100:05d}.00.fits" for i in range(5)]  # 14000–14400
_DARK_A = [f"/data/L0/20240405/KP.20240405.{18000 + i*100:05d}.00.fits" for i in range(3)]  # 18000–18200
_SCI_A  = [f"/data/L0/20240405/KP.20240405.{50000 + i*100:05d}.00.fits" for i in range(2)]  # 50000–50100


def _make_mini_db():
    rows = (
        [{"FILENAME": f, "IMTYPE": "Bias",   "OBJECT": "autocal-bias", "TARGNAME": None} for f in _BIAS_A]
      + [{"FILENAME": f, "IMTYPE": "Bias",   "OBJECT": "autocal-bias", "TARGNAME": None} for f in _BIAS_B]
      + [{"FILENAME": f, "IMTYPE": "Dark",   "OBJECT": "autocal-dark", "TARGNAME": None} for f in _DARK_A]
      + [{"FILENAME": f, "IMTYPE": "Object", "OBJECT": "185144",       "TARGNAME": "185144"} for f in _SCI_A]
    )
    df = pd.DataFrame(rows)
    df["EXPTIME"] = 60.0
    df["ELAPSED"] = 60.0
    return _detect_calibration_stack_clusters(df)


def _write_test_csv(tmp_path, db):
    """Write a mini_db CSV in the expected location for L0/20240405."""
    data_dir = tmp_path / "L0" / "20240405"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "KP.20240405_L0.csv"
    db.to_csv(csv_path, index=False)
    return str(data_dir)


# ---------------------------------------------------------------------------
# _utc_to_hst
# ---------------------------------------------------------------------------


class TestUtcToHst:
    def test_midday_no_rollover(self):
        # 12:00 UTC = 02:00 HST same day (43200 - 36000 = 7200)
        result = _utc_to_hst("20240405.43200.00")
        assert result == "20240405.07200.00"

    def test_rollover_to_previous_day(self):
        # 03:00 UTC = 17:00 HST previous day (10800 - 36000 < 0)
        result = _utc_to_hst("20240405.10800.00")
        assert result == "20240404.61200.00"

    def test_frame_str_preserved(self):
        result = _utc_to_hst("20240405.43200.71")
        assert result.endswith(".71")

    def test_exact_offset_boundary(self):
        # 10:00 UTC = 00:00 HST (36000 - 36000 = 0)
        result = _utc_to_hst("20240405.36000.00")
        assert result == "20240405.00000.00"


# ---------------------------------------------------------------------------
# _detect_calibration_stack_clusters
# ---------------------------------------------------------------------------


class TestDetectCalibrationStackClusters:

    @pytest.fixture(scope="class")
    def db(self):
        return _make_mini_db()

    def test_cal_start_columns_exist(self, db):
        assert "CAL_START" in db.columns
        assert "CAL_END" in db.columns

    def test_bias_cluster_a_cal_start(self, db):
        rows = db[db["FILENAME"].isin(_BIAS_A)]
        assert (rows["CAL_START"] == "20240405.03600.00").all()

    def test_bias_cluster_a_cal_end(self, db):
        rows = db[db["FILENAME"].isin(_BIAS_A)]
        assert (rows["CAL_END"] == "20240405.04000.00").all()

    def test_bias_cluster_b_cal_start(self, db):
        rows = db[db["FILENAME"].isin(_BIAS_B)]
        assert (rows["CAL_START"] == "20240405.14000.00").all()

    def test_bias_cluster_b_cal_end(self, db):
        rows = db[db["FILENAME"].isin(_BIAS_B)]
        assert (rows["CAL_END"] == "20240405.14400.00").all()

    def test_two_bias_clusters_detected(self, db):
        bias = db[db["OBJECT"] == "autocal-bias"]
        assert bias["CAL_START"].nunique() == 2

    def test_dark_cluster_cal_start(self, db):
        rows = db[db["FILENAME"].isin(_DARK_A)]
        assert (rows["CAL_START"] == "20240405.18000.00").all()

    def test_science_frames_cal_start_empty(self, db):
        sci = db[db["IMTYPE"] == "Object"]
        assert (sci["CAL_START"] == "").all()

    def test_science_frames_cal_end_empty(self, db):
        sci = db[db["IMTYPE"] == "Object"]
        assert (sci["CAL_END"] == "").all()

    def test_row_order_preserved(self, db):
        # Science frames should still be at the end
        assert db.iloc[-1]["IMTYPE"] == "Object"


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
        lists = build_l0_file_lists(data_dir, "bias")
        assert len(lists) == 2

    def test_bias_cluster_a_files(self, data_dir):
        lists = build_l0_file_lists(data_dir, "bias")
        assert lists[0] == sorted(_BIAS_A)

    def test_bias_cluster_b_files(self, data_dir):
        lists = build_l0_file_lists(data_dir, "bias")
        assert lists[1] == sorted(_BIAS_B)

    def test_files_are_sorted(self, data_dir):
        for lst in build_l0_file_lists(data_dir, "bias"):
            assert lst == sorted(lst)

    def test_small_clusters_merged_issues_warning(self, data_dir):
        # min_file_count=6: both bias clusters (5 files each) fall below → merged
        with pytest.warns(UserWarning, match="merged into one list"):
            build_l0_file_lists(data_dir, "bias", min_file_count=6)

    def test_small_clusters_merged_returns_one_list(self, data_dir):
        with pytest.warns(UserWarning):
            lists = build_l0_file_lists(data_dir, "bias", min_file_count=6)
        assert len(lists) == 1
        assert lists[0] == sorted(_BIAS_A + _BIAS_B)

    def test_raises_when_no_frames_found(self, data_dir):
        with pytest.raises(ValueError, match="No 'flat' calibration frames found"):
            build_l0_file_lists(data_dir, "flat")

    def test_raises_when_merged_below_min(self, data_dir):
        # dark cluster has only 3 files; merged total still < min_file_count=5
        with pytest.raises(ValueError, match="need at least"):
            build_l0_file_lists(data_dir, "dark")

    def test_invalid_imtype_raises(self, data_dir):
        with pytest.raises(ValueError, match="imtype must be one of"):
            build_l0_file_lists(data_dir, "wls")

    def test_rebuilds_db_if_csv_missing(self, tmp_path):
        data_dir = str(tmp_path / "L0" / "20240405")
        os.makedirs(data_dir)
        with patch("kpfpipe.utils.pipeline.build_mini_database") as mock_bmd:
            mock_bmd.return_value = _make_mini_db()
            lists = build_l0_file_lists(data_dir, "bias")
        mock_bmd.assert_called_once_with(data_dir)
        assert len(lists) == 2


# ---------------------------------------------------------------------------
# build_l0_file_lists (real data)
# ---------------------------------------------------------------------------


class TestBuildL0FileListsRealData:

    @pytest.fixture(scope="class")
    def l0_dir(self):
        return str(TESTDATA_L0_DIR)

    def test_bias_returns_single_cluster(self, l0_dir):
        lists = build_l0_file_lists(l0_dir, "bias")
        assert len(lists) == 1
        assert len(lists[0]) == 5
        assert lists[0] == sorted(lists[0])

    def test_flat_returns_single_cluster(self, l0_dir):
        lists = build_l0_file_lists(l0_dir, "flat")
        assert len(lists) == 1
        assert len(lists[0]) == 5
        assert lists[0] == sorted(lists[0])

    def test_dark_clusters_merged_issues_warning(self, l0_dir):
        with pytest.warns(UserWarning, match="merged into one list"):
            build_l0_file_lists(l0_dir, "dark")

    def test_dark_clusters_merged_returns_one_list(self, l0_dir):
        with pytest.warns(UserWarning):
            lists = build_l0_file_lists(l0_dir, "dark")
        assert len(lists) == 1
        assert len(lists[0]) == 5
        assert lists[0] == sorted(lists[0])


# ---------------------------------------------------------------------------
# build_filepath
# ---------------------------------------------------------------------------


class TestBuildFilepath:

    def test_master_bias_with_obs_id(self):
        path = build_filepath("KP.20240405.03600.00", "/data", "L1", master="bias")
        assert path == "/data/masters/20240405/KP.20240405.03600.00_master_bias_L1.fits"

    def test_master_flat_with_obs_id(self):
        path = build_filepath("KP.20240405.14000.00", "/data", "L1", master="flat")
        assert path == "/data/masters/20240405/KP.20240405.14000.00_master_flat_L1.fits"

    def test_master_with_datecode_deprecated(self):
        path = build_filepath("20240405", "/data", "L1", master="bias")
        assert path == "/data/masters/20240405/kpf_20240405_bias_L1.fits"

    def test_science_l0(self):
        path = build_filepath("KP.20240405.49597.71", "/data", "L0")
        assert path == "/data/L0/20240405/KP.20240405.49597.71.fits"

    def test_science_l1(self):
        path = build_filepath("KP.20240405.49597.71", "/data", "L1")
        assert path == "/data/L1/20240405/KP.20240405.49597.71_L1.fits"

    def test_invalid_master_type_raises(self):
        with pytest.raises(ValueError, match="'master' must be"):
            build_filepath("KP.20240405.03600.00", "/data", "L1", master="wls")

    def test_invalid_master_level_raises(self):
        with pytest.raises(ValueError, match="'level' for master products"):
            build_filepath("KP.20240405.03600.00", "/data", "L0", master="bias")

    def test_science_invalid_obs_id_raises(self):
        with pytest.raises(ValueError, match="must be a valid obs_id"):
            build_filepath("20240405", "/data", "L1")


# ---------------------------------------------------------------------------
# build_mini_database (real L0 data from tests/testdata/)
# ---------------------------------------------------------------------------


class TestBuildMiniDatabase:

    @pytest.fixture(scope="class")
    def mini_db(self):
        return build_mini_database(str(TESTDATA_L0_DIR))

    def test_has_required_columns(self, mini_db):
        for col in ("FILENAME", "IMTYPE", "OBJECT", "CAL_START", "CAL_END"):
            assert col in mini_db.columns

    def test_all_files_are_fits(self, mini_db):
        assert mini_db["FILENAME"].str.endswith(".fits").all()

    def test_cal_start_empty_for_science(self, mini_db):
        sci = mini_db[mini_db["IMTYPE"] == "Object"]
        if not sci.empty:
            assert (sci["CAL_START"] == "").all()

    def test_cal_start_nonempty_for_bias(self, mini_db):
        bias = mini_db[mini_db["OBJECT"] == "autocal-bias"]
        assert (bias["CAL_START"] != "").all()

    def test_csv_written(self):
        csv_path = TESTDATA_L0_DIR / "KP.20240405_L0.csv"
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# Masters recipe integration (real L0 data from tests/testdata/)
# ---------------------------------------------------------------------------


class TestMastersRecipe:
    """End-to-end recipe test: build_l0_file_lists → Bias.make_master_l1 → to_fits."""

    @pytest.fixture(scope="class")
    def recipe_output(self, tmp_path_factory):
        from kpfpipe.modules.masters.bias import Bias
        from kpfpipe.utils.kpf import get_obs_id

        tmp_path = tmp_path_factory.mktemp("recipe_out")
        data_root_out = str(tmp_path)

        output_paths = []
        for files in build_l0_file_lists(str(TESTDATA_L0_DIR), "bias"):
            bias_handler = Bias(files)
            bias_l1      = bias_handler.make_master_l1()
            out_path     = build_filepath(get_obs_id(files[0]), data_root_out, "L1", master="bias")
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
