"""
Tests for pipeline utils and the kpf_drp_masters recipe.

Unit tests use synthetic DataFrames and temp directories — no real data needed.
Integration tests are gated on KPF_TESTDATA pointing to an L0 directory,
e.g. /Users/research/data/kpf/L0/20240405.
"""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from kpfpipe.utils.pipeline import (
    _utc_to_hst,
    _detect_calibration_stack_clusters,
    build_l0_file_list,
    build_filepath,
    build_mini_database,
    get_calibration_stack_clusters,
)


# ---------------------------------------------------------------------------
# Test data setup
# ---------------------------------------------------------------------------

L0_DIR = os.environ.get("KPF_TESTDATA")

needs_l0_data = pytest.mark.skipif(
    L0_DIR is None or not os.path.isdir(L0_DIR),
    reason="L0 test data not available (set KPF_TESTDATA env var)",
)

# Synthetic filenames: KP.YYYYMMDD.SSSSS.FF.fits
# Two bias clusters separated by a >2hr gap; one dark cluster; science frames.
_BIAS_A = [f"/data/L0/20240405/KP.20240405.0{3600 + i*100:04d}.00.fits" for i in range(4)]  # 03600–03900
_BIAS_B = [f"/data/L0/20240405/KP.20240405.{14000 + i*100:05d}.00.fits" for i in range(4)]  # 14000–14300
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
        assert (rows["CAL_END"] == "20240405.03900.00").all()

    def test_bias_cluster_b_cal_start(self, db):
        rows = db[db["FILENAME"].isin(_BIAS_B)]
        assert (rows["CAL_START"] == "20240405.14000.00").all()

    def test_bias_cluster_b_cal_end(self, db):
        rows = db[db["FILENAME"].isin(_BIAS_B)]
        assert (rows["CAL_END"] == "20240405.14300.00").all()

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
# get_calibration_stack_clusters
# ---------------------------------------------------------------------------


class TestGetCalibrationStackClusters:

    @pytest.fixture(scope="class")
    def db(self):
        return _make_mini_db()

    def test_two_bias_clusters(self, db):
        clusters = get_calibration_stack_clusters(db, "bias")
        assert len(clusters) == 2

    def test_bias_cluster_a_files(self, db):
        clusters = get_calibration_stack_clusters(db, "bias")
        assert clusters[0] == sorted(_BIAS_A)

    def test_bias_cluster_b_files(self, db):
        clusters = get_calibration_stack_clusters(db, "bias")
        assert clusters[1] == sorted(_BIAS_B)

    def test_one_dark_cluster(self, db):
        clusters = get_calibration_stack_clusters(db, "dark")
        assert len(clusters) == 1
        assert clusters[0] == sorted(_DARK_A)

    def test_no_flat_clusters(self, db):
        clusters = get_calibration_stack_clusters(db, "flat")
        assert clusters == []

    def test_invalid_imtype_raises(self, db):
        with pytest.raises(ValueError, match="imtype must be one of"):
            get_calibration_stack_clusters(db, "wls")

    def test_files_are_sorted(self, db):
        for cluster in get_calibration_stack_clusters(db, "bias"):
            assert cluster == sorted(cluster)


# ---------------------------------------------------------------------------
# build_l0_file_list
# ---------------------------------------------------------------------------


def _write_test_csv(tmp_path, db):
    """Write a mini_db CSV in the expected location for L0/20240405."""
    data_dir = tmp_path / "L0" / "20240405"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "KP.20240405_L0.csv"
    db.to_csv(csv_path, index=False)
    return str(data_dir)


class TestBuildL0FileList:

    @pytest.fixture(scope="class")
    def data_dir(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("l0data")
        db = _make_mini_db()
        return _write_test_csv(tmp_path, db)

    def test_selects_cluster_before_utc_time(self, data_dir):
        # 20000s is after bias cluster B starts at 14000s
        files = build_l0_file_list(data_dir, "bias", "20240405.20000.00")
        assert files == sorted(_BIAS_B)

    def test_selects_earlier_cluster_when_between_two(self, data_dir):
        # 13000s is after cluster A ends (03900) but before cluster B starts (14000)
        files = build_l0_file_list(data_dir, "bias", "20240405.13000.00")
        assert files == sorted(_BIAS_A)

    def test_raises_when_no_cluster_before_utc_time(self, data_dir):
        with pytest.raises(ValueError, match="No 'bias' calibration cluster found before"):
            build_l0_file_list(data_dir, "bias", "20240405.03000.00")

    def test_raises_when_gap_exceeds_24h(self, data_dir):
        # Cluster B starts at 14000s on 20240405; 14001s on 20240406 = 86401s gap
        with pytest.raises(ValueError, match="exceeds 24-hour limit"):
            build_l0_file_list(data_dir, "bias", "20240406.14001.00")

    def test_invalid_imtype_raises(self, data_dir):
        with pytest.raises(ValueError, match="imtype must be one of"):
            build_l0_file_list(data_dir, "wls", "20240405.20000.00")

    def test_rebuilds_db_if_csv_missing(self, tmp_path):
        data_dir = str(tmp_path / "L0" / "20240405")
        os.makedirs(data_dir)
        with patch("kpfpipe.utils.pipeline.build_mini_database") as mock_bmd:
            mock_bmd.return_value = _make_mini_db()
            files = build_l0_file_list(data_dir, "bias", "20240405.20000.00")
        mock_bmd.assert_called_once_with(data_dir)
        assert files == sorted(_BIAS_B)


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
# build_mini_database (integration, real L0 data)
# ---------------------------------------------------------------------------


@needs_l0_data
class TestBuildMiniDatabase:

    @pytest.fixture(scope="class")
    def mini_db(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("minidb")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Point data_dir at the real L0 directory so headers can be read,
            # but write the CSV to a temp location by monkey-patching normpath.
            pass
        return build_mini_database(L0_DIR)

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
        datecode = os.path.basename(L0_DIR)
        level = os.path.basename(os.path.dirname(L0_DIR))
        csv_path = os.path.join(L0_DIR, f"KP.{datecode}_{level}.csv")
        assert os.path.isfile(csv_path)


# ---------------------------------------------------------------------------
# Masters recipe integration (real L0 data)
# ---------------------------------------------------------------------------


@needs_l0_data
class TestMastersRecipe:
    """End-to-end recipe test: build_mini_database → get_calibration_stack_clusters
    → Bias.make_master_l1 → to_fits."""

    @pytest.fixture(scope="class")
    def recipe_output(self, tmp_path_factory):
        from unittest.mock import patch
        from kpfpipe.modules.masters.bias import Bias
        from kpfpipe.utils.kpf import get_obs_id

        tmp_path = tmp_path_factory.mktemp("recipe_out")
        data_root_in  = os.path.dirname(os.path.dirname(L0_DIR))
        data_root_out = str(tmp_path)
        datecode      = os.path.basename(L0_DIR)

        mini_db = build_mini_database(L0_DIR)

        output_paths = []
        for files in get_calibration_stack_clusters(mini_db, "bias"):
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
        from astropy.io import fits
        from kpfpipe.data_models.masters import KPFMasterL1
        for path in recipe_output:
            ml1 = KPFMasterL1.from_fits(path)
            assert ml1.data["GREEN_IMG"] is not None
            assert ml1.data["RED_IMG"] is not None
