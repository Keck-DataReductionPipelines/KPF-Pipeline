"""
Unit and regression tests for the master bias module.

Uses mocked stack_frames for unit tests (no real data needed).
Real-data regression tests are gated on KPF_TESTDATA env var.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.modules.masters.bias import Bias

from ._dtype_policy import (
    L1_IMAGE,
    MASK_DISK,
    MASK_MEM,
    assert_dtype,
    assert_roundtrip_dtype,
)
from ._masters import make_l1_arrays

TESTDATA_L0_DIR = Path(__file__).parent.parent / "testdata" / "L0" / "20240405"
TESTDATA_BIAS_FILES = sorted(
    [
        str(TESTDATA_L0_DIR / "KP.20240405.03637.74.fits"),
        str(TESTDATA_L0_DIR / "KP.20240405.03687.64.fits"),
        str(TESTDATA_L0_DIR / "KP.20240405.03737.52.fits"),
        str(TESTDATA_L0_DIR / "KP.20240405.03787.42.fits"),
        str(TESTDATA_L0_DIR / "KP.20240405.03837.33.fits"),
    ]
)

CHIPS = ["GREEN", "RED"]
NROW, NCOL = 10, 10  # small arrays for unit tests
# make_l1_arrays() — shared synthetic stack_frames builder — lives in _masters.py

FILE_LIST = [f"KP.20240101.{i:05d}.00.fits" for i in range(8)]


# ---------------------------------------------------------------------------
# Dtype provenance (shared master-L1 path; see tests/regression/_dtype_policy.py)
# ---------------------------------------------------------------------------


class TestDtypeProvenance:
    """Master IMG/SNR are float32; MASK is bool in memory, uint8 (8-bit) on disk."""

    @pytest.fixture(scope="class")
    def master(self):
        bias = Bias(FILE_LIST)
        with patch.object(bias, "stack_frames", return_value=make_l1_arrays()):
            return bias.make_master_l1()

    def test_img_snr_float32(self, master):
        for ext in ("GREEN_IMG", "RED_IMG", "GREEN_SNR", "RED_SNR"):
            assert_dtype(master.data[ext], L1_IMAGE, ext)

    def test_mask_bool_in_memory(self, master):
        for ext in ("GREEN_MASK", "RED_MASK"):
            assert_dtype(master.data[ext], MASK_MEM, ext)

    def test_roundtrip_img_float32_mask_uint8(self, master, tmp_path):
        assert_roundtrip_dtype(KPFMasterL1, master, "GREEN_IMG", L1_IMAGE, tmp_path)
        assert_roundtrip_dtype(
            KPFMasterL1,
            master,
            "GREEN_MASK",
            MASK_MEM,
            tmp_path,
            expected_disk=MASK_DISK,
        )


# ---------------------------------------------------------------------------
# Unit tests (mocked stack_frames)
# ---------------------------------------------------------------------------


class TestMasterBiasUnit:
    """Unit tests using a mocked stack_frames — no real data needed."""

    @pytest.fixture(scope="class")
    def master_bias(self):
        synthetic = make_l1_arrays()
        bias = Bias(FILE_LIST)
        with patch.object(bias, "stack_frames", return_value=synthetic):
            return bias.make_master_l1()

    def test_returns_kpf_master_l1(self, master_bias):
        assert isinstance(master_bias, KPFMasterL1)

    @pytest.mark.parametrize(
        "ext",
        ["GREEN_IMG", "RED_IMG", "GREEN_SNR", "RED_SNR", "GREEN_MASK", "RED_MASK"],
    )
    def test_extension_shape(self, master_bias, ext):
        assert master_bias.data[ext].shape == (NROW, NCOL)

    def test_mask_is_boolean(self, master_bias):
        assert master_bias.data["GREEN_MASK"].dtype == bool
        assert master_bias.data["RED_MASK"].dtype == bool

    def test_snr_non_negative(self, master_bias):
        assert np.all(master_bias.data["GREEN_SNR"] >= 0)
        assert np.all(master_bias.data["RED_SNR"] >= 0)

    def test_receipt_entry(self, master_bias):
        assert "master_bias" in master_bias.receipt["FUNCTION"].values

    def test_bunit_is_electrons(self, master_bias):
        for chip in CHIPS:
            assert master_bias.headers[f"{chip}_IMG"].get("BUNIT") == "electrons"

    def test_datalvl_class_attribute(self, master_bias):
        assert master_bias._DATALVL == "ML1"


# ---------------------------------------------------------------------------
# info() smoke tests
# ---------------------------------------------------------------------------


class TestMasterBiasInfo:
    """Smoke tests for Bias.info() in both pre- and post-perform states."""

    def test_info_before_make_master_l1(self, capsys):
        bias = Bias(FILE_LIST)
        bias.info()
        out = capsys.readouterr().out
        assert "Bias" in out
        assert "make_master_l1() has not been called" in out

    def test_info_after_make_master_l1(self, capsys):
        synthetic = make_l1_arrays()
        bias = Bias(FILE_LIST)
        with patch.object(bias, "stack_frames", return_value=synthetic):
            bias.make_master_l1()
        bias.info()
        out = capsys.readouterr().out
        assert "Bias" in out
        assert "make_master_l1() has not been called" not in out
        for chip in CHIPS:
            assert chip in out


# ---------------------------------------------------------------------------
# FITS round-trip (mocked stack_frames)
# ---------------------------------------------------------------------------


class TestMasterBiasRoundTrip:
    """Test that master bias output survives a FITS write/read cycle."""

    def test_roundtrip_arrays(self):
        synthetic = make_l1_arrays()
        bias = Bias(FILE_LIST)
        with patch.object(bias, "stack_frames", return_value=synthetic):
            ml1 = bias.make_master_l1()

        with tempfile.TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, "master_bias.fits")
            ml1.to_fits(fn)
            ml1_read = KPFMasterL1.from_fits(fn)

        np.testing.assert_array_almost_equal(
            ml1_read.data["GREEN_IMG"], ml1.data["GREEN_IMG"], decimal=4
        )
        np.testing.assert_array_almost_equal(
            ml1_read.data["RED_IMG"], ml1.data["RED_IMG"], decimal=4
        )

    def test_roundtrip_datalvl(self):
        synthetic = make_l1_arrays()
        bias = Bias(FILE_LIST)
        with patch.object(bias, "stack_frames", return_value=synthetic):
            ml1 = bias.make_master_l1()

        with tempfile.TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, "master_bias.fits")
            ml1.to_fits(fn)
            ml1_read = KPFMasterL1.from_fits(fn)

        val = ml1_read.headers["PRIMARY"].get("DATALVL")
        assert val == "ML1"

    def test_roundtrip_mask_dtype(self):
        synthetic = make_l1_arrays()
        bias = Bias(FILE_LIST)
        with patch.object(bias, "stack_frames", return_value=synthetic):
            ml1 = bias.make_master_l1()

        with tempfile.TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, "master_bias.fits")
            ml1.to_fits(fn)
            ml1_read = KPFMasterL1.from_fits(fn)

        assert ml1_read.data["GREEN_MASK"].dtype == bool
        assert ml1_read.data["RED_MASK"].dtype == bool


# ---------------------------------------------------------------------------
# master_path integration with save_master
# ---------------------------------------------------------------------------


class TestMasterBiasSaveMaster:
    """make_master_l1(master_path=...) should write the FITS via save_master."""

    def test_master_path_writes_fits(self, tmp_path):
        synthetic = make_l1_arrays()
        bias = Bias(FILE_LIST)
        master_path = tmp_path / "master_bias.fits"
        with patch.object(bias, "stack_frames", return_value=synthetic):
            bias.make_master_l1(filepath=str(master_path))
        assert master_path.exists()

    def test_master_path_creates_parent_dir(self, tmp_path):
        synthetic = make_l1_arrays()
        bias = Bias(FILE_LIST)
        master_path = tmp_path / "nested" / "subdir" / "master_bias.fits"
        with patch.object(bias, "stack_frames", return_value=synthetic):
            bias.make_master_l1(filepath=str(master_path))
        assert master_path.exists()

    def test_master_path_overwrites_existing(self, tmp_path):
        synthetic = make_l1_arrays()
        bias = Bias(FILE_LIST)
        master_path = tmp_path / "master_bias.fits"
        master_path.touch()
        with patch.object(bias, "stack_frames", return_value=synthetic):
            bias.make_master_l1(filepath=str(master_path))
        assert master_path.read_bytes()[:6] == b"SIMPLE"

    def test_save_master_before_make_raises(self):
        bias = Bias(FILE_LIST)
        with pytest.raises(RuntimeError, match="run make_master_l1"):
            bias.save_master("L1", "/tmp/should_not_be_created.fits")

    def test_save_master_refuses_overwrite_by_default(self, tmp_path):
        synthetic = make_l1_arrays()
        bias = Bias(FILE_LIST)
        master_path = tmp_path / "master_bias.fits"
        master_path.touch()
        with patch.object(bias, "stack_frames", return_value=synthetic):
            bias.make_master_l1()  # populates ml1_obj
        with pytest.raises(FileExistsError, match="overwrite=True"):
            bias.save_master("L1", str(master_path))


# ---------------------------------------------------------------------------
# Regression tests (real L0 data)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMasterBiasRegression:
    """Regression tests against a real stack of L0 bias frames."""

    @pytest.fixture(scope="class")
    def master_bias(self):
        return Bias(TESTDATA_BIAS_FILES).make_master_l1()

    def test_returns_kpf_master_l1(self, master_bias):
        assert isinstance(master_bias, KPFMasterL1)

    def test_img_near_zero(self, master_bias):
        assert abs(np.nanmedian(master_bias.data["GREEN_IMG"])) < 5.0
        assert abs(np.nanmedian(master_bias.data["RED_IMG"])) < 5.0

    def test_snr_positive(self, master_bias):
        good = master_bias.data["GREEN_MASK"]
        assert np.all(master_bias.data["GREEN_SNR"][good] > 0)

    def test_snr_never_negative(self, master_bias):
        # SNR is non-negative by construction (|counts| / sqrt(var)); bad pixels
        # are exactly zero, never negative.
        for chip in ("GREEN", "RED"):
            assert np.all(master_bias.data[f"{chip}_SNR"] >= 0)

    def test_mask_has_good_pixels(self, master_bias):
        assert np.sum(master_bias.data["GREEN_MASK"]) > 0
        assert np.sum(master_bias.data["RED_MASK"]) > 0

    def test_img_snr_dtype_is_float32(self, master_bias):
        for chip in ("GREEN", "RED"):
            assert master_bias.data[f"{chip}_IMG"].dtype == np.float32
            assert master_bias.data[f"{chip}_SNR"].dtype == np.float32

    def test_receipt_chain(self, master_bias):
        assert "master_bias" in master_bias.receipt["FUNCTION"].values


# ---------------------------------------------------------------------------
# Signature: a bias applies no calibrations, so no bias/dark/flat kwargs
# ---------------------------------------------------------------------------


class TestMasterBiasSignature:
    @pytest.mark.parametrize("kwarg", ["bias", "dark", "flat"])
    def test_calibration_kwargs_rejected(self, kwarg):
        with pytest.raises(TypeError):
            Bias(FILE_LIST).make_master_l1(**{kwarg: True})
