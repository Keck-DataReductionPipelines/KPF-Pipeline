"""
Unit and regression tests for the master bias module.

Uses mocked stack_frames for unit tests (no real data needed).
Real-data regression tests are gated on KPF_TESTDATA env var.
"""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.modules.masters.bias import Bias


L0_DIR = os.environ.get("KPF_TESTDATA")

needs_l0_data = pytest.mark.skipif(
    L0_DIR is None or not os.path.isdir(L0_DIR),
    reason="L0 test data not available (set KPF_TESTDATA env var)",
)

CHIPS = ["GREEN", "RED"]
NROW, NCOL = 10, 10  # small arrays for unit tests


def make_l1_arrays(rng=None):
    """Return a synthetic stack_frames output dict."""
    if rng is None:
        rng = np.random.default_rng(42)
    arrays = {}
    for chip in CHIPS:
        arrays[f"{chip}_IMG"] = rng.normal(0.0, 5.0, (NROW, NCOL)).astype(np.float32)
        arrays[f"{chip}_SNR"] = np.abs(rng.normal(10.0, 1.0, (NROW, NCOL))).astype(np.float32)
        arrays[f"{chip}_MASK"] = np.ones((NROW, NCOL), dtype=bool)
    return arrays


FILE_LIST = [f"KP.20240101.{i:05d}.00.fits" for i in range(8)]


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

    def test_green_img_shape(self, master_bias):
        assert master_bias.data["GREEN_IMG"].shape == (NROW, NCOL)

    def test_red_img_shape(self, master_bias):
        assert master_bias.data["RED_IMG"].shape == (NROW, NCOL)

    def test_green_snr_shape(self, master_bias):
        assert master_bias.data["GREEN_SNR"].shape == (NROW, NCOL)

    def test_red_snr_shape(self, master_bias):
        assert master_bias.data["RED_SNR"].shape == (NROW, NCOL)

    def test_green_mask_shape(self, master_bias):
        assert master_bias.data["GREEN_MASK"].shape == (NROW, NCOL)

    def test_red_mask_shape(self, master_bias):
        assert master_bias.data["RED_MASK"].shape == (NROW, NCOL)

    def test_mask_is_boolean(self, master_bias):
        assert master_bias.data["GREEN_MASK"].dtype == bool
        assert master_bias.data["RED_MASK"].dtype == bool

    def test_snr_non_negative(self, master_bias):
        assert np.all(master_bias.data["GREEN_SNR"] >= 0)
        assert np.all(master_bias.data["RED_SNR"] >= 0)

    def test_receipt_entry(self, master_bias):
        assert "master_bias" in master_bias.receipt["Module_Name"].values

    def test_datalvl_class_attribute(self, master_bias):
        assert master_bias._DATALVL == "ML1"


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

        datalvl = ml1_read.headers["PRIMARY"]["DATALVL"]
        val = datalvl[0] if isinstance(datalvl, tuple) else datalvl
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
# Regression tests (real L0 data)
# ---------------------------------------------------------------------------


@needs_l0_data
class TestMasterBiasRegression:
    """Regression tests against a real stack of L0 bias frames."""

    @pytest.fixture(scope="class")
    def l0_bias_list(self):
        fns = sorted(
            os.path.join(L0_DIR, f)
            for f in os.listdir(L0_DIR)
            if f.endswith(".fits")
        )
        if len(fns) < 2:
            pytest.skip("Need at least two L0 files for stacking")
        return fns[:10]

    @pytest.fixture(scope="class")
    def master_bias(self, l0_bias_list):
        return Bias(l0_bias_list).make_master_l1()

    def test_returns_kpf_master_l1(self, master_bias):
        assert isinstance(master_bias, KPFMasterL1)

    def test_img_near_zero(self, master_bias):
        assert abs(np.nanmedian(master_bias.data["GREEN_IMG"])) < 5.0
        assert abs(np.nanmedian(master_bias.data["RED_IMG"])) < 5.0

    def test_snr_positive(self, master_bias):
        good = master_bias.data["GREEN_MASK"]
        assert np.all(master_bias.data["GREEN_SNR"][good] > 0)

    def test_mask_has_good_pixels(self, master_bias):
        assert np.sum(master_bias.data["GREEN_MASK"]) > 0
        assert np.sum(master_bias.data["RED_MASK"]) > 0

    def test_receipt_chain(self, master_bias):
        assert "master_bias" in master_bias.receipt["Module_Name"].values
