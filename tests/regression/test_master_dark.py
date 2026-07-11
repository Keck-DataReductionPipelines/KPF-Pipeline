"""Unit and regression tests for the master dark module (`Dark`).

Unit tests mock stack_frames (no real data). TestMasterDarkRegression builds a
real master dark from the bundled L0 darks: the five frames span two default-gap
clusters and HST midnight, so it widens cluster_gap_seconds and lifts the
midnight boundary to group them into one stack, bias-subtracting each frame. The
shared stacking engine these exercise (`BaseMasterModule`) is unit-tested in
test_master_base.py.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.modules.masters.dark import Dark
from kpfpipe.utils.io import FileHandler

from ._masters import make_l1_arrays

CHIPS = ["GREEN", "RED"]
NROW, NCOL = 10, 10  # small arrays for unit tests
# make_l1_arrays() — shared synthetic stack_frames builder — lives in _masters.py

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"

FILE_LIST = [f"KP.20240101.{i:05d}.00.fits" for i in range(8)]


# ---------------------------------------------------------------------------
# Unit tests (mocked stack_frames)
# ---------------------------------------------------------------------------


class TestMasterDarkUnit:
    """Unit tests using a mocked stack_frames — no real data needed."""

    @pytest.fixture(scope="class")
    def master_dark(self):
        synthetic = make_l1_arrays()
        dark = Dark(FILE_LIST)
        with patch.object(dark, "stack_frames", return_value=synthetic):
            return dark.make_master_l1()

    def test_returns_kpf_master_l1(self, master_dark):
        assert isinstance(master_dark, KPFMasterL1)

    @pytest.mark.parametrize("ext", ["GREEN_IMG", "RED_IMG", "GREEN_SNR", "RED_SNR"])
    def test_extension_shape(self, master_dark, ext):
        assert master_dark.data[ext].shape == (NROW, NCOL)

    def test_mask_is_boolean(self, master_dark):
        assert master_dark.data["GREEN_MASK"].dtype == bool
        assert master_dark.data["RED_MASK"].dtype == bool

    def test_snr_non_negative(self, master_dark):
        assert np.all(master_dark.data["GREEN_SNR"] >= 0)
        assert np.all(master_dark.data["RED_SNR"] >= 0)

    def test_receipt_entry(self, master_dark):
        assert "master_dark" in master_dark.receipt["FUNCTION"].values

    def test_bunit_is_rate(self, master_dark):
        for chip in CHIPS:
            bunit = master_dark.headers[f"{chip}_IMG"].get("BUNIT")
            assert bunit == "electrons/sec"

    def test_datalvl_class_attribute(self, master_dark):
        assert master_dark._DATALVL == "ML1"


# ---------------------------------------------------------------------------
# info() smoke tests
# ---------------------------------------------------------------------------


class TestMasterDarkInfo:
    """Smoke tests for Dark.info() in both pre- and post-perform states."""

    def test_info_before_make_master_l1(self, capsys):
        dark = Dark(FILE_LIST)
        dark.info()
        out = capsys.readouterr().out
        assert "Dark" in out
        assert "make_master_l1() has not been called" in out

    def test_info_after_make_master_l1(self, capsys):
        synthetic = make_l1_arrays()
        dark = Dark(FILE_LIST)
        with patch.object(dark, "stack_frames", return_value=synthetic):
            dark.make_master_l1()
        dark.info()
        out = capsys.readouterr().out
        assert "Dark" in out
        assert "make_master_l1() has not been called" not in out
        for chip in CHIPS:
            assert chip in out


# ---------------------------------------------------------------------------
# FITS round-trip (mocked stack_frames)
# ---------------------------------------------------------------------------


class TestMasterDarkRoundTrip:
    """Test that master dark output survives a FITS write/read cycle."""

    def test_roundtrip_arrays(self):
        synthetic = make_l1_arrays()
        dark = Dark(FILE_LIST)
        with patch.object(dark, "stack_frames", return_value=synthetic):
            ml1 = dark.make_master_l1()

        with tempfile.TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, "master_dark.fits")
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
        dark = Dark(FILE_LIST)
        with patch.object(dark, "stack_frames", return_value=synthetic):
            ml1 = dark.make_master_l1()

        with tempfile.TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, "master_dark.fits")
            ml1.to_fits(fn)
            ml1_read = KPFMasterL1.from_fits(fn)

        assert ml1_read.headers["PRIMARY"].get("DATALVL") == "ML1"


# ---------------------------------------------------------------------------
# Signature: a dark is only bias-subtracted, so make_master_l1 takes bias only
# ---------------------------------------------------------------------------


class TestMasterDarkSignature:
    @pytest.mark.parametrize("kwarg", ["dark", "flat"])
    def test_dark_flat_kwargs_rejected(self, kwarg):
        with pytest.raises(TypeError):
            Dark(FILE_LIST).make_master_l1(**{kwarg: True})

    def test_bias_kwarg_accepted(self):
        synthetic = make_l1_arrays()
        dark = Dark(FILE_LIST)
        with patch.object(dark, "stack_frames", return_value=synthetic):
            ml1 = dark.make_master_l1(bias=False)
        assert isinstance(ml1, KPFMasterL1)


# ---------------------------------------------------------------------------
# Regression: a real master dark from the bundled L0 darks (bias-subtracted,
# real flag_outliers rejection, real bad-pixel cleaning)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMasterDarkRegression:
    """End-to-end master dark from real L0 frames. The five bundled darks span
    two default-gap clusters *and* HST midnight, so they are grouped with
    groupby='obs_night' (the whole night in one stack, as the masters recipe does
    for darks); each frame is bias-subtracted against the bundled master bias."""

    @pytest.fixture(scope="class")
    def master_dark(self):
        file_handler = FileHandler({"KPF_DATA_INPUT": str(TESTDATA_DIR)})
        file_handler.build_mini_database("20240405")
        files = file_handler.build_calibration_stacks(
            "dark",
            min_stack_size=5,
            groupby="obs_night",
        )
        assert len(files) == 1 and len(files[0]) == 5
        config = {"KPF_MASTERS_OUTPUT": str(TESTDATA_DIR)}
        return Dark(files[0], config=config).make_master_l1()

    def test_returns_kpf_master_l1(self, master_dark):
        assert isinstance(master_dark, KPFMasterL1)

    def test_dark_current_is_small_and_positive(self, master_dark):
        # KPF dark current is a small positive rate (electrons/sec).
        for chip in CHIPS:
            median = np.nanmedian(master_dark.data[f"{chip}_IMG"])
            assert 0 < median < 1.0

    def test_bunit_is_rate(self, master_dark):
        for chip in CHIPS:
            bunit = master_dark.headers[f"{chip}_IMG"].get("BUNIT")
            assert bunit == "electrons/sec"

    def test_snr_never_negative(self, master_dark):
        for chip in CHIPS:
            assert np.all(master_dark.data[f"{chip}_SNR"] >= 0)

    def test_mask_mostly_good(self, master_dark):
        # A clean detector stack should keep the large majority of pixels.
        for chip in CHIPS:
            mask = master_dark.data[f"{chip}_MASK"]
            assert mask.dtype == bool
            assert np.mean(mask) > 0.9

    def test_bias_subtracted_via_receipt(self, master_dark):
        assert "master_dark" in master_dark.receipt["FUNCTION"].values
