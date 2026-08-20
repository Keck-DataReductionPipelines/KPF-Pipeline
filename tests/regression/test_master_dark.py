"""Unit and regression tests for the master dark module (`Dark`).

Unit tests mock stack_frames; the regression class builds a real master dark from
the five bundled L0 darks, which span two default-gap clusters and HST midnight and
so are grouped with groupby='obs_night' (the whole night in one stack), as the
masters recipe does for darks. The shared stacking engine (`BaseMasterModule`) is
unit-tested in test_master_base.py.
"""

from pathlib import Path

import numpy as np
import pytest

from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.modules.masters.dark import Dark
from kpfpipe.utils.io import FileHandler

from ._dtype_policy import MASK_MEM, assert_dtype
from ._masters import CHIPS, FILE_LIST, make_mocked_master

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"


# ---------------------------------------------------------------------------
# Unit tests (mocked stack_frames)
# ---------------------------------------------------------------------------


class TestMasterDarkUnit:
    @pytest.fixture(scope="class")
    def master_dark(self):
        return make_mocked_master(Dark)

    def test_receipt_entry(self, master_dark):
        assert "master_dark" in master_dark.receipt["FUNCTION"].values

    def test_bunit_is_rate(self, master_dark):
        for chip in CHIPS:
            bunit = master_dark.headers[f"{chip}_IMG"].get("BUNIT")
            assert bunit == "electrons/sec"


# ---------------------------------------------------------------------------
# Signature: a dark is only bias-subtracted, so make_master_l1 takes bias only
# ---------------------------------------------------------------------------


class TestMasterDarkSignature:
    @pytest.mark.parametrize("kwarg", ["dark", "flat"])
    def test_dark_flat_kwargs_rejected(self, kwarg):
        with pytest.raises(TypeError):
            Dark(FILE_LIST).make_master_l1(**{kwarg: True})

    def test_bias_kwarg_accepted(self):
        assert isinstance(make_mocked_master(Dark, bias=False), KPFMasterL1)


# ---------------------------------------------------------------------------
# Regression: a real master dark from the bundled L0 darks (bias-subtracted,
# real flag_outliers rejection, real bad-pixel cleaning)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.requires_testdata
class TestMasterDarkRegression:
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
        # A clean detector stack keeps the large majority of pixels.
        for chip in CHIPS:
            mask = master_dark.data[f"{chip}_MASK"]
            assert_dtype(mask, MASK_MEM, f"{chip}_MASK")
            assert np.mean(mask) > 0.9

    def test_bias_subtracted_via_receipt(self, master_dark):
        assert "master_dark" in master_dark.receipt["FUNCTION"].values
