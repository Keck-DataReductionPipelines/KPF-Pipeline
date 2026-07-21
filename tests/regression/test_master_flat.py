"""Unit and regression tests for the master flat module (`Flat`).

Unit tests mock stack_frames (no real data). TestMasterFlatRegression builds a
real master flat from the bundled L0 flats: the five frames fall within one
time-of-day cluster, so it groups them with groupby='time_of_day' (as the masters
recipe does for flats) before bias- and dark-subtracting each frame. The shared
stacking engine these exercise (`BaseMasterModule`) is unit-tested in
test_master_base.py.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.modules.masters.flat import Flat
from kpfpipe.utils.io import FileHandler

from ._masters import make_l1_arrays

CHIPS = ["GREEN", "RED"]
# make_l1_arrays() -- shared synthetic stack_frames builder -- lives in _masters.py

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"

FILE_LIST = [f"KP.20240101.{i:05d}.00.fits" for i in range(8)]


# ---------------------------------------------------------------------------
# Unit tests (mocked stack_frames)
# ---------------------------------------------------------------------------


class TestMasterFlatUnit:
    """Unit tests using a mocked stack_frames -- no real data needed."""

    @pytest.fixture(scope="class")
    def master_flat(self):
        synthetic = make_l1_arrays()
        flat = Flat(FILE_LIST)
        with patch.object(flat, "stack_frames", return_value=synthetic):
            return flat.make_master_l1()

    def test_receipt_entry(self, master_flat):
        assert "master_flat" in master_flat.receipt["FUNCTION"].values

    def test_bunit_is_unset(self, master_flat):
        # A flat is a unitless relative throughput, so it carries no BUNIT.
        for chip in CHIPS:
            assert master_flat.headers[f"{chip}_IMG"].get("BUNIT") is None


# ---------------------------------------------------------------------------
# Signature: a flat is bias- and dark-subtracted, so make_master_l1 takes bias
# and dark but not flat.
# ---------------------------------------------------------------------------


class TestMasterFlatSignature:
    def test_flat_kwarg_rejected(self):
        with pytest.raises(TypeError):
            Flat(FILE_LIST).make_master_l1(flat=True)

    @pytest.mark.parametrize("kwarg", ["bias", "dark"])
    def test_bias_dark_kwargs_accepted(self, kwarg):
        synthetic = make_l1_arrays()
        flat = Flat(FILE_LIST)
        with patch.object(flat, "stack_frames", return_value=synthetic):
            ml1 = flat.make_master_l1(**{kwarg: False})
        assert isinstance(ml1, KPFMasterL1)


# ---------------------------------------------------------------------------
# Regression: a real master flat from the bundled L0 flats (bias- and dark-
# subtracted, real flag_outliers rejection, real bad-pixel cleaning)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMasterFlatRegression:
    """End-to-end master flat from real L0 frames. The five bundled flats fall
    within a single time-of-day cluster, so they are grouped with
    groupby='time_of_day' (as the masters recipe does for flats); each frame is
    bias- and dark-subtracted against the bundled master bias and dark."""

    @pytest.fixture(scope="class")
    def master_flat(self):
        file_handler = FileHandler({"KPF_DATA_INPUT": str(TESTDATA_DIR)})
        file_handler.build_mini_database("20240405")
        files = file_handler.build_calibration_stacks(
            "flat",
            min_stack_size=5,
            groupby="time_of_day",
        )
        assert len(files) == 1 and len(files[0]) == 5
        config = {"KPF_MASTERS_OUTPUT": str(TESTDATA_DIR)}
        return Flat(files[0], config=config).make_master_l1()

    def test_returns_kpf_master_l1(self, master_flat):
        assert isinstance(master_flat, KPFMasterL1)

    def test_flat_is_illuminated(self, master_flat):
        # A flat is an illuminated exposure, so the stacked image has real
        # positive signal.
        for chip in CHIPS:
            median = np.nanmedian(master_flat.data[f"{chip}_IMG"])
            assert median > 0

    def test_bunit_is_unset(self, master_flat):
        for chip in CHIPS:
            assert master_flat.headers[f"{chip}_IMG"].get("BUNIT") is None

    def test_snr_never_negative(self, master_flat):
        for chip in CHIPS:
            assert np.all(master_flat.data[f"{chip}_SNR"] >= 0)

    def test_mask_mostly_good(self, master_flat):
        # A clean detector stack keeps the large majority of pixels. The final
        # trend outlier pass scales its threshold by a local (not global) noise
        # scale, so the flat's bright orders are not over-flagged against the
        # dark inter-order floor.
        for chip in CHIPS:
            mask = master_flat.data[f"{chip}_MASK"]
            assert mask.dtype == bool
            assert np.mean(mask) > 0.9

    def test_calibrated_via_receipt(self, master_flat):
        assert "master_flat" in master_flat.receipt["FUNCTION"].values
