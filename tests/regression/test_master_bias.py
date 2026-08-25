"""Unit and regression tests for the master bias module (`Bias`).

Unit tests mock stack_frames (no real data); the regression tests stack the
bundled L0 bias frames. The shared engine (`BaseMasterModule`) these exercise is
unit-tested in test_master_base.py.
"""

from pathlib import Path

import numpy as np
import pytest

from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.modules.masters.bias import Bias
from kpfpipe.utils.io import kpf_filepath
from kpfpipe.utils.kpf import get_obs_id

from ._dtype_policy import L1_IMAGE, assert_dtype
from ._masters import (
    CHIPS,
    FILE_LIST,
    MASTER_NAME,
    NCOL,
    NROW,
    make_mocked_master,
    mocked_stack,
)

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


# ---------------------------------------------------------------------------
# Unit tests (mocked stack_frames)
# ---------------------------------------------------------------------------


class TestMasterBiasUnit:
    @pytest.fixture(scope="class")
    def master_bias(self):
        return make_mocked_master(Bias)

    def test_returns_kpf_master_l1(self, master_bias):
        assert isinstance(master_bias, KPFMasterL1)

    @pytest.mark.parametrize(
        "ext",
        ["GREEN_IMG", "RED_IMG", "GREEN_SNR", "RED_SNR", "GREEN_MASK", "RED_MASK"],
    )
    def test_extension_shape(self, master_bias, ext):
        assert master_bias.data[ext].shape == (NROW, NCOL)

    def test_receipt_entry(self, master_bias):
        assert "master_bias" in master_bias.receipt["FUNCTION"].values

    def test_bunit_is_electrons(self, master_bias):
        for chip in CHIPS:
            assert master_bias.headers[f"{chip}_IMG"].get("BUNIT") == "electrons"


# ---------------------------------------------------------------------------
# info() smoke tests
# ---------------------------------------------------------------------------


class TestMasterBiasInfo:
    def test_info_before_make_master_l1(self, capsys):
        bias = Bias(FILE_LIST)
        bias.info()
        out = capsys.readouterr().out
        assert "Bias" in out
        assert "make_master_l1() has not been called" in out

    def test_info_after_make_master_l1(self, capsys):
        bias = Bias(FILE_LIST)
        with mocked_stack(bias):
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
    """The mask/BITPIX dtype round-trip lives in test_master_base.py, which
    exercises the same shared write path and also checks the on-disk BITPIX."""

    @pytest.fixture
    def reread(self, tmp_path):
        ml1 = make_mocked_master(Bias)
        fn = str(tmp_path / MASTER_NAME)
        ml1.to_fits(fn)
        return ml1, KPFMasterL1.from_fits(fn)

    def test_roundtrip_arrays(self, reread):
        ml1, ml1_read = reread
        for chip in CHIPS:
            np.testing.assert_array_almost_equal(
                ml1_read.data[f"{chip}_IMG"], ml1.data[f"{chip}_IMG"], decimal=4
            )

    def test_roundtrip_datalvl(self, reread):
        _, ml1_read = reread
        assert ml1_read.headers["PRIMARY"].get("DATALVL") == "ML1"


# ---------------------------------------------------------------------------
# Signature: a bias applies no calibrations, so no bias/dark/flat kwargs
# ---------------------------------------------------------------------------


class TestMasterBiasSignature:
    @pytest.mark.parametrize("kwarg", ["bias", "dark", "flat"])
    def test_calibration_kwargs_rejected(self, kwarg):
        module = Bias(FILE_LIST)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            module.make_master_l1(**{kwarg: True})


# ---------------------------------------------------------------------------
# Regression tests (real L0 data)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.requires_testdata
class TestMasterBiasRegression:
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
        for chip in CHIPS:
            for suffix in ("IMG", "SNR"):
                ext = f"{chip}_{suffix}"
                assert_dtype(master_bias.data[ext], L1_IMAGE, ext)

    def test_receipt_chain(self, master_bias):
        assert "master_bias" in master_bias.receipt["FUNCTION"].values

    # ------------------------------------------------------------------
    # The persisted product: written through kpf_filepath, as the masters
    # recipe writes it, then read back. These assertions arrived from a
    # recipe test that re-stacked this same five-frame bias to reach them.
    # ------------------------------------------------------------------

    @pytest.fixture(scope="class")
    def master_bias_on_disk(self, master_bias, tmp_path_factory):
        out_path = kpf_filepath(
            get_obs_id(TESTDATA_BIAS_FILES[0]),
            "L1",
            data_root=str(tmp_path_factory.mktemp("master_bias_out")),
            master="bias",
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        master_bias.to_fits(out_path)
        return out_path, KPFMasterL1.from_fits(out_path)

    def test_written_to_the_convention_path(self, master_bias_on_disk):
        out_path, _ = master_bias_on_disk
        assert Path(out_path).is_file()
        assert Path(out_path).name.endswith("_master_bias_L1.fits")

    def test_round_trip_preserves_chip_images(self, master_bias, master_bias_on_disk):
        _, read_back = master_bias_on_disk
        for chip in CHIPS:
            np.testing.assert_array_equal(
                read_back.data[f"{chip}_IMG"], master_bias.data[f"{chip}_IMG"]
            )

    def test_input_files_extension_present(self, master_bias_on_disk):
        _, read_back = master_bias_on_disk
        assert "INPUT_FILES" in read_back.extensions

    def test_input_files_records_the_stacked_frames(self, master_bias_on_disk):
        # NOTE: production records the *requested* file list, not the frames that
        # survived stacking -- masters/base.py passes l0_file_list into
        # set_input_files unchanged, while _load_frame may drop QC failures within
        # the tolerated budget. All five frames here pass, so the count is the
        # same under either semantic; when that provenance defect is fixed, this
        # is the assertion that will need revisiting.
        _, read_back = master_bias_on_disk
        assert len(read_back.data["INPUT_FILES"]) == len(TESTDATA_BIAS_FILES)

    def test_input_files_all_fits(self, master_bias_on_disk):
        _, read_back = master_bias_on_disk
        filenames = read_back.data["INPUT_FILES"]["FILENAME"].tolist()
        assert all(name.endswith(".fits") for name in filenames)
