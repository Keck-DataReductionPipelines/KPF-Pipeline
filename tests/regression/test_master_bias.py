"""Unit and regression tests for the master bias module (`Bias`).

Unit tests mock stack_frames (no real data); the regression tests stack the
bundled L0 bias frames. The shared engine (`BaseMasterModule`) these exercise is
unit-tested in test_master_base.py.
"""

from pathlib import Path

import numpy as np
import pytest

from kpfpipe import CHIPS
from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.modules.masters.bias import Bias
from kpfpipe.utils.io import kpf_filepath
from kpfpipe.utils.kpf import get_obs_id

from ._dtype_policy import L1_IMAGE, MASK_MEM, assert_dtype
from ._masters import (
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


class TestMasterBiasUnit:
    @pytest.fixture(scope="class")
    def master_bias(self):
        return make_mocked_master(Bias)

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

    def test_roundtrip_datalvl(self, master_bias, tmp_path):
        # The IMG round trip and the mask/BITPIX dtypes are covered in
        # test_data_models_l1.py and test_master_base.py.
        fn = str(tmp_path / MASTER_NAME)
        master_bias.to_fits(fn)
        assert KPFMasterL1.from_fits(fn).headers["PRIMARY"].get("DATALVL") == "ML1"


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


class TestMasterBiasSignature:
    @pytest.mark.parametrize("kwarg", ["bias", "dark", "flat"])
    def test_calibration_kwargs_raises(self, kwarg):
        module = Bias(FILE_LIST)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            module.make_master_l1(**{kwarg: True})


@pytest.mark.slow
@pytest.mark.requires_testdata
class TestMasterBiasRegression:
    @pytest.fixture(scope="class")
    def master_bias(self):
        return Bias(TESTDATA_BIAS_FILES).make_master_l1()

    def test_img_near_zero(self, master_bias):
        assert abs(np.nanmedian(master_bias.data["GREEN_IMG"])) < 5.0
        assert abs(np.nanmedian(master_bias.data["RED_IMG"])) < 5.0

    def test_snr_positive_on_good_pixels_never_negative(self, master_bias):
        # SNR is non-negative by construction (|counts| / sqrt(var)); bad pixels
        # are exactly zero, never negative, and good pixels strictly positive.
        for chip in CHIPS:
            snr = master_bias.data[f"{chip}_SNR"]
            assert np.all(snr >= 0)
            assert np.all(snr[master_bias.data[f"{chip}_MASK"]] > 0)

    def test_mask_mostly_good(self, master_bias):
        # A clean detector stack keeps the large majority of pixels.
        for chip in CHIPS:
            mask = master_bias.data[f"{chip}_MASK"]
            assert_dtype(mask, MASK_MEM, f"{chip}_MASK")
            assert np.mean(mask) > 0.9

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="prod bug: image_assembly.py:403-426 -- stitch_ffi adds read "
        "noise as a std, not rn**2; the master SNR (masters/base.py:817) sums "
        "that VAR, so the master variance is ~3x low",
    )
    @pytest.mark.parametrize("chip, rn_min", [("GREEN", 4.41), ("RED", 6.02)])
    def test_master_variance_floor_is_read_noise_over_n(
        self, master_bias, chip, rn_min
    ):
        # The variance of an N-frame mean bias is at least rn**2 / N, and
        # (IMG/SNR)**2 is exactly that variance. rn_min is an oracle: the weakest
        # amp's read noise measured on KP.20240405.03637.74 (GREEN 4.41, RED
        # 6.02 e-); the 0.9 factor absorbs frame-to-frame read-noise drift. Not a
        # two-sided rn**2/N check: the |data| Poisson term of a zero-signal frame
        # legitimately adds ~0.8 rn on top.
        img = master_bias.data[f"{chip}_IMG"]
        snr = master_bias.data[f"{chip}_SNR"]
        use = master_bias.data[f"{chip}_MASK"] & (snr > 0)
        var_mean = np.median((img[use] / snr[use]) ** 2)
        assert var_mean * len(TESTDATA_BIAS_FILES) >= 0.9 * rn_min**2

    def test_img_snr_dtype_is_float32(self, master_bias):
        for chip in CHIPS:
            for suffix in ("IMG", "SNR"):
                ext = f"{chip}_{suffix}"
                assert_dtype(master_bias.data[ext], L1_IMAGE, ext)

    def test_receipt_chain(self, master_bias):
        assert "master_bias" in master_bias.receipt["FUNCTION"].values

    # The persisted product: written to its convention path, then read back.
    # These assertions arrived from a recipe test that re-stacked this same
    # five-frame bias to reach them.

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

    def test_input_files_records_the_stacked_frames(self, master_bias_on_disk):
        # INPUT_FILES records the frames that survived stacking, which is also
        # what names the master. All five frames here pass, so every requested
        # file is present.
        _, read_back = master_bias_on_disk
        assert len(read_back.data["INPUT_FILES"]) == len(TESTDATA_BIAS_FILES)

    def test_input_files_all_fits(self, master_bias_on_disk):
        _, read_back = master_bias_on_disk
        filenames = read_back.data["INPUT_FILES"]["FILENAME"].tolist()
        assert all(name.endswith(".fits") for name in filenames)
