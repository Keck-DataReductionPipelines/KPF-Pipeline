"""Tests for the ImageProcessing module (L1 bias and dark subtraction).

Synthetic arrays and a MockL1 throughout; the only FITS files are the minimal
masters written into tmp_path, since some paths require one on disk.
"""

import os

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from kpfpipe.data_models.base import KPFDataModel
from kpfpipe.data_models.masters.level1 import KPFMasterL1
from kpfpipe.modules.image_processing import ImageProcessing

from ._dtype_policy import L1_IMAGE, assert_dtype

_SHAPE = (4, 4)
_CCD_VALUE = 10.0
_VAR_VALUE = 12.0  # science image variance [electrons^2]
_BIAS_VALUE = 3.0
_BIAS_SNR = 50.0  # master bias SNR; var contribution = (IMG/SNR)^2
_DARK_VALUE = 2.0  # master dark rate [electrons/sec]
_DARK_SNR = 40.0  # master dark SNR
_EXPTIME = 4.0  # != 1 so dark scaling is observable


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class MockL1:
    def __init__(self):
        self.obs_id = "KP.20240405.40113.57"
        # Mirrors a real L1: dark scaling reads the EPRV-standard PRIMARY EXPTIME,
        # and the calibration flags/paths live in the RECEIPT table.
        primary = fits.Header()
        primary["EXPTIME"] = _EXPTIME
        self.headers = {
            "PRIMARY": primary,
            "INSTRUMENT_HEADER": fits.Header(),
            "RECEIPT": fits.Header(),
            "QUALITY_CONTROL": fits.Header(),
        }
        self.data = {
            "GREEN_CCD": np.full(_SHAPE, _CCD_VALUE, dtype=np.float32),
            "RED_CCD": np.full(_SHAPE, _CCD_VALUE, dtype=np.float32),
            "GREEN_VAR": np.full(_SHAPE, _VAR_VALUE, dtype=np.float32),
            "RED_VAR": np.full(_SHAPE, _VAR_VALUE, dtype=np.float32),
        }
        self.receipt = pd.DataFrame(columns=["FUNCTION", "ARGS", "STATUS"])

    def receipt_add_entry(self, name, args, status):
        self.receipt.loc[len(self.receipt)] = [name, args, status]

    # The parse under test is production code, so borrow it rather than mock it.
    receipt_read_entry = KPFDataModel.receipt_read_entry


def _make_module(bias_file=None, bias_dir=None, dark_file=None, dark_dir=None):
    # {bias,dark}file hold the master's full path (no separate dir argument), so
    # join the optional dir onto the filename when composing the receipt entry.
    def path(name, dirname):
        if name is None:
            return None
        return os.path.join(dirname, name) if dirname is not None else name

    l1 = MockL1()
    l1.receipt_add_entry(
        "calibration_association",
        f"biasfile={path(bias_file, bias_dir)}, "
        f"darkfile={path(dark_file, dark_dir)}, flatfile=None, wlsfile=None",
        "PASS",
    )
    return ImageProcessing(l1)


def _write_master(path, value, snr):
    """Write a minimal master FITS file: per-chip IMG = value, SNR = snr."""
    hdus = [fits.PrimaryHDU()]
    for chip in ("GREEN", "RED"):
        hdus.append(
            fits.ImageHDU(
                data=np.full(_SHAPE, value, dtype=np.float32), name=f"{chip}_IMG"
            )
        )
        hdus.append(
            fits.ImageHDU(
                data=np.full(_SHAPE, snr, dtype=np.float32), name=f"{chip}_SNR"
            )
        )
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _write_master_bias(path):
    _write_master(path, _BIAS_VALUE, _BIAS_SNR)


def _write_master_dark(path):
    _write_master(path, _DARK_VALUE, _DARK_SNR)


def _master_bias(tmp_path):
    path = str(tmp_path / "master_bias.fits")
    _write_master_bias(path)
    return KPFMasterL1.from_fits(path)


def _master_dark(tmp_path):
    path = str(tmp_path / "master_dark.fits")
    _write_master_dark(path)
    return KPFMasterL1.from_fits(path)


# ---------------------------------------------------------------------------
# TestDtypeProvenance
# ---------------------------------------------------------------------------


class TestDtypeProvenance:
    """L1 CCD/VAR must stay float32 through bias and dark subtraction; the
    dark * exptime (float64) scaling is the prime upcast trap."""

    def _module_bias_dark(self, tmp_path):
        _write_master_bias(str(tmp_path / "master_bias.fits"))
        _write_master_dark(str(tmp_path / "master_dark.fits"))
        return _make_module(
            bias_file="master_bias.fits",
            bias_dir=str(tmp_path),
            dark_file="master_dark.fits",
            dark_dir=str(tmp_path),
        )

    def test_ccd_var_stay_float32_after_bias_dark(self, tmp_path):
        mod = self._module_bias_dark(tmp_path)
        mod.perform()
        for ext in ("GREEN_CCD", "RED_CCD", "GREEN_VAR", "RED_VAR"):
            assert_dtype(mod.l1_obj.data[ext], L1_IMAGE, f"{ext} after bias+dark")


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_none_config(self):
        ip = ImageProcessing(MockL1())
        assert ip.chips == ["GREEN", "RED"]

    def test_dict_config_overrides_chips(self):
        l1 = MockL1()
        ip = ImageProcessing(l1, config={"chips": ["GREEN"]})
        assert ip.chips == ["GREEN"]

    def test_invalid_config_raises(self):
        with pytest.raises(TypeError, match="config must be"):
            ImageProcessing(MockL1(), config=42)

    def test_paths_none_before_perform(self):
        mod = ImageProcessing(MockL1())
        assert mod._bias_path is None
        assert mod._dark_path is None

    def test_master_caches_none_before_perform(self):
        mod = ImageProcessing(MockL1())
        assert mod._bias_ml1 is None
        assert mod._dark_ml1 is None


# ---------------------------------------------------------------------------
# TestSubtractBias
# ---------------------------------------------------------------------------


class TestSubtractBias:
    def test_subtracts_correct_values_green(self, tmp_path):
        mod = _make_module()
        mod.subtract_bias("GREEN", _master_bias(tmp_path))
        expected = _CCD_VALUE - _BIAS_VALUE
        np.testing.assert_allclose(mod.l1_obj.data["GREEN_CCD"], expected)

    def test_subtracts_correct_values_red(self, tmp_path):
        mod = _make_module()
        mod.subtract_bias("RED", _master_bias(tmp_path))
        expected = _CCD_VALUE - _BIAS_VALUE
        np.testing.assert_allclose(mod.l1_obj.data["RED_CCD"], expected)

    def test_modifies_in_place(self, tmp_path):
        mod = _make_module()
        original = mod.l1_obj.data["GREEN_CCD"]
        mod.subtract_bias("GREEN", _master_bias(tmp_path))
        assert mod.l1_obj.data["GREEN_CCD"] is original

    def test_chip_name_case_insensitive(self, tmp_path):
        mod = _make_module()
        mod.subtract_bias("green", _master_bias(tmp_path))
        np.testing.assert_allclose(
            mod.l1_obj.data["GREEN_CCD"], _CCD_VALUE - _BIAS_VALUE
        )

    def test_propagates_variance(self, tmp_path):
        mod = _make_module()
        mod.subtract_bias("GREEN", _master_bias(tmp_path))
        expected = _VAR_VALUE + (_BIAS_VALUE / _BIAS_SNR) ** 2
        np.testing.assert_allclose(mod.l1_obj.data["GREEN_VAR"], expected, rtol=1e-5)

    def test_zero_snr_pixel_adds_no_variance(self, tmp_path):
        # A master pixel with SNR == 0 (bad / zero-flux) must not inject inf/NaN.
        path = str(tmp_path / "master_bias.fits")
        snr = np.full(_SHAPE, _BIAS_SNR, dtype=np.float32)
        snr[0, 0] = 0.0
        hdus = [fits.PrimaryHDU()]
        for chip in ("GREEN", "RED"):
            hdus.append(
                fits.ImageHDU(
                    data=np.full(_SHAPE, _BIAS_VALUE, dtype=np.float32),
                    name=f"{chip}_IMG",
                )
            )
            hdus.append(fits.ImageHDU(data=snr.copy(), name=f"{chip}_SNR"))
        fits.HDUList(hdus).writeto(path, overwrite=True)

        mod = _make_module()
        mod.subtract_bias("GREEN", KPFMasterL1.from_fits(path))
        var = mod.l1_obj.data["GREEN_VAR"]
        assert np.all(np.isfinite(var))
        np.testing.assert_allclose(var[0, 0], _VAR_VALUE)
        np.testing.assert_allclose(
            var[1, 1], _VAR_VALUE + (_BIAS_VALUE / _BIAS_SNR) ** 2, rtol=1e-5
        )


# ---------------------------------------------------------------------------
# TestSubtractDark
# ---------------------------------------------------------------------------


class TestSubtractDark:
    def test_subtracts_scaled_by_exptime_green(self, tmp_path):
        mod = _make_module()
        mod.subtract_dark("GREEN", _master_dark(tmp_path))
        expected = _CCD_VALUE - _DARK_VALUE * _EXPTIME
        np.testing.assert_allclose(mod.l1_obj.data["GREEN_CCD"], expected)

    def test_subtracts_scaled_by_exptime_red(self, tmp_path):
        mod = _make_module()
        mod.subtract_dark("RED", _master_dark(tmp_path))
        expected = _CCD_VALUE - _DARK_VALUE * _EXPTIME
        np.testing.assert_allclose(mod.l1_obj.data["RED_CCD"], expected)

    def test_scaling_is_applied_not_raw_subtraction(self, tmp_path):
        # Guard against a literal copy of subtract_bias (no exptime factor):
        # the result must reflect dark_IMG * EXPTIME, not dark_IMG alone.
        mod = _make_module()
        mod.subtract_dark("GREEN", _master_dark(tmp_path))
        unscaled = _CCD_VALUE - _DARK_VALUE
        assert not np.allclose(mod.l1_obj.data["GREEN_CCD"], unscaled)

    def test_modifies_in_place(self, tmp_path):
        mod = _make_module()
        original = mod.l1_obj.data["GREEN_CCD"]
        mod.subtract_dark("GREEN", _master_dark(tmp_path))
        assert mod.l1_obj.data["GREEN_CCD"] is original

    def test_chip_name_case_insensitive(self, tmp_path):
        mod = _make_module()
        mod.subtract_dark("green", _master_dark(tmp_path))
        np.testing.assert_allclose(
            mod.l1_obj.data["GREEN_CCD"], _CCD_VALUE - _DARK_VALUE * _EXPTIME
        )

    def test_propagates_variance_scaled_by_exptime(self, tmp_path):
        mod = _make_module()
        mod.subtract_dark("GREEN", _master_dark(tmp_path))
        expected = _VAR_VALUE + (_EXPTIME * _DARK_VALUE / _DARK_SNR) ** 2
        np.testing.assert_allclose(mod.l1_obj.data["GREEN_VAR"], expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# TestPerform
# ---------------------------------------------------------------------------


class TestPerform:
    @pytest.fixture
    def mod_with_bias(self, tmp_path, monkeypatch):
        bias_path = str(tmp_path / "master_bias.fits")
        _write_master_bias(bias_path)
        mod = _make_module(bias_file="master_bias.fits", bias_dir=str(tmp_path))
        # These tests exercise bias subtraction in isolation; dark defaults on.
        mod.dark = False
        return mod

    def test_returns_l1_obj(self, mod_with_bias):
        result = mod_with_bias.perform()
        assert result is mod_with_bias.l1_obj

    def test_bias_subtracted_green(self, mod_with_bias):
        mod_with_bias.perform()
        np.testing.assert_allclose(
            mod_with_bias.l1_obj.data["GREEN_CCD"], _CCD_VALUE - _BIAS_VALUE
        )

    def test_bias_subtracted_red(self, mod_with_bias):
        mod_with_bias.perform()
        np.testing.assert_allclose(
            mod_with_bias.l1_obj.data["RED_CCD"], _CCD_VALUE - _BIAS_VALUE
        )

    def test_bias_path_recorded(self, mod_with_bias, tmp_path):
        mod_with_bias.perform()
        assert mod_with_bias._bias_path == str(tmp_path / "master_bias.fits")

    def test_biassub_header_set(self, mod_with_bias):
        mod_with_bias.perform()
        assert (
            mod_with_bias.l1_obj.receipt_read_entry("image_processing")["biassub"]
            == "1"
        )

    def test_receipt_entry_added(self, mod_with_bias):
        mod_with_bias.perform()
        entry = mod_with_bias.l1_obj.receipt_read_entry("image_processing")
        assert entry == {"biassub": "1", "darksub": "0", "flatdiv": "0"}

    def test_chips_override_processes_only_requested(self, mod_with_bias):
        mod_with_bias.perform(chips=["GREEN"])
        np.testing.assert_allclose(
            mod_with_bias.l1_obj.data["GREEN_CCD"], _CCD_VALUE - _BIAS_VALUE
        )
        np.testing.assert_allclose(mod_with_bias.l1_obj.data["RED_CCD"], _CCD_VALUE)

    def test_raises_when_headers_missing(self):
        mod = _make_module()  # no biasfile
        with pytest.raises(FileNotFoundError, match="biasfile"):
            mod.perform()

    def test_bias_false_skips_subtraction(self):
        mod = _make_module()  # no biasfile needed when bias is off
        result = mod.perform(bias=False, dark=False)
        np.testing.assert_allclose(result.data["GREEN_CCD"], _CCD_VALUE)
        np.testing.assert_allclose(result.data["RED_CCD"], _CCD_VALUE)
        assert result.receipt_read_entry("image_processing")["biassub"] == "0"
        assert mod._bias_path is None

    def test_darksub_header_false_when_dark_off(self, mod_with_bias):
        mod_with_bias.perform()
        assert (
            mod_with_bias.l1_obj.receipt_read_entry("image_processing")["darksub"]
            == "0"
        )

    def test_flatdiv_header_false_when_flat_off(self, mod_with_bias):
        # All three applied-step flags are written on every run: QCL1 gates
        # flat_ok on the 0 as much as on the 1.
        mod_with_bias.perform()
        assert (
            mod_with_bias.l1_obj.receipt_read_entry("image_processing")["flatdiv"]
            == "0"
        )

    def test_flat_true_raises_not_implemented(self, mod_with_bias):
        with pytest.raises(NotImplementedError, match="flat"):
            mod_with_bias.perform(flat=True)

    def test_bias_filepath_overrides_headers(self, tmp_path):
        bias_path = str(tmp_path / "explicit_bias.fits")
        _write_master_bias(bias_path)
        mod = _make_module(bias_file="wrong.fits", bias_dir="/wrong/dir")
        result = mod.perform(bias=bias_path, dark=False)
        np.testing.assert_allclose(result.data["GREEN_CCD"], _CCD_VALUE - _BIAS_VALUE)
        np.testing.assert_allclose(result.data["RED_CCD"], _CCD_VALUE - _BIAS_VALUE)
        assert result.receipt_read_entry("image_processing")["biassub"] == "1"
        assert mod._bias_path == bias_path

    def test_bias_master_l1_object_used_directly(self, tmp_path):
        # A preloaded KPFMasterL1 bypasses disk I/O entirely.
        bias_path = str(tmp_path / "preloaded_bias.fits")
        _write_master_bias(bias_path)
        preloaded = KPFMasterL1.from_fits(bias_path)
        mod = _make_module()  # no biasfile needed
        result = mod.perform(bias=preloaded, dark=False)
        np.testing.assert_allclose(result.data["GREEN_CCD"], _CCD_VALUE - _BIAS_VALUE)
        assert result.receipt_read_entry("image_processing")["biassub"] == "1"
        # _bias_path comes from the in-memory object's own filename.
        assert mod._bias_path == bias_path

    def test_bias_invalid_type_raises_type_error(self):
        mod = _make_module()
        with pytest.raises(TypeError, match="bias must be"):
            mod.perform(bias=42)

    def test_flat_master_l1_raises_not_implemented(self, mod_with_bias, tmp_path):
        # A KPFMasterL1 instance must trip the flat stub too, not just flat=True.
        flat_path = str(tmp_path / "master_flat.fits")
        _write_master_bias(flat_path)
        flat_master = KPFMasterL1.from_fits(flat_path)
        with pytest.raises(NotImplementedError, match="flat"):
            mod_with_bias.perform(flat=flat_master)

    def test_bias_file_not_on_disk_raises(self, tmp_path):
        mod = _make_module(bias_file="missing.fits", bias_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="missing.fits"):
            mod.perform(dark=False)

    def test_bias_explicit_path_missing_raises(self, tmp_path):
        mod = _make_module()
        with pytest.raises(FileNotFoundError):
            mod.perform(bias=str(tmp_path / "nonexistent.fits"), dark=False)


# ---------------------------------------------------------------------------
# TestPerformDark
# ---------------------------------------------------------------------------


class TestPerformDark:
    @pytest.fixture
    def mod_with_bias_dark(self, tmp_path):
        _write_master_bias(str(tmp_path / "master_bias.fits"))
        _write_master_dark(str(tmp_path / "master_dark.fits"))
        return _make_module(
            bias_file="master_bias.fits",
            bias_dir=str(tmp_path),
            dark_file="master_dark.fits",
            dark_dir=str(tmp_path),
        )

    def test_dark_only_subtracted(self, tmp_path):
        _write_master_dark(str(tmp_path / "master_dark.fits"))
        mod = _make_module(dark_file="master_dark.fits", dark_dir=str(tmp_path))
        mod.perform(bias=False, dark=True)
        np.testing.assert_allclose(
            mod.l1_obj.data["GREEN_CCD"], _CCD_VALUE - _DARK_VALUE * _EXPTIME
        )

    def test_defaults_apply_bias_and_dark(self, mod_with_bias_dark):
        # No explicit toggles: _DEFAULTS enables both bias and dark.
        mod_with_bias_dark.perform()
        expected = _CCD_VALUE - _BIAS_VALUE - _DARK_VALUE * _EXPTIME
        np.testing.assert_allclose(
            mod_with_bias_dark.l1_obj.data["GREEN_CCD"], expected
        )
        assert (
            mod_with_bias_dark.l1_obj.receipt_read_entry("image_processing")["biassub"]
            == "1"
        )
        assert (
            mod_with_bias_dark.l1_obj.receipt_read_entry("image_processing")["darksub"]
            == "1"
        )

    def test_bias_then_dark_combined(self, mod_with_bias_dark):
        mod_with_bias_dark.perform(bias=True, dark=True)
        # Bias comes off first, then the exposure-scaled dark.
        expected = _CCD_VALUE - _BIAS_VALUE - _DARK_VALUE * _EXPTIME
        np.testing.assert_allclose(
            mod_with_bias_dark.l1_obj.data["GREEN_CCD"], expected
        )
        np.testing.assert_allclose(mod_with_bias_dark.l1_obj.data["RED_CCD"], expected)

    def test_darksub_header_true(self, mod_with_bias_dark):
        mod_with_bias_dark.perform(bias=True, dark=True)
        assert (
            mod_with_bias_dark.l1_obj.receipt_read_entry("image_processing")["darksub"]
            == "1"
        )

    def test_dark_path_recorded(self, mod_with_bias_dark, tmp_path):
        mod_with_bias_dark.perform(bias=True, dark=True)
        assert mod_with_bias_dark._dark_path == str(tmp_path / "master_dark.fits")

    def test_dark_filepath_overrides_headers(self, tmp_path):
        dark_path = str(tmp_path / "explicit_dark.fits")
        _write_master_dark(dark_path)
        mod = _make_module(dark_file="wrong.fits", dark_dir="/wrong/dir")
        mod.perform(bias=False, dark=dark_path)
        np.testing.assert_allclose(
            mod.l1_obj.data["GREEN_CCD"], _CCD_VALUE - _DARK_VALUE * _EXPTIME
        )
        assert mod._dark_path == dark_path

    def test_dark_master_l1_object_used_directly(self, tmp_path):
        dark_path = str(tmp_path / "preloaded_dark.fits")
        _write_master_dark(dark_path)
        preloaded = KPFMasterL1.from_fits(dark_path)
        mod = _make_module()  # no darkfile needed
        mod.perform(bias=False, dark=preloaded)
        np.testing.assert_allclose(
            mod.l1_obj.data["GREEN_CCD"], _CCD_VALUE - _DARK_VALUE * _EXPTIME
        )
        assert mod._dark_path == dark_path

    def test_dark_invalid_type_raises_type_error(self):
        mod = _make_module()
        with pytest.raises(TypeError, match="dark must be"):
            mod.perform(bias=False, dark=42)

    def test_perform_propagates_bias_and_dark_variance(self, mod_with_bias_dark):
        mod_with_bias_dark.perform()
        expected = (
            _VAR_VALUE
            + (_BIAS_VALUE / _BIAS_SNR) ** 2
            + (_EXPTIME * _DARK_VALUE / _DARK_SNR) ** 2
        )
        np.testing.assert_allclose(
            mod_with_bias_dark.l1_obj.data["GREEN_VAR"], expected, rtol=1e-5
        )

    def test_dark_missing_header_raises(self):
        mod = _make_module()
        with pytest.raises(FileNotFoundError, match="darkfile"):
            mod.perform(bias=False)

    def test_dark_file_not_on_disk_raises(self, tmp_path):
        mod = _make_module(dark_file="missing.fits", dark_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="missing.fits"):
            mod.perform(bias=False)


# ---------------------------------------------------------------------------
# TestCalibrationGuard
# ---------------------------------------------------------------------------


class TestCalibrationGuard:
    """perform() must refuse to subtract a calibration already applied."""

    def _bias_module(self, tmp_path):
        _write_master_bias(str(tmp_path / "master_bias.fits"))
        mod = _make_module(bias_file="master_bias.fits", bias_dir=str(tmp_path))
        mod.dark = False
        return mod

    def _dark_module(self, tmp_path):
        _write_master_dark(str(tmp_path / "master_dark.fits"))
        mod = _make_module(dark_file="master_dark.fits", dark_dir=str(tmp_path))
        mod.bias = False
        return mod

    def test_calibration_applied_false_before_perform(self, tmp_path):
        mod = self._bias_module(tmp_path)
        assert ImageProcessing.calibration_applied(mod.l1_obj, "bias") is False

    def test_calibration_applied_true_after_perform(self, tmp_path):
        mod = self._bias_module(tmp_path)
        mod.perform()
        assert ImageProcessing.calibration_applied(mod.l1_obj, "bias") is True

    def test_second_bias_perform_raises(self, tmp_path):
        mod = self._bias_module(tmp_path)
        mod.perform()
        with pytest.raises(RuntimeError, match="bias already subtracted"):
            mod.perform()

    def test_second_dark_perform_raises(self, tmp_path):
        mod = self._dark_module(tmp_path)
        mod.perform()
        with pytest.raises(RuntimeError, match="dark already subtracted"):
            mod.perform()

    def test_guard_does_not_mutate_on_raise(self, tmp_path):
        # A blocked re-run must leave the (already-calibrated) data untouched.
        mod = self._bias_module(tmp_path)
        mod.perform()
        once = mod.l1_obj.data["GREEN_CCD"].copy()
        with pytest.raises(RuntimeError):
            mod.perform()
        np.testing.assert_array_equal(mod.l1_obj.data["GREEN_CCD"], once)

    def test_sequential_bias_then_dark_preserves_both_flags(self, tmp_path):
        # Applying dark after bias must not clear the recorded bias flag.
        _write_master_bias(str(tmp_path / "master_bias.fits"))
        _write_master_dark(str(tmp_path / "master_dark.fits"))
        mod = _make_module(
            bias_file="master_bias.fits",
            bias_dir=str(tmp_path),
            dark_file="master_dark.fits",
            dark_dir=str(tmp_path),
        )
        mod.perform(bias=True, dark=False)
        mod.perform(bias=False, dark=True)
        assert mod.l1_obj.receipt_read_entry("image_processing")["biassub"] == "1"
        assert mod.l1_obj.receipt_read_entry("image_processing")["darksub"] == "1"
        expected = _CCD_VALUE - _BIAS_VALUE - _DARK_VALUE * _EXPTIME
        np.testing.assert_allclose(mod.l1_obj.data["GREEN_CCD"], expected)


# ---------------------------------------------------------------------------
# TestMasterCache
# ---------------------------------------------------------------------------


class TestMasterCache:
    def test_resolve_caches_within_instance(self, tmp_path):
        # Resolving the same calibration twice (once per chip) must reuse the
        # cached master rather than re-read from disk.
        path = str(tmp_path / "master_bias.fits")
        _write_master_bias(path)
        mod = _make_module()
        first = mod._resolve_master("bias", path)
        second = mod._resolve_master("bias", path)
        assert first is second
        assert mod._bias_ml1 is first

    def test_separate_instances_do_not_share(self, tmp_path):
        path = str(tmp_path / "master_bias.fits")
        _write_master_bias(path)
        a = _make_module()
        b = _make_module()
        assert a._resolve_master("bias", path) is not b._resolve_master("bias", path)

    def test_load_master_reads_master_from_disk(self, tmp_path):
        path = str(tmp_path / "master_bias.fits")
        _write_master_bias(path)
        master = ImageProcessing._load_master(path)
        assert isinstance(master, KPFMasterL1)
        np.testing.assert_allclose(master.data["GREEN_IMG"], _BIAS_VALUE)
        np.testing.assert_allclose(master.data["GREEN_SNR"], _BIAS_SNR)

    def test_load_master_is_stateless(self, tmp_path):
        # Deliberate contract: caching lives in _resolve_master, so each call here
        # re-reads rather than returning a shared object.
        path = str(tmp_path / "master_bias.fits")
        _write_master_bias(path)
        assert ImageProcessing._load_master(path) is not ImageProcessing._load_master(
            path
        )

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Master file not found"):
            ImageProcessing._load_master(str(tmp_path / "nope.fits"))
