"""
Tests for the ImageProcessing module (L1 bias subtraction).

Unit tests use synthetic arrays and MockL1 objects; no real data or FITS
files are required except where a master bias file must exist on disk.
"""

import numpy as np
import pytest
from astropy.io import fits

from kpfpipe.data_models.masters.level1 import KPFMasterL1
from kpfpipe.modules.image_processing import ImageProcessing

_SHAPE = (4, 4)
_CCD_VALUE = 10.0
_BIAS_VALUE = 3.0
_DARK_VALUE = 2.0  # master dark rate [electrons/sec]
_EXPTIME = 4.0  # != 1 so dark scaling is observable


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class MockL1:
    def __init__(self):
        self.obs_id = "KP.20240405.40113.57"
        self.headers = {"PRIMARY": {"EXPTIME": _EXPTIME}}
        self.data = {
            "GREEN_CCD": np.full(_SHAPE, _CCD_VALUE, dtype=np.float32),
            "RED_CCD": np.full(_SHAPE, _CCD_VALUE, dtype=np.float32),
        }
        self._receipt = []

    def receipt_add_entry(self, name, status):
        self._receipt.append((name, status))


class MockMasterBias:
    data = {
        "GREEN_IMG": np.full(_SHAPE, _BIAS_VALUE, dtype=np.float32),
        "RED_IMG": np.full(_SHAPE, _BIAS_VALUE, dtype=np.float32),
    }


class MockMasterDark:
    data = {
        "GREEN_IMG": np.full(_SHAPE, _DARK_VALUE, dtype=np.float32),
        "RED_IMG": np.full(_SHAPE, _DARK_VALUE, dtype=np.float32),
    }


def _make_module(bias_file=None, bias_dir=None, dark_file=None, dark_dir=None):
    l1 = MockL1()
    if bias_file is not None:
        l1.headers["PRIMARY"]["BIASFILE"] = bias_file
    if bias_dir is not None:
        l1.headers["PRIMARY"]["BIASDIR"] = bias_dir
    if dark_file is not None:
        l1.headers["PRIMARY"]["DARKFILE"] = dark_file
    if dark_dir is not None:
        l1.headers["PRIMARY"]["DARKDIR"] = dark_dir
    return ImageProcessing(l1)


def _write_master(path, value):
    """Write a minimal master FITS file (GREEN_IMG/RED_IMG = value) to path."""
    primary = fits.PrimaryHDU()
    green = fits.ImageHDU(
        data=np.full(_SHAPE, value, dtype=np.float32), name="GREEN_IMG"
    )
    red = fits.ImageHDU(data=np.full(_SHAPE, value, dtype=np.float32), name="RED_IMG")
    fits.HDUList([primary, green, red]).writeto(path, overwrite=True)


def _write_master_bias(path):
    """Write a minimal master bias FITS file to path."""
    _write_master(path, _BIAS_VALUE)


def _write_master_dark(path):
    """Write a minimal master dark FITS file to path."""
    _write_master(path, _DARK_VALUE)


@pytest.fixture(autouse=True)
def _clear_master_cache():
    """Isolate the shared ImageProcessing master cache between tests."""
    ImageProcessing.clear_master_cache()
    yield
    ImageProcessing.clear_master_cache()


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
        with pytest.raises(TypeError):
            ImageProcessing(MockL1(), config=42)

    def test_results_none_before_perform(self):
        assert ImageProcessing(MockL1())._results is None

    def test_bias_path_none_before_load(self):
        assert ImageProcessing(MockL1())._bias_path is None

    def test_dark_path_none_before_load(self):
        assert ImageProcessing(MockL1())._dark_path is None


# ---------------------------------------------------------------------------
# TestLoadBias
# ---------------------------------------------------------------------------


class TestLoadBias:
    def test_raises_when_biasfile_missing(self):
        mod = _make_module(bias_dir="/some/dir")
        with pytest.raises(FileNotFoundError, match="BIASFILE"):
            mod.load_bias()

    def test_raises_when_biasdir_missing(self):
        mod = _make_module(bias_file="master_bias.fits")
        with pytest.raises(FileNotFoundError, match="BIASDIR"):
            mod.load_bias()

    def test_raises_when_file_not_on_disk(self, tmp_path):
        mod = _make_module(bias_file="missing.fits", bias_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="missing.fits"):
            mod.load_bias()

    def test_sets_bias_path_attribute(self, tmp_path):
        bias_path = str(tmp_path / "master_bias.fits")
        _write_master_bias(bias_path)
        mod = _make_module(bias_file="master_bias.fits", bias_dir=str(tmp_path))
        mod.load_bias()
        assert mod._bias_path == bias_path

    def test_returns_kpfmaster_l1(self, tmp_path):
        bias_path = str(tmp_path / "master_bias.fits")
        _write_master_bias(bias_path)
        mod = _make_module(bias_file="master_bias.fits", bias_dir=str(tmp_path))
        result = mod.load_bias()
        assert isinstance(result, KPFMasterL1)

    def test_explicit_path_overrides_headers(self, tmp_path):
        bias_path = str(tmp_path / "master_bias.fits")
        _write_master_bias(bias_path)
        # Headers point nowhere valid — explicit path should win.
        mod = _make_module(bias_file="wrong.fits", bias_dir="/wrong/dir")
        result = mod.load_bias(bias_path=bias_path)
        assert isinstance(result, KPFMasterL1)
        assert mod._bias_path == bias_path

    def test_explicit_path_raises_when_missing(self, tmp_path):
        mod = _make_module()
        with pytest.raises(FileNotFoundError):
            mod.load_bias(bias_path=str(tmp_path / "nonexistent.fits"))


# ---------------------------------------------------------------------------
# TestSubtractBias
# ---------------------------------------------------------------------------


class TestSubtractBias:
    def test_subtracts_correct_values_green(self):
        mod = _make_module()
        mod.subtract_bias(MockMasterBias(), "GREEN")
        expected = _CCD_VALUE - _BIAS_VALUE
        np.testing.assert_allclose(mod.l1_obj.data["GREEN_CCD"], expected)

    def test_subtracts_correct_values_red(self):
        mod = _make_module()
        mod.subtract_bias(MockMasterBias(), "RED")
        expected = _CCD_VALUE - _BIAS_VALUE
        np.testing.assert_allclose(mod.l1_obj.data["RED_CCD"], expected)

    def test_modifies_in_place(self):
        mod = _make_module()
        original = mod.l1_obj.data["GREEN_CCD"]
        mod.subtract_bias(MockMasterBias(), "GREEN")
        assert mod.l1_obj.data["GREEN_CCD"] is original

    def test_chip_name_case_insensitive(self):
        mod = _make_module()
        mod.subtract_bias(MockMasterBias(), "green")
        np.testing.assert_allclose(
            mod.l1_obj.data["GREEN_CCD"], _CCD_VALUE - _BIAS_VALUE
        )


# ---------------------------------------------------------------------------
# TestLoadDark
# ---------------------------------------------------------------------------


class TestLoadDark:
    def test_raises_when_darkfile_missing(self):
        mod = _make_module(dark_dir="/some/dir")
        with pytest.raises(FileNotFoundError, match="DARKFILE"):
            mod.load_dark()

    def test_raises_when_darkdir_missing(self):
        mod = _make_module(dark_file="master_dark.fits")
        with pytest.raises(FileNotFoundError, match="DARKDIR"):
            mod.load_dark()

    def test_raises_when_file_not_on_disk(self, tmp_path):
        mod = _make_module(dark_file="missing.fits", dark_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="missing.fits"):
            mod.load_dark()

    def test_sets_dark_path_attribute(self, tmp_path):
        dark_path = str(tmp_path / "master_dark.fits")
        _write_master_dark(dark_path)
        mod = _make_module(dark_file="master_dark.fits", dark_dir=str(tmp_path))
        mod.load_dark()
        assert mod._dark_path == dark_path

    def test_returns_kpfmaster_l1(self, tmp_path):
        dark_path = str(tmp_path / "master_dark.fits")
        _write_master_dark(dark_path)
        mod = _make_module(dark_file="master_dark.fits", dark_dir=str(tmp_path))
        assert isinstance(mod.load_dark(), KPFMasterL1)

    def test_explicit_path_overrides_headers(self, tmp_path):
        dark_path = str(tmp_path / "master_dark.fits")
        _write_master_dark(dark_path)
        mod = _make_module(dark_file="wrong.fits", dark_dir="/wrong/dir")
        result = mod.load_dark(dark_path=dark_path)
        assert isinstance(result, KPFMasterL1)
        assert mod._dark_path == dark_path


# ---------------------------------------------------------------------------
# TestSubtractDark
# ---------------------------------------------------------------------------


class TestSubtractDark:
    def test_subtracts_scaled_by_exptime_green(self):
        mod = _make_module()
        mod.subtract_dark(MockMasterDark(), "GREEN")
        expected = _CCD_VALUE - _DARK_VALUE * _EXPTIME
        np.testing.assert_allclose(mod.l1_obj.data["GREEN_CCD"], expected)

    def test_subtracts_scaled_by_exptime_red(self):
        mod = _make_module()
        mod.subtract_dark(MockMasterDark(), "RED")
        expected = _CCD_VALUE - _DARK_VALUE * _EXPTIME
        np.testing.assert_allclose(mod.l1_obj.data["RED_CCD"], expected)

    def test_scaling_is_applied_not_raw_subtraction(self):
        # Guard against a literal copy of subtract_bias (no exptime factor):
        # the result must reflect dark_IMG * EXPTIME, not dark_IMG alone.
        mod = _make_module()
        mod.subtract_dark(MockMasterDark(), "GREEN")
        unscaled = _CCD_VALUE - _DARK_VALUE
        assert not np.allclose(mod.l1_obj.data["GREEN_CCD"], unscaled)

    def test_modifies_in_place(self):
        mod = _make_module()
        original = mod.l1_obj.data["GREEN_CCD"]
        mod.subtract_dark(MockMasterDark(), "GREEN")
        assert mod.l1_obj.data["GREEN_CCD"] is original

    def test_chip_name_case_insensitive(self):
        mod = _make_module()
        mod.subtract_dark(MockMasterDark(), "green")
        np.testing.assert_allclose(
            mod.l1_obj.data["GREEN_CCD"], _CCD_VALUE - _DARK_VALUE * _EXPTIME
        )


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

    def test_results_keyed_by_bias(self, mod_with_bias, tmp_path):
        mod_with_bias.perform()
        assert "bias" in mod_with_bias._results
        assert mod_with_bias._results["bias"] == str(tmp_path / "master_bias.fits")

    def test_biasub_header_set(self, mod_with_bias):
        mod_with_bias.perform()
        assert mod_with_bias.l1_obj.headers["PRIMARY"]["BIASUB"][0] is True

    def test_receipt_entry_added(self, mod_with_bias):
        mod_with_bias.perform()
        assert ("image_processing", "PASS") in mod_with_bias.l1_obj._receipt

    def test_chips_override_processes_only_requested(self, mod_with_bias):
        mod_with_bias.perform(chips=["GREEN"])
        np.testing.assert_allclose(
            mod_with_bias.l1_obj.data["GREEN_CCD"], _CCD_VALUE - _BIAS_VALUE
        )
        # RED_CCD should be untouched
        np.testing.assert_allclose(mod_with_bias.l1_obj.data["RED_CCD"], _CCD_VALUE)

    def test_raises_when_headers_missing(self):
        mod = _make_module()  # no BIASFILE / BIASDIR
        with pytest.raises(FileNotFoundError):
            mod.perform()

    def test_bias_false_skips_subtraction(self):
        mod = _make_module()  # no BIASFILE / BIASDIR needed when bias is off
        result = mod.perform(bias=False, dark=False)
        # CCDs untouched
        np.testing.assert_allclose(result.data["GREEN_CCD"], _CCD_VALUE)
        np.testing.assert_allclose(result.data["RED_CCD"], _CCD_VALUE)
        # BIASUB header reflects the choice
        assert result.headers["PRIMARY"]["BIASUB"][0] is False
        # No bias path recorded
        assert "bias" not in mod._results

    def test_darksub_header_false_when_dark_off(self, mod_with_bias):
        mod_with_bias.perform()
        assert mod_with_bias.l1_obj.headers["PRIMARY"]["DARKSUB"][0] is False

    def test_flat_true_raises_not_implemented(self, mod_with_bias):
        with pytest.raises(NotImplementedError, match="flat"):
            mod_with_bias.perform(flat=True)

    def test_bias_filepath_overrides_headers(self, tmp_path):
        # explicit filepath should be used even when headers point elsewhere
        bias_path = str(tmp_path / "explicit_bias.fits")
        _write_master_bias(bias_path)
        mod = _make_module(bias_file="wrong.fits", bias_dir="/wrong/dir")
        result = mod.perform(bias=bias_path, dark=False)
        np.testing.assert_allclose(result.data["GREEN_CCD"], _CCD_VALUE - _BIAS_VALUE)
        np.testing.assert_allclose(result.data["RED_CCD"], _CCD_VALUE - _BIAS_VALUE)
        assert result.headers["PRIMARY"]["BIASUB"][0] is True
        assert mod._results["bias"] == bias_path

    def test_bias_master_l1_object_used_directly(self, tmp_path):
        # supplying a KPFMasterL1 instance should bypass disk I/O entirely
        bias_path = str(tmp_path / "preloaded_bias.fits")
        _write_master_bias(bias_path)
        preloaded = KPFMasterL1.from_fits(bias_path)
        mod = _make_module()  # no BIASFILE / BIASDIR needed
        result = mod.perform(bias=preloaded, dark=False)
        np.testing.assert_allclose(result.data["GREEN_CCD"], _CCD_VALUE - _BIAS_VALUE)
        assert result.headers["PRIMARY"]["BIASUB"][0] is True
        # _bias_path should reflect the in-memory object's filename
        assert mod._results["bias"] == bias_path

    def test_bias_invalid_type_raises_type_error(self):
        mod = _make_module()
        with pytest.raises(TypeError, match="bias must be"):
            mod.perform(bias=42)

    def test_flat_master_l1_raises_not_implemented(self, mod_with_bias, tmp_path):
        # KPFMasterL1 instance should also trip the flat stub.
        flat_path = str(tmp_path / "master_flat.fits")
        _write_master_bias(flat_path)
        flat_master = KPFMasterL1.from_fits(flat_path)
        with pytest.raises(NotImplementedError, match="flat"):
            mod_with_bias.perform(flat=flat_master)


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
        # No explicit toggles: the _DEFAULTS now enable bias and dark.
        mod_with_bias_dark.perform()
        expected = _CCD_VALUE - _BIAS_VALUE - _DARK_VALUE * _EXPTIME
        np.testing.assert_allclose(
            mod_with_bias_dark.l1_obj.data["GREEN_CCD"], expected
        )
        assert mod_with_bias_dark.l1_obj.headers["PRIMARY"]["BIASUB"][0] is True
        assert mod_with_bias_dark.l1_obj.headers["PRIMARY"]["DARKSUB"][0] is True

    def test_bias_then_dark_combined(self, mod_with_bias_dark):
        mod_with_bias_dark.perform(bias=True, dark=True)
        # bias removed first, then exposure-scaled dark.
        expected = _CCD_VALUE - _BIAS_VALUE - _DARK_VALUE * _EXPTIME
        np.testing.assert_allclose(
            mod_with_bias_dark.l1_obj.data["GREEN_CCD"], expected
        )
        np.testing.assert_allclose(mod_with_bias_dark.l1_obj.data["RED_CCD"], expected)

    def test_darksub_header_true(self, mod_with_bias_dark):
        mod_with_bias_dark.perform(bias=True, dark=True)
        assert mod_with_bias_dark.l1_obj.headers["PRIMARY"]["DARKSUB"][0] is True

    def test_results_keyed_by_dark(self, mod_with_bias_dark, tmp_path):
        mod_with_bias_dark.perform(bias=True, dark=True)
        assert mod_with_bias_dark._results["dark"] == str(tmp_path / "master_dark.fits")

    def test_dark_filepath_overrides_headers(self, tmp_path):
        dark_path = str(tmp_path / "explicit_dark.fits")
        _write_master_dark(dark_path)
        mod = _make_module(dark_file="wrong.fits", dark_dir="/wrong/dir")
        mod.perform(bias=False, dark=dark_path)
        np.testing.assert_allclose(
            mod.l1_obj.data["GREEN_CCD"], _CCD_VALUE - _DARK_VALUE * _EXPTIME
        )
        assert mod._results["dark"] == dark_path

    def test_dark_master_l1_object_used_directly(self, tmp_path):
        dark_path = str(tmp_path / "preloaded_dark.fits")
        _write_master_dark(dark_path)
        preloaded = KPFMasterL1.from_fits(dark_path)
        mod = _make_module()  # no DARKFILE / DARKDIR needed
        mod.perform(bias=False, dark=preloaded)
        np.testing.assert_allclose(
            mod.l1_obj.data["GREEN_CCD"], _CCD_VALUE - _DARK_VALUE * _EXPTIME
        )
        assert mod._results["dark"] == dark_path

    def test_dark_invalid_type_raises_type_error(self):
        mod = _make_module()
        with pytest.raises(TypeError, match="dark must be"):
            mod.perform(bias=False, dark=42)


# ---------------------------------------------------------------------------
# TestMasterCache
# ---------------------------------------------------------------------------


class TestMasterCache:
    def test_same_path_returns_cached_object(self, tmp_path):
        path = str(tmp_path / "master_bias.fits")
        _write_master_bias(path)
        first = ImageProcessing._load_master(path)
        second = ImageProcessing._load_master(path)
        assert first is second

    def test_clear_master_cache_forces_reload(self, tmp_path):
        path = str(tmp_path / "master_bias.fits")
        _write_master_bias(path)
        first = ImageProcessing._load_master(path)
        ImageProcessing.clear_master_cache()
        assert ImageProcessing._load_master(path) is not first

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Master file not found"):
            ImageProcessing._load_master(str(tmp_path / "nope.fits"))
