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
_CCD_VALUE  = 10.0
_BIAS_VALUE =  3.0


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

class MockL1:
    def __init__(self):
        self.obs_id = 'KP.20240405.40113.57'
        self.headers = {'PRIMARY': {}}
        self.data = {
            'GREEN_CCD': np.full(_SHAPE, _CCD_VALUE, dtype=np.float32),
            'RED_CCD':   np.full(_SHAPE, _CCD_VALUE, dtype=np.float32),
        }
        self._receipt = []

    def receipt_add_entry(self, name, status):
        self._receipt.append((name, status))


class MockMasterBias:
    data = {
        'GREEN_IMG': np.full(_SHAPE, _BIAS_VALUE, dtype=np.float32),
        'RED_IMG':   np.full(_SHAPE, _BIAS_VALUE, dtype=np.float32),
    }


def _make_module(bias_file=None, bias_dir=None):
    l1 = MockL1()
    if bias_file is not None:
        l1.headers['PRIMARY']['BIASFILE'] = bias_file
    if bias_dir is not None:
        l1.headers['PRIMARY']['BIASDIR'] = bias_dir
    return ImageProcessing(l1)


def _write_master_bias(path):
    """Write a minimal master bias FITS file to path."""
    primary = fits.PrimaryHDU()
    green   = fits.ImageHDU(data=np.full(_SHAPE, _BIAS_VALUE, dtype=np.float32), name='GREEN_IMG')
    red     = fits.ImageHDU(data=np.full(_SHAPE, _BIAS_VALUE, dtype=np.float32), name='RED_IMG')
    fits.HDUList([primary, green, red]).writeto(path, overwrite=True)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:

    def test_none_config(self):
        ip = ImageProcessing(MockL1())
        assert ip.chips == ['GREEN', 'RED']

    def test_dict_config_overrides_chips(self):
        l1 = MockL1()
        ip = ImageProcessing(l1, config={'chips': ['GREEN']})
        assert ip.chips == ['GREEN']

    def test_invalid_config_raises(self):
        with pytest.raises(TypeError):
            ImageProcessing(MockL1(), config=42)

    def test_results_none_before_perform(self):
        assert ImageProcessing(MockL1())._results is None

    def test_bias_path_none_before_load(self):
        assert ImageProcessing(MockL1())._bias_path is None


# ---------------------------------------------------------------------------
# TestLoadBias
# ---------------------------------------------------------------------------

class TestLoadBias:

    def test_raises_when_biasfile_missing(self):
        mod = _make_module(bias_dir='/some/dir')
        with pytest.raises(FileNotFoundError, match="BIASFILE"):
            mod.load_bias()

    def test_raises_when_biasdir_missing(self):
        mod = _make_module(bias_file='master_bias.fits')
        with pytest.raises(FileNotFoundError, match="BIASDIR"):
            mod.load_bias()

    def test_raises_when_file_not_on_disk(self, tmp_path):
        mod = _make_module(bias_file='missing.fits', bias_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="missing.fits"):
            mod.load_bias()

    def test_sets_bias_path_attribute(self, tmp_path):
        bias_path = str(tmp_path / 'master_bias.fits')
        _write_master_bias(bias_path)
        mod = _make_module(bias_file='master_bias.fits', bias_dir=str(tmp_path))
        mod.load_bias()
        assert mod._bias_path == bias_path

    def test_returns_kpfmaster_l1(self, tmp_path):
        bias_path = str(tmp_path / 'master_bias.fits')
        _write_master_bias(bias_path)
        mod = _make_module(bias_file='master_bias.fits', bias_dir=str(tmp_path))
        result = mod.load_bias()
        assert isinstance(result, KPFMasterL1)

    def test_explicit_path_overrides_headers(self, tmp_path):
        bias_path = str(tmp_path / 'master_bias.fits')
        _write_master_bias(bias_path)
        # Headers point nowhere valid — explicit path should win.
        mod = _make_module(bias_file='wrong.fits', bias_dir='/wrong/dir')
        result = mod.load_bias(bias_path=bias_path)
        assert isinstance(result, KPFMasterL1)
        assert mod._bias_path == bias_path

    def test_explicit_path_raises_when_missing(self, tmp_path):
        mod = _make_module()
        with pytest.raises(FileNotFoundError):
            mod.load_bias(bias_path=str(tmp_path / 'nonexistent.fits'))


# ---------------------------------------------------------------------------
# TestSubtractBias
# ---------------------------------------------------------------------------

class TestSubtractBias:

    def test_subtracts_correct_values_green(self):
        mod = _make_module()
        mod.subtract_bias(MockMasterBias(), 'GREEN')
        expected = _CCD_VALUE - _BIAS_VALUE
        np.testing.assert_allclose(mod.l1_obj.data['GREEN_CCD'], expected)

    def test_subtracts_correct_values_red(self):
        mod = _make_module()
        mod.subtract_bias(MockMasterBias(), 'RED')
        expected = _CCD_VALUE - _BIAS_VALUE
        np.testing.assert_allclose(mod.l1_obj.data['RED_CCD'], expected)

    def test_modifies_in_place(self):
        mod = _make_module()
        original = mod.l1_obj.data['GREEN_CCD']
        mod.subtract_bias(MockMasterBias(), 'GREEN')
        assert mod.l1_obj.data['GREEN_CCD'] is original

    def test_chip_name_case_insensitive(self):
        mod = _make_module()
        mod.subtract_bias(MockMasterBias(), 'green')
        np.testing.assert_allclose(mod.l1_obj.data['GREEN_CCD'], _CCD_VALUE - _BIAS_VALUE)


# ---------------------------------------------------------------------------
# TestPerform
# ---------------------------------------------------------------------------

class TestPerform:

    @pytest.fixture
    def mod_with_bias(self, tmp_path, monkeypatch):
        bias_path = str(tmp_path / 'master_bias.fits')
        _write_master_bias(bias_path)
        mod = _make_module(bias_file='master_bias.fits', bias_dir=str(tmp_path))
        return mod

    def test_returns_l1_obj(self, mod_with_bias):
        result = mod_with_bias.perform()
        assert result is mod_with_bias.l1_obj

    def test_bias_subtracted_green(self, mod_with_bias):
        mod_with_bias.perform()
        np.testing.assert_allclose(
            mod_with_bias.l1_obj.data['GREEN_CCD'], _CCD_VALUE - _BIAS_VALUE
        )

    def test_bias_subtracted_red(self, mod_with_bias):
        mod_with_bias.perform()
        np.testing.assert_allclose(
            mod_with_bias.l1_obj.data['RED_CCD'], _CCD_VALUE - _BIAS_VALUE
        )

    def test_results_keyed_by_bias(self, mod_with_bias, tmp_path):
        mod_with_bias.perform()
        assert 'bias' in mod_with_bias._results
        assert mod_with_bias._results['bias'] == str(tmp_path / 'master_bias.fits')

    def test_biasub_header_set(self, mod_with_bias):
        mod_with_bias.perform()
        assert mod_with_bias.l1_obj.headers['PRIMARY']['BIASUB'][0] is True

    def test_receipt_entry_added(self, mod_with_bias):
        mod_with_bias.perform()
        assert ('image_processing', 'PASS') in mod_with_bias.l1_obj._receipt

    def test_chips_override_processes_only_requested(self, mod_with_bias):
        mod_with_bias.perform(chips=['GREEN'])
        np.testing.assert_allclose(
            mod_with_bias.l1_obj.data['GREEN_CCD'], _CCD_VALUE - _BIAS_VALUE
        )
        # RED_CCD should be untouched
        np.testing.assert_allclose(
            mod_with_bias.l1_obj.data['RED_CCD'], _CCD_VALUE
        )

    def test_raises_when_headers_missing(self):
        mod = _make_module()  # no BIASFILE / BIASDIR
        with pytest.raises(FileNotFoundError):
            mod.perform()
