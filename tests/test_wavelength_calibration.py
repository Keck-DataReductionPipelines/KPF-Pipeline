"""Tests for the WavelengthCalibration module."""

import numpy as np
import pytest

from kpfpipe import DETECTOR
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.masters.level2 import KPFMasterL2
from kpfpipe.modules.wavelength_calibration import WavelengthCalibration


NORDER_GREEN = DETECTOR['norder']['GREEN']
NORDER_RED   = DETECTOR['norder']['RED']

_FIBERS = ['SKY', 'SCI1', 'SCI2', 'SCI3', 'CAL']
_CHIPS  = ['GREEN', 'RED']

# Small detector width for fast tests; the WLS copy is agnostic to NCOL.
NCOL = 32


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_master_l2(seed=42):
    """Build a KPFMasterL2 with deterministic, distinct per-fiber WAVE arrays.

    Each (chip, fiber) gets its own random-but-reproducible block so we can
    later confirm that exactly the right block lands on the science L2.
    """
    master = KPFMasterL2()
    master.headers['PRIMARY']['INSTRUME'] = 'KPF'
    master.headers['PRIMARY']['DATE-OBS'] = '2024-04-05T01:00:37'

    rng = np.random.default_rng(seed)
    for chip in _CHIPS:
        norder = NORDER_GREEN if chip == 'GREEN' else NORDER_RED
        for fiber in _FIBERS:
            arr = rng.uniform(4000.0, 8000.0, size=(norder, NCOL)).astype(np.float32)
            master.data[f'{chip}_{fiber}_WAVE'] = arr
    return master


def _make_science_l2(wls_path=None):
    """Build a minimal KPF2 for wavecal tests."""
    l2 = KPF2()
    l2.headers['PRIMARY']['INSTRUME'] = 'KPF'
    l2.headers['PRIMARY']['DATE-OBS'] = '2024-04-05T11:08:33'
    if wls_path is not None:
        l2.headers['PRIMARY']['WLSFILE'] = wls_path
    return l2


@pytest.fixture
def master_wls_path(tmp_path):
    """Write a synthetic KPFMasterL2 to disk and return its path."""
    master = _make_master_l2()
    path = str(tmp_path / 'kpf_ML2_20240405T010037.fits')
    master.to_fits(path)
    return path


# ---------------------------------------------------------------------------
# Constructor / config plumbing
# ---------------------------------------------------------------------------

class TestConstructor:

    def test_default_chips_and_fibers(self):
        mod = WavelengthCalibration(_make_science_l2())
        assert mod.chips  == _CHIPS
        assert mod.fibers == _FIBERS

    def test_dict_config_override(self):
        mod = WavelengthCalibration(
            _make_science_l2(),
            config={'chips': ['GREEN'], 'fibers': ['SCI2']},
        )
        assert mod.chips  == ['GREEN']
        assert mod.fibers == ['SCI2']

    def test_invalid_config_type(self):
        with pytest.raises(TypeError):
            WavelengthCalibration(_make_science_l2(), config='not a dict')

    def test_results_is_none_before_perform(self):
        mod = WavelengthCalibration(_make_science_l2())
        assert mod._results is None

    def test_wls_path_is_none_before_load(self):
        mod = WavelengthCalibration(_make_science_l2())
        assert mod._wls_path is None


# ---------------------------------------------------------------------------
# load_wls()
# ---------------------------------------------------------------------------

class TestLoadWLS:

    def test_raises_when_wlsfile_missing(self):
        mod = WavelengthCalibration(_make_science_l2())
        with pytest.raises(KeyError, match='WLSFILE'):
            mod.load_wls()

    def test_raises_when_file_does_not_exist(self, tmp_path):
        bogus = str(tmp_path / 'does_not_exist.fits')
        mod = WavelengthCalibration(_make_science_l2(wls_path=bogus))
        with pytest.raises(FileNotFoundError, match='Master WLS file not found'):
            mod.load_wls()

    def test_reads_wlsfile_from_primary(self, master_wls_path):
        mod = WavelengthCalibration(_make_science_l2(wls_path=master_wls_path))
        loaded = mod.load_wls()
        assert isinstance(loaded, KPFMasterL2)
        assert mod._wls_path == master_wls_path

    def test_explicit_path_overrides_header(self, master_wls_path):
        # Set WLSFILE to a bogus value to make sure the override wins.
        mod = WavelengthCalibration(_make_science_l2(wls_path='/tmp/bogus.fits'))
        loaded = mod.load_wls(wls_path=master_wls_path)
        assert isinstance(loaded, KPFMasterL2)
        assert mod._wls_path == master_wls_path


# ---------------------------------------------------------------------------
# perform()
# ---------------------------------------------------------------------------

class TestPerform:

    def test_returns_l2_obj(self, master_wls_path):
        l2 = _make_science_l2(wls_path=master_wls_path)
        mod = WavelengthCalibration(l2)
        assert mod.perform() is l2

    def test_adds_receipt_entry(self, master_wls_path):
        l2 = _make_science_l2(wls_path=master_wls_path)
        WavelengthCalibration(l2).perform()
        assert (l2.receipt['Module_Name'] == 'wavelength_calibration').any()

    def test_results_populated_after_perform(self, master_wls_path):
        l2 = _make_science_l2(wls_path=master_wls_path)
        mod = WavelengthCalibration(l2)
        mod.perform()
        assert mod._results == {
            'wls_path': master_wls_path,
            'chips':  _CHIPS,
            'fibers': _FIBERS,
        }

    def test_copies_all_wave_arrays(self, master_wls_path):
        # Every (chip, fiber) WAVE array on the science L2 should match the master.
        l2 = _make_science_l2(wls_path=master_wls_path)
        WavelengthCalibration(l2).perform()

        master = KPFMasterL2.from_fits(master_wls_path)
        for chip in _CHIPS:
            for fiber in _FIBERS:
                key = f'{chip}_{fiber}_WAVE'
                np.testing.assert_array_equal(l2.data[key], master.data[key])

    def test_explicit_path_bypasses_header(self, master_wls_path):
        # WLSFILE header is bogus, but wls_path override is valid → perform() succeeds.
        l2 = _make_science_l2(wls_path='/tmp/bogus.fits')
        WavelengthCalibration(l2).perform(wls_path=master_wls_path)

        master = KPFMasterL2.from_fits(master_wls_path)
        np.testing.assert_array_equal(
            l2.data['GREEN_SCI2_WAVE'], master.data['GREEN_SCI2_WAVE']
        )

    def test_subset_chips_and_fibers(self, master_wls_path):
        # Only the requested (chip, fiber) blocks are copied; the rest stay zero.
        l2 = _make_science_l2(wls_path=master_wls_path)
        WavelengthCalibration(l2).perform(chips=['GREEN'], fibers=['SCI2'])

        master = KPFMasterL2.from_fits(master_wls_path)
        np.testing.assert_array_equal(
            l2.data['GREEN_SCI2_WAVE'], master.data['GREEN_SCI2_WAVE']
        )
        # An un-requested block remains zero (default fill from KPF2's
        # _KPF2DataDict alloc on first chip-prefix write).
        assert not np.any(l2.data['RED_SCI2_WAVE'])

    def test_raises_when_wlsfile_missing(self):
        l2 = _make_science_l2()  # no WLSFILE
        with pytest.raises(KeyError, match='WLSFILE'):
            WavelengthCalibration(l2).perform()
