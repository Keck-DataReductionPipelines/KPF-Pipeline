"""
Tests for the SpectralExtraction module (L1 → L2).

Unit tests for extraction algorithms use synthetic arrays and require no
real data. Integration tests (perform()) monkeypatch extract_ffi so they
also require no real data. Regression tests using real L1 FITS files are
skipped if KPF_TESTDATA is not set.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from kpfpipe import DETECTOR
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.modules.spectral_extraction import SpectralExtraction
from kpfpipe.data_models.level0 import KPF0

NORDER_GREEN = DETECTOR['norder']['GREEN']
NORDER_RED   = DETECTOR['norder']['RED']
NCOL         = DETECTOR['ccd']['ncol']

TESTDATA_L0_DIR = Path(__file__).parent / 'testdata' / 'L0' / '20240405'
L0_FILE = str(TESTDATA_L0_DIR / 'KP.20240405.00020.86.fits')


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_l1(tmp_path):
    """Minimal KPF1 object sufficient for to_kpf2() and SpectralExtraction init."""
    fn = str(tmp_path / "kpf_L1_20240101T000000.fits")
    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-01-01T00:00:00"
    green_ccd = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="GREEN_CCD")
    green_var = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="GREEN_VAR")
    red_ccd   = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_CCD")
    red_var   = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_VAR")
    fits.HDUList([primary, green_ccd, green_var, red_ccd, red_var]).writeto(fn, overwrite=True)
    return KPF1.from_fits(fn)


# ---------------------------------------------------------------------------
# Box extraction (unit tests — no data or fixtures required)
# ---------------------------------------------------------------------------

class TestBoxExtraction:

    def test_basic_extraction(self):
        D = np.ones((5, 10), dtype=np.float32)
        V = np.ones((5, 10), dtype=np.float32)
        flux, var = SpectralExtraction._box_extraction(D, V)
        assert flux.shape == (10,)
        assert var.shape == (10,)
        np.testing.assert_allclose(flux, 5.0)

    def test_with_weights(self):
        """Weights < 1 at edges should reduce total flux."""
        D = np.ones((5, 10), dtype=np.float32)
        V = np.ones((5, 10), dtype=np.float32)
        W = np.ones((5, 10), dtype=np.float32)
        W[0, :] = 0.5
        W[-1, :] = 0.5
        flux_full, _ = SpectralExtraction._box_extraction(D, V)
        flux_w,    _ = SpectralExtraction._box_extraction(D, V, W=W)
        assert np.all(flux_w < flux_full)

    def test_with_sky_subtraction(self):
        D = np.full((5, 10), 10.0, dtype=np.float32)
        V = np.ones((5, 10), dtype=np.float32)
        S = np.full((5, 10),  3.0, dtype=np.float32)
        flux, _ = SpectralExtraction._box_extraction(D, V, S=S)
        np.testing.assert_allclose(flux, 5 * (10.0 - 3.0))

    def test_with_mask(self):
        """Masking a pixel should redistribute weight via M normalisation."""
        D = np.ones((5, 10), dtype=np.float32)
        V = np.ones((5, 10), dtype=np.float32)
        M = np.ones((5, 10), dtype=np.float32)
        M[2, :] = 0  # mask centre row
        flux, _ = SpectralExtraction._box_extraction(D, V, M=M)
        # Sum should still equal nrow (mask re-normalises)
        np.testing.assert_allclose(flux, 5.0)

    def test_fully_masked_column_raises(self):
        D = np.ones((5, 10), dtype=np.float32)
        V = np.ones((5, 10), dtype=np.float32)
        M = np.ones((5, 10), dtype=np.float32)
        M[:, 3] = 0  # fully mask column 3
        with pytest.raises(ValueError, match="Fully masked"):
            SpectralExtraction._box_extraction(D, V, M=M)

    def test_variance_propagation(self):
        D = np.ones((5, 10), dtype=np.float32)
        V = np.full((5, 10), 4.0, dtype=np.float32)
        _, var = SpectralExtraction._box_extraction(D, V)
        np.testing.assert_allclose(var, 5 * 4.0)


# ---------------------------------------------------------------------------
# Unimplemented extraction stubs
# ---------------------------------------------------------------------------

class TestUnimplementedExtraction:

    def test_optimal_raises(self):
        D = V = np.ones((5, 10))
        with pytest.raises(NotImplementedError):
            SpectralExtraction._optimal_extraction(D, V)

    def test_flat_relative_raises(self):
        D = V = np.ones((5, 10))
        with pytest.raises(NotImplementedError):
            SpectralExtraction._flat_relative_extraction(D, V)


# ---------------------------------------------------------------------------
# perform() shape tests (monkeypatched — no real data or trace files needed)
# ---------------------------------------------------------------------------

class TestPerformShapes:
    """Verify perform() assembles GREEN and RED arrays into correctly shaped
    KPF2 traces. extract_ffi is monkeypatched to return pre-built arrays."""

    @pytest.fixture
    def mock_ffi_arrays(self):
        """Return pre-built (chip, fiber) arrays matching real detector dims."""
        chips  = ['GREEN', 'RED']
        fibers = ['CAL', 'SCI1', 'SCI2', 'SCI3', 'SKY']
        norder = {'GREEN': NORDER_GREEN, 'RED': NORDER_RED}
        arrays = {}
        for chip in chips:
            for fiber in fibers:
                n = norder[chip]
                arrays[f'{chip}_{fiber}_FLUX'] = np.ones((n, NCOL), dtype=np.float32)
                arrays[f'{chip}_{fiber}_VAR']  = np.ones((n, NCOL), dtype=np.float32)
        return arrays

    def test_returns_kpf2(self, minimal_l1, mock_ffi_arrays, monkeypatch):
        monkeypatch.setattr(SpectralExtraction, 'extract_ffi',
                            lambda self, chip, fibers, method: {
                                k: v for k, v in mock_ffi_arrays.items()
                                if k.startswith(chip)
                            })
        se = SpectralExtraction(minimal_l1)
        l2 = se.perform()
        assert isinstance(l2, KPF2)
        assert l2.level == 2

    def test_green_trace_shape(self, minimal_l1, mock_ffi_arrays, monkeypatch):
        monkeypatch.setattr(SpectralExtraction, 'extract_ffi',
                            lambda self, chip, fibers, method: {
                                k: v for k, v in mock_ffi_arrays.items()
                                if k.startswith(chip)
                            })
        se = SpectralExtraction(minimal_l1)
        l2 = se.perform()
        assert l2.data['GREEN_SCI2_FLUX'].shape == (NORDER_GREEN, NCOL)

    def test_red_trace_shape(self, minimal_l1, mock_ffi_arrays, monkeypatch):
        monkeypatch.setattr(SpectralExtraction, 'extract_ffi',
                            lambda self, chip, fibers, method: {
                                k: v for k, v in mock_ffi_arrays.items()
                                if k.startswith(chip)
                            })
        se = SpectralExtraction(minimal_l1)
        l2 = se.perform()
        assert l2.data['RED_SCI2_FLUX'].shape == (NORDER_RED, NCOL)

    def test_full_trace_shape(self, minimal_l1, mock_ffi_arrays, monkeypatch):
        monkeypatch.setattr(SpectralExtraction, 'extract_ffi',
                            lambda self, chip, fibers, method: {
                                k: v for k, v in mock_ffi_arrays.items()
                                if k.startswith(chip)
                            })
        se = SpectralExtraction(minimal_l1)
        l2 = se.perform()
        assert l2.data['SCI2_FLUX'].shape == (NORDER_GREEN + NORDER_RED, NCOL)

    def test_all_fibers_populated(self, minimal_l1, mock_ffi_arrays, monkeypatch):
        monkeypatch.setattr(SpectralExtraction, 'extract_ffi',
                            lambda self, chip, fibers, method: {
                                k: v for k, v in mock_ffi_arrays.items()
                                if k.startswith(chip)
                            })
        se = SpectralExtraction(minimal_l1)
        l2 = se.perform()
        for fiber in ['CAL', 'SCI1', 'SCI2', 'SCI3', 'SKY']:
            assert l2.data[f'{fiber}_FLUX'].shape == (NORDER_GREEN + NORDER_RED, NCOL)
            assert l2.data[f'{fiber}_VAR'].shape  == (NORDER_GREEN + NORDER_RED, NCOL)

    def test_green_red_slices_independent(self, minimal_l1, monkeypatch):
        """GREEN and RED slices should contain distinct values."""
        def mock_extract(self, chip, fibers, method):
            fill = 1.0 if chip == 'GREEN' else 2.0
            n = NORDER_GREEN if chip == 'GREEN' else NORDER_RED
            return {
                f'{chip}_SCI2_FLUX': np.full((n, NCOL), fill, dtype=np.float32),
                f'{chip}_SCI2_VAR':  np.full((n, NCOL), fill, dtype=np.float32),
            }
        monkeypatch.setattr(SpectralExtraction, 'extract_ffi', mock_extract)

        se = SpectralExtraction(minimal_l1, config={'fibers': ['SCI2']})
        l2 = se.perform(fibers=['SCI2'])

        np.testing.assert_array_equal(l2.data['GREEN_SCI2_FLUX'], 1.0)
        np.testing.assert_array_equal(l2.data['RED_SCI2_FLUX'],   2.0)

    def test_receipt_chain(self, minimal_l1, mock_ffi_arrays, monkeypatch):
        monkeypatch.setattr(SpectralExtraction, 'extract_ffi',
                            lambda self, chip, fibers, method: {
                                k: v for k, v in mock_ffi_arrays.items()
                                if k.startswith(chip)
                            })
        se = SpectralExtraction(minimal_l1)
        l2 = se.perform()
        modules = l2.receipt['Module_Name'].values
        assert 'to_kpf2' in modules
        assert 'spectral_extraction' in modules


# ---------------------------------------------------------------------------
# Regression tests (real L0 data → assemble L1 → extract)
# ---------------------------------------------------------------------------

class TestSpectralExtractionRealData:

    @pytest.fixture(scope="class")
    def l2_from_flat(self):
        l0 = KPF0.from_fits(L0_FILE)
        ia = ImageAssembly(l0)
        l1 = ia.perform()
        se = SpectralExtraction(l1)
        return se.perform(), se

    def test_returns_kpf2(self, l2_from_flat):
        l2, _ = l2_from_flat
        assert isinstance(l2, KPF2)

    def test_green_sci2_flux_shape(self, l2_from_flat):
        l2, _ = l2_from_flat
        assert l2.data['GREEN_SCI2_FLUX'].shape == (NORDER_GREEN, NCOL)

    def test_red_sci2_flux_shape(self, l2_from_flat):
        l2, _ = l2_from_flat
        assert l2.data['RED_SCI2_FLUX'].shape == (NORDER_RED, NCOL)

    def test_full_trace_shape(self, l2_from_flat):
        l2, _ = l2_from_flat
        assert l2.data['SCI2_FLUX'].shape == (NORDER_GREEN + NORDER_RED, NCOL)

    def test_flux_positive(self, l2_from_flat):
        """Flat lamp flux should be positive after extraction."""
        l2, _ = l2_from_flat
        assert np.nanmedian(l2.data['GREEN_SCI2_FLUX']) > 0
        assert np.nanmedian(l2.data['RED_SCI2_FLUX']) > 0

    def test_variance_positive(self, l2_from_flat):
        l2, _ = l2_from_flat
        assert np.all(l2.data['GREEN_SCI2_VAR'] >= 0)
        assert np.all(l2.data['RED_SCI2_VAR'] >= 0)

    def test_receipt_chain(self, l2_from_flat):
        l2, _ = l2_from_flat
        modules = l2.receipt['Module_Name'].values
        assert 'image_assembly' in modules
        assert 'spectral_extraction' in modules
