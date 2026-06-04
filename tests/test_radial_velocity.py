"""
Tests for the RadialVelocity module (KPF2 -> per-chip/fiber CCFs and RVs).

Static-method unit tests (_compute_ccf, _compute_rv) build synthetic spectra
and CCFs with no fixtures. Build-helper tests use a header-only KPF2 and read
the real on-disk line masks. Integration tests (compute_ccf/compute_rv/perform)
use a synthetic KPF2 with absorption injected at a monkeypatched line mask, and
a narrow velocity grid for speed.
"""

import numpy as np
import pytest

from kpfpipe.data_models.level2 import KPF2, NORDER_GREEN, NORDER_RED
from kpfpipe.modules.radial_velocity import RadialVelocity, SPEED_OF_LIGHT_KMS

NORDER = NORDER_GREEN + NORDER_RED

# Narrow CCF grid for fast integration tests: +/-10 km/s at 0.25 km/s -> 81 steps.
_STEP_RANGE = [-40, 40]
_NVEL = _STEP_RANGE[1] - _STEP_RANGE[0] + 1
_V_INJECT = 1.5                                   # injected RV [km/s], on the grid
_MASK_CENTERS = np.linspace(5005.0, 5045.0, 30)   # vacuum line centers [Å]
NCOL = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mask(centers, weights=None, mask_width_kms=0.5):
    """Build a line-mask dict matching _build_ccf_line_mask's structure."""
    centers = np.asarray(centers, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(centers)
    half_width = centers * (mask_width_kms / SPEED_OF_LIGHT_KMS)
    return {
        'center': centers,
        'weight': np.asarray(weights, dtype=np.float64),
        'start': centers - half_width,
        'end': centers + half_width,
    }


def _absorption_spectrum(wave, centers, weights=None, depth=0.6, sigma_kms=4.0):
    """Unit continuum with Gaussian absorption lines at `centers`."""
    if weights is None:
        weights = np.ones_like(centers)
    flux = np.ones_like(wave)
    for center, weight in zip(centers, weights):
        sigma_a = center * sigma_kms / SPEED_OF_LIGHT_KMS
        flux -= depth * (weight / np.max(weights)) * np.exp(-0.5 * ((wave - center) / sigma_a) ** 2)
    return flux


# ---------------------------------------------------------------------------
# _compute_ccf (staticmethod)
# ---------------------------------------------------------------------------

class TestComputeCCF:

    def _order(self, v_dip=0.0, z=0.0):
        wave = np.linspace(5000.0, 5050.0, 2000)
        centers = np.linspace(5008.0, 5042.0, 20)
        mask = _make_mask(centers)
        # Observed absorption that the CCF should align at velocity step v_dip.
        lam_obs = centers * (1.0 + v_dip / SPEED_OF_LIGHT_KMS) / (1.0 + z)
        flux = _absorption_spectrum(wave, lam_obs)
        vel = np.arange(-402, 403) * 0.25
        return wave, flux, mask, vel

    def test_dip_at_injected_velocity(self):
        wave, flux, mask, vel = self._order(v_dip=3.0, z=0.0)
        ccf = RadialVelocity._compute_ccf(wave, flux, mask, vel, 0.0)
        assert vel[np.argmin(ccf)] == pytest.approx(3.0, abs=0.3)

    def test_zero_velocity_dip(self):
        wave, flux, mask, vel = self._order(v_dip=0.0, z=0.0)
        ccf = RadialVelocity._compute_ccf(wave, flux, mask, vel, 0.0)
        assert vel[np.argmin(ccf)] == pytest.approx(0.0, abs=0.3)

    def test_barycorr_z_folds_in(self):
        z = 5.0 / SPEED_OF_LIGHT_KMS
        wave, flux, mask, vel = self._order(v_dip=2.0, z=z)
        ccf = RadialVelocity._compute_ccf(wave, flux, mask, vel, z)
        assert vel[np.argmin(ccf)] == pytest.approx(2.0, abs=0.3)

    def test_reversed_order_matches(self):
        wave, flux, mask, vel = self._order(v_dip=0.0)
        ccf_asc = RadialVelocity._compute_ccf(wave, flux, mask, vel, 0.0)
        ccf_desc = RadialVelocity._compute_ccf(wave[::-1], flux[::-1], mask, vel, 0.0)
        np.testing.assert_allclose(ccf_asc, ccf_desc)

    def test_constant_wave_returns_zeros(self):
        wave, flux, mask, vel = self._order()
        ccf = RadialVelocity._compute_ccf(np.full_like(wave, 5000.0), flux, mask, vel, 0.0)
        assert not np.any(ccf)

    def test_nan_wave_returns_zeros(self):
        wave, flux, mask, vel = self._order()
        wave = wave.copy()
        wave[1000] = np.nan
        ccf = RadialVelocity._compute_ccf(wave, flux, mask, vel, 0.0)
        assert not np.any(ccf)

    def test_no_lines_in_order_returns_zeros(self):
        wave = np.linspace(6000.0, 6050.0, 2000)
        flux = np.ones_like(wave)
        mask = _make_mask(np.linspace(5008.0, 5042.0, 20))   # all outside the order
        vel = np.arange(-402, 403) * 0.25
        ccf = RadialVelocity._compute_ccf(wave, flux, mask, vel, 0.0)
        assert not np.any(ccf)


# ---------------------------------------------------------------------------
# _compute_rv (staticmethod)
# ---------------------------------------------------------------------------

class TestComputeRV:

    def _ccf(self, v0=0.0, sigma=4.0, baseline=100.0, depth=30.0):
        vel = np.arange(-402, 403) * 0.25
        ccf = baseline - depth * np.exp(-0.5 * ((vel - v0) / sigma) ** 2)
        wave = np.linspace(5000.0, 5050.0, 4000)
        return vel, ccf, wave

    def test_recovers_injected_rv(self):
        vel, ccf, wave = self._ccf(v0=2.5)
        rv, rv_err = RadialVelocity._compute_rv(vel, ccf, wave, 50.0, 11)
        assert rv == pytest.approx(2.5, abs=0.05)

    def test_error_finite_and_positive(self):
        vel, ccf, wave = self._ccf(v0=0.0)
        _, rv_err = RadialVelocity._compute_rv(vel, ccf, wave, 50.0, 11)
        assert np.isfinite(rv_err) and rv_err > 0

    def test_even_window_raises(self):
        vel, ccf, wave = self._ccf()
        with pytest.raises(ValueError, match="odd"):
            RadialVelocity._compute_rv(vel, ccf, wave, 50.0, 8)

    def test_flat_ccf_returns_nan(self):
        vel = np.arange(-402, 403) * 0.25
        ccf = np.full_like(vel, 100.0)
        wave = np.linspace(5000.0, 5050.0, 4000)
        rv, rv_err = RadialVelocity._compute_rv(vel, ccf, wave, 50.0, 11)
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_window_off_grid_returns_rv_with_nan_error(self):
        vel = np.arange(-402, 403) * 0.25
        v0 = vel[3]   # dip within rv_window_pts//2 of the low edge
        ccf = 100.0 - 30.0 * np.exp(-0.5 * ((vel - v0) / 4.0) ** 2)
        wave = np.linspace(5000.0, 5050.0, 4000)
        rv, rv_err = RadialVelocity._compute_rv(vel, ccf, wave, 50.0, 11)
        assert np.isfinite(rv) and np.isnan(rv_err)

    def test_window_size_controls_points(self):
        # A wider points window still recovers the same center.
        vel, ccf, wave = self._ccf(v0=1.0)
        rv9, _ = RadialVelocity._compute_rv(vel, ccf, wave, 50.0, 9)
        rv21, _ = RadialVelocity._compute_rv(vel, ccf, wave, 50.0, 21)
        assert rv9 == pytest.approx(1.0, abs=0.05)
        assert rv21 == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# _build_ccf_line_mask / _build_ccf_velocity_grid  (header-only fixture)
# ---------------------------------------------------------------------------

@pytest.fixture
def header_kpf2():
    """KPF2 with only the header keywords the build helpers need."""
    kpf2 = KPF2()
    kpf2.headers['INSTRUMENT_HEADER']['TARGTEFF'] = 5772.0
    kpf2.headers['INSTRUMENT_HEADER']['TARGRADV'] = 0.0
    return kpf2


class TestBuildLineMask:

    def test_keys_and_shapes(self, header_kpf2):
        mask = RadialVelocity(header_kpf2)._build_ccf_line_mask()
        assert set(mask) == {'center', 'weight', 'start', 'end'}
        n = mask['center'].size
        assert all(mask[k].shape == (n,) for k in mask)

    def test_top_hat_edges_bracket_center(self, header_kpf2):
        mask = RadialVelocity(header_kpf2)._build_ccf_line_mask()
        assert np.all(mask['start'] < mask['center'])
        assert np.all(mask['center'] < mask['end'])

    def test_cached(self, header_kpf2):
        rv = RadialVelocity(header_kpf2)
        assert rv._build_ccf_line_mask() is rv._build_ccf_line_mask()

    def test_targteff_selects_mask(self, header_kpf2):
        m_g2 = RadialVelocity(header_kpf2)._build_ccf_line_mask()
        header_kpf2.headers['INSTRUMENT_HEADER']['TARGTEFF'] = 4000.0   # different bin
        m_other = RadialVelocity(header_kpf2)._build_ccf_line_mask()
        assert (m_g2['center'].size != m_other['center'].size
                or not np.array_equal(m_g2['center'], m_other['center']))

    @pytest.mark.parametrize('teff', [0.0, -100.0, 'nan'])
    def test_invalid_targteff_raises(self, header_kpf2, teff):
        header_kpf2.headers['INSTRUMENT_HEADER']['TARGTEFF'] = teff
        with pytest.raises(ValueError, match="TARGTEFF"):
            RadialVelocity(header_kpf2)._build_ccf_line_mask()

    def test_missing_targteff_raises(self, header_kpf2):
        del header_kpf2.headers['INSTRUMENT_HEADER']['TARGTEFF']
        with pytest.raises(ValueError, match="TARGTEFF"):
            RadialVelocity(header_kpf2)._build_ccf_line_mask()


class TestBuildVelocityGrid:

    def test_centered_on_targradv(self, header_kpf2):
        header_kpf2.headers['INSTRUMENT_HEADER']['TARGRADV'] = 10.0
        grid = RadialVelocity(header_kpf2)._build_ccf_velocity_grid()
        assert grid.mean() == pytest.approx(10.0)

    def test_default_size_and_step(self, header_kpf2):
        grid = RadialVelocity(header_kpf2)._build_ccf_velocity_grid()
        assert grid.size == 805                       # arange(-402, 403)
        np.testing.assert_allclose(np.diff(grid), 0.25)

    def test_symmetric_about_targradv(self, header_kpf2):
        grid = RadialVelocity(header_kpf2)._build_ccf_velocity_grid()
        np.testing.assert_allclose(grid.min(), -grid.max())

    def test_cached(self, header_kpf2):
        rv = RadialVelocity(header_kpf2)
        assert rv._build_ccf_velocity_grid() is rv._build_ccf_velocity_grid()

    def test_missing_targradv_raises(self, header_kpf2):
        del header_kpf2.headers['INSTRUMENT_HEADER']['TARGRADV']
        with pytest.raises(ValueError, match="TARGRADV"):
            RadialVelocity(header_kpf2)._build_ccf_velocity_grid()


# ---------------------------------------------------------------------------
# compute_ccf / compute_rv / perform  (synthetic KPF2 + monkeypatched mask)
# ---------------------------------------------------------------------------

@pytest.fixture
def rv_kpf2():
    """KPF2 with identical per-order synthetic spectra (absorption at _MASK_CENTERS
    shifted by _V_INJECT) and zero barycentric redshift."""
    kpf2 = KPF2()
    kpf2.headers['INSTRUMENT_HEADER']['TARGTEFF'] = 5772.0
    kpf2.headers['INSTRUMENT_HEADER']['TARGRADV'] = 0.0

    wave_1d = np.linspace(5000.0, 5050.0, NCOL)
    lam_obs = _MASK_CENTERS * (1.0 + _V_INJECT / SPEED_OF_LIGHT_KMS)   # z = 0
    flux_1d = _absorption_spectrum(wave_1d, lam_obs)

    for chip, n in [('GREEN', NORDER_GREEN), ('RED', NORDER_RED)]:
        for fiber in ['SCI1', 'SCI2', 'SCI3']:
            kpf2.set_data(f'{chip}_{fiber}_WAVE',
                          np.tile(wave_1d, (n, 1)).astype(np.float64))
            kpf2.set_data(f'{chip}_{fiber}_FLUX',
                          np.tile(flux_1d, (n, 1)).astype(np.float64))
    kpf2.set_data('BARYCORR_Z', np.zeros(NORDER))
    return kpf2


@pytest.fixture
def rv_module(rv_kpf2, monkeypatch):
    """RadialVelocity on a narrow grid with the line mask stubbed to _MASK_CENTERS."""
    mask = _make_mask(_MASK_CENTERS)
    monkeypatch.setattr(RadialVelocity, '_build_ccf_line_mask',
                        lambda self, mask_width_kms=None: mask)
    return RadialVelocity(rv_kpf2, config={'ccf_step_range': _STEP_RANGE})


class TestComputeCCFPublic:

    def test_returns_array_with_shape(self, rv_module):
        ccf = rv_module.compute_ccf('GREEN', 'SCI2')
        assert ccf.shape == (NORDER_GREEN, _NVEL)

    def test_red_chip_shape(self, rv_module):
        ccf = rv_module.compute_ccf('RED', 'SCI1')
        assert ccf.shape == (NORDER_RED, _NVEL)

    def test_dip_at_injected_velocity(self, rv_module):
        ccf = rv_module.compute_ccf('GREEN', 'SCI2')
        vel = rv_module._build_ccf_velocity_grid()
        assert vel[np.argmin(ccf[0])] == pytest.approx(_V_INJECT, abs=0.3)

    def test_caches_ccf(self, rv_module):
        ccf = rv_module.compute_ccf('GREEN', 'SCI2')
        assert rv_module._ccf['GREEN_SCI2'] is ccf

    def test_lowercase_chip_accepted(self, rv_module):
        ccf = rv_module.compute_ccf('green', 'sci2')
        assert ccf.shape == (NORDER_GREEN, _NVEL)

    def test_missing_barycorr_z_raises(self, rv_kpf2, monkeypatch):
        monkeypatch.setattr(RadialVelocity, '_build_ccf_line_mask',
                            lambda self, mask_width_kms=None: _make_mask(_MASK_CENTERS))
        rv_kpf2.set_data('BARYCORR_Z', np.array([]))
        rv = RadialVelocity(rv_kpf2, config={'ccf_step_range': _STEP_RANGE})
        with pytest.raises(ValueError, match="BARYCORR_Z"):
            rv.compute_ccf('GREEN', 'SCI2')


class TestComputeRVPublic:

    def test_returns_rv_dict(self, rv_module):
        rv_module.compute_ccf('GREEN', 'SCI2')
        res = rv_module.compute_rv('GREEN', 'SCI2')
        assert set(res) == {'rv', 'rv_err'}
        assert res['rv'].shape == (NORDER_GREEN,)
        assert res['rv_err'].shape == (NORDER_GREEN,)

    def test_raises_without_ccf(self, rv_module):
        with pytest.raises(RuntimeError, match="compute_ccf"):
            rv_module.compute_rv('GREEN', 'SCI2')

    def test_recovers_injected_rv(self, rv_module):
        rv_module.compute_ccf('GREEN', 'SCI2')
        rv = rv_module.compute_rv('GREEN', 'SCI2')['rv']
        np.testing.assert_allclose(rv, _V_INJECT, atol=0.1)

    def test_errors_finite_and_positive(self, rv_module):
        rv_module.compute_ccf('GREEN', 'SCI2')
        rv_err = rv_module.compute_rv('GREEN', 'SCI2')['rv_err']
        assert np.all(np.isfinite(rv_err)) and np.all(rv_err > 0)


class TestPerform:

    def test_result_structure(self, rv_module):
        ccf_arrays, rv_arrays = rv_module.perform()
        exts = {f'{c}_{f}' for c in ('GREEN', 'RED') for f in ('SCI1', 'SCI2', 'SCI3')}
        assert set(ccf_arrays) == exts
        assert set(rv_arrays) == exts
        assert set(rv_arrays['GREEN_SCI2']) == {'rv', 'rv_err'}

    def test_ccf_shapes(self, rv_module):
        ccf_arrays, _ = rv_module.perform()
        assert ccf_arrays['GREEN_SCI2'].shape == (NORDER_GREEN, _NVEL)
        assert ccf_arrays['RED_SCI1'].shape == (NORDER_RED, _NVEL)

    def test_recovers_injected_rv_all_chips_fibers(self, rv_module):
        _, rv_arrays = rv_module.perform()
        for chip in ('GREEN', 'RED'):
            for fiber in ('SCI1', 'SCI2', 'SCI3'):
                rv = rv_arrays[f'{chip}_{fiber}']['rv']
                np.testing.assert_allclose(rv, _V_INJECT, atol=0.1)

    def test_explicit_chips_and_fibers(self, rv_module):
        ccf_arrays, rv_arrays = rv_module.perform(chips=['GREEN'], fibers=['SCI1'])
        assert set(ccf_arrays) == {'GREEN_SCI1'}
        assert set(rv_arrays) == {'GREEN_SCI1'}


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestConstructor:

    def test_invalid_config_type_raises(self, header_kpf2):
        with pytest.raises(TypeError, match="None, dict, or ConfigHandler"):
            RadialVelocity(header_kpf2, config="not-a-config")

    def test_dict_config_overrides_default(self, header_kpf2):
        rv = RadialVelocity(header_kpf2, config={'mask_width_kms': 1.0})
        assert rv.mask_width_kms == 1.0

    def test_defaults_applied(self, header_kpf2):
        rv = RadialVelocity(header_kpf2)
        assert rv.mask_width_kms == 0.5
        assert rv.ccf_step_size == 0.25
