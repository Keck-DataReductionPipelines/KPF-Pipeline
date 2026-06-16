"""
Tests for the RadialVelocity module (KPF2 -> KPF4: per-orderlet CCFs and RVs).

Static-method unit tests (_compute_ccf, _compute_rv) build synthetic spectra
and CCFs with no fixtures. Build-helper tests use a header-only KPF2 and read
the real on-disk line masks. Integration tests (compute_ccf/compute_rv/perform)
use a synthetic KPF2 with absorption injected at a monkeypatched line mask, and
a narrow velocity grid for speed.
"""

import numpy as np
import pytest

from kpfpipe.data_models.level2 import KPF2, NORDER_GREEN, NORDER_RED
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.modules.radial_velocity import RadialVelocity, SPEED_OF_LIGHT_KMS

NORDER = NORDER_GREEN + NORDER_RED
_FIBERS = ['CAL', 'SCI1', 'SCI2', 'SCI3', 'SKY']   # all orderlets

# Narrow CCF grid for fast integration tests; wide enough for the second-pass
# +/-3 sigma window (sigma ~ 4 km/s) to stay on-grid: +/-15 km/s at 0.25 km/s.
_RANGE_KMS = [-15.0, 15.0]
_STEP_KMS = 0.25                                   # matches the module default
_NVEL = round((_RANGE_KMS[1] - _RANGE_KMS[0]) / _STEP_KMS) + 1
_V_INJECT = 1.5                                   # injected RV [km/s], on the grid
_MASK_CENTERS = np.linspace(5015.0, 5035.0, 30)   # vacuum line centers [Å]
# Wide enough that the default compute_ccf clip (clip_edge_pixels=[500, 500])
# trims the order edges but leaves the 5015-5035 Å mask lines well inside.
NCOL = 2000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mask(centers, weights=None, width=0.5):
    """Build a line-mask dict matching _build_ccf_line_mask's structure."""
    centers = np.asarray(centers, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(centers)
    half_width = centers * (width / SPEED_OF_LIGHT_KMS)
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

    def test_descending_wave_raises(self):
        wave, flux, mask, vel = self._order(v_dip=0.0)
        with pytest.raises(ValueError, match="descending"):
            RadialVelocity._compute_ccf(wave[::-1], flux[::-1], mask, vel, 0.0)

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
        rv, rv_err = RadialVelocity._compute_rv(vel, ccf, wave, [-50.0, 50.0], 11)
        assert rv == pytest.approx(2.5, abs=0.05)

    def test_error_finite_and_positive(self):
        vel, ccf, wave = self._ccf(v0=0.0)
        _, rv_err = RadialVelocity._compute_rv(vel, ccf, wave, [-50.0, 50.0], 11)
        assert np.isfinite(rv_err) and rv_err > 0

    def test_even_window_pts_allowed(self):
        # min_npts is a minimum point count, not a centered odd window.
        vel, ccf, wave = self._ccf(v0=1.0)
        rv, _ = RadialVelocity._compute_rv(vel, ccf, wave, [-50.0, 50.0], 8)
        assert rv == pytest.approx(1.0, abs=0.05)

    def test_flat_ccf_returns_nan(self):
        vel = np.arange(-402, 403) * 0.25
        ccf = np.full_like(vel, 100.0)
        wave = np.linspace(5000.0, 5050.0, 4000)
        rv, rv_err = RadialVelocity._compute_rv(vel, ccf, wave, [-50.0, 50.0], 11)
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_nonfinite_ccf_returns_nan(self):
        # Non-finite CCF values fail loudly (NaN) rather than being masked out.
        vel, ccf, wave = self._ccf(v0=1.0)
        ccf[10] = np.nan
        rv, rv_err = RadialVelocity._compute_rv(vel, ccf, wave, [-50.0, 50.0], 11)
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_second_pass_off_grid_returns_first_pass_rv(self):
        # Dip near the low edge: the first-pass fit succeeds, but the +/-3 sigma
        # second-pass window runs off the grid -> first-pass RV, NaN error.
        vel = np.arange(-402, 403) * 0.25
        v0 = vel[20]
        ccf = 100.0 - 30.0 * np.exp(-0.5 * ((vel - v0) / 4.0) ** 2)
        wave = np.linspace(5000.0, 5050.0, 4000)
        rv, rv_err = RadialVelocity._compute_rv(vel, ccf, wave, [-50.0, 50.0], 11)
        assert rv == pytest.approx(v0, abs=0.1) and np.isnan(rv_err)

    def test_min_points_floor_does_not_change_result(self):
        # A larger min-points floor below the +/-3 sigma window doesn't shift the fit.
        vel, ccf, wave = self._ccf(v0=1.0)
        rv9, _ = RadialVelocity._compute_rv(vel, ccf, wave, [-50.0, 50.0], 9)
        rv21, _ = RadialVelocity._compute_rv(vel, ccf, wave, [-50.0, 50.0], 21)
        assert rv9 == pytest.approx(1.0, abs=0.05)
        assert rv21 == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# _build_ccf_line_mask / _build_ccf_velocity_grid  (header-only fixture)
# ---------------------------------------------------------------------------

@pytest.fixture
def header_kpf2():
    """KPF2 with only the header keywords the build/dispatch helpers need."""
    kpf2 = KPF2()
    kpf2.headers['INSTRUMENT_HEADER']['TARGTEFF'] = 5772.0
    kpf2.headers['INSTRUMENT_HEADER']['TARGRADV'] = 0.0
    kpf2.headers['INSTRUMENT_HEADER']['SCI-OBJ'] = 'Target'
    kpf2.headers['INSTRUMENT_HEADER']['SKY-OBJ'] = 'Sky'
    kpf2.headers['INSTRUMENT_HEADER']['CAL-OBJ'] = 'None'
    return kpf2


class TestDispatch:
    """Illumination-source resolution and per-source CCF configuration."""

    @pytest.mark.parametrize('raw, source', [
        ('Target', 'target'), ('Sky', 'sky'), ('Th_gold', 'thar'),
        ('Th_daily', 'thar'), ('LFCFiber', 'lfc'), ('EtalonFiber', 'etalon'),
        ('None', 'none'), ('TARGET', 'target'),
    ])
    def test_normalize_source(self, raw, source):
        assert RadialVelocity._normalize_source(raw) == source

    def test_normalize_source_unrecognized_raises(self):
        with pytest.raises(ValueError, match="unrecognized illumination"):
            RadialVelocity._normalize_source('Frobnicator')

    def test_resolve_illumination_per_fiber(self, header_kpf2):
        rv = RadialVelocity(header_kpf2)
        assert rv._resolve_illumination('GREEN', 'SCI2') == 'target'
        assert rv._resolve_illumination('RED', 'SKY') == 'sky'
        assert rv._resolve_illumination('GREEN', 'CAL') == 'none'

    def test_resolve_unknown_fiber_raises(self, header_kpf2):
        with pytest.raises(ValueError, match="unknown fiber"):
            RadialVelocity(header_kpf2)._resolve_illumination('GREEN', 'BOGUS')

    def test_resolve_missing_keyword_raises(self, header_kpf2):
        del header_kpf2.headers['INSTRUMENT_HEADER']['CAL-OBJ']
        with pytest.raises(ValueError, match="CAL-OBJ"):
            RadialVelocity(header_kpf2)._resolve_illumination('GREEN', 'CAL')

    def test_config_target(self, header_kpf2):
        header_kpf2.headers['INSTRUMENT_HEADER']['TARGRADV'] = 11.1
        mask, barycorr, center = RadialVelocity(header_kpf2)._ccf_config_for_source('target')
        assert mask == 'G2_espresso' and barycorr is True and center == 11.1

    def test_config_sky(self, header_kpf2):
        mask, barycorr, center = RadialVelocity(header_kpf2)._ccf_config_for_source('sky')
        assert mask == 'G2_espresso' and barycorr is True and center == 0.0

    def test_config_thar(self, header_kpf2):
        mask, barycorr, center = RadialVelocity(header_kpf2)._ccf_config_for_source('thar')
        assert mask == 'thar' and barycorr is False and center == 0.0

    def test_config_none_skips(self, header_kpf2):
        assert RadialVelocity(header_kpf2)._ccf_config_for_source('none') is None

    @pytest.mark.parametrize('source', ['etalon', 'lfc'])
    def test_config_unimplemented_raises(self, header_kpf2, source):
        with pytest.raises(NotImplementedError, match=source):
            RadialVelocity(header_kpf2)._ccf_config_for_source(source)


class TestStellarMaskName:

    def test_targteff_selects_mask(self, header_kpf2):
        rv = RadialVelocity(header_kpf2)
        assert rv._stellar_mask_name() == 'G2_espresso'      # 5772 K -> G2
        header_kpf2.headers['INSTRUMENT_HEADER']['TARGTEFF'] = 4000.0
        assert RadialVelocity(header_kpf2)._stellar_mask_name() == 'K6_espresso'

    @pytest.mark.parametrize('teff', [0.0, -100.0, 'nan'])
    def test_invalid_targteff_raises(self, header_kpf2, teff):
        header_kpf2.headers['INSTRUMENT_HEADER']['TARGTEFF'] = teff
        with pytest.raises(ValueError, match="TARGTEFF"):
            RadialVelocity(header_kpf2)._stellar_mask_name()

    def test_missing_targteff_raises(self, header_kpf2):
        del header_kpf2.headers['INSTRUMENT_HEADER']['TARGTEFF']
        with pytest.raises(ValueError, match="TARGTEFF"):
            RadialVelocity(header_kpf2)._stellar_mask_name()

    def test_missing_targradv_raises(self, header_kpf2):
        del header_kpf2.headers['INSTRUMENT_HEADER']['TARGRADV']
        with pytest.raises(ValueError, match="TARGRADV"):
            RadialVelocity(header_kpf2)._systemic_rv()


class TestBuildLineMask:

    def test_keys_and_shapes(self, header_kpf2):
        mask = RadialVelocity(header_kpf2)._build_ccf_line_mask('GREEN', 'SCI2', 'G2_espresso')
        assert set(mask) == {'center', 'weight', 'start', 'end'}
        n = mask['center'].size
        assert all(mask[k].shape == (n,) for k in mask)

    def test_top_hat_edges_bracket_center(self, header_kpf2):
        mask = RadialVelocity(header_kpf2)._build_ccf_line_mask('GREEN', 'SCI2', 'G2_espresso')
        assert np.all(mask['start'] < mask['center'])
        assert np.all(mask['center'] < mask['end'])

    def test_cached(self, header_kpf2):
        rv = RadialVelocity(header_kpf2)
        assert (rv._build_ccf_line_mask('GREEN', 'SCI2', 'G2_espresso')
                is rv._build_ccf_line_mask('GREEN', 'SCI2', 'G2_espresso'))

    def test_thar_mask_uniform_weights(self, header_kpf2):
        mask = RadialVelocity(header_kpf2)._build_ccf_line_mask('GREEN', 'CAL', 'thar')
        assert np.all(mask['weight'] == 1.0)
        # ThAr centers are deduped and sorted (lines recur across overlapping orders).
        assert np.all(np.diff(mask['center']) > 0)


class TestBuildVelocityGrid:

    def test_centered_on_center(self, header_kpf2):
        grid = RadialVelocity(header_kpf2)._build_ccf_velocity_grid('GREEN', 'SCI2', 10.0)
        assert grid.mean() == pytest.approx(10.0)

    def test_default_size_and_step(self, header_kpf2):
        grid = RadialVelocity(header_kpf2)._build_ccf_velocity_grid('GREEN', 'SCI2', 0.0)
        assert grid.size == 801                       # [-100, 100] km/s at 0.25 -> arange(-400, 401)
        np.testing.assert_allclose(np.diff(grid), 0.25)

    def test_symmetric_about_zero_center(self, header_kpf2):
        grid = RadialVelocity(header_kpf2)._build_ccf_velocity_grid('GREEN', 'SCI2', 0.0)
        np.testing.assert_allclose(grid.min(), -grid.max())

    def test_cached(self, header_kpf2):
        rv = RadialVelocity(header_kpf2)
        assert (rv._build_ccf_velocity_grid('GREEN', 'SCI2', 0.0)
                is rv._build_ccf_velocity_grid('GREEN', 'SCI2', 0.0))


# ---------------------------------------------------------------------------
# compute_ccf / compute_rv / perform  (synthetic KPF2 + monkeypatched mask)
# ---------------------------------------------------------------------------

@pytest.fixture
def rv_kpf2():
    """KPF2 with identical per-order synthetic spectra (absorption at _MASK_CENTERS
    shifted by _V_INJECT) for every orderlet and zero barycentric correction."""
    kpf2 = KPF2()
    kpf2.headers['INSTRUMENT_HEADER']['TARGTEFF'] = 5772.0
    kpf2.headers['INSTRUMENT_HEADER']['TARGRADV'] = 0.0
    # Illumination sources: SCI on a star, SKY on sky, CAL dark (skipped).
    kpf2.headers['INSTRUMENT_HEADER']['SCI-OBJ'] = 'Target'
    kpf2.headers['INSTRUMENT_HEADER']['SKY-OBJ'] = 'Sky'
    kpf2.headers['INSTRUMENT_HEADER']['CAL-OBJ'] = 'None'

    wave_1d = np.linspace(5000.0, 5050.0, NCOL)
    lam_obs = _MASK_CENTERS * (1.0 + _V_INJECT / SPEED_OF_LIGHT_KMS)   # z = 0
    flux_1d = _absorption_spectrum(wave_1d, lam_obs)

    for chip, n in [('GREEN', NORDER_GREEN), ('RED', NORDER_RED)]:
        for fiber in _FIBERS:
            kpf2.set_data(f'{chip}_{fiber}_WAVE',
                          np.tile(wave_1d, (n, 1)).astype(np.float64))
            kpf2.set_data(f'{chip}_{fiber}_FLUX',
                          np.tile(flux_1d, (n, 1)).astype(np.float64))
    # Per-order barycentric extensions (populated together by BarycentricCorrection).
    kpf2.set_data('BARYCORR_Z', np.zeros(NORDER))
    kpf2.set_data('BARYCORR_KMS', np.zeros(NORDER))
    kpf2.set_data('BJD_TDB', np.zeros(NORDER))
    return kpf2


@pytest.fixture
def rv_module(rv_kpf2, monkeypatch):
    """RadialVelocity on a narrow grid with the line mask stubbed to _MASK_CENTERS."""
    mask = _make_mask(_MASK_CENTERS)
    monkeypatch.setattr(RadialVelocity, '_build_ccf_line_mask',
                        lambda self, chip, fiber, mask_name, width=None: mask)
    return RadialVelocity(rv_kpf2, config={'ccf_window': _RANGE_KMS})


class TestComputeCCFPublic:

    def test_returns_velocity_and_ccf(self, rv_module):
        res = rv_module.compute_ccf('GREEN', 'SCI2')
        assert set(res) == {'velocity', 'ccf'}
        assert res['velocity'].shape == (_NVEL,)
        assert res['ccf'].shape == (NORDER_GREEN, _NVEL)

    def test_red_chip_shape(self, rv_module):
        res = rv_module.compute_ccf('RED', 'SCI1')
        assert res['ccf'].shape == (NORDER_RED, _NVEL)

    def test_dip_at_injected_velocity(self, rv_module):
        res = rv_module.compute_ccf('GREEN', 'SCI2')
        vel, ccf = res['velocity'], res['ccf']
        assert vel[np.argmin(ccf[0])] == pytest.approx(_V_INJECT, abs=0.3)

    def test_caches_ccf(self, rv_module):
        res = rv_module.compute_ccf('GREEN', 'SCI2')
        assert rv_module._ccf['GREEN_SCI2'] is res['ccf']

    def test_lowercase_chip_accepted(self, rv_module):
        res = rv_module.compute_ccf('green', 'sci2')
        assert res['ccf'].shape == (NORDER_GREEN, _NVEL)

    def test_missing_barycorr_z_raises(self, rv_kpf2, monkeypatch):
        monkeypatch.setattr(RadialVelocity, '_build_ccf_line_mask',
                            lambda self, chip, fiber, mask_name, width=None: _make_mask(_MASK_CENTERS))
        rv_kpf2.set_data('BARYCORR_Z', np.array([]))
        rv = RadialVelocity(rv_kpf2, config={'ccf_window': _RANGE_KMS})
        with pytest.raises(ValueError, match="BARYCORR_Z"):
            rv.compute_ccf('GREEN', 'SCI2')

    def test_all_zero_ccf_raises(self, rv_module):
        # No usable signal across the whole orderlet -> fail loudly instead of
        # silently returning an all-zero CCF cube.
        flux = np.asarray(rv_module.l2_obj.data['GREEN_SCI2_FLUX'])
        rv_module.l2_obj.set_data('GREEN_SCI2_FLUX', np.zeros_like(flux))
        with pytest.raises(RuntimeError, match="identically zero"):
            rv_module.compute_ccf('GREEN', 'SCI2')

    def test_clip_edge_pixels_zero_keeps_all(self, rv_module):
        # clip_edge_pixels=[0, 0] is a no-op (no pixels removed).
        full = rv_module.compute_ccf('GREEN', 'SCI2', clip_edge_pixels=[0, 0])['ccf']
        assert np.any(full)

    def test_clip_edge_pixels_too_large_raises(self, rv_module):
        # Clipping more pixels than the order has fails loudly.
        with pytest.raises(ValueError, match="removes all"):
            rv_module.compute_ccf('GREEN', 'SCI2', clip_edge_pixels=[NCOL, NCOL])


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

    _ILLUMINATED = ['SCI1', 'SCI2', 'SCI3', 'SKY']   # CAL-OBJ='None' -> skipped

    def test_returns_kpf4_with_per_orderlet_extensions(self, rv_module):
        l4 = rv_module.perform()
        assert isinstance(l4, KPF4)
        for fiber in self._ILLUMINATED:
            assert l4.data[f'{fiber}_CCF'].shape == (NORDER, _NVEL)
            table = l4.data[f'{fiber}_RV']
            assert len(table) == NORDER
            assert set(table.columns) >= {
                'ORDER_INDEX', 'BJD_TDB', 'BERV', 'WAVE_START', 'WAVE_END', 'RV', 'RV_ERR'}

    def test_unilluminated_fiber_skipped(self, rv_module):
        # CAL-OBJ='None' -> no CCF cube or RV table written.
        l4 = rv_module.perform()
        assert l4.data['CAL_CCF'].size == 0
        assert len(l4.data['CAL_RV']) == 0

    def test_ccf_chip_halves_populated(self, rv_module):
        l4 = rv_module.perform()
        assert l4.data['GREEN_SCI2_CCF'].shape == (NORDER_GREEN, _NVEL)
        assert l4.data['RED_SCI2_CCF'].shape == (NORDER_RED, _NVEL)
        assert np.any(l4.data['GREEN_SCI2_CCF'])
        assert np.any(l4.data['RED_SCI2_CCF'])

    def test_recovers_injected_rv_illuminated_orderlets(self, rv_module):
        l4 = rv_module.perform()
        for fiber in self._ILLUMINATED:
            rv = np.asarray(l4.data[f'{fiber}_RV']['RV'])
            np.testing.assert_allclose(rv, _V_INJECT, atol=0.1)

    def test_per_fiber_ccf_and_rv_headers(self, rv_module):
        # EPRV L4 keywords on each orderlet's CCF/RV extension; RVMETHOD on PRIMARY.
        l4 = rv_module.perform()
        ccf_hdr = l4.headers['SCI2_CCF']
        assert ccf_hdr['VELNSTEP'][0] == _NVEL
        assert ccf_hdr['VELSTEP'][0] == pytest.approx(0.25)
        assert ccf_hdr['VELSTART'][0] == pytest.approx(_RANGE_KMS[0])   # center 0 (TARGRADV=0)
        assert ccf_hdr['CCFMASK'][0] == 'G2_espresso'                   # 5772 K -> G2
        rv_hdr = l4.headers['SCI2_RV']
        assert rv_hdr['RVMETHOD'][0] == 'CCF'
        assert rv_hdr['SKYRMVD'][0] is False
        assert rv_hdr['TELLRMVD'][0] is False
        assert l4.headers['PRIMARY']['RVMETHOD'][0] == 'CCF'

    def test_thar_mask_recorded_for_cal(self, rv_module):
        # CAL on a ThAr lamp -> CCFMASK 'thar', instrument frame (no barycorr).
        rv_module.l2_obj.headers['INSTRUMENT_HEADER']['CAL-OBJ'] = 'Th_gold'
        l4 = rv_module.perform(fibers=['CAL'])
        assert l4.headers['CAL_CCF']['CCFMASK'][0] == 'thar'
        rv = np.asarray(l4.data['CAL_RV']['RV'])
        np.testing.assert_allclose(rv, _V_INJECT, atol=0.1)

    def test_etalon_fiber_raises(self, rv_module):
        rv_module.l2_obj.headers['INSTRUMENT_HEADER']['CAL-OBJ'] = 'EtalonFiber'
        with pytest.raises(NotImplementedError, match='etalon'):
            rv_module.perform(fibers=['CAL'])

    def test_explicit_chips_and_fibers(self, rv_module):
        l4 = rv_module.perform(chips=['GREEN'], fibers=['SCI1'])
        # SCI1 green half populated, red half left zero; other orderlets untouched.
        assert np.any(l4.data['GREEN_SCI1_CCF'])
        assert not np.any(l4.data['RED_SCI1_CCF'])
        assert l4.data['SCI2_CCF'].size == 0
        assert len(l4.data['SCI2_RV']) == 0
        rv = np.asarray(l4.data['SCI1_RV']['RV'])
        assert np.all(np.isfinite(rv[:NORDER_GREEN]))
        assert np.all(np.isnan(rv[NORDER_GREEN:]))


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestConstructor:

    def test_invalid_config_type_raises(self, header_kpf2):
        with pytest.raises(TypeError, match="None, dict, or ConfigHandler"):
            RadialVelocity(header_kpf2, config="not-a-config")

    def test_dict_config_overrides_default(self, header_kpf2):
        rv = RadialVelocity(header_kpf2, config={'ccf_mask_width': 1.0})
        assert rv.ccf_mask_width == 1.0

    def test_defaults_applied(self, header_kpf2):
        rv = RadialVelocity(header_kpf2)
        assert rv.ccf_mask_width == 0.5
        assert rv.ccf_step_size == 0.25
