"""
Tests for the BarycentricCorrection module (KPF2 → KPF2).

Static-method unit tests require no fixtures. Integration tests use a
synthetic KPF2 with a small EXPMETER_SCI table and populated SCI2_WAVE.
Gaia and barycorrpy calls are stubbed via monkeypatching.
"""

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from kpfpipe import DETECTOR
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.modules.barycentric_correction import (
    BarycentricCorrection,
    NORDER_GREEN,
    NORDER_RED,
    NORDER,
)

NCOL = 50   # reduced column count for speed


# ---------------------------------------------------------------------------
# Synthetic KPF2 fixture
# ---------------------------------------------------------------------------

# Exposure meter: 3 readings of 60s, with 60s gaps, starting at T0.
#   Reading 0: 00:00:00 → 00:01:00
#   Gap:       00:01:00 → 00:02:00
#   Reading 1: 00:02:00 → 00:03:00
#   Gap:       00:03:00 → 00:04:00
#   Reading 2: 00:04:00 → 00:05:00
# DATE-BEG = 00:00:00, DATE-END = 00:05:00 (shutter open for full range)
_T0 = '2024-01-01T'
_WAVE_COLS  = ['5000', '5100', '5200', '5300']   # 100Å spacing → dispersion = 100Å
_FLUX_VALUE = 100.0                               # uniform ADU per reading


def _make_expmeter_table():
    begs = [f'{_T0}00:00:00.000', f'{_T0}00:02:00.000', f'{_T0}00:04:00.000']
    ends = [f'{_T0}00:01:00.000', f'{_T0}00:03:00.000', f'{_T0}00:05:00.000']
    data = {'Date-Beg': begs, 'Date-End': ends}
    for wc in _WAVE_COLS:
        data[wc] = np.full(3, _FLUX_VALUE)
    return Table(data)


@pytest.fixture
def synthetic_kpf2():
    """KPF2 with synthetic EXPMETER_SCI and wavelength arrays.

    SCI2_WAVE is filled with 5000.0 — order_center = 5000 for every order,
    so np.interp returns the per-channel value at 5000Å (i.e. the first
    PHOTON_JD) for every order. Tests that need per-order variation
    populate SCI2_WAVE themselves.
    """
    kpf2 = KPF2()

    # KPF-native keywords live in INSTRUMENT_HEADER on L2 (preserved L1 PRIMARY).
    kpf2.headers['INSTRUMENT_HEADER']['DATE-BEG'] = f'{_T0}00:00:00.000'
    kpf2.headers['INSTRUMENT_HEADER']['DATE-END'] = f'{_T0}00:05:00.000'
    kpf2.headers['INSTRUMENT_HEADER']['GAIAID']   = 'DR3 1234567890123456789'

    kpf2.set_data('EXPMETER_SCI', _make_expmeter_table())

    for chip in ['GREEN', 'RED']:
        n = NORDER_GREEN if chip == 'GREEN' else NORDER_RED
        for fiber in ['SKY', 'SCI1', 'SCI2', 'SCI3', 'CAL']:
            kpf2.set_data(f'{chip}_{fiber}_WAVE',
                          np.full((n, NCOL), 5000.0, dtype=np.float32))

    return kpf2


def _fake_skycoord():
    """Deterministic ICRS SkyCoord with realistic proper motion / parallax."""
    return SkyCoord(
        ra=180.0 * u.deg, dec=0.0 * u.deg,
        pm_ra_cosdec=0.0 * u.mas / u.yr,
        pm_dec=0.0 * u.mas / u.yr,
        distance=100.0 * u.pc,
        obstime=Time(2016.0, format='jyear'),
        frame='icrs',
    )


# ---------------------------------------------------------------------------
# Static helpers — unchanged behavior across the split
# ---------------------------------------------------------------------------

class TestStrictlyIncreasing:

    def _make_time(self, seconds):
        jd0 = 2460310.5
        return Time(jd0 + np.array(seconds) / 86400.0, format='jd', scale='utc')

    def test_increasing(self):
        assert BarycentricCorrection._strictly_increasing(self._make_time([0, 1, 2, 3])) is True

    def test_constant_fails(self):
        assert BarycentricCorrection._strictly_increasing(self._make_time([0, 1, 1, 2])) is False

    def test_decreasing_fails(self):
        assert BarycentricCorrection._strictly_increasing(self._make_time([3, 2, 1, 0])) is False

    def test_single_element(self):
        assert BarycentricCorrection._strictly_increasing(self._make_time([0])) is True


class TestInterpolate:

    def _make_times(self, seconds):
        jd0 = 2460310.5
        return Time(jd0 + np.array(seconds, dtype=float) / 86400.0,
                    format='jd', scale='utc')

    def test_gap_midpoint_time(self):
        t_beg = self._make_times([0, 120])
        t_end = self._make_times([60, 180])
        f = np.ones((2, 3))

        t_gap, _ = BarycentricCorrection._interpolate(t_beg, t_end, f)
        np.testing.assert_allclose(t_gap[0].jd, (t_end[0].jd + t_beg[1].jd) / 2)

    def test_gap_flux_equal_exposure_flux_when_same_duration(self):
        t_beg = self._make_times([0, 120])
        t_end = self._make_times([60, 180])
        f = np.full((2, 4), _FLUX_VALUE)

        _, f_gap = BarycentricCorrection._interpolate(t_beg, t_end, f)
        np.testing.assert_allclose(f_gap[0], _FLUX_VALUE, rtol=1e-6)

    def test_output_shape(self):
        t_beg = self._make_times([0, 120, 240])
        t_end = self._make_times([60, 180, 300])
        f = np.ones((3, 4))

        t_gap, f_gap = BarycentricCorrection._interpolate(t_beg, t_end, f)
        assert len(t_gap) == 2
        assert f_gap.shape == (2, 4)

    def test_zero_gap_gives_zero_flux(self):
        t_beg = self._make_times([0, 60])
        t_end = self._make_times([60, 120])
        f = np.ones((2, 3))

        _, f_gap = BarycentricCorrection._interpolate(t_beg, t_end, f)
        np.testing.assert_allclose(f_gap[0], 0.0, atol=1e-10)


class TestExtrapolate:

    def _t(self, sec):
        return Time(2460310.5 + sec / 86400.0, format='jd', scale='utc')

    def test_extrapolate_before_first_reading(self):
        t0    = self._t(0)
        t_beg = self._t(60)
        t_end = self._t(120)
        f = np.full(4, _FLUX_VALUE)

        t_ext, _ = BarycentricCorrection._extrapolate(t0, t_beg, t_end, f)
        np.testing.assert_allclose(t_ext.jd, (t0.jd + t_beg.jd) / 2)

    def test_extrapolate_before_flux_proportional(self):
        t0    = self._t(0)
        t_beg = self._t(60)
        t_end = self._t(120)
        f = np.full(4, _FLUX_VALUE)

        _, f_ext = BarycentricCorrection._extrapolate(t0, t_beg, t_end, f)
        np.testing.assert_allclose(f_ext, _FLUX_VALUE, rtol=1e-6)

    def test_extrapolate_after_last_reading(self):
        t_beg = self._t(0)
        t_end = self._t(60)
        t0    = self._t(120)
        f = np.full(4, _FLUX_VALUE)

        t_ext, _ = BarycentricCorrection._extrapolate(t0, t_beg, t_end, f)
        np.testing.assert_allclose(t_ext.jd, (t_end.jd + t0.jd) / 2)

    def test_t0_inside_raises(self):
        t_beg = self._t(0)
        t_end = self._t(120)
        t0    = self._t(60)
        f = np.ones(4)

        with pytest.raises(ValueError):
            BarycentricCorrection._extrapolate(t0, t_beg, t_end, f)


class TestFixBadExposures:

    def test_clean_array_unchanged(self):
        rng = np.random.default_rng(0)
        f = rng.normal(100.0, 2.0, (60, 20))
        f_fixed = BarycentricCorrection._fix_bad_exposures(f)
        np.testing.assert_allclose(f_fixed, f, rtol=1e-4)

    def test_outlier_repaired(self):
        rng = np.random.default_rng(1)
        f = rng.normal(100.0, 2.0, (60, 20))
        f[30, 10] = 1e6

        f_fixed = BarycentricCorrection._fix_bad_exposures(f)
        assert abs(f_fixed[30, 10] - 100.0) < 20.0

    def test_output_shape_preserved(self):
        rng = np.random.default_rng(2)
        f = rng.normal(50.0, 1.0, (60, 20))
        assert BarycentricCorrection._fix_bad_exposures(f).shape == f.shape


# ---------------------------------------------------------------------------
# _get_timestamps / _get_normalized_flux
# ---------------------------------------------------------------------------

class TestGetTimestamps:

    def test_returns_three_time_arrays(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        t_beg, t_mid, t_end = bc._get_timestamps()
        assert len(t_beg) == 3 and len(t_mid) == 3 and len(t_end) == 3

    def test_mid_is_between_beg_and_end(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        t_beg, t_mid, t_end = bc._get_timestamps()
        assert np.all(t_mid.jd > t_beg.jd)
        assert np.all(t_mid.jd < t_end.jd)

    def test_non_monotonic_raises(self, synthetic_kpf2):
        bad_table = Table({
            'Date-Beg': ['2024-01-01T00:04:00.000', '2024-01-01T00:02:00.000',
                         '2024-01-01T00:00:00.000'],
            'Date-End': ['2024-01-01T00:05:00.000', '2024-01-01T00:03:00.000',
                         '2024-01-01T00:01:00.000'],
            '5000': [1.0, 1.0, 1.0],
        })
        synthetic_kpf2.set_data('EXPMETER_SCI', bad_table)
        bc = BarycentricCorrection(synthetic_kpf2)
        with pytest.raises(ValueError, match="strictly increasing"):
            bc._get_timestamps()


class TestGetNormalizedFlux:

    def test_returns_wavelengths_and_flux(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        w, f = bc._get_normalized_flux()
        assert w.shape == (len(_WAVE_COLS),)
        assert f.shape == (3, len(_WAVE_COLS))

    def test_wavelengths_match_column_labels(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        w, _ = bc._get_normalized_flux()
        np.testing.assert_array_equal(w, [float(c) for c in _WAVE_COLS])

    def test_non_numeric_columns_excluded(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        _, f = bc._get_normalized_flux()
        assert f.shape[1] == len(_WAVE_COLS)

    def test_gain_and_dispersion_applied(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        _, f = bc._get_normalized_flux()
        np.testing.assert_allclose(f, _FLUX_VALUE * 1.48424 / 100.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# flux_weighted_midpoint (now returns (w, t_fwm))
# ---------------------------------------------------------------------------

class TestFluxWeightedMidpoint:

    def test_uniform_flux_gives_geometric_midpoint(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        w, t_fwm = bc.flux_weighted_midpoint(interpolate=False, extrapolate=False,
                                              fix_bad_exposures=False)
        _, t_mid, _ = bc._get_timestamps()
        np.testing.assert_allclose(np.mean(t_fwm.jd), np.mean(t_mid.jd), atol=1e-6)

    def test_output_shapes(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        w, t_fwm = bc.flux_weighted_midpoint(interpolate=False, extrapolate=False,
                                              fix_bad_exposures=False)
        assert w.shape == (len(_WAVE_COLS),)
        assert len(t_fwm) == len(_WAVE_COLS)

    def test_negative_flux_raises(self, synthetic_kpf2, monkeypatch):
        def mock_flux(self):
            w = np.array([float(c) for c in _WAVE_COLS])
            return w, np.full((3, len(_WAVE_COLS)), -1.0)
        monkeypatch.setattr(BarycentricCorrection, '_get_normalized_flux', mock_flux)

        bc = BarycentricCorrection(synthetic_kpf2)
        with pytest.raises(ValueError, match="negative"):
            bc.flux_weighted_midpoint()

    def test_interpolate_shifts_midpoint_with_front_weighted_flux(self, synthetic_kpf2):
        data = {'Date-Beg': ['2024-01-01T00:00:00.000',
                              '2024-01-01T00:02:00.000',
                              '2024-01-01T00:04:00.000'],
                'Date-End': ['2024-01-01T00:01:00.000',
                              '2024-01-01T00:03:00.000',
                              '2024-01-01T00:05:00.000'],
                '5000': [1000.0, 1.0, 1.0],
                '5100': [1000.0, 1.0, 1.0]}
        synthetic_kpf2.set_data('EXPMETER_SCI', Table(data))
        bc = BarycentricCorrection(synthetic_kpf2)

        _, t_no = bc.flux_weighted_midpoint(interpolate=False, extrapolate=False,
                                             fix_bad_exposures=False)
        _, t_yes = bc.flux_weighted_midpoint(interpolate=True, extrapolate=False,
                                              fix_bad_exposures=False)
        assert np.mean(t_yes.jd) >= np.mean(t_no.jd) - 1e-9


# ---------------------------------------------------------------------------
# map_to_orders
# ---------------------------------------------------------------------------

class TestMapToOrders:

    def test_output_shape(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        w_em = np.array([5000.0, 5100.0, 5200.0, 5300.0])
        t_fwm = Time(np.linspace(2460310.5, 2460310.6, 4), format='jd', scale='utc')
        t_per_order = bc.map_to_orders(w_em, t_fwm)
        assert len(t_per_order) == NORDER

    def test_constant_wave_gives_constant_time(self, synthetic_kpf2):
        """SCI2_WAVE filled with 5000.0 → every order interpolates at 5000Å
        and gets the first per-channel t_fwm."""
        bc = BarycentricCorrection(synthetic_kpf2)
        w_em = np.array([5000.0, 5100.0, 5200.0, 5300.0])
        jds = np.array([10.0, 20.0, 30.0, 40.0])
        t_fwm = Time(2460310.5 + jds / 86400.0, format='jd', scale='utc')
        t_per_order = bc.map_to_orders(w_em, t_fwm)
        np.testing.assert_allclose(t_per_order.jd, t_fwm.jd[0])

    def test_orders_get_distinct_times_when_waves_vary(self, synthetic_kpf2):
        """Set SCI2_WAVE so GREEN orders straddle 5000Å and RED orders 5300Å."""
        bc = BarycentricCorrection(synthetic_kpf2)
        green = np.full((NORDER_GREEN, NCOL), 5000.0, dtype=np.float32)
        red   = np.full((NORDER_RED, NCOL),   5300.0, dtype=np.float32)
        synthetic_kpf2.set_data('GREEN_SCI2_WAVE', green)
        synthetic_kpf2.set_data('RED_SCI2_WAVE',   red)

        w_em = np.array([5000.0, 5100.0, 5200.0, 5300.0])
        jds  = np.array([10.0, 20.0, 30.0, 40.0])
        t_fwm = Time(2460310.5 + jds / 86400.0, format='jd', scale='utc')

        t_per_order = bc.map_to_orders(w_em, t_fwm)
        # GREEN orders → t_fwm[0] (=10s), RED orders → t_fwm[-1] (=40s)
        np.testing.assert_allclose(t_per_order.jd[:NORDER_GREEN], t_fwm.jd[0])
        np.testing.assert_allclose(t_per_order.jd[NORDER_GREEN:], t_fwm.jd[-1])

    def test_empty_sci2_wave_raises(self, synthetic_kpf2, monkeypatch):
        """If SCI2_WAVE is empty, map_to_orders should fail loudly."""
        bc = BarycentricCorrection(synthetic_kpf2)
        # Force the underlying TRACE3_WAVE to empty by replacing the alias target
        synthetic_kpf2.data['TRACE3_WAVE'] = np.array([])
        w_em = np.array([5000.0, 5100.0])
        t_fwm = Time([2460310.5, 2460310.6], format='jd', scale='utc')
        with pytest.raises(KeyError, match="SCI2_WAVE"):
            bc.map_to_orders(w_em, t_fwm)


# ---------------------------------------------------------------------------
# perform() — Gaia and barycorrpy stubbed
# ---------------------------------------------------------------------------

class TestPerform:

    DELTA_RV_MPS = 30000.0   # constant 30 km/s shift across all orders

    @pytest.fixture
    def bc_monkeypatched(self, synthetic_kpf2, monkeypatch):
        """BarycentricCorrection with Gaia + barycorrpy stubbed."""
        def mock_query(gaia_id):
            return _fake_skycoord()

        def mock_compute(skycoord, obs_times, location):
            n = len(np.atleast_1d(obs_times.jd))
            bc_vel = np.full(n, TestPerform.DELTA_RV_MPS, dtype=float)
            # BJD = JD + 500s (light-travel approx); same for every order in this stub
            bjd_tdb = np.atleast_1d(obs_times.jd) + 500.0 / 86400.0
            return bc_vel, bjd_tdb

        monkeypatch.setattr(BarycentricCorrection, '_query_gaia', staticmethod(mock_query))
        monkeypatch.setattr(BarycentricCorrection, '_compute_barycorr', staticmethod(mock_compute))
        # Disable fix_bad_exposures: the 3×4 uniform-flux fixture triggers a
        # degenerate triangulation inside scipy.griddata. Filter itself is
        # exercised by TestFixBadExposures with a noisy 60×20 array.
        return BarycentricCorrection(synthetic_kpf2, config={'fix_bad_exposures': False})

    def test_returns_kpf2(self, bc_monkeypatched):
        assert isinstance(bc_monkeypatched.perform(), KPF2)

    def test_returns_same_object(self, bc_monkeypatched):
        original = bc_monkeypatched.kpf2_obj
        assert bc_monkeypatched.perform() is original

    def test_bjd_tdb_extension_populated(self, bc_monkeypatched):
        kpf2 = bc_monkeypatched.perform()
        bjd = np.asarray(kpf2.data['BJD_TDB'])
        assert bjd.shape == (NORDER,)
        assert np.all(np.isfinite(bjd))

    def test_barycorr_kms_extension_populated(self, bc_monkeypatched):
        kpf2 = bc_monkeypatched.perform()
        kms = np.asarray(kpf2.data['BARYCORR_KMS'])
        assert kms.shape == (NORDER,)
        np.testing.assert_allclose(kms, TestPerform.DELTA_RV_MPS / 1000.0)

    def test_barycorr_z_extension_populated(self, bc_monkeypatched):
        kpf2 = bc_monkeypatched.perform()
        z = np.asarray(kpf2.data['BARYCORR_Z'])
        assert z.shape == (NORDER,)
        # All orders share the same delta_rv → all z values identical
        assert np.std(z) < 1e-12
        # 30 km/s redshift is < 1: positive RV means moving away
        assert 0.9 < z[0] < 1.0

    def test_wave_arrays_scaled(self, bc_monkeypatched):
        kpf2 = bc_monkeypatched.kpf2_obj
        orig = {f'{chip}_{fiber}_WAVE': kpf2.data[f'{chip}_{fiber}_WAVE'].copy()
                for chip in ['GREEN', 'RED']
                for fiber in ['SKY', 'SCI1', 'SCI2', 'SCI3', 'CAL']}

        bc_monkeypatched.perform()

        z_extension = np.asarray(kpf2.data['BARYCORR_Z'])
        for key, before in orig.items():
            after = kpf2.data[key]
            chip = key.split('_')[0]
            z = z_extension[:NORDER_GREEN] if chip == 'GREEN' else z_extension[NORDER_GREEN:]
            np.testing.assert_allclose(after, before * z[:, None], rtol=1e-5,
                                       err_msg=f"{key} not scaled correctly")

    def test_per_ccd_primary_keywords(self, bc_monkeypatched):
        kpf2 = bc_monkeypatched.perform()
        primary = kpf2.headers['PRIMARY']
        for key in ['CCD1BJD', 'CCD1BKMS', 'CCD1BZ',
                    'CCD2BJD', 'CCD2BKMS', 'CCD2BZ']:
            assert key in primary, f"{key} missing from PRIMARY"

        def _v(k):
            x = primary[k]
            return x[0] if isinstance(x, tuple) else x

        # All orders had the same delta_rv → green and red means are equal
        np.testing.assert_allclose(_v('CCD1BKMS'), _v('CCD2BKMS'))
        np.testing.assert_allclose(_v('CCD1BZ'),   _v('CCD2BZ'))

    def test_receipt_entry_added(self, bc_monkeypatched):
        bc_monkeypatched.perform()
        modules = bc_monkeypatched.kpf2_obj.receipt['Module_Name'].values
        assert 'barycentric_correction' in modules

    def test_results_populated(self, bc_monkeypatched):
        assert bc_monkeypatched._results is None
        bc_monkeypatched.perform()
        results = bc_monkeypatched._results
        assert set(results.keys()) == {'bjd_tdb', 'bary_kms', 'bary_z'}
        for v in results.values():
            assert len(v) == NORDER

    def test_zero_rv_leaves_wavelengths_unchanged(self, synthetic_kpf2, monkeypatch):
        """delta_rv == 0 → z == 1 → WAVE arrays unchanged."""
        def mock_query(gaia_id):
            return _fake_skycoord()

        def mock_compute(skycoord, obs_times, location):
            n = len(np.atleast_1d(obs_times.jd))
            return np.zeros(n), np.atleast_1d(obs_times.jd)

        monkeypatch.setattr(BarycentricCorrection, '_query_gaia', staticmethod(mock_query))
        monkeypatch.setattr(BarycentricCorrection, '_compute_barycorr', staticmethod(mock_compute))

        bc = BarycentricCorrection(synthetic_kpf2, config={'fix_bad_exposures': False})
        orig = bc.kpf2_obj.data['GREEN_SCI2_WAVE'].copy()
        bc.perform()
        np.testing.assert_allclose(bc.kpf2_obj.data['GREEN_SCI2_WAVE'], orig, rtol=1e-7)
