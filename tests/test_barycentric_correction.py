"""
Tests for the BarycentricCorrection module (KPF2 → KPF2).

Static-method unit tests require no fixtures. Integration tests use a
synthetic KPF2 object with a small EXPMETER_SCI table. Tests that would
require a live Gaia query or barycorrpy call are covered via monkeypatching.
"""

import numpy as np
import pytest
import astropy.units as u
from astropy.table import Table
from astropy.time import Time

from kpfpipe import DETECTOR
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.modules.barycentric_correction import BarycentricCorrection

NORDER_GREEN = DETECTOR['norder']['GREEN']
NORDER_RED   = DETECTOR['norder']['RED']
NCOL         = 50   # reduced column count for speed


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
    """KPF2 with synthetic EXPMETER_SCI and wavelength arrays."""
    kpf2 = KPF2()

    kpf2.headers['PRIMARY']['DATE-BEG'] = f'{_T0}00:00:00.000'
    kpf2.headers['PRIMARY']['DATE-END'] = f'{_T0}00:05:00.000'
    kpf2.headers['PRIMARY']['GAIAID'] = 'DR3 1234567890123456789'

    kpf2.set_data('EXPMETER_SCI', _make_expmeter_table())

    for chip in ['GREEN', 'RED']:
        n = NORDER_GREEN if chip == 'GREEN' else NORDER_RED
        for fiber in ['SKY', 'SCI1', 'SCI2', 'SCI3', 'CAL']:
            kpf2.set_data(f'{chip}_{fiber}_WAVE',
                          np.full((n, NCOL), 5000.0, dtype=np.float32))

    return kpf2


# ---------------------------------------------------------------------------
# _strictly_increasing
# ---------------------------------------------------------------------------

class TestStrictlyIncreasing:

    def _make_time(self, seconds):
        jd0 = 2460310.5  # arbitrary JD base
        return Time(jd0 + np.array(seconds) / 86400.0, format='jd', scale='utc')

    def test_increasing(self):
        t = self._make_time([0, 1, 2, 3])
        assert BarycentricCorrection._strictly_increasing(t) is True

    def test_constant_fails(self):
        t = self._make_time([0, 1, 1, 2])
        assert BarycentricCorrection._strictly_increasing(t) is False

    def test_decreasing_fails(self):
        t = self._make_time([3, 2, 1, 0])
        assert BarycentricCorrection._strictly_increasing(t) is False

    def test_single_element(self):
        t = self._make_time([0])
        # slice t[:-1] and t[1:] are both empty → np.all of empty → True
        assert BarycentricCorrection._strictly_increasing(t) is True


# ---------------------------------------------------------------------------
# _interpolate
# ---------------------------------------------------------------------------

class TestInterpolate:

    def _make_times(self, seconds):
        jd0 = 2460310.5
        return Time(jd0 + np.array(seconds, dtype=float) / 86400.0,
                    format='jd', scale='utc')

    def test_gap_midpoint_time(self):
        """Gap midpoint should fall halfway between end of reading 0 and start of reading 1."""
        t_beg = self._make_times([0, 120])    # two 60s readings, 60s gap between
        t_end = self._make_times([60, 180])
        f = np.ones((2, 3))

        t_gap, _ = BarycentricCorrection._interpolate(t_beg, t_end, f)

        expected_jd = (t_end[0].jd + t_beg[1].jd) / 2
        np.testing.assert_allclose(t_gap[0].jd, expected_jd)

    def test_gap_flux_equal_exposure_flux_when_same_duration(self):
        """If gap duration == exposure duration and flux is uniform, f_gap == f."""
        t_beg = self._make_times([0, 120])
        t_end = self._make_times([60, 180])
        f = np.full((2, 4), _FLUX_VALUE)

        _, f_gap = BarycentricCorrection._interpolate(t_beg, t_end, f)

        np.testing.assert_allclose(f_gap[0], _FLUX_VALUE, rtol=1e-6)

    def test_output_shape(self):
        """n readings → n-1 gaps."""
        t_beg = self._make_times([0, 120, 240])
        t_end = self._make_times([60, 180, 300])
        f = np.ones((3, 4))

        t_gap, f_gap = BarycentricCorrection._interpolate(t_beg, t_end, f)

        assert len(t_gap) == 2
        assert f_gap.shape == (2, 4)

    def test_zero_gap_gives_zero_flux(self):
        """Back-to-back readings (zero gap) should produce zero gap flux."""
        t_beg = self._make_times([0, 60])    # reading 1 starts exactly when reading 0 ends
        t_end = self._make_times([60, 120])
        f = np.ones((2, 3))

        _, f_gap = BarycentricCorrection._interpolate(t_beg, t_end, f)

        np.testing.assert_allclose(f_gap[0], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# _extrapolate
# ---------------------------------------------------------------------------

class TestExtrapolate:

    def _t(self, sec):
        return Time(2460310.5 + sec / 86400.0, format='jd', scale='utc')

    def test_extrapolate_before_first_reading(self):
        """Gap before the first reading: midpoint is halfway between t0 and t_beg."""
        t0   = self._t(0)
        t_beg = self._t(60)
        t_end = self._t(120)
        f = np.full(4, _FLUX_VALUE)

        t_ext, f_ext = BarycentricCorrection._extrapolate(t0, t_beg, t_end, f)

        expected_jd = (t0.jd + t_beg.jd) / 2
        np.testing.assert_allclose(t_ext.jd, expected_jd)

    def test_extrapolate_before_flux_proportional(self):
        """Gap flux equals exposure flux when gap and exposure have equal duration."""
        t0    = self._t(0)
        t_beg = self._t(60)   # 60s gap before first reading
        t_end = self._t(120)  # 60s reading
        f = np.full(4, _FLUX_VALUE)

        _, f_ext = BarycentricCorrection._extrapolate(t0, t_beg, t_end, f)

        np.testing.assert_allclose(f_ext, _FLUX_VALUE, rtol=1e-6)

    def test_extrapolate_after_last_reading(self):
        """Gap after the last reading: midpoint is halfway between t_end and t0."""
        t_beg = self._t(0)
        t_end = self._t(60)
        t0    = self._t(120)
        f = np.full(4, _FLUX_VALUE)

        t_ext, f_ext = BarycentricCorrection._extrapolate(t0, t_beg, t_end, f)

        expected_jd = (t_end.jd + t0.jd) / 2
        np.testing.assert_allclose(t_ext.jd, expected_jd)

    def test_t0_inside_raises(self):
        """t0 between t_beg and t_end should raise ValueError."""
        t_beg = self._t(0)
        t_end = self._t(120)
        t0    = self._t(60)
        f = np.ones(4)

        with pytest.raises(ValueError):
            BarycentricCorrection._extrapolate(t0, t_beg, t_end, f)


# ---------------------------------------------------------------------------
# _fix_bad_exposures
# ---------------------------------------------------------------------------

class TestFixBadExposures:

    def test_clean_array_unchanged(self):
        rng = np.random.default_rng(0)
        # Use a realistically-noisy array so mad_std > 0 and no outliers are flagged
        f = rng.normal(100.0, 2.0, (60, 20))
        f_fixed = BarycentricCorrection._fix_bad_exposures(f)
        np.testing.assert_allclose(f_fixed, f, rtol=1e-4)

    def test_outlier_repaired(self):
        rng = np.random.default_rng(1)
        f = rng.normal(100.0, 2.0, (60, 20))
        f[30, 10] = 1e6   # inject a large outlier

        f_fixed = BarycentricCorrection._fix_bad_exposures(f)

        assert abs(f_fixed[30, 10] - 100.0) < 20.0, (
            f"Outlier not repaired: f_fixed[30,10] = {f_fixed[30, 10]}"
        )

    def test_output_shape_preserved(self):
        rng = np.random.default_rng(2)
        f = rng.normal(50.0, 1.0, (60, 20))
        f_fixed = BarycentricCorrection._fix_bad_exposures(f)
        assert f_fixed.shape == f.shape


# ---------------------------------------------------------------------------
# _get_timestamps and _get_normalized_flux
# ---------------------------------------------------------------------------

class TestGetTimestamps:

    def test_returns_three_time_arrays(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        t_beg, t_mid, t_end = bc._get_timestamps()
        assert len(t_beg) == 3
        assert len(t_mid) == 3
        assert len(t_end) == 3

    def test_mid_is_between_beg_and_end(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        t_beg, t_mid, t_end = bc._get_timestamps()
        assert np.all(t_mid.jd > t_beg.jd)
        assert np.all(t_mid.jd < t_end.jd)

    def test_beg_strictly_increasing(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        t_beg, _, _ = bc._get_timestamps()
        assert BarycentricCorrection._strictly_increasing(t_beg)

    def test_non_monotonic_raises(self, synthetic_kpf2):
        """Reversed timestamps should raise ValueError."""
        from astropy.table import Table
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

    def test_output_shape(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        f = bc._get_normalized_flux()
        assert f.shape == (3, 4)   # 3 time steps, 4 wavelength channels

    def test_non_numeric_columns_excluded(self, synthetic_kpf2):
        """Date-Beg and Date-End should not appear in the flux array."""
        bc = BarycentricCorrection(synthetic_kpf2)
        f = bc._get_normalized_flux()
        # 4 numeric wave columns in the fixture → ncol = 4
        assert f.shape[1] == len(_WAVE_COLS)

    def test_gain_and_dispersion_applied(self, synthetic_kpf2):
        """With 100Å uniform spacing, f_norm = f_raw * 1.48424 / 100."""
        bc = BarycentricCorrection(synthetic_kpf2)
        f = bc._get_normalized_flux()
        expected = _FLUX_VALUE * 1.48424 / 100.0
        np.testing.assert_allclose(f, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# flux_weighted_midpoint
# ---------------------------------------------------------------------------

class TestFluxWeightedMidpoint:

    def test_uniform_flux_gives_geometric_midpoint(self, synthetic_kpf2):
        """With uniform flux and symmetric timestamps the FWM should equal
        the mean of the three reading midpoints (ignoring gaps and edges)."""
        bc = BarycentricCorrection(synthetic_kpf2)
        t_fwm = bc.flux_weighted_midpoint(interpolate=False, extrapolate=False,
                                           fix_bad_exposures=False)
        t_beg, t_mid, _ = bc._get_timestamps()
        expected_jd = np.mean(t_mid.jd)
        np.testing.assert_allclose(
            np.mean(t_fwm.jd), expected_jd, atol=1e-6
        )

    def test_output_shape(self, synthetic_kpf2):
        """One FWM time per wavelength channel."""
        bc = BarycentricCorrection(synthetic_kpf2)
        t_fwm = bc.flux_weighted_midpoint(interpolate=False, extrapolate=False,
                                           fix_bad_exposures=False)
        assert len(t_fwm) == len(_WAVE_COLS)

    def test_negative_flux_raises(self, synthetic_kpf2, monkeypatch):
        """Negative expmeter values should raise ValueError."""
        def mock_flux(self):
            return np.full((3, 4), -1.0)
        monkeypatch.setattr(BarycentricCorrection, '_get_normalized_flux', mock_flux)

        bc = BarycentricCorrection(synthetic_kpf2)
        with pytest.raises(ValueError, match="negative"):
            bc.flux_weighted_midpoint()

    def test_interpolate_shifts_midpoint_later_when_front_weighted(self, synthetic_kpf2):
        """When flux is front-weighted, the interpolated gap between bright
        reading 0 and dim reading 1 has a large flux sample at T+90s — later
        than reading 0's midpoint — so FWM shifts later with interpolation."""
        from astropy.table import Table

        # Front-weighted flux: first reading much brighter than the rest
        data = {'Date-Beg': ['2024-01-01T00:00:00.000',
                              '2024-01-01T00:02:00.000',
                              '2024-01-01T00:04:00.000'],
                'Date-End': ['2024-01-01T00:01:00.000',
                              '2024-01-01T00:03:00.000',
                              '2024-01-01T00:05:00.000'],
                '5000': [1000.0, 1.0, 1.0],
                '5100': [1000.0, 1.0, 1.0],
                }
        synthetic_kpf2.set_data('EXPMETER_SCI', Table(data))
        bc = BarycentricCorrection(synthetic_kpf2)

        t_no_interp = bc.flux_weighted_midpoint(interpolate=False, extrapolate=False,
                                                fix_bad_exposures=False)
        t_with_interp = bc.flux_weighted_midpoint(interpolate=True, extrapolate=False,
                                                   fix_bad_exposures=False)

        # Gap sample at T+90s is bright (≈500), pulling FWM later than without it
        assert np.mean(t_with_interp.jd) >= np.mean(t_no_interp.jd) - 1e-9


# ---------------------------------------------------------------------------
# perform() — Gaia and barycorrpy are monkeypatched
# ---------------------------------------------------------------------------

class TestPerform:

    Z_TEST = 1.0 - 1e-4   # arbitrary but non-trivial shift factor

    @pytest.fixture
    def bc_monkeypatched(self, synthetic_kpf2, monkeypatch):
        """BarycentricCorrection with network calls stubbed out."""
        def mock_fwm(self, **kwargs):
            t0 = Time('2024-01-01T00:02:30.000', format='isot', scale='utc')
            return Time(np.full(len(_WAVE_COLS), t0.jd), format='jd', scale='utc')

        def mock_barycorr(self, obs_time):
            return TestPerform.Z_TEST, 30000.0 * u.m / u.s

        monkeypatch.setattr(BarycentricCorrection, 'flux_weighted_midpoint', mock_fwm)
        monkeypatch.setattr(BarycentricCorrection, 'barycentric_correction', mock_barycorr)
        return BarycentricCorrection(synthetic_kpf2)

    def test_returns_kpf2(self, bc_monkeypatched):
        result = bc_monkeypatched.perform()
        assert isinstance(result, KPF2)

    def test_returns_same_object(self, bc_monkeypatched):
        """perform() should modify the KPF2 in-place and return the same object."""
        original = bc_monkeypatched.kpf2_obj
        result = bc_monkeypatched.perform()
        assert result is original

    def test_wavelength_arrays_scaled(self, bc_monkeypatched):
        """All WAVE extensions should be multiplied by z."""
        kpf2 = bc_monkeypatched.kpf2_obj
        # Record original values before perform
        orig = {}
        for chip in ['GREEN', 'RED']:
            for fiber in ['SKY', 'SCI1', 'SCI2', 'SCI3', 'CAL']:
                orig[f'{chip}_{fiber}_WAVE'] = kpf2.data[f'{chip}_{fiber}_WAVE'].copy()

        bc_monkeypatched.perform()

        for key, original_wave in orig.items():
            np.testing.assert_allclose(
                kpf2.data[key], original_wave * TestPerform.Z_TEST, rtol=1e-5,
                err_msg=f"{key} not scaled correctly"
            )

    def test_receipt_entry_added(self, bc_monkeypatched):
        bc_monkeypatched.perform()
        modules = bc_monkeypatched.kpf2_obj.receipt['Module_Name'].values
        assert 'barycentric_correction' in modules

    def test_results_populated(self, bc_monkeypatched):
        assert bc_monkeypatched._results is None
        bc_monkeypatched.perform()
        assert bc_monkeypatched._results is not None
        assert 'obs_time' in bc_monkeypatched._results
        assert 'delta_rv' in bc_monkeypatched._results
        assert 'z' in bc_monkeypatched._results

    def test_z_equals_one_leaves_wavelengths_unchanged(self, synthetic_kpf2, monkeypatch):
        """A z of exactly 1.0 should leave all wavelength arrays unchanged."""
        def mock_fwm(self, **kwargs):
            t0 = Time('2024-01-01T00:02:30.000', format='isot', scale='utc')
            return Time(np.full(len(_WAVE_COLS), t0.jd), format='jd', scale='utc')

        def mock_barycorr(self, obs_time):
            return 1.0, 0.0 * u.m / u.s

        monkeypatch.setattr(BarycentricCorrection, 'flux_weighted_midpoint', mock_fwm)
        monkeypatch.setattr(BarycentricCorrection, 'barycentric_correction', mock_barycorr)

        bc = BarycentricCorrection(synthetic_kpf2)
        orig = bc.kpf2_obj.data['GREEN_SCI2_WAVE'].copy()
        bc.perform()
        np.testing.assert_array_equal(bc.kpf2_obj.data['GREEN_SCI2_WAVE'], orig)
