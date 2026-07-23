"""Tests for the BarycentricCorrection module (KPF2 -> KPF2).

Static-method unit tests require no fixtures. Integration tests use a
synthetic KPF2 with a small EXPMETER_SCI table, populated SCI2_WAVE, and the
SCI2 catalog cards on PRIMARY. barycorrpy calls are stubbed via monkeypatching.
"""

import logging

import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time

from kpfpipe import DETECTOR
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.modules.barycentric_correction import BarycentricCorrection

from ._dtype_policy import BARYCORR, BJD, assert_dtype

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
NORDER = NORDER_GREEN + NORDER_RED
NCOL = 50  # reduced column count for speed


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
_T0 = "2024-01-01T"
_WAVE_COLS = ["5000", "5100", "5200", "5300"]  # 100Å spacing → dispersion = 100Å
_FLUX_VALUE = 100.0  # uniform ADU per reading

# SCI2 catalog cards, as KPF0.to_kpf1 writes them from AstroQuery's canonical record:
# ICRS, RA/Dec sexagesimal (RA hourangle, Dec deg), PM arcsec/yr, parallax mas, rv
# km/s, epoch Julian years. RA 180 deg / Dec 0 / 100 pc, matching the astrometry the
# stubbed Gaia query used to supply.
_CATALOG_CARDS = {
    "CSRC3": "gaia",
    "CID3": "1234567890123456789",
    "CRA3": "12:00:00.0000",
    "CDEC3": "+00:00:00.000",
    "CPMR3": 0.0,
    "CPMD3": 0.0,
    "CPLX3": 10.0,
    "CEPCH3": 2016.0,
    "CEQNX3": 2000.0,
}


def _make_expmeter_table():
    begs = [f"{_T0}00:00:00.000", f"{_T0}00:02:00.000", f"{_T0}00:04:00.000"]
    ends = [f"{_T0}00:01:00.000", f"{_T0}00:03:00.000", f"{_T0}00:05:00.000"]
    data = {"Date-Beg": begs, "Date-End": ends}
    for wc in _WAVE_COLS:
        data[wc] = np.full(3, _FLUX_VALUE)
    return Table(data)


@pytest.fixture
def synthetic_kpf2():
    """KPF2 with synthetic EXPMETER_SCI and wavelength arrays.

    SCI2_WAVE is filled with 5000.0 -- order_center = 5000 for every order,
    so np.interp returns the per-channel value at 5000Å (i.e. the first
    PHOTON_JD) for every order. Tests that need per-order variation
    populate SCI2_WAVE themselves.
    """
    kpf2 = KPF2()

    # KPF-native keywords live in INSTRUMENT_HEADER on L2 (preserved L1 PRIMARY).
    kpf2.headers["INSTRUMENT_HEADER"]["DATE-BEG"] = f"{_T0}00:00:00.000"
    kpf2.headers["INSTRUMENT_HEADER"]["DATE-END"] = f"{_T0}00:05:00.000"

    # Target astrometry reaches L2 on the EPRV PRIMARY C*# cards (AstroQuery ->
    # KPF0.to_kpf1 -> KPF1.to_kpf2); BarycentricCorrection issues no query.
    for key, value in _CATALOG_CARDS.items():
        kpf2.headers["PRIMARY"][key] = value

    kpf2.set_data("EXPMETER_SCI", _make_expmeter_table())

    for chip in ["GREEN", "RED"]:
        n = NORDER_GREEN if chip == "GREEN" else NORDER_RED
        for fiber in ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]:
            kpf2.set_data(
                f"{chip}_{fiber}_WAVE", np.full((n, NCOL), 5000.0, dtype=np.float64)
            )

    return kpf2


# ---------------------------------------------------------------------------
# Static helpers -- unchanged behavior across the split
# ---------------------------------------------------------------------------


class TestInterpolate:
    def _make_times(self, seconds):
        jd0 = 2460310.5
        return Time(
            jd0 + np.array(seconds, dtype=float) / 86400.0, format="jd", scale="utc"
        )

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
        return Time(2460310.5 + sec / 86400.0, format="jd", scale="utc")

    def test_extrapolate_before_first_reading(self):
        t0 = self._t(0)
        t_beg = self._t(60)
        t_end = self._t(120)
        f = np.full(4, _FLUX_VALUE)

        t_ext, _ = BarycentricCorrection._extrapolate(t0, t_beg, t_end, f)
        np.testing.assert_allclose(t_ext.jd, (t0.jd + t_beg.jd) / 2)

    def test_extrapolate_before_flux_proportional(self):
        t0 = self._t(0)
        t_beg = self._t(60)
        t_end = self._t(120)
        f = np.full(4, _FLUX_VALUE)

        _, f_ext = BarycentricCorrection._extrapolate(t0, t_beg, t_end, f)
        np.testing.assert_allclose(f_ext, _FLUX_VALUE, rtol=1e-6)

    def test_extrapolate_after_last_reading(self):
        t_beg = self._t(0)
        t_end = self._t(60)
        t0 = self._t(120)
        f = np.full(4, _FLUX_VALUE)

        t_ext, _ = BarycentricCorrection._extrapolate(t0, t_beg, t_end, f)
        np.testing.assert_allclose(t_ext.jd, (t_end.jd + t0.jd) / 2)

    def test_t0_inside_raises(self):
        t_beg = self._t(0)
        t_end = self._t(120)
        t0 = self._t(60)
        f = np.ones(4)

        with pytest.raises(ValueError):
            BarycentricCorrection._extrapolate(t0, t_beg, t_end, f)


class TestFixExpmeterOutliers:
    def test_clean_array_unchanged(self):
        rng = np.random.default_rng(0)
        f = rng.normal(100.0, 2.0, (60, 20))
        f_fixed = BarycentricCorrection._fix_expmeter_outliers(f)
        np.testing.assert_allclose(f_fixed, f, rtol=1e-4)

    def test_outlier_repaired(self):
        rng = np.random.default_rng(1)
        f = rng.normal(100.0, 2.0, (60, 20))
        f[30, 10] = 1e6

        f_fixed = BarycentricCorrection._fix_expmeter_outliers(f)
        assert abs(f_fixed[30, 10] - 100.0) < 20.0

    def test_output_shape_preserved(self):
        rng = np.random.default_rng(2)
        f = rng.normal(50.0, 1.0, (60, 20))
        assert BarycentricCorrection._fix_expmeter_outliers(f).shape == f.shape


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
        bad_table = Table(
            {
                "Date-Beg": [
                    "2024-01-01T00:04:00.000",
                    "2024-01-01T00:02:00.000",
                    "2024-01-01T00:00:00.000",
                ],
                "Date-End": [
                    "2024-01-01T00:05:00.000",
                    "2024-01-01T00:03:00.000",
                    "2024-01-01T00:01:00.000",
                ],
                "5000": [1.0, 1.0, 1.0],
            }
        )
        synthetic_kpf2.set_data("EXPMETER_SCI", bad_table)
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
# _compute_barycorr (staticmethod) -- the one non-stubbed regression pin
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestComputeBarycorrReference:
    """The real barycorrpy handoff, pinned to a golden value.

    Every perform() test stubs _compute_barycorr, so this is the only guard that
    the real barycorrpy wiring stays correct: the ICRS ra/dec/pm/parallax, the
    J2000 epoch, UTC->JD, Keck lat/lon/alt, and the m/s units and sign. It calls
    barycorrpy directly (no stub) and may download JPL ephemeris on first run,
    hence @slow. Golden values are pinned to barycorrpy==0.4.4 (see pyproject).
    """

    def test_matches_barycorrpy_reference(self):
        # Fixed Tau Ceti-like astrometry at a fixed UTC epoch, observed from Keck.
        astrometry = {
            "ra": 26.0213,
            "dec": -15.9395,
            "pmra": -1721.05,
            "pmdec": 854.16,
            "px": 273.96,
            "epoch": Time("J2000.0").jd,
        }
        t = Time("2024-06-15T09:00:00.000", scale="utc")
        bc_vel, bjd_tdb = BarycentricCorrection._compute_barycorr(
            astrometry, t, BarycentricCorrection.KECK_LOCATION
        )
        # Physical sanity: BERV within Earth's orbital +/-30 km/s; BJD_TDB within
        # a few minutes of JD_UTC (clock + Romer light-travel delay).
        assert abs(bc_vel[0]) < 3.0e4
        assert abs(bjd_tdb[0] - t.utc.jd) < 0.01
        # Regression pins (1 cm/s, ~0.1 s): catch any frame/unit/sign rewiring,
        # well above barycorrpy/ephemeris/IERS numerical noise at the pinned version.
        assert bc_vel[0] == pytest.approx(24627.871206121636, abs=1e-2)
        assert bjd_tdb[0] == pytest.approx(2460476.873644211, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_flux_weighted_midpoint_times (output = 'expmeter' | 'orders' | 'ccds')
# ---------------------------------------------------------------------------


class TestFluxWeightedMidpointExpmeter:
    def test_uniform_flux_gives_geometric_midpoint(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        _, t_fwm = bc.compute_flux_weighted_midpoint_times(
            output="expmeter",
            interpolate=False,
            extrapolate=False,
            fix_expmeter_outliers=False,
        )
        _, t_mid, _ = bc._get_timestamps()
        np.testing.assert_allclose(np.mean(t_fwm.jd), np.mean(t_mid.jd), atol=1e-6)

    def test_output_shapes(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        w, t_fwm = bc.compute_flux_weighted_midpoint_times(
            output="expmeter",
            interpolate=False,
            extrapolate=False,
            fix_expmeter_outliers=False,
        )
        assert w.shape == (len(_WAVE_COLS),)
        assert len(t_fwm) == len(_WAVE_COLS)

    def test_small_negative_flux_preserved(self, synthetic_kpf2, monkeypatch):
        """Small negative fluctuations (background-subtraction noise near zero
        flux) are preserved with their real weight, not floored: flooring would
        bias the flux-weighted midpoint. Only a non-positive *channel total*
        raises (the meaningful guard); a single negative sample does not."""

        def mock_flux(self):
            w = np.array([float(c) for c in _WAVE_COLS])
            f = np.full((3, len(_WAVE_COLS)), 100.0)
            f[0, 0] = -20.0  # one small negative; channel total stays positive (+180)
            return w, f

        monkeypatch.setattr(BarycentricCorrection, "_get_normalized_flux", mock_flux)

        bc = BarycentricCorrection(synthetic_kpf2)
        _, t_mid, _ = bc._get_timestamps()
        _, t_fwm = bc.compute_flux_weighted_midpoint_times(
            output="expmeter",
            interpolate=False,
            extrapolate=False,
            fix_expmeter_outliers=False,
        )

        # Compare midpoints as seconds from the first reading: JD magnitudes
        # (~2.46e6) make any relative tolerance meaningless, so reduce to O(100 s)
        # offsets where absolute tolerances actually mean what they say.
        f0 = np.array([-20.0, 100.0, 100.0])  # channel 0 carries the negative sample
        floored = np.clip(f0, 0.0, None)
        ref = t_mid.jd[0]

        def offset_seconds(weights):
            return (np.sum(t_mid.jd * weights) / np.sum(weights) - ref) * 86400.0

        actual_s = (t_fwm.jd[0] - ref) * 86400.0
        expected_s = offset_seconds(f0)  # negative-preserving weighting
        floored_s = offset_seconds(floored)  # what flooring to zero would give

        assert np.isfinite(actual_s)
        # Matches the exact negative-preserving midpoint to <1 ms...
        np.testing.assert_allclose(actual_s, expected_s, atol=1e-3)
        # ...and is unambiguously not the floored result (the two differ by the
        # negative sample's ~20 s pull on the earliest reading).
        assert abs(actual_s - floored_s) > 1.0

    def test_zero_flux_channel_raises(self, synthetic_kpf2):
        """A channel with zero flux across all readings → 0/0 midpoint; fail loudly."""
        data = {
            "Date-Beg": [
                "2024-01-01T00:00:00.000",
                "2024-01-01T00:02:00.000",
                "2024-01-01T00:04:00.000",
            ],
            "Date-End": [
                "2024-01-01T00:01:00.000",
                "2024-01-01T00:03:00.000",
                "2024-01-01T00:05:00.000",
            ],
            "5000": [100.0, 100.0, 100.0],
            "5100": [0.0, 0.0, 0.0],
        }
        synthetic_kpf2.set_data("EXPMETER_SCI", Table(data))
        bc = BarycentricCorrection(synthetic_kpf2)
        with pytest.raises(ValueError, match="total flux"):
            bc.compute_flux_weighted_midpoint_times(
                output="expmeter",
                interpolate=False,
                extrapolate=False,
                fix_expmeter_outliers=False,
            )

    def test_interpolate_shifts_midpoint_with_front_weighted_flux(self, synthetic_kpf2):
        """Front-weighted flux: interpolation sees a bright sample at the
        first gap (~T+90s) that pulls the FWM strictly later."""
        data = {
            "Date-Beg": [
                "2024-01-01T00:00:00.000",
                "2024-01-01T00:02:00.000",
                "2024-01-01T00:04:00.000",
            ],
            "Date-End": [
                "2024-01-01T00:01:00.000",
                "2024-01-01T00:03:00.000",
                "2024-01-01T00:05:00.000",
            ],
            "5000": [1000.0, 1.0, 1.0],
            "5100": [1000.0, 1.0, 1.0],
        }
        synthetic_kpf2.set_data("EXPMETER_SCI", Table(data))
        bc = BarycentricCorrection(synthetic_kpf2)

        _, t_no = bc.compute_flux_weighted_midpoint_times(
            output="expmeter",
            interpolate=False,
            extrapolate=False,
            fix_expmeter_outliers=False,
        )
        _, t_yes = bc.compute_flux_weighted_midpoint_times(
            output="expmeter",
            interpolate=True,
            extrapolate=False,
            fix_expmeter_outliers=False,
        )
        # Shift should be sub-minute → tens of microdays, easily > 1e-9 days
        assert np.mean(t_yes.jd) > np.mean(t_no.jd) + 1e-7

    def test_extrapolate_shifts_midpoint_when_shutter_brackets_expmeter(
        self, synthetic_kpf2
    ):
        """DATE-BEG before t_beg[0] and DATE-END after t_end[-1] → extrapolated
        gap samples on both sides pull the FWM. Bright first reading + faint
        last → leading extrapolation dominates → midpoint earlier than with
        extrapolate=False."""
        data = {
            "Date-Beg": [
                "2024-01-01T00:02:00.000",  # readings inset from shutter
                "2024-01-01T00:02:30.000",
                "2024-01-01T00:03:00.000",
            ],
            "Date-End": [
                "2024-01-01T00:02:20.000",
                "2024-01-01T00:02:50.000",
                "2024-01-01T00:03:20.000",
            ],
            "5000": [1000.0, 1.0, 1.0],
            "5100": [1000.0, 1.0, 1.0],
        }
        synthetic_kpf2.set_data("EXPMETER_SCI", Table(data))
        # Shutter open 00:00:00 → 00:05:00; readings only 00:02:00–00:03:20
        synthetic_kpf2.headers["INSTRUMENT_HEADER"]["DATE-BEG"] = (
            "2024-01-01T00:00:00.000"
        )
        synthetic_kpf2.headers["INSTRUMENT_HEADER"]["DATE-END"] = (
            "2024-01-01T00:05:00.000"
        )

        bc = BarycentricCorrection(synthetic_kpf2)
        _, t_no = bc.compute_flux_weighted_midpoint_times(
            output="expmeter",
            interpolate=False,
            extrapolate=False,
            fix_expmeter_outliers=False,
        )
        _, t_yes = bc.compute_flux_weighted_midpoint_times(
            output="expmeter",
            interpolate=False,
            extrapolate=True,
            fix_expmeter_outliers=False,
        )
        # Bright first reading + 2-minute leading shutter gap → big leading
        # extrapolation pulls FWM clearly earlier than the no-extrap case.
        assert np.mean(t_yes.jd) < np.mean(t_no.jd) - 1e-7


class TestFluxWeightedMidpointOrders:
    _KWARGS = dict(interpolate=False, extrapolate=False, fix_expmeter_outliers=False)

    def test_output_shape(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        w, t = bc.compute_flux_weighted_midpoint_times(output="orders", **self._KWARGS)
        assert w.shape == (NORDER,)
        assert len(t) == NORDER

    def test_constant_wave_gives_constant_time(self, synthetic_kpf2):
        """SCI2_WAVE filled with 5000.0 → every order interpolates at 5000Å
        and gets the value from the first per-channel t_fwm."""
        bc = BarycentricCorrection(synthetic_kpf2)
        _, t = bc.compute_flux_weighted_midpoint_times(output="orders", **self._KWARGS)
        # All per-channel times are equal (uniform flux), so all per-order
        # interpolated times equal that shared value.
        assert np.std(t.jd) < 1e-12

    def test_orders_get_distinct_times_when_waves_vary(self, synthetic_kpf2):
        """Front-weighted flux + GREEN orders at 5000Å vs RED at 5100Å:
        GREEN orders should get an earlier midpoint than RED."""
        synthetic_kpf2.set_data(
            "GREEN_SCI2_WAVE", np.full((NORDER_GREEN, NCOL), 5000.0, dtype=np.float64)
        )
        synthetic_kpf2.set_data(
            "RED_SCI2_WAVE", np.full((NORDER_RED, NCOL), 5100.0, dtype=np.float64)
        )
        # Front-weighted at 5000Å, back-weighted at 5100Å
        data = {
            "Date-Beg": [
                "2024-01-01T00:00:00.000",
                "2024-01-01T00:02:00.000",
                "2024-01-01T00:04:00.000",
            ],
            "Date-End": [
                "2024-01-01T00:01:00.000",
                "2024-01-01T00:03:00.000",
                "2024-01-01T00:05:00.000",
            ],
            "5000": [1000.0, 1.0, 1.0],
            "5100": [1.0, 1.0, 1000.0],
        }
        synthetic_kpf2.set_data("EXPMETER_SCI", Table(data))

        bc = BarycentricCorrection(synthetic_kpf2)
        _, t = bc.compute_flux_weighted_midpoint_times(output="orders", **self._KWARGS)
        assert t.jd[:NORDER_GREEN].mean() < t.jd[NORDER_GREEN:].mean()

    def test_empty_sci2_wave_raises(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        synthetic_kpf2.data["TRACE3_WAVE"] = np.array([])
        with pytest.raises(KeyError, match="SCI2_WAVE"):
            bc.compute_flux_weighted_midpoint_times(output="orders", **self._KWARGS)


class TestFluxWeightedMidpointCcds:
    _KWARGS = dict(interpolate=False, extrapolate=False, fix_expmeter_outliers=False)

    def test_output_shape(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        w, t = bc.compute_flux_weighted_midpoint_times(output="ccds", **self._KWARGS)
        assert w.shape == (2,)
        assert len(t) == 2

    def test_ccd_values_equal_chip_means_with_uniform_weights(self, synthetic_kpf2):
        """With no SCI2_FLUX (uniform weights), the ccds output should equal the
        plain per-chip means of the orders output."""
        bc = BarycentricCorrection(synthetic_kpf2)
        _, t_orders = bc.compute_flux_weighted_midpoint_times(
            output="orders", **self._KWARGS
        )
        _, t_ccds = bc.compute_flux_weighted_midpoint_times(
            output="ccds", **self._KWARGS
        )
        np.testing.assert_allclose(
            t_ccds.jd,
            [t_orders.jd[:NORDER_GREEN].mean(), t_orders.jd[NORDER_GREEN:].mean()],
        )

    def test_flux_weighting_biases_ccd_time_toward_bright_orders(self, synthetic_kpf2):
        """A bright bluest GREEN order pulls the chip summary toward its
        (earlier) midpoint relative to the unweighted chip mean."""
        green_waves = np.linspace(5000.0, 5300.0, NORDER_GREEN, dtype=np.float64)
        synthetic_kpf2.set_data(
            "GREEN_SCI2_WAVE", np.repeat(green_waves[:, None], NCOL, axis=1)
        )
        synthetic_kpf2.set_data(
            "RED_SCI2_WAVE", np.full((NORDER_RED, NCOL), 5300.0, dtype=np.float64)
        )
        # Chromatic flux gradient: bluest channel early, reddest channel late.
        data = {
            "Date-Beg": [
                "2024-01-01T00:00:00.000",
                "2024-01-01T00:02:00.000",
                "2024-01-01T00:04:00.000",
            ],
            "Date-End": [
                "2024-01-01T00:01:00.000",
                "2024-01-01T00:03:00.000",
                "2024-01-01T00:05:00.000",
            ],
            "5000": [1000.0, 1.0, 1.0],
            "5300": [1.0, 1.0, 1000.0],
        }
        synthetic_kpf2.set_data("EXPMETER_SCI", Table(data))

        bc = BarycentricCorrection(synthetic_kpf2)
        _, t_orders = bc.compute_flux_weighted_midpoint_times(
            output="orders", **self._KWARGS
        )
        unweighted_green = t_orders.jd[:NORDER_GREEN].mean()

        # Make the bluest (earliest) GREEN order dominate the flux weighting.
        flux = np.ones((NORDER, NCOL), dtype=float)
        flux[0] = 1000.0
        synthetic_kpf2.set_data("SCI2_FLUX", flux)

        _, t_ccds = bc.compute_flux_weighted_midpoint_times(
            output="ccds", **self._KWARGS
        )
        assert t_ccds.jd[0] < unweighted_green

    def test_nan_flux_order_gets_zero_weight(self, synthetic_kpf2):
        """A failed-extraction order (all-NaN SCI2 flux) gets zero weight rather
        than poisoning its chip summary, so the ccds time stays finite."""
        flux = np.ones((NORDER, NCOL), dtype=float)
        flux[NORDER_GREEN] = np.nan  # first RED order failed extraction
        synthetic_kpf2.set_data("SCI2_FLUX", flux)

        bc = BarycentricCorrection(synthetic_kpf2)
        _, t_ccds = bc.compute_flux_weighted_midpoint_times(
            output="ccds", **self._KWARGS
        )
        assert np.all(np.isfinite(t_ccds.jd))

    def test_all_zero_weight_chip_raises(self, synthetic_kpf2):
        """A whole chip with zero SCI2 flux → undefined weighted mean; fail loudly."""
        flux = np.ones((NORDER, NCOL), dtype=float)
        flux[NORDER_GREEN:] = 0.0  # all RED orders have zero flux
        synthetic_kpf2.set_data("SCI2_FLUX", flux)
        bc = BarycentricCorrection(synthetic_kpf2)
        with pytest.raises(ValueError, match="weights are zero"):
            bc.compute_flux_weighted_midpoint_times(output="ccds", **self._KWARGS)

    def test_green_and_red_resolve_distinct_values(self, synthetic_kpf2):
        """With GREEN orders at 5000Å and RED orders at 5100Å plus a chromatic
        flux gradient, 'ccds' should report distinguishable times per chip."""
        synthetic_kpf2.set_data(
            "GREEN_SCI2_WAVE", np.full((NORDER_GREEN, NCOL), 5000.0, dtype=np.float64)
        )
        synthetic_kpf2.set_data(
            "RED_SCI2_WAVE", np.full((NORDER_RED, NCOL), 5100.0, dtype=np.float64)
        )
        # Front-weighted at 5000Å, back-weighted at 5100Å → distinct midpoints
        data = {
            "Date-Beg": [
                "2024-01-01T00:00:00.000",
                "2024-01-01T00:02:00.000",
                "2024-01-01T00:04:00.000",
            ],
            "Date-End": [
                "2024-01-01T00:01:00.000",
                "2024-01-01T00:03:00.000",
                "2024-01-01T00:05:00.000",
            ],
            "5000": [1000.0, 1.0, 1.0],
            "5100": [1.0, 1.0, 1000.0],
        }
        synthetic_kpf2.set_data("EXPMETER_SCI", Table(data))

        bc = BarycentricCorrection(synthetic_kpf2)
        w, t = bc.compute_flux_weighted_midpoint_times(output="ccds", **self._KWARGS)
        assert w[0] == 5000.0 and w[1] == 5100.0
        assert t.jd[0] < t.jd[1], (
            f"GREEN should be earlier than RED with front-then-back-weighted flux; "
            f"got GREEN={t.jd[0]}, RED={t.jd[1]}"
        )


class TestFluxWeightedMidpointFormat:
    def test_invalid_format_raises(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        with pytest.raises(ValueError, match="output"):
            bc.compute_flux_weighted_midpoint_times(output="bogus")

    def test_default_is_orders(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        w, t = bc.compute_flux_weighted_midpoint_times(
            interpolate=False,
            extrapolate=False,
            fix_expmeter_outliers=False,
        )
        assert w.shape == (NORDER,)


# ---------------------------------------------------------------------------
# _get_astrometry -- read the PRIMARY C*# cards, convert, sanitize
# ---------------------------------------------------------------------------


class TestGetAstrometry:
    """The C*# cards are AstroQuery's canonical record; this is the unit
    conversion into barycorrpy's argument set, plus the parallax sanitation the
    faithful-catalog-record contract pushes downstream."""

    def test_converts_cards_to_barycorrpy_units(self, synthetic_kpf2):
        primary = synthetic_kpf2.headers["PRIMARY"]
        primary["CPMR3"] = 0.5  # arcsec/yr
        primary["CPMD3"] = -0.3  # arcsec/yr

        astrometry = BarycentricCorrection(synthetic_kpf2)._get_astrometry()

        # RA sexagesimal hourangle / Dec sexagesimal deg -> deg
        assert astrometry["ra"] == pytest.approx(180.0)
        assert astrometry["dec"] == pytest.approx(0.0)
        # arcsec/yr -> mas/yr
        assert astrometry["pmra"] == pytest.approx(500.0)
        assert astrometry["pmdec"] == pytest.approx(-300.0)
        # parallax stays mas; epoch Julian years -> JD
        assert astrometry["px"] == pytest.approx(10.0)
        assert astrometry["epoch"] == pytest.approx(Time(2016.0, format="jyear").jd)

    def test_records_provenance_from_csrc(self, synthetic_kpf2):
        synthetic_kpf2.headers["PRIMARY"]["CSRC3"] = "simbad"
        bc = BarycentricCorrection(synthetic_kpf2)
        bc._get_astrometry()
        assert bc._astrometry_source == "simbad"

    def test_result_is_cached(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        assert bc._get_astrometry() is bc._get_astrometry()

    # NaN is not representable in a FITS header, so an unmeasured parallax arrives as
    # an absent card (covered below) or a blank one, never as NaN.
    @pytest.mark.parametrize("parallax", [-0.4, 0.0, ""])
    def test_unusable_parallax_becomes_zero(self, caplog, synthetic_kpf2, parallax):
        """AstroQuery persists catalog values faithfully -- a negative Gaia parallax
        is routine -- so BarycentricCorrection degrades to px=0 rather than
        computing a negative distance."""
        synthetic_kpf2.headers["PRIMARY"]["CPLX3"] = parallax
        bc = BarycentricCorrection(synthetic_kpf2)
        with caplog.at_level(logging.WARNING):
            astrometry = bc._get_astrometry()
        assert astrometry["px"] == 0.0
        assert "CPLX3" in caplog.text

    def test_missing_parallax_becomes_zero(self, caplog, synthetic_kpf2):
        """CPLX# is Required=No in the EPRV standard and is skipped from PRIMARY
        when the catalog measured none."""
        del synthetic_kpf2.headers["PRIMARY"]["CPLX3"]
        bc = BarycentricCorrection(synthetic_kpf2)
        with caplog.at_level(logging.WARNING):
            astrometry = bc._get_astrometry()
        assert astrometry["px"] == 0.0
        assert "CPLX3" in caplog.text

    @pytest.mark.parametrize("card", ["CRA3", "CDEC3", "CPMR3", "CPMD3", "CEPCH3"])
    def test_missing_position_card_raises(self, synthetic_kpf2, card):
        """The position block is all-or-nothing: AstroQuery's merge refuses to emit a
        canonical row without it, so a gap means the pipeline is misordered."""
        del synthetic_kpf2.headers["PRIMARY"][card]
        bc = BarycentricCorrection(synthetic_kpf2)
        with pytest.raises(ValueError, match=card):
            bc._get_astrometry()

    def test_blank_position_card_raises(self, synthetic_kpf2):
        synthetic_kpf2.headers["PRIMARY"]["CRA3"] = ""
        bc = BarycentricCorrection(synthetic_kpf2)
        with pytest.raises(ValueError, match="run AstroQuery"):
            bc._get_astrometry()


# ---------------------------------------------------------------------------
# perform() -- barycorrpy stubbed
# ---------------------------------------------------------------------------


class TestPerform:
    DELTA_RV_MPS = 30000.0  # constant 30 km/s shift across all orders

    @pytest.fixture
    def bc_monkeypatched(self, synthetic_kpf2, monkeypatch):
        """BarycentricCorrection with barycorrpy stubbed."""

        def mock_compute(astrometry, obs_times, location, rv_mps=0.0):
            n = len(np.atleast_1d(obs_times.jd))
            bc_vel = np.full(n, TestPerform.DELTA_RV_MPS, dtype=float)
            # BJD = JD + 500s (light-travel approx); same for every order in this stub
            bjd_tdb = np.atleast_1d(obs_times.jd) + 500.0 / 86400.0
            return bc_vel, bjd_tdb

        def passthrough(f, kernel_size=5):
            return f.copy()

        monkeypatch.setattr(
            BarycentricCorrection, "_compute_barycorr", staticmethod(mock_compute)
        )
        # Stub _fix_expmeter_outliers: the 3×4 uniform-flux fixture triggers a
        # degenerate triangulation inside scipy.griddata. Filter itself is
        # exercised by TestFixExpmeterOutliers with a noisy 60×20 array.
        monkeypatch.setattr(
            BarycentricCorrection, "_fix_expmeter_outliers", staticmethod(passthrough)
        )
        return BarycentricCorrection(synthetic_kpf2)

    def test_returns_same_object(self, bc_monkeypatched):
        original = bc_monkeypatched.l2_obj
        result = bc_monkeypatched.perform()
        assert result is original
        assert isinstance(result, KPF2)

    def test_bjd_tdb_extension_populated(self, bc_monkeypatched):
        kpf2 = bc_monkeypatched.perform()
        bjd = np.asarray(kpf2.data["BJD_TDB"])
        assert bjd.shape == (NORDER,)
        assert np.all(np.isfinite(bjd))
        assert bjd.dtype == np.float64  # EPRV: BJD_TDB is 64-bit

    def test_barycorr_kms_extension_populated(self, bc_monkeypatched):
        kpf2 = bc_monkeypatched.perform()
        kms = np.asarray(kpf2.data["BARYCORR_KMS"])
        assert kms.shape == (NORDER,)
        np.testing.assert_allclose(kms, TestPerform.DELTA_RV_MPS / 1000.0)

    def test_barycorr_z_extension_populated(self, bc_monkeypatched):
        from astropy.constants import c

        kpf2 = bc_monkeypatched.perform()
        z = np.asarray(kpf2.data["BARYCORR_Z"])
        assert z.shape == (NORDER,)
        # All orders share the same delta_rv → all z values identical
        assert np.std(z) < 1e-12
        # BARYCORR_Z is the small redshift z (~ v/c), same sign as the velocity:
        # +30 km/s (receding) -> small positive z, not the Doppler factor (~1).
        assert z[0] == pytest.approx(
            TestPerform.DELTA_RV_MPS / c.to("m/s").value, rel=1e-4
        )
        assert 0.0 < z[0] < 1e-3

    def test_barycorr_and_bjd_are_float64(self, bc_monkeypatched):
        # Correction arrays feed the RV redshift; float64 protects RV precision.
        kpf2 = bc_monkeypatched.perform()
        assert_dtype(kpf2.data["BARYCORR_KMS"], BARYCORR, "BARYCORR_KMS")
        assert_dtype(kpf2.data["BARYCORR_Z"], BARYCORR, "BARYCORR_Z")
        assert_dtype(kpf2.data["BJD_TDB"], BJD, "BJD_TDB")

    def test_wave_arrays_untouched(self, bc_monkeypatched):
        """BarycentricCorrection should track but not apply: WAVE arrays unchanged."""
        kpf2 = bc_monkeypatched.l2_obj
        orig = {
            f"{chip}_{fiber}_WAVE": kpf2.data[f"{chip}_{fiber}_WAVE"].copy()
            for chip in ["GREEN", "RED"]
            for fiber in ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]
        }

        bc_monkeypatched.perform()

        for key, before in orig.items():
            np.testing.assert_array_equal(
                kpf2.data[key],
                before,
                err_msg=f"{key} should be unmodified",
            )

    def test_per_ccd_primary_keywords(self, bc_monkeypatched):
        kpf2 = bc_monkeypatched.perform()
        bjd = kpf2.headers["BJD_TDB"]
        kms = kpf2.headers["BARYCORR_KMS"]
        z = kpf2.headers["BARYCORR_Z"]
        for key in ("CCD1BJD", "CCD2BJD"):
            assert key in bjd, f"{key} missing from BJD_TDB"
        for key in ("CCD1BKMS", "CCD2BKMS"):
            assert key in kms, f"{key} missing from BARYCORR_KMS"
        for key in ("CCD1BZ", "CCD2BZ"):
            assert key in z, f"{key} missing from BARYCORR_Z"

        # All orders had the same delta_rv → green and red means are equal
        np.testing.assert_allclose(kms.get("CCD1BKMS"), kms.get("CCD2BKMS"))
        np.testing.assert_allclose(z.get("CCD1BZ"), z.get("CCD2BZ"))

    def test_ctype1_axis_label(self, bc_monkeypatched):
        # These are 1-D per-order arrays: CTYPE1 names the order axis (registered
        # content). CTYPE2 is N/A (no second axis).
        kpf2 = bc_monkeypatched.perform()
        for ext in ("BJD_TDB", "BARYCORR_KMS", "BARYCORR_Z"):
            hdr = kpf2.headers[ext]
            assert hdr["CTYPE1"] == "Order-N", ext
            assert "CTYPE2" not in hdr, ext

    def test_receipt_entry_added(self, bc_monkeypatched):
        bc_monkeypatched.perform()
        modules = bc_monkeypatched.l2_obj.receipt["FUNCTION"].values
        assert "barycentric_correction" in modules

    def test_crv_converted_km_to_m_and_passed_through(
        self, synthetic_kpf2, monkeypatch
    ):
        """CRV3 (km/s) should arrive at _compute_barycorr as m/s."""
        synthetic_kpf2.headers["PRIMARY"]["CRV3"] = 81.87  # km/s
        captured = {}

        def mock_compute(astrometry, obs_times, location, rv_mps=0.0):
            captured["rv_mps"] = rv_mps
            n = len(np.atleast_1d(obs_times.jd))
            return np.zeros(n), np.atleast_1d(obs_times.jd)

        def passthrough(f, kernel_size=5):
            return f.copy()

        monkeypatch.setattr(
            BarycentricCorrection, "_compute_barycorr", staticmethod(mock_compute)
        )
        monkeypatch.setattr(
            BarycentricCorrection, "_fix_expmeter_outliers", staticmethod(passthrough)
        )

        BarycentricCorrection(synthetic_kpf2).perform()
        assert captured["rv_mps"] == pytest.approx(81870.0)

    def test_missing_crv_defaults_to_zero(self, bc_monkeypatched, monkeypatch):
        """No CRV3 on PRIMARY (no catalog rv, no TARGRADV) → rv_mps=0."""
        # synthetic_kpf2 fixture does not set CRV3
        captured = {}

        def mock_compute(astrometry, obs_times, location, rv_mps=0.0):
            captured["rv_mps"] = rv_mps
            n = len(np.atleast_1d(obs_times.jd))
            return np.full(n, TestPerform.DELTA_RV_MPS), np.atleast_1d(obs_times.jd)

        monkeypatch.setattr(
            BarycentricCorrection, "_compute_barycorr", staticmethod(mock_compute)
        )
        bc_monkeypatched.perform()
        assert captured["rv_mps"] == 0.0

    def test_state_populated(self, bc_monkeypatched):
        for attr in ("_ccd_bjd", "_ccd_kms", "_ccd_z"):
            assert getattr(bc_monkeypatched, attr) is None
        kpf2 = bc_monkeypatched.perform()
        assert bc_monkeypatched._astrometry_source == "gaia"
        for key in ("BJD_TDB", "BARYCORR_KMS", "BARYCORR_Z"):
            assert len(kpf2.data[key]) == NORDER
        for attr in ("_ccd_bjd", "_ccd_kms", "_ccd_z"):
            assert len(getattr(bc_monkeypatched, attr)) == 2

    def test_real_outlier_filter_runs_end_to_end(self, synthetic_kpf2, monkeypatch):
        """Exercise fix_expmeter_outliers=True through perform() with a
        noisy expmeter table (3×4 uniform fixture would crash griddata)."""
        rng = np.random.default_rng(0)
        ntime, nwave = 60, 20
        wave_cols = [str(5000.0 + 10 * i) for i in range(nwave)]
        data = {
            "Date-Beg": [f"2024-01-01T00:{m:02d}:00.000" for m in range(ntime)],
            "Date-End": [f"2024-01-01T00:{m:02d}:30.000" for m in range(ntime)],
        }
        for wc in wave_cols:
            data[wc] = rng.normal(100.0, 2.0, ntime).astype(float)
        # Inject a clear outlier so the filter has work to do
        data[wave_cols[5]][30] = 1e6
        synthetic_kpf2.set_data("EXPMETER_SCI", Table(data))
        synthetic_kpf2.headers["INSTRUMENT_HEADER"]["DATE-END"] = (
            "2024-01-01T01:00:00.000"
        )

        def mock_compute(astrometry, obs_times, location, rv_mps=0.0):
            n = len(np.atleast_1d(obs_times.jd))
            return np.full(n, 1000.0), np.atleast_1d(obs_times.jd)

        monkeypatch.setattr(
            BarycentricCorrection, "_compute_barycorr", staticmethod(mock_compute)
        )

        # No monkeypatch on _fix_expmeter_outliers -- real filter runs.
        kpf2 = BarycentricCorrection(synthetic_kpf2).perform(fix_expmeter_outliers=True)
        assert np.all(np.isfinite(np.asarray(kpf2.data["BJD_TDB"])))


# ---------------------------------------------------------------------------
# Constructor + missing-header error paths
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_invalid_config_type_raises(self, synthetic_kpf2):
        with pytest.raises(TypeError, match="None, dict, or ConfigHandler"):
            BarycentricCorrection(synthetic_kpf2, config="not-a-config")


class TestMissingHeader:
    """perform() should fail loudly when required header keys are absent."""

    def test_missing_catalog_cards_raises(self, synthetic_kpf2, monkeypatch):
        # Stub _fix_expmeter_outliers so we don't hit the griddata degeneracy
        # before reaching the catalog-card lookup.
        def passthrough(f, kernel_size=5):
            return f.copy()

        monkeypatch.setattr(
            BarycentricCorrection, "_fix_expmeter_outliers", staticmethod(passthrough)
        )

        del synthetic_kpf2.headers["PRIMARY"]["CRA3"]
        bc = BarycentricCorrection(synthetic_kpf2)
        with pytest.raises(ValueError, match="CRA3"):
            bc.perform()

    def test_missing_date_beg_raises_when_extrapolating(self, synthetic_kpf2):
        del synthetic_kpf2.headers["INSTRUMENT_HEADER"]["DATE-BEG"]
        bc = BarycentricCorrection(synthetic_kpf2)
        with pytest.raises(KeyError, match="DATE-BEG"):
            bc.compute_flux_weighted_midpoint_times(
                output="expmeter",
                interpolate=False,
                extrapolate=True,
                fix_expmeter_outliers=False,
            )

    def test_missing_date_beg_ok_without_extrapolate(self, synthetic_kpf2):
        """DATE-BEG is only needed when extrapolate=True."""
        del synthetic_kpf2.headers["INSTRUMENT_HEADER"]["DATE-BEG"]
        bc = BarycentricCorrection(synthetic_kpf2)
        # Should not raise:
        bc.compute_flux_weighted_midpoint_times(
            output="expmeter",
            interpolate=False,
            extrapolate=False,
            fix_expmeter_outliers=False,
        )
