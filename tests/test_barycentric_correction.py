"""
Tests for the BarycentricCorrection module (KPF2 → KPF2).

Static-method unit tests require no fixtures. Integration tests use a
synthetic KPF2 with a small EXPMETER_SCI table and populated SCI2_WAVE.
Gaia and barycorrpy calls are stubbed via monkeypatching.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from kpfpipe import DETECTOR
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.modules.barycentric_correction import BarycentricCorrection
from kpfpipe.utils.validation import strictly_increasing

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

    SCI2_WAVE is filled with 5000.0 — order_center = 5000 for every order,
    so np.interp returns the per-channel value at 5000Å (i.e. the first
    PHOTON_JD) for every order. Tests that need per-order variation
    populate SCI2_WAVE themselves.
    """
    kpf2 = KPF2()

    # KPF-native keywords live in INSTRUMENT_HEADER on L2 (preserved L1 PRIMARY).
    kpf2.headers["INSTRUMENT_HEADER"]["DATE-BEG"] = f"{_T0}00:00:00.000"
    kpf2.headers["INSTRUMENT_HEADER"]["DATE-END"] = f"{_T0}00:05:00.000"
    kpf2.headers["INSTRUMENT_HEADER"]["GAIAID"] = "DR3 1234567890123456789"

    kpf2.set_data("EXPMETER_SCI", _make_expmeter_table())

    for chip in ["GREEN", "RED"]:
        n = NORDER_GREEN if chip == "GREEN" else NORDER_RED
        for fiber in ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]:
            kpf2.set_data(
                f"{chip}_{fiber}_WAVE", np.full((n, NCOL), 5000.0, dtype=np.float32)
            )

    return kpf2


def _fake_skycoord():
    """Deterministic ICRS SkyCoord with realistic proper motion / parallax."""
    return SkyCoord(
        ra=180.0 * u.deg,
        dec=0.0 * u.deg,
        pm_ra_cosdec=0.0 * u.mas / u.yr,
        pm_dec=0.0 * u.mas / u.yr,
        distance=100.0 * u.pc,
        obstime=Time(2016.0, format="jyear"),
        frame="icrs",
    )


# ---------------------------------------------------------------------------
# Static helpers — unchanged behavior across the split
# ---------------------------------------------------------------------------


class TestStrictlyIncreasing:
    def _make_time(self, seconds):
        jd0 = 2460310.5
        return Time(jd0 + np.array(seconds) / 86400.0, format="jd", scale="utc")

    def test_increasing(self):
        assert strictly_increasing(self._make_time([0, 1, 2, 3]).jd) is True

    def test_constant_fails(self):
        assert strictly_increasing(self._make_time([0, 1, 1, 2]).jd) is False

    def test_decreasing_fails(self):
        assert strictly_increasing(self._make_time([3, 2, 1, 0]).jd) is False

    def test_single_element(self):
        assert strictly_increasing(self._make_time([0]).jd) is True


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

    def test_negative_flux_raises(self, synthetic_kpf2, monkeypatch):
        def mock_flux(self):
            w = np.array([float(c) for c in _WAVE_COLS])
            return w, np.full((3, len(_WAVE_COLS)), -1.0)

        monkeypatch.setattr(BarycentricCorrection, "_get_normalized_flux", mock_flux)

        bc = BarycentricCorrection(synthetic_kpf2)
        with pytest.raises(ValueError, match="negative"):
            bc.compute_flux_weighted_midpoint_times(output="expmeter")

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
            "GREEN_SCI2_WAVE", np.full((NORDER_GREEN, NCOL), 5000.0, dtype=np.float32)
        )
        synthetic_kpf2.set_data(
            "RED_SCI2_WAVE", np.full((NORDER_RED, NCOL), 5100.0, dtype=np.float32)
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
        green_waves = np.linspace(5000.0, 5300.0, NORDER_GREEN, dtype=np.float32)
        synthetic_kpf2.set_data(
            "GREEN_SCI2_WAVE", np.repeat(green_waves[:, None], NCOL, axis=1)
        )
        synthetic_kpf2.set_data(
            "RED_SCI2_WAVE", np.full((NORDER_RED, NCOL), 5300.0, dtype=np.float32)
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
            "GREEN_SCI2_WAVE", np.full((NORDER_GREEN, NCOL), 5000.0, dtype=np.float32)
        )
        synthetic_kpf2.set_data(
            "RED_SCI2_WAVE", np.full((NORDER_RED, NCOL), 5100.0, dtype=np.float32)
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
# _gaia_astrometry input validation
# ---------------------------------------------------------------------------


class TestQueryGaiaValidation:
    """Reject non-numeric source IDs before they hit the ADQL query string."""

    @pytest.mark.parametrize(
        "bad_id",
        [
            "foo",  # not a number at all
            "12345 OR 1=1",  # injection attempt
            "12345; DROP TABLE",
            "",
            "12.34",  # decimal
            "-12345",  # negative
        ],
    )
    def test_non_numeric_raises(self, synthetic_kpf2, bad_id):
        synthetic_kpf2.headers["INSTRUMENT_HEADER"]["GAIAID"] = bad_id
        bc = BarycentricCorrection(synthetic_kpf2)
        with pytest.raises(ValueError, match="all digits"):
            bc._gaia_astrometry()


# ---------------------------------------------------------------------------
# _get_skycoord — Gaia first, WMKO header fallback, else raise
# ---------------------------------------------------------------------------


class TestAstrometryResolution:
    @staticmethod
    def _add_wmko_keys(kpf2):
        inst = kpf2.headers["INSTRUMENT_HEADER"]
        inst["TARGRA"] = "10:59:27.50"
        inst["TARGDEC"] = "+40:25:50.0"
        inst["TARGEPOC"] = 2000.0
        inst["TARGFRAM"] = "FK5"
        inst["TARGPLAX"] = 72.0  # mas
        inst["TARGPMRA"] = 0.0  # time-s/yr
        inst["TARGPMDC"] = 0.0  # arcsec/yr

    def test_gaia_used_when_enabled(self, synthetic_kpf2, monkeypatch):
        sentinel = _fake_skycoord()
        monkeypatch.setattr(
            BarycentricCorrection, "_gaia_astrometry", lambda self: sentinel
        )
        # defaults: use_gaia_astrometry=True, use_wmko_fallback=False
        assert BarycentricCorrection(synthetic_kpf2)._get_skycoord() is sentinel

    def test_falls_back_to_wmko_on_gaia_error(self, synthetic_kpf2, monkeypatch):
        self._add_wmko_keys(synthetic_kpf2)

        def boom(self):
            raise ConnectionError("gaia server down")

        monkeypatch.setattr(BarycentricCorrection, "_gaia_astrometry", boom)

        bc = BarycentricCorrection(synthetic_kpf2, config={"use_wmko_fallback": True})
        with pytest.warns(UserWarning, match="ConnectionError"):
            sc = bc._get_skycoord()
        assert sc.icrs.distance.to(u.pc).value == pytest.approx(1e3 / 72.0)

    def test_wmko_only_when_gaia_disabled(self, synthetic_kpf2, monkeypatch):
        self._add_wmko_keys(synthetic_kpf2)

        def fail(self):
            raise AssertionError("Gaia should not be queried when disabled")

        monkeypatch.setattr(BarycentricCorrection, "_gaia_astrometry", fail)

        bc = BarycentricCorrection(
            synthetic_kpf2,
            config={"use_gaia_astrometry": False, "use_wmko_fallback": True},
        )
        sc = bc._get_skycoord()
        assert sc.ra.deg == pytest.approx(164.8645833)

    def test_raises_and_surfaces_gaia_error_when_both_unavailable(
        self, synthetic_kpf2, monkeypatch
    ):
        def boom(self):
            raise ValueError("Gaia source_id must be all digits; got 'foo'")

        monkeypatch.setattr(BarycentricCorrection, "_gaia_astrometry", boom)

        bc = BarycentricCorrection(synthetic_kpf2)  # wmko fallback off by default
        with pytest.raises(ValueError, match="all digits"):
            bc._get_skycoord()

    def test_wmko_proper_motion_and_parallax_units(self, synthetic_kpf2):
        self._add_wmko_keys(synthetic_kpf2)
        inst = synthetic_kpf2.headers["INSTRUMENT_HEADER"]
        inst["TARGPMRA"] = 0.01  # time-s/yr
        inst["TARGPMDC"] = -0.5  # arcsec/yr

        sc = BarycentricCorrection(synthetic_kpf2)._wmko_astrometry()
        expected_pmra = 0.01 * 15.0 * np.cos(sc.dec.rad) * 1e3  # -> mas/yr (cosdec)
        assert sc.pm_ra_cosdec.to(u.mas / u.yr).value == pytest.approx(expected_pmra)
        assert sc.pm_dec.to(u.mas / u.yr).value == pytest.approx(-500.0)
        assert sc.distance.to(u.pc).value == pytest.approx(1e3 / 72.0)


# ---------------------------------------------------------------------------
# perform() — Gaia and barycorrpy stubbed
# ---------------------------------------------------------------------------


class TestPerform:
    DELTA_RV_MPS = 30000.0  # constant 30 km/s shift across all orders

    @pytest.fixture
    def bc_monkeypatched(self, synthetic_kpf2, monkeypatch):
        """BarycentricCorrection with Gaia + barycorrpy stubbed."""

        def mock_query(self):
            return _fake_skycoord()

        def mock_compute(skycoord, obs_times, location, rv_mps=0.0):
            n = len(np.atleast_1d(obs_times.jd))
            bc_vel = np.full(n, TestPerform.DELTA_RV_MPS, dtype=float)
            # BJD = JD + 500s (light-travel approx); same for every order in this stub
            bjd_tdb = np.atleast_1d(obs_times.jd) + 500.0 / 86400.0
            return bc_vel, bjd_tdb

        def passthrough(f, kernel_size=5):
            return f.copy()

        monkeypatch.setattr(BarycentricCorrection, "_gaia_astrometry", mock_query)
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

    def test_returns_kpf2(self, bc_monkeypatched):
        assert isinstance(bc_monkeypatched.perform(), KPF2)

    def test_returns_same_object(self, bc_monkeypatched):
        original = bc_monkeypatched.l2_obj
        assert bc_monkeypatched.perform() is original

    def test_bjd_tdb_extension_populated(self, bc_monkeypatched):
        kpf2 = bc_monkeypatched.perform()
        bjd = np.asarray(kpf2.data["BJD_TDB"])
        assert bjd.shape == (NORDER,)
        assert np.all(np.isfinite(bjd))

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

    def test_per_ccd_instrument_header_keywords(self, bc_monkeypatched):
        kpf2 = bc_monkeypatched.perform()
        inst = kpf2.headers["INSTRUMENT_HEADER"]
        for key in ["CCD1BJD", "CCD1BKMS", "CCD1BZ", "CCD2BJD", "CCD2BKMS", "CCD2BZ"]:
            assert key in inst, f"{key} missing from INSTRUMENT_HEADER"

        def _v(k):
            x = inst[k]
            return x[0] if isinstance(x, tuple) else x

        # All orders had the same delta_rv → green and red means are equal
        np.testing.assert_allclose(_v("CCD1BKMS"), _v("CCD2BKMS"))
        np.testing.assert_allclose(_v("CCD1BZ"), _v("CCD2BZ"))

    def test_receipt_entry_added(self, bc_monkeypatched):
        bc_monkeypatched.perform()
        modules = bc_monkeypatched.l2_obj.receipt["Module_Name"].values
        assert "barycentric_correction" in modules

    def test_targradv_converted_km_to_m_and_passed_through(
        self, synthetic_kpf2, monkeypatch
    ):
        """TARGRADV (km/s) should arrive at _compute_barycorr as m/s."""
        synthetic_kpf2.headers["INSTRUMENT_HEADER"]["TARGRADV"] = 81.87  # km/s
        captured = {}

        def mock_query(self):
            return _fake_skycoord()

        def mock_compute(skycoord, obs_times, location, rv_mps=0.0):
            captured["rv_mps"] = rv_mps
            n = len(np.atleast_1d(obs_times.jd))
            return np.zeros(n), np.atleast_1d(obs_times.jd)

        def passthrough(f, kernel_size=5):
            return f.copy()

        monkeypatch.setattr(BarycentricCorrection, "_gaia_astrometry", mock_query)
        monkeypatch.setattr(
            BarycentricCorrection, "_compute_barycorr", staticmethod(mock_compute)
        )
        monkeypatch.setattr(
            BarycentricCorrection, "_fix_expmeter_outliers", staticmethod(passthrough)
        )

        BarycentricCorrection(synthetic_kpf2).perform()
        assert captured["rv_mps"] == pytest.approx(81870.0)

    def test_missing_targradv_defaults_to_zero(self, bc_monkeypatched, monkeypatch):
        """No TARGRADV in INSTRUMENT_HEADER → rv_mps=0 passed through."""
        # synthetic_kpf2 fixture does not set TARGRADV
        captured = {}

        def mock_compute(skycoord, obs_times, location, rv_mps=0.0):
            captured["rv_mps"] = rv_mps
            n = len(np.atleast_1d(obs_times.jd))
            return np.full(n, TestPerform.DELTA_RV_MPS), np.atleast_1d(obs_times.jd)

        monkeypatch.setattr(
            BarycentricCorrection, "_compute_barycorr", staticmethod(mock_compute)
        )
        bc_monkeypatched.perform()
        assert captured["rv_mps"] == 0.0

    def test_results_populated(self, bc_monkeypatched):
        assert bc_monkeypatched._results is None
        bc_monkeypatched.perform()
        results = bc_monkeypatched._results
        assert set(results.keys()) == {
            "bjd_tdb",
            "bary_kms",
            "bary_z",
            "ccd_bjd",
            "ccd_kms",
            "ccd_z",
            "astrometry_source",
        }
        assert results["astrometry_source"] == "Gaia DR3"
        for key in ("bjd_tdb", "bary_kms", "bary_z"):
            assert len(results[key]) == NORDER
        for key in ("ccd_bjd", "ccd_kms", "ccd_z"):
            assert len(results[key]) == 2

    def test_records_gaia_provenance(self, bc_monkeypatched):
        kpf2 = bc_monkeypatched.perform()
        assert kpf2.headers["INSTRUMENT_HEADER"]["ASTRSRC"] == "Gaia DR3"

    def test_perform_falls_back_and_records_wmko_provenance(
        self, synthetic_kpf2, monkeypatch
    ):
        TestAstrometryResolution._add_wmko_keys(synthetic_kpf2)

        def boom(self):
            raise ConnectionError("gaia down")

        def mock_compute(skycoord, obs_times, location, rv_mps=0.0):
            n = len(np.atleast_1d(obs_times.jd))
            return np.zeros(n), np.atleast_1d(obs_times.jd)

        monkeypatch.setattr(BarycentricCorrection, "_gaia_astrometry", boom)
        monkeypatch.setattr(
            BarycentricCorrection, "_compute_barycorr", staticmethod(mock_compute)
        )
        monkeypatch.setattr(
            BarycentricCorrection,
            "_fix_expmeter_outliers",
            staticmethod(lambda f, kernel_size=5: f.copy()),
        )

        bc = BarycentricCorrection(synthetic_kpf2)  # defaults: gaia on, wmko off
        with pytest.warns(UserWarning, match="ConnectionError"):
            kpf2 = bc.perform(
                use_wmko_fallback=True
            )  # override the toggle for this call

        assert kpf2.headers["INSTRUMENT_HEADER"]["ASTRSRC"] == "WMKO header"
        assert bc._results["astrometry_source"] == "WMKO header"

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

        def mock_query(self):
            return _fake_skycoord()

        def mock_compute(skycoord, obs_times, location, rv_mps=0.0):
            n = len(np.atleast_1d(obs_times.jd))
            return np.full(n, 1000.0), np.atleast_1d(obs_times.jd)

        monkeypatch.setattr(BarycentricCorrection, "_gaia_astrometry", mock_query)
        monkeypatch.setattr(
            BarycentricCorrection, "_compute_barycorr", staticmethod(mock_compute)
        )

        # No monkeypatch on _fix_expmeter_outliers — real filter runs.
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
    """perform() should fail loudly when required INSTRUMENT_HEADER keys are absent."""

    def test_missing_gaiaid_raises(self, synthetic_kpf2, monkeypatch):
        # Stub _fix_expmeter_outliers so we don't hit the griddata degeneracy
        # before reaching the GAIAID lookup.
        def passthrough(f, kernel_size=5):
            return f.copy()

        monkeypatch.setattr(
            BarycentricCorrection, "_fix_expmeter_outliers", staticmethod(passthrough)
        )

        del synthetic_kpf2.headers["INSTRUMENT_HEADER"]["GAIAID"]
        bc = BarycentricCorrection(synthetic_kpf2)
        # With the WMKO fallback disabled (default), the Gaia-side KeyError is
        # surfaced inside the "no target astrometry" error.
        with pytest.raises(ValueError, match="GAIAID"):
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
