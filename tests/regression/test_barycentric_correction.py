"""Tests for the BarycentricCorrection module (KPF2 -> KPF2).

Static-method unit tests require no fixtures. Integration tests use a synthetic
KPF2 with a small EXPMETER_SCI table, populated SCI2_WAVE, and the SCI2 catalog
cards on PRIMARY; barycorrpy is stubbed except in TestComputeBarycorrReference.
"""

import logging

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Distance, SkyCoord
from astropy.table import Table
from astropy.time import Time

from kpfpipe import DETECTOR
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.modules.barycentric_correction import BarycentricCorrection
from kpfpipe.utils.astro import KECK_LOCATION

from ._dtype_policy import BARYCORR, BJD, assert_dtype

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
NORDER = DETECTOR["numorder"]
NCOL = 50  # reduced column count for speed


# ---------------------------------------------------------------------------
# Synthetic KPF2 fixture
# ---------------------------------------------------------------------------

# Exposure meter: 3 readings of 60 s at 00:00, 00:02 and 00:04, separated by 60 s
# gaps. The shutter (DATE-BEG/DATE-END) is open 00:00:00 to 00:05:00, so it
# exactly brackets the readings unless a test says otherwise.
_T0 = "2024-01-01T"
_WAVE_COLS = ["5000", "5100", "5200", "5300"]  # 100 Angstrom spacing = dispersion
_FLUX_VALUE = 100.0  # uniform ADU per reading

# SCI2 catalog cards as KPF0.to_kpf1 writes them: ICRS, RA/Dec sexagesimal (hourangle
# / deg), PM arcsec/yr, parallax mas, epoch Julian years. RA 180 deg, Dec 0, 100 pc.
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
    "CRV3": 0.0,  # present so the module does not take its warn-and-default path
}


def _expmeter_table(data):
    """EXPMETER_SCI table from Date-Beg/Date-End plus flux columns.

    The module prefers the shutter-corrected columns, which real EXPMETER_SCI
    always carries; mirroring the uncorrected ones keeps expected times unchanged.
    """
    return Table(
        {**data, "Date-Beg-Corr": data["Date-Beg"], "Date-End-Corr": data["Date-End"]}
    )


def _make_expmeter_table():
    data = {
        "Date-Beg": [f"{_T0}00:00:00.000", f"{_T0}00:02:00.000", f"{_T0}00:04:00.000"],
        "Date-End": [f"{_T0}00:01:00.000", f"{_T0}00:03:00.000", f"{_T0}00:05:00.000"],
    }
    for wc in _WAVE_COLS:
        data[wc] = np.full(3, _FLUX_VALUE)
    return _expmeter_table(data)


@pytest.fixture
def synthetic_kpf2():
    """KPF2 with synthetic EXPMETER_SCI and wavelength arrays.

    Every order's center is 5000.0, so all orders interpolate to the same
    per-channel time. Tests needing per-order variation set SCI2_WAVE themselves.
    """
    kpf2 = KPF2()

    # KPF-native keywords live in INSTRUMENT_HEADER on L2 (the preserved L1 PRIMARY).
    kpf2.headers["INSTRUMENT_HEADER"]["DATE-BEG"] = f"{_T0}00:00:00.000"
    kpf2.headers["INSTRUMENT_HEADER"]["DATE-END"] = f"{_T0}00:05:00.000"

    # Target astrometry reaches L2 on the PRIMARY C*# cards; the module never queries.
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
# Static helpers
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

        with pytest.raises(ValueError, match="t0 must be before"):
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
        bad_table = _expmeter_table(
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

    def test_gain_and_dispersion_applied(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        _, f = bc._get_normalized_flux()
        np.testing.assert_allclose(f, _FLUX_VALUE * 1.48424 / 100.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# _compute_barycorr (staticmethod) -- the one non-stubbed regression pin
# ---------------------------------------------------------------------------


class TestComputeBarycorrReference:
    """The real barycorrpy handoff, pinned to a golden value.

    Every perform() test stubs _compute_barycorr, so this is the only guard on
    the real wiring: ICRS ra/dec/pm/parallax, the J2000 epoch, UTC->JD, the Keck
    site, and the m/s units and sign. Golden values are pinned to
    barycorrpy==0.4.4.

    Unmarked: it reads no truth frames, which is what ``slow`` means here. It
    carried the marker as a hedge against a cold-cache ephemeris download, which
    tests/conftest.py now allows through rather than blocking.
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
            astrometry, t, KECK_LOCATION
        )
        # Physical sanity: BERV within Earth's orbital +/-30 km/s; BJD_TDB within
        # a few minutes of JD_UTC (clock + Romer light-travel delay).
        assert abs(bc_vel[0]) < 3.0e4
        assert abs(bjd_tdb[0] - t.utc.jd) < 0.01
        # Pins at 1 cm/s and ~0.1 s: tight enough to catch a frame/unit/sign
        # rewiring, loose enough to clear ephemeris/IERS noise at the pinned version.
        assert bc_vel[0] == pytest.approx(24627.84694194215, abs=1e-2)
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
            fix_outliers=False,
        )
        _, t_mid, _ = bc._get_timestamps()
        np.testing.assert_allclose(np.mean(t_fwm.jd), np.mean(t_mid.jd), atol=1e-6)

    def test_output_shapes(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        w, t_fwm = bc.compute_flux_weighted_midpoint_times(
            output="expmeter",
            interpolate=False,
            extrapolate=False,
            fix_outliers=False,
        )
        assert w.shape == (len(_WAVE_COLS),)
        assert len(t_fwm) == len(_WAVE_COLS)

    def test_small_negative_flux_preserved(self, synthetic_kpf2, monkeypatch):
        # Small negatives are background-subtraction noise near zero flux;
        # flooring them to zero would bias the flux-weighted midpoint. Only a
        # nonpositive channel *total* raises.
        def mock_flux(self):
            w = np.array([float(c) for c in _WAVE_COLS])
            f = np.full((3, len(_WAVE_COLS)), 100.0)
            f[0, 0] = -20.0  # channel total stays positive (+180)
            return w, f

        monkeypatch.setattr(BarycentricCorrection, "_get_normalized_flux", mock_flux)

        bc = BarycentricCorrection(synthetic_kpf2)
        _, t_mid, _ = bc._get_timestamps()
        _, t_fwm = bc.compute_flux_weighted_midpoint_times(
            output="expmeter",
            interpolate=False,
            extrapolate=False,
            fix_outliers=False,
        )

        # Compare midpoints as seconds from the first reading: JD magnitudes
        # (~2.46e6) swamp any relative tolerance, so reduce to O(100 s) offsets
        # where absolute tolerances mean what they say.
        f0 = np.array([-20.0, 100.0, 100.0])
        floored = np.clip(f0, 0.0, None)
        ref = t_mid.jd[0]

        def offset_seconds(weights):
            return (np.sum(t_mid.jd * weights) / np.sum(weights) - ref) * 86400.0

        actual_s = (t_fwm.jd[0] - ref) * 86400.0
        expected_s = offset_seconds(f0)
        floored_s = offset_seconds(floored)

        assert np.isfinite(actual_s)
        np.testing.assert_allclose(actual_s, expected_s, atol=1e-3)
        # The two weightings differ by the negative sample's ~20 s pull, so this
        # separates the actual result from the floored one unambiguously.
        assert abs(actual_s - floored_s) > 1.0

    def test_zero_flux_channel_raises(self, synthetic_kpf2):
        # Zero flux across all readings makes the midpoint 0/0; fail loudly.
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
        synthetic_kpf2.set_data("EXPMETER_SCI", _expmeter_table(data))
        bc = BarycentricCorrection(synthetic_kpf2)
        with pytest.raises(ValueError, match="total flux"):
            bc.compute_flux_weighted_midpoint_times(
                output="expmeter",
                interpolate=False,
                extrapolate=False,
                fix_outliers=False,
            )

    def test_interpolate_shifts_midpoint_with_front_weighted_flux(self, synthetic_kpf2):
        # Front-weighted flux makes interpolation add a bright sample at the
        # first gap (~T+90 s), which pulls the FWM strictly later.
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
        synthetic_kpf2.set_data("EXPMETER_SCI", _expmeter_table(data))
        bc = BarycentricCorrection(synthetic_kpf2)

        _, t_no = bc.compute_flux_weighted_midpoint_times(
            output="expmeter",
            interpolate=False,
            extrapolate=False,
            fix_outliers=False,
        )
        _, t_yes = bc.compute_flux_weighted_midpoint_times(
            output="expmeter",
            interpolate=True,
            extrapolate=False,
            fix_outliers=False,
        )
        # The shift is sub-minute (tens of microdays), comfortably clearing the
        # 1e-7 day (~9 ms) margin below.
        assert np.mean(t_yes.jd) > np.mean(t_no.jd) + 1e-7

    def test_extrapolate_shifts_midpoint_when_shutter_brackets_expmeter(
        self, synthetic_kpf2
    ):
        # The shutter brackets the readings on both sides, so extrapolation adds
        # a gap sample at each end. A bright first reading plus a 2-minute
        # leading gap makes the leading sample dominate and pull the FWM earlier.
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
        synthetic_kpf2.set_data("EXPMETER_SCI", _expmeter_table(data))
        # Shutter open 00:00:00 to 00:05:00; readings only 00:02:00 to 00:03:20.
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
            fix_outliers=False,
        )
        _, t_yes = bc.compute_flux_weighted_midpoint_times(
            output="expmeter",
            interpolate=False,
            extrapolate=True,
            fix_outliers=False,
        )
        assert np.mean(t_yes.jd) < np.mean(t_no.jd) - 1e-7


class TestFluxWeightedMidpointOrders:
    _KWARGS = dict(interpolate=False, extrapolate=False, fix_outliers=False)

    def test_output_shape(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        w, t = bc.compute_flux_weighted_midpoint_times(output="orders", **self._KWARGS)
        assert w.shape == (NORDER,)
        assert len(t) == NORDER

    def test_constant_wave_gives_constant_time(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        _, t = bc.compute_flux_weighted_midpoint_times(output="orders", **self._KWARGS)
        # Uniform flux makes every per-channel time equal, and the fixture's flat
        # SCI2_WAVE makes every order interpolate to that shared value.
        assert np.std(t.jd) < 1e-12

    def test_orders_get_distinct_times_when_waves_vary(self, synthetic_kpf2):
        synthetic_kpf2.set_data(
            "GREEN_SCI2_WAVE", np.full((NORDER_GREEN, NCOL), 5000.0, dtype=np.float64)
        )
        synthetic_kpf2.set_data(
            "RED_SCI2_WAVE", np.full((NORDER_RED, NCOL), 5100.0, dtype=np.float64)
        )
        # Front-weighted at 5000 A (GREEN), back-weighted at 5100 A (RED), so
        # GREEN's midpoint must land earlier than RED's.
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
        synthetic_kpf2.set_data("EXPMETER_SCI", _expmeter_table(data))

        bc = BarycentricCorrection(synthetic_kpf2)
        _, t = bc.compute_flux_weighted_midpoint_times(output="orders", **self._KWARGS)
        assert t.jd[:NORDER_GREEN].mean() < t.jd[NORDER_GREEN:].mean()

    def test_empty_sci2_wave_raises(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        synthetic_kpf2.data["TRACE3_WAVE"] = np.array([])
        with pytest.raises(KeyError, match="SCI2_WAVE"):
            bc.compute_flux_weighted_midpoint_times(output="orders", **self._KWARGS)


class TestFluxWeightedMidpointCcds:
    _KWARGS = dict(interpolate=False, extrapolate=False, fix_outliers=False)

    def test_output_shape(self, synthetic_kpf2):
        bc = BarycentricCorrection(synthetic_kpf2)
        w, t = bc.compute_flux_weighted_midpoint_times(output="ccds", **self._KWARGS)
        assert w.shape == (2,)
        assert len(t) == 2

    def test_ccd_values_equal_chip_means_with_uniform_weights(self, synthetic_kpf2):
        # With no SCI2_FLUX the order weights are uniform, so the chip summary
        # reduces to a plain mean.
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
        synthetic_kpf2.set_data("EXPMETER_SCI", _expmeter_table(data))

        bc = BarycentricCorrection(synthetic_kpf2)
        _, t_orders = bc.compute_flux_weighted_midpoint_times(
            output="orders", **self._KWARGS
        )
        unweighted_green = t_orders.jd[:NORDER_GREEN].mean()

        # Make the bluest (earliest) GREEN order dominate the flux weighting.
        flux = np.ones((NORDER, NCOL), dtype=np.float32)
        flux[0] = 1000.0
        synthetic_kpf2.set_data("SCI2_FLUX", flux)

        _, t_ccds = bc.compute_flux_weighted_midpoint_times(
            output="ccds", **self._KWARGS
        )
        assert t_ccds.jd[0] < unweighted_green

    # The all-NaN order under test makes nanpercentile warn; the weighting handles
    # it one line later (nan_to_num -> zero weight), which is what this asserts.
    @pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
    def test_nan_flux_order_gets_zero_weight(self, synthetic_kpf2):
        flux = np.ones((NORDER, NCOL), dtype=np.float32)
        flux[NORDER_GREEN] = np.nan  # first RED order failed extraction
        synthetic_kpf2.set_data("SCI2_FLUX", flux)

        bc = BarycentricCorrection(synthetic_kpf2)
        _, t_ccds = bc.compute_flux_weighted_midpoint_times(
            output="ccds", **self._KWARGS
        )
        assert np.all(np.isfinite(t_ccds.jd))

    def test_all_zero_weight_chip_raises(self, synthetic_kpf2):
        # A whole chip at zero flux makes the weighted mean undefined.
        flux = np.ones((NORDER, NCOL), dtype=np.float32)
        flux[NORDER_GREEN:] = 0.0
        synthetic_kpf2.set_data("SCI2_FLUX", flux)
        bc = BarycentricCorrection(synthetic_kpf2)
        with pytest.raises(ValueError, match="weights are zero"):
            bc.compute_flux_weighted_midpoint_times(output="ccds", **self._KWARGS)

    def test_green_and_red_resolve_distinct_values(self, synthetic_kpf2):
        synthetic_kpf2.set_data(
            "GREEN_SCI2_WAVE", np.full((NORDER_GREEN, NCOL), 5000.0, dtype=np.float64)
        )
        synthetic_kpf2.set_data(
            "RED_SCI2_WAVE", np.full((NORDER_RED, NCOL), 5100.0, dtype=np.float64)
        )
        # Front-weighted at 5000 A, back-weighted at 5100 A: distinct midpoints.
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
        synthetic_kpf2.set_data("EXPMETER_SCI", _expmeter_table(data))

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
            fix_outliers=False,
        )
        assert w.shape == (NORDER,)


# ---------------------------------------------------------------------------
# _get_astrometry -- read the PRIMARY C*# cards, convert, sanitize
# ---------------------------------------------------------------------------


class TestGetAstrometry:
    """Convert the C*# cards to barycorrpy's units and sanitize the parallax."""

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
        # A nonpositive parallax is routine for faint Gaia sources, so it degrades
        # to px=0 rather than becoming a negative distance.
        synthetic_kpf2.headers["PRIMARY"]["CPLX3"] = parallax
        bc = BarycentricCorrection(synthetic_kpf2)
        with caplog.at_level(logging.WARNING):
            astrometry = bc._get_astrometry()
        assert astrometry["px"] == 0.0
        assert "CPLX3" in caplog.text

    def test_missing_parallax_becomes_zero(self, caplog, synthetic_kpf2):
        # CPLX# is optional, so it is absent when the catalog measured no parallax.
        del synthetic_kpf2.headers["PRIMARY"]["CPLX3"]
        bc = BarycentricCorrection(synthetic_kpf2)
        with caplog.at_level(logging.WARNING):
            astrometry = bc._get_astrometry()
        assert astrometry["px"] == 0.0
        assert "CPLX3" in caplog.text

    @pytest.mark.parametrize("card", ["CRA3", "CDEC3", "CPMR3", "CPMD3", "CEPCH3"])
    def test_missing_position_card_raises(self, synthetic_kpf2, card):
        # The position block is all-or-nothing: AstroQuery never emits a canonical
        # row without it, so a gap means the pipeline ran out of order.
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
            # BJD = JD + 500 s stands in for the light-travel delay.
            bjd_tdb = np.atleast_1d(obs_times.jd) + 500.0 / 86400.0
            return bc_vel, bjd_tdb

        def passthrough(f, kernel_size=5):
            return f.copy()

        monkeypatch.setattr(
            BarycentricCorrection, "_compute_barycorr", staticmethod(mock_compute)
        )
        # The 3x4 uniform-flux fixture triggers a degenerate triangulation inside
        # scipy.griddata; TestFixExpmeterOutliers exercises the real filter.
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
        assert_dtype(bjd, BJD, "BJD_TDB")

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
        assert np.std(z) < 1e-12
        # BARYCORR_Z is the small redshift z (~ v/c), not the Doppler factor
        # (~1), and carries the same sign as the velocity.
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
        # BarycentricCorrection tracks the correction; it never applies it.
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
        for key in ("BJDGREEN", "BJDRED"):
            assert key in bjd, f"{key} missing from BJD_TDB"
        for key in ("BVGREEN", "BVRED"):
            assert key in kms, f"{key} missing from BARYCORR_KMS"
        for key in ("BZGREEN", "BZRED"):
            assert key in z, f"{key} missing from BARYCORR_Z"

        # The stub gives every order the same delta_rv, so the chip means match.
        np.testing.assert_allclose(kms.get("BVGREEN"), kms.get("BVRED"))
        np.testing.assert_allclose(z.get("BZGREEN"), z.get("BZRED"))

    def test_ctype1_axis_label(self, bc_monkeypatched):
        # These are 1-D per-order arrays, so CTYPE1 names the order axis and
        # there is no second axis to label.
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

    def test_missing_crv_defaults_to_zero(self, caplog, bc_monkeypatched, monkeypatch):
        del bc_monkeypatched.l2_obj.headers["PRIMARY"]["CRV3"]
        captured = {}

        def mock_compute(astrometry, obs_times, location, rv_mps=0.0):
            captured["rv_mps"] = rv_mps
            n = len(np.atleast_1d(obs_times.jd))
            return np.full(n, TestPerform.DELTA_RV_MPS), np.atleast_1d(obs_times.jd)

        monkeypatch.setattr(
            BarycentricCorrection, "_compute_barycorr", staticmethod(mock_compute)
        )
        with caplog.at_level(logging.WARNING):
            bc_monkeypatched.perform()
        assert captured["rv_mps"] == 0.0
        assert "CRV3" in caplog.text

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
        # The real filter needs a noisy table; the 3x4 uniform fixture would
        # crash griddata.
        rng = np.random.default_rng(0)
        ntime, nwave = 60, 20
        wave_cols = [str(5000.0 + 10 * i) for i in range(nwave)]
        data = {
            "Date-Beg": [f"2024-01-01T00:{m:02d}:00.000" for m in range(ntime)],
            "Date-End": [f"2024-01-01T00:{m:02d}:30.000" for m in range(ntime)],
        }
        for wc in wave_cols:
            data[wc] = rng.normal(100.0, 2.0, ntime).astype(float)
        data[wave_cols[5]][30] = 1e6  # a clear outlier, so the filter has work
        synthetic_kpf2.set_data("EXPMETER_SCI", _expmeter_table(data))
        synthetic_kpf2.headers["INSTRUMENT_HEADER"]["DATE-END"] = (
            "2024-01-01T01:00:00.000"
        )

        def mock_compute(astrometry, obs_times, location, rv_mps=0.0):
            n = len(np.atleast_1d(obs_times.jd))
            return np.full(n, 1000.0), np.atleast_1d(obs_times.jd)

        monkeypatch.setattr(
            BarycentricCorrection, "_compute_barycorr", staticmethod(mock_compute)
        )

        # _fix_expmeter_outliers is deliberately not stubbed here.
        kpf2 = BarycentricCorrection(synthetic_kpf2).perform(fix_expmeter_outliers=True)
        assert np.all(np.isfinite(np.asarray(kpf2.data["BJD_TDB"])))


# ---------------------------------------------------------------------------
# perform(skycoord=...) -- interactive astrometry override
# ---------------------------------------------------------------------------


def _user_skycoord(ra_deg=90.0):
    """A fully-specified SkyCoord, as an interactive caller would build one."""
    return SkyCoord(
        ra=ra_deg * u.deg,
        dec=30.0 * u.deg,
        pm_ra_cosdec=100.0 * u.mas / u.yr,
        pm_dec=-50.0 * u.mas / u.yr,
        distance=Distance(parallax=25.0 * u.mas),
        obstime=Time(2020.0, format="jyear"),
        frame="icrs",
    )


class TestSkycoordOverride:
    """A user-supplied SkyCoord bypasses the PRIMARY C*# cards.

    That lets an L2 already in hand be re-corrected without re-reducing from L0.
    """

    @staticmethod
    def _capture(monkeypatch):
        captured = {}

        def mock_compute(astrometry, obs_times, location, rv_mps=0.0):
            captured.update(astrometry)
            n = len(np.atleast_1d(obs_times.jd))
            return np.zeros(n), np.atleast_1d(obs_times.jd)

        monkeypatch.setattr(
            BarycentricCorrection, "_compute_barycorr", staticmethod(mock_compute)
        )
        monkeypatch.setattr(
            BarycentricCorrection,
            "_fix_expmeter_outliers",
            staticmethod(lambda f, kernel_size=5: f.copy()),
        )
        return captured

    def test_converts_skycoord_to_barycorrpy_units(self, synthetic_kpf2, monkeypatch):
        captured = self._capture(monkeypatch)
        BarycentricCorrection(synthetic_kpf2).perform(skycoord=_user_skycoord())

        assert captured["ra"] == pytest.approx(90.0)
        assert captured["dec"] == pytest.approx(30.0)
        assert captured["pmra"] == pytest.approx(100.0)
        assert captured["pmdec"] == pytest.approx(-50.0)
        assert captured["px"] == pytest.approx(25.0)
        assert captured["epoch"] == pytest.approx(Time(2020.0, format="jyear").jd)

    def test_overrides_the_header_cards(self, synthetic_kpf2, monkeypatch):
        # The fixture's CRA3 is 12h (180 deg); the override must win.
        captured = self._capture(monkeypatch)
        BarycentricCorrection(synthetic_kpf2).perform(skycoord=_user_skycoord())
        assert captured["ra"] == pytest.approx(90.0)

    def test_works_without_catalog_cards(self, synthetic_kpf2, monkeypatch):
        # An L2 whose C*# cards are absent (AstroQuery never ran) is still
        # correctable -- the point of the escape hatch.
        for card in ("CRA3", "CDEC3", "CPMR3", "CPMD3", "CEPCH3"):
            del synthetic_kpf2.headers["PRIMARY"][card]
        captured = self._capture(monkeypatch)

        kpf2 = BarycentricCorrection(synthetic_kpf2).perform(skycoord=_user_skycoord())

        assert captured["ra"] == pytest.approx(90.0)
        assert len(kpf2.data["BJD_TDB"]) == NORDER

    def test_header_cards_not_modified(self, synthetic_kpf2, monkeypatch):
        self._capture(monkeypatch)
        BarycentricCorrection(synthetic_kpf2).perform(skycoord=_user_skycoord())
        assert synthetic_kpf2.headers["PRIMARY"]["CRA3"] == "12:00:00.0000"

    def test_records_user_provenance(self, synthetic_kpf2, monkeypatch):
        self._capture(monkeypatch)
        bc = BarycentricCorrection(synthetic_kpf2)
        bc.perform(skycoord=_user_skycoord())
        assert bc._astrometry_source == "user SkyCoord"

    def test_override_is_not_cached(self, synthetic_kpf2, monkeypatch):
        # A cached override would make a later header-path call silently reuse
        # the user's values.
        captured = self._capture(monkeypatch)
        bc = BarycentricCorrection(synthetic_kpf2)
        bc.perform(skycoord=_user_skycoord())
        assert bc._astrometry is None

        bc.perform()  # no override: back to the header
        assert captured["ra"] == pytest.approx(180.0)

    def test_default_reads_the_header(self, synthetic_kpf2, monkeypatch):
        captured = self._capture(monkeypatch)
        BarycentricCorrection(synthetic_kpf2).perform()
        assert captured["ra"] == pytest.approx(180.0)

    @pytest.mark.parametrize(
        "kwargs, error, match",
        [
            # No proper motion: the frame carries no velocity data.
            (
                {"distance": Distance(parallax=25.0 * u.mas)},
                TypeError,
                "no associated differentials",
            ),
            # No distance: .distance is a dimensionless 1.0, not convertible to pc.
            (
                {
                    "pm_ra_cosdec": 100.0 * u.mas / u.yr,
                    "pm_dec": -50.0 * u.mas / u.yr,
                },
                u.UnitConversionError,
                "not convertible",
            ),
        ],
    )
    def test_incomplete_skycoord_raises(
        self, synthetic_kpf2, monkeypatch, kwargs, error, match
    ):
        # Deliberately unvalidated: astropy raises on its own for a SkyCoord
        # missing components the correction needs.
        self._capture(monkeypatch)
        incomplete = SkyCoord(
            ra=90.0 * u.deg,
            dec=30.0 * u.deg,
            obstime=Time(2020.0, format="jyear"),
            frame="icrs",
            **kwargs,
        )
        with pytest.raises(error, match=match):
            BarycentricCorrection(synthetic_kpf2).perform(skycoord=incomplete)


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
        # Stub the outlier filter so the griddata degeneracy does not fire before
        # the catalog-card lookup under test.
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
                fix_outliers=False,
            )

    def test_missing_date_beg_ok_without_extrapolate(self, synthetic_kpf2):
        # DATE-BEG is read only to extrapolate, so this must not raise.
        del synthetic_kpf2.headers["INSTRUMENT_HEADER"]["DATE-BEG"]
        bc = BarycentricCorrection(synthetic_kpf2)
        bc.compute_flux_weighted_midpoint_times(
            output="expmeter",
            interpolate=False,
            extrapolate=False,
            fix_outliers=False,
        )
