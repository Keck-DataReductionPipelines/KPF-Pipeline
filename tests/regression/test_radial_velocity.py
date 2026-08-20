"""Tests for the RadialVelocity module (KPF4 -> KPF4: fit per-order CCFs to RVs).

Static-method unit tests (_compute_rv_1d, _ccf_noise_corr_length,
_pixel_velocity_scale) build synthetic CCFs with no fixtures. Integration tests
build a synthetic KPF2, run CrossCorrelation (mask monkeypatched, narrow grid) to
get a CCF-bearing L4, then exercise RadialVelocity on it.
"""

import copy
import logging

import numpy as np
import pytest
from astropy.io import fits

from kpfpipe.data_models.level2 import KPF2, NORDER_GREEN, NORDER_RED
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.modules.cross_correlation import CrossCorrelation
from kpfpipe.modules.radial_velocity import RadialVelocity

from ._catalog import seed_sci2_cards
from ._dtype_policy import RV_FLOAT, assert_dtype
from ._science import (
    MASK_CENTERS,
    NCOL,
    RANGE_KMS,
    SPEED_OF_LIGHT_KMS,
    V_INJECT,
    absorption_spectrum,
    make_mask,
)

NORDER = NORDER_GREEN + NORDER_RED
# Fiber order is the module's own config-overridable default, not the canonical
# slicer order -- spelled out so a reordering in production shows up here.
_FIBERS = ["CAL", "SCI1", "SCI2", "SCI3", "SKY"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_l4(
    monkeypatch,
    sci_obj="Target",
    sky_obj="Sky",
    cal_obj="None",
    berv=0.0,
    bjd=0.0,
    perform_fibers=None,
):
    """Synthetic KPF2 (absorption at MASK_CENTERS, shifted by V_INJECT) run
    through CrossCorrelation (mask stubbed, narrow grid) into a KPF4."""
    kpf2 = KPF2()
    seed_sci2_cards(kpf2, sci_obj=sci_obj, sky_obj=sky_obj, cal_obj=cal_obj)

    wave_1d = np.linspace(5000.0, 5050.0, NCOL)
    lam_obs = MASK_CENTERS * (1.0 + V_INJECT / SPEED_OF_LIGHT_KMS)
    flux_1d = absorption_spectrum(wave_1d, lam_obs)
    for chip, n in [("GREEN", NORDER_GREEN), ("RED", NORDER_RED)]:
        for fiber in _FIBERS:
            kpf2.set_data(
                f"{chip}_{fiber}_WAVE", np.tile(wave_1d, (n, 1)).astype(np.float64)
            )
            kpf2.set_data(
                f"{chip}_{fiber}_FLUX", np.tile(flux_1d, (n, 1)).astype(np.float64)
            )
            kpf2.set_data(
                f"{chip}_{fiber}_VAR", np.tile(flux_1d, (n, 1)).astype(np.float64)
            )
    kpf2.set_data("BARYCORR_Z", np.zeros(NORDER))
    kpf2.set_data("BARYCORR_KMS", np.full(NORDER, berv))
    kpf2.set_data("BJD_TDB", np.full(NORDER, bjd))

    mask = make_mask(MASK_CENTERS)
    monkeypatch.setattr(
        CrossCorrelation,
        "_build_line_mask",
        lambda self, chip, fiber, mask_width=None: mask,
    )
    return CrossCorrelation(kpf2, config={"ccf_window": RANGE_KMS}).perform(
        fibers=perform_fibers
    )


# ---------------------------------------------------------------------------
# _compute_rv_1d / _ccf_noise_corr_length (staticmethods)
# ---------------------------------------------------------------------------


class TestComputeRV:
    def _ccf(self, v0=0.0, sigma=4.0, baseline=100.0, depth=30.0):
        vel = np.arange(-402, 403) * 0.25
        ccf = baseline - depth * np.exp(-0.5 * ((vel - v0) / sigma) ** 2)
        ccf_var = ccf.copy()  # photon-noise proxy at the CCF count scale
        vps = RadialVelocity._pixel_velocity_scale(5000.0, 5050.0, 4000)
        return vel, ccf, ccf_var, vps

    def test_recovers_injected_rv(self):
        vel, ccf, ccf_var, vps = self._ccf(v0=2.5)
        rv, _ = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        assert rv == pytest.approx(2.5, abs=0.05)

    def test_error_finite_and_positive(self):
        vel, ccf, ccf_var, vps = self._ccf(v0=0.0)
        _, rv_err = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        assert np.isfinite(rv_err) and rv_err > 0

    def test_returns_float64(self):
        vel, ccf, ccf_var, vps = self._ccf(v0=1.0)
        rv, rv_err = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        assert_dtype(np.asarray(rv), RV_FLOAT, "RV scalar")
        assert_dtype(np.asarray(rv_err), RV_FLOAT, "RV_ERR scalar")

    def test_even_window_pts_allowed(self):
        # min_npts is a minimum point count, not a centered odd window.
        vel, ccf, ccf_var, vps = self._ccf(v0=1.0)
        rv, _ = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 8
        )
        assert rv == pytest.approx(1.0, abs=0.05)

    def test_nonphysical_variance_raises(self):
        # A non-positive CCF variance in the fit window is corrupt flux, not a
        # recoverable "no error" case -> fail loudly rather than return NaN.
        vel, ccf, ccf_var, vps = self._ccf(v0=0.0)
        ccf_var[402] = -1.0  # index of v = 0, inside the +/-3 sigma fit window
        with pytest.raises(ValueError, match="non-physical"):
            RadialVelocity._compute_rv_1d(
                vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
            )

    def test_flat_ccf_returns_nan(self):
        vel = np.arange(-402, 403) * 0.25
        ccf = np.full_like(vel, 100.0)
        ccf_var = ccf
        vps = RadialVelocity._pixel_velocity_scale(5000.0, 5050.0, 4000)
        rv, rv_err = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_nonfinite_ccf_returns_nan(self):
        # Non-finite CCF values are not quietly masked out; the fit returns NaN.
        vel, ccf, ccf_var, vps = self._ccf(v0=1.0)
        ccf[10] = np.nan
        rv, rv_err = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_second_pass_off_grid_returns_first_pass_rv(self):
        # Dip near the low edge: the first-pass fit succeeds, but the +/-3 sigma
        # second-pass window runs off the grid -> first-pass RV, NaN error.
        vel = np.arange(-402, 403) * 0.25
        v0 = vel[20]
        ccf = 100.0 - 30.0 * np.exp(-0.5 * ((vel - v0) / 4.0) ** 2)
        ccf_var = ccf
        vps = RadialVelocity._pixel_velocity_scale(5000.0, 5050.0, 4000)
        rv, rv_err = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        assert rv == pytest.approx(v0, abs=0.1) and np.isnan(rv_err)

    def test_narrow_window_returns_nan(self):
        # A first-pass window narrower than min_npts grid points -> NaN, NaN.
        vel, ccf, ccf_var, vps = self._ccf(v0=0.0)
        rv, rv_err = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-0.1, 0.1], 11
        )
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_first_pass_fit_failure_returns_nan(self, monkeypatch):
        # optimize_lsq raising on the first pass degrades to NaN, not a crash.
        vel, ccf, ccf_var, vps = self._ccf(v0=0.0)

        def boom(*args, **kwargs):
            raise RuntimeError("singular matrix")

        monkeypatch.setattr("kpfpipe.modules.radial_velocity.optimize_lsq", boom)
        rv, rv_err = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_nonfinite_fit_params_return_nan(self, monkeypatch):
        vel, ccf, ccf_var, vps = self._ccf(v0=0.0)

        def bad_fit(*args, **kwargs):
            return np.array([100.0, 30.0, np.nan, 4.0]), None

        monkeypatch.setattr("kpfpipe.modules.radial_velocity.optimize_lsq", bad_fit)
        rv, rv_err = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_second_pass_fit_failure_keeps_first_pass_rv(self, monkeypatch):
        vel, ccf, ccf_var, vps = self._ccf(v0=0.0)
        calls = []

        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return np.array([100.0, 30.0, 0.0, 4.0]), None
            raise RuntimeError("refinement failed")

        monkeypatch.setattr("kpfpipe.modules.radial_velocity.optimize_lsq", flaky)
        rv, _ = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        assert np.isfinite(rv) and rv == pytest.approx(0.0, abs=1e-9)

    def test_error_scales_with_correlation_length(self):
        # The photon error is 1/sqrt(N_scale) with N_scale = dv / corr_length, so
        # widening the mask hole lengthens the noise correlation and inflates
        # rv_err by exactly sqrt(L2/L1).
        vel, ccf, ccf_var, vps = self._ccf(v0=0.0)
        _, err_a = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        _, err_b = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 1.0, [-50.0, 50.0], 11
        )
        la = RadialVelocity._ccf_noise_corr_length(vps, 0.5)
        lb = RadialVelocity._ccf_noise_corr_length(vps, 1.0)
        assert err_b / err_a == pytest.approx((lb / la) ** 0.5, rel=1e-6)


class TestPixelVelocityScale:
    def test_matches_linear_dispersion(self):
        # For a linear order, the endpoint scale equals the median-of-diffs scale.
        wave = np.linspace(5000.0, 5050.0, 4000)
        vps = RadialVelocity._pixel_velocity_scale(wave[0], wave[-1], wave.size)
        expected = (
            SPEED_OF_LIGHT_KMS * np.median(np.abs(np.diff(wave))) / np.median(wave)
        )
        assert vps == pytest.approx(expected, rel=1e-9)


class TestCCFNoiseCorrLength:
    def test_matches_measured_value(self):
        # Native pixel 0.885 km/s, mask hole full width 1.0 km/s -> trapezoid-ACF
        # integral length, verified against the frame's measured autocorrelation.
        assert RadialVelocity._ccf_noise_corr_length(0.885, 1.0) == pytest.approx(
            1.418, abs=1e-3
        )

    def test_equal_widths_give_1p5w(self):
        # pixel width == mask-hole width == w -> 1.5 w.
        assert RadialVelocity._ccf_noise_corr_length(0.6, 0.6) == pytest.approx(0.9)

    def test_symmetric_in_the_two_widths(self):
        # L depends only on the {pixel, mask-hole} pair, not which is larger.
        a = RadialVelocity._ccf_noise_corr_length(1.0, 0.8)
        b = RadialVelocity._ccf_noise_corr_length(0.8, 1.0)
        assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# compute_order_by_order_rvs / compute_weighted_rvs (L4 CCFs loaded)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rv_l4():
    """A CCF-bearing L4 from CrossCorrelation (SCI on a star, SKY on sky, CAL dark).

    Module-scoped: the ~1s CrossCorrelation.perform() build runs once and is shared
    read-only. The mask stub is needed only during that build, hence a self-managed
    MonkeyPatch context instead of the function-scoped ``monkeypatch`` fixture.
    """
    with pytest.MonkeyPatch.context() as mp:
        return _make_l4(mp)


@pytest.fixture
def rv_module(rv_l4):
    """RadialVelocity on a per-test copy of the CCF-bearing L4; caches not yet loaded.

    ``perform()`` mutates the L4 in place, so each test deepcopies the shared
    ``rv_l4``. The copy is a memcpy -- the ~1s cost was the CCF compute, not the
    data -- so isolation is cheap while the CCFs are still built only once.
    """
    return RadialVelocity(copy.deepcopy(rv_l4), config={"rv_window": RANGE_KMS})


@pytest.fixture
def rv_loaded(rv_module):
    """RadialVelocity with the L4 CCFs loaded into the caches."""
    rv_module._load_ccfs(["GREEN", "RED"], _FIBERS)
    return rv_module


class TestComputeRVPublic:
    def test_returns_rv_dict(self, rv_loaded):
        res = rv_loaded.compute_order_by_order_rvs("GREEN", "SCI2")
        assert set(res) == {"rv", "rv_err"}
        assert res["rv"].shape == (NORDER_GREEN,)
        assert res["rv_err"].shape == (NORDER_GREEN,)

    def test_raises_without_loaded_ccf(self, rv_module):
        with pytest.raises(RuntimeError, match="not loaded"):
            rv_module.compute_order_by_order_rvs("GREEN", "SCI2")

    def test_recovers_injected_rv(self, rv_loaded):
        rv = rv_loaded.compute_order_by_order_rvs("GREEN", "SCI2")["rv"]
        np.testing.assert_allclose(rv, V_INJECT, atol=0.1)

    def test_per_ccd_rv_recovers_injected(self, rv_loaded):
        # The per-CCD error comes from the unweighted-summed CCF.
        ccd_rv, ccd_rv_err = rv_loaded.compute_weighted_rvs(
            ["GREEN"], "SCI2", combine_fibers=False, combine_ccds=False
        )["GREEN"]
        assert ccd_rv == pytest.approx(V_INJECT, abs=0.1)
        assert np.isfinite(ccd_rv_err) and ccd_rv_err > 0

    def test_per_ccd_rv_sums_science_fibers(self, rv_loaded):
        # combine_fibers=True sums the three science fibers' cached CCFs before
        # fitting, giving the SCI-combined per-CCD RV.
        ccd_rv, ccd_rv_err = rv_loaded.compute_weighted_rvs(
            ["GREEN"], ["SCI1", "SCI2", "SCI3"], combine_fibers=True, combine_ccds=False
        )["GREEN"]
        assert ccd_rv == pytest.approx(V_INJECT, abs=0.1)
        assert np.isfinite(ccd_rv_err) and ccd_rv_err > 0

    def test_combine_ccds_returns_tuple_and_recovers_injected(self, rv_loaded):
        # combine_ccds=True combines at the RV level, not the CCF level.
        out = rv_loaded.compute_weighted_rvs(
            ["GREEN", "RED"], "SCI2", combine_fibers=False, combine_ccds=True
        )
        assert isinstance(out, tuple) and len(out) == 2
        assert out[0] == pytest.approx(V_INJECT, abs=0.1)
        assert np.isfinite(out[1]) and out[1] > 0

    def test_combine_fibers_requires_three_sci(self, rv_loaded):
        with pytest.raises(ValueError, match="three science fibers"):
            rv_loaded.compute_weighted_rvs(
                ["GREEN"], "SCI2", combine_fibers=True, combine_ccds=False
            )

    def test_no_combine_fibers_requires_single(self, rv_loaded):
        with pytest.raises(ValueError, match="single fiber"):
            rv_loaded.compute_weighted_rvs(
                ["GREEN"],
                ["SCI1", "SCI2", "SCI3"],
                combine_fibers=False,
                combine_ccds=False,
            )

    def test_combine_ccfs_invalid_fibers_raises(self, rv_loaded):
        # _combine_ccfs accepts a single fiber or exactly the three SCI fibers.
        with pytest.raises(ValueError, match="single fiber or exactly"):
            rv_loaded._combine_ccfs("GREEN", ["SCI1", "SCI2"])

    def test_per_ccd_raises_without_loaded_ccf(self, rv_module):
        with pytest.raises(RuntimeError, match="not loaded"):
            rv_module.compute_weighted_rvs(
                ["GREEN"], "SCI2", combine_fibers=False, combine_ccds=False
            )

    def test_errors_finite_and_positive(self, rv_loaded):
        rv_err = rv_loaded.compute_order_by_order_rvs("GREEN", "SCI2")["rv_err"]
        assert np.all(np.isfinite(rv_err)) and np.all(rv_err > 0)

    def test_nonphysical_order_degrades_to_nan(self, rv_loaded):
        # One bad order (negative variance) degrades to NaN instead of aborting
        # the whole frame.
        bad = 5
        rv_loaded._ccf_var["GREEN_SCI2"][bad] = -1.0
        res = rv_loaded.compute_order_by_order_rvs("GREEN", "SCI2")
        assert np.isnan(res["rv"][bad]) and np.isnan(res["rv_err"][bad])
        others = np.delete(np.arange(NORDER_GREEN), bad)
        assert np.all(np.isfinite(res["rv"][others]))
        assert np.all(np.isfinite(res["rv_err"][others]))

    def test_order_weights_read_from_rvn(self, rv_loaded):
        # _get_order_weights slices the WEIGHT column of the L4 RVn table.
        w = rv_loaded._get_order_weights("GREEN", "SCI2")
        assert w.shape == (NORDER_GREEN,) and np.all(w >= 0)
        expected = np.asarray(
            rv_loaded.l4_obj.data["GREEN_SCI2_RV"]["WEIGHT"], dtype=float
        )
        np.testing.assert_array_equal(w, expected)


# ---------------------------------------------------------------------------
# perform (KPF4 -> KPF4)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPerform:
    _ILLUMINATED = ["SCI1", "SCI2", "SCI3", "SKY"]  # CAL-OBJ='None' -> skipped

    def test_fills_rv_columns_preserving_metadata(self, rv_module):
        l4 = rv_module.perform()
        assert isinstance(l4, KPF4)
        for fiber in self._ILLUMINATED:
            table = l4.data[f"{fiber}_RV"]
            assert len(table) == NORDER
            assert np.any(np.isfinite(np.asarray(table["RV"], dtype=float)))
            # The CrossCorrelation-seeded metadata columns survive the RV fill.
            assert set(table.columns) >= {
                "ORDER_INDEX",
                "ORDER_ID",
                "ECHELLE_ORDER",
                "BJD_TDB",
                "BERV",
                "WAVE_START",
                "WAVE_END",
                "WEIGHT",
                "RV",
                "RV_ERR",
            }

    def test_unilluminated_fiber_skipped(self, rv_module):
        # CAL-OBJ='None' -> CrossCorrelation wrote no CAL CCF or RV table.
        l4 = rv_module.perform()
        assert l4.data["CAL_CCF"].size == 0
        assert len(l4.data["CAL_RV"]) == 0

    def test_recovers_injected_rv_illuminated_orderlets(self, rv_module):
        l4 = rv_module.perform()
        for fiber in self._ILLUMINATED:
            rv = np.asarray(l4.data[f"{fiber}_RV"]["RV"])
            np.testing.assert_allclose(rv, V_INJECT, atol=0.1)

    def test_rv_headers(self, rv_module):
        l4 = rv_module.perform()
        rv_hdr = l4.headers["SCI2_RV"]
        assert rv_hdr["RVMETHOD"] == "CCF"
        assert rv_hdr["SKYRMVD"] is False
        assert rv_hdr["TELLRMVD"] is False
        assert l4.headers["PRIMARY"]["RVMETHOD"] == "CCF"

    def test_per_ccd_rv_keywords(self, rv_module):
        # Per-orderlet legacy RVs route to their RV# table header as CCD<n>RV<sfx>
        # (CCD1=GREEN, CCD2=RED; SCI2 has suffix '2' and lives on RV3).
        l4 = rv_module.perform()
        rv_hdr = l4.headers["RV3"]
        assert rv_hdr["CCD1RV2"] == pytest.approx(V_INJECT, abs=0.1)
        assert rv_hdr["CCD2RV2"] == pytest.approx(V_INJECT, abs=0.1)
        assert rv_hdr["CCD1ERV2"] > 0 and rv_hdr["CCD2ERV2"] > 0
        # The per-orderlet keywords do not leak onto PRIMARY.
        assert "CCD1RV2" not in l4.headers["PRIMARY"]

    def test_combined_rv_populated(self, rv_module):
        # PRIMARY: EPRV RV/RVERR plus the KPF SCI-combined per-CCD CCD1RV/CCD2RV.
        l4 = rv_module.perform()
        prim = l4.headers["PRIMARY"]
        assert prim["CCD1RV"] == pytest.approx(V_INJECT, abs=0.1)
        assert prim["CCD2RV"] == pytest.approx(V_INJECT, abs=0.1)
        assert prim["CCD1ERV"] > 0 and prim["CCD2ERV"] > 0
        assert prim["RV"] == pytest.approx(V_INJECT, abs=0.1)
        assert prim["RVERR"] > 0
        assert prim["RVMETHOD"] == "CCF"
        assert "CCD1RV" not in l4.headers["RV3"]
        assert "RV" not in l4.headers["RV3"]

    def test_combined_rv_is_weighted_ccd_combine(self, rv_module):
        # PRIMARY RV = (CCD1RV*Wg + CCD2RV*Wr)/(Wg+Wr), Wg/Wr the summed order
        # weights; RVERR = inverse-variance combination of the per-CCD errors.
        l4 = rv_module.perform()
        prim = l4.headers["PRIMARY"]
        wg = np.nansum(rv_module._get_order_weights("GREEN", "SCI1"))
        wr = np.nansum(rv_module._get_order_weights("RED", "SCI1"))
        expect_rv = (prim["CCD1RV"] * wg + prim["CCD2RV"] * wr) / (wg + wr)
        expect_err = (1.0 / prim["CCD1ERV"] ** 2 + 1.0 / prim["CCD2ERV"] ** 2) ** -0.5
        assert prim["RV"] == pytest.approx(expect_rv, abs=1e-9)
        assert prim["RVERR"] == pytest.approx(expect_err, rel=1e-9)

    def test_primary_berv_bjdtdb_from_per_order(self, rv_module):
        # PRIMARY BERV/BJDTDB are the WEIGHT-weighted mean of the rep SCI fiber's
        # per-order columns, so constant per-order values reproduce that value.
        # These columns never feed the CCF compute, so overwriting them on the
        # shared L4 is cheaper than rebuilding it via _make_l4.
        l4 = rv_module.l4_obj
        table = l4.data["SCI1_RV"]
        table["BERV"] = -12.3
        table["BJD_TDB"] = 2460123.5
        l4.set_data("SCI1_RV", table)
        prim = rv_module.perform().headers["PRIMARY"]
        assert prim["BERV"] == pytest.approx(-12.3)
        assert prim["BJDTDB"] == pytest.approx(2460123.5)

    def test_primary_berv_undefined_when_berv_nan(self, rv_module):
        # An all-NaN BERV column leaves BERV UNDEFINED while BJDTDB, from the
        # still-finite BJD_TDB column, stays defined.
        l4 = rv_module.l4_obj
        table = l4.data["SCI1_RV"]
        table["BERV"] = np.nan
        l4.set_data("SCI1_RV", table)
        prim = rv_module.perform().headers["PRIMARY"]
        assert prim["BERV"] is None
        assert prim["BJDTDB"] is not None

    def test_no_science_illuminated_raises(self, monkeypatch):
        # SCI requested by default, but the L4 carries no SCI CCFs -> fail loudly.
        l4 = _make_l4(monkeypatch, sci_obj="None")
        rv = RadialVelocity(l4, config={"rv_window": RANGE_KMS})
        with pytest.raises(ValueError, match="none illuminated"):
            rv.perform()

    def test_cal_only_run_skips_combine(self, monkeypatch, caplog):
        # A calibration-only run must not raise: PRIMARY RV is left UNDEFINED.
        l4 = _make_l4(monkeypatch, sci_obj="None", sky_obj="None", cal_obj="Th_gold")
        with caplog.at_level(logging.INFO, logger="kpfpipe"):
            prim = (
                RadialVelocity(l4, config={"rv_window": RANGE_KMS})
                .perform(fibers=["CAL"])
                .headers["PRIMARY"]
            )
        assert prim.get("RV") is None
        assert "no science orderlet requested" in caplog.text

    def test_single_chip_combine_warns(self, rv_module, caplog):
        # With one chip present the combined RV is that chip's RV alone.
        with caplog.at_level(logging.INFO, logger="kpfpipe"):
            l4 = rv_module.perform(chips=["GREEN"])
        prim = l4.headers["PRIMARY"]
        assert prim["RV"] == pytest.approx(prim["CCD1RV"], abs=1e-9)
        assert "only chip GREEN present" in caplog.text

    def test_l4_serializes_to_fits(self, rv_module, tmp_path):
        # SCI2's per-orderlet RVs land on the RV3 table.
        l4 = rv_module.perform(fibers=["SCI1", "SCI2", "SCI3"])
        path = tmp_path / "kpf_SL4_20240405T000000.fits"
        l4.to_fits(str(path))
        with fits.open(path) as hdul:
            rv = hdul["RV3"].header
            assert rv["RVMETHOD"] == "CCF"
            assert rv["CCD1RV2"] == pytest.approx(V_INJECT, abs=0.1)
            assert "CCD1RV2" not in hdul["PRIMARY"].header
            # Only the EPRV combined RV belongs on PRIMARY.
            assert hdul["PRIMARY"].header["RV"] == pytest.approx(V_INJECT, abs=0.1)
            assert hdul["PRIMARY"].header["RVERR"] > 0
            rv_table = hdul["RV3"].data
            assert rv_table["ORDER_ID"][0] == "GREEN_SCI2_0"
            assert rv_table["ECHELLE_ORDER"][0] == 137
            assert np.all(np.isfinite(rv_table["RV"]))

    def test_failed_combined_fit_written_as_undefined(self, rv_module, monkeypatch):
        # A failed fit is written as a FITS UNDEFINED card (present, value None),
        # never a bare NaN.
        monkeypatch.setattr(
            rv_module, "_compute_rv_1d", lambda *args, **kwargs: (np.nan, np.nan)
        )
        l4 = rv_module.perform(fibers=["SCI1", "SCI2", "SCI3"])
        rv_hdr = l4.headers["RV3"]  # SCI2 per-orderlet RVs -> RV3
        assert "CCD1RV2" in rv_hdr and rv_hdr["CCD1RV2"] is None
        assert "CCD2RV2" in rv_hdr and rv_hdr["CCD2RV2"] is None

    def test_explicit_chips_and_fibers(self, rv_module):
        l4 = rv_module.perform(chips=["GREEN"], fibers=["SCI1", "SCI2", "SCI3"])
        # GREEN-only: science green RVs filled, red rows left NaN.
        rv = np.asarray(l4.data["SCI1_RV"]["RV"])
        assert np.all(np.isfinite(rv[:NORDER_GREEN]))
        assert np.all(np.isnan(rv[NORDER_GREEN:]))
        # SKY not requested -> its RV column stays NaN (CrossCorrelation seeded it).
        assert np.all(np.isnan(np.asarray(l4.data["SKY_RV"]["RV"])))


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_invalid_config_type_raises(self):
        with pytest.raises(TypeError, match="None, dict, or ConfigHandler"):
            RadialVelocity(KPF4(), config="not-a-config")

    def test_rv_window_default(self):
        assert RadialVelocity(KPF4()).rv_window == [-25.0, 25.0]

    def test_dict_config_overrides_default(self):
        rv = RadialVelocity(KPF4(), config={"rv_window": [-10.0, 10.0]})
        assert rv.rv_window == [-10.0, 10.0]
