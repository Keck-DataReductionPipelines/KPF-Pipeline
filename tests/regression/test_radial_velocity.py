"""Tests for the RadialVelocity module (KPF4 -> KPF4: fit per-order CCFs to RVs).

Static-method unit tests (_compute_rv_1d, _ccf_noise_corr_length,
_pixel_velocity_scale) build synthetic CCFs with no fixtures. Integration tests
build a synthetic KPF2, run CrossCorrelation (mask monkeypatched, narrow grid) to
get a CCF-bearing L4, then exercise RadialVelocity on it.
"""

import copy
import logging

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

from kpfpipe import CHIPS, DETECTOR, FIBERS
from kpfpipe.data_models.level2 import KPF2, NORDER_GREEN
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.modules.cross_correlation import CrossCorrelation
from kpfpipe.modules.radial_velocity import RadialVelocity
from kpfpipe.utils.astro import compute_doppler_factor

from ._catalog import seed_sci2_cards
from ._dtype_policy import RV_FLOAT, assert_dtype
from ._science import (
    MASK_CENTERS,
    NCOL,
    RANGE_KMS,
    SPEED_OF_LIGHT_KMS,
    absorption_spectrum,
    make_mask,
)

NORDER = DETECTOR["numorder"]
NORDER_RED = DETECTOR["norder"]["RED"]

# Injected RV [km/s], deliberately OFF the 0.25 km/s CCF grid so recovery
# exercises the Gaussian fit rather than landing on a grid node.
V_TRUE = 1.37
# Noiseless RV recovery tolerance [km/s] = 1 m/s, the scale of an EPRV
# instrument-error budget term. The CCF chain's own noiseless bias is ~0.3 m/s
# for these 4 km/s lines: the mask holes have a fixed velocity width, so the CCF
# continuum tilts as (1 + z(v)) and the constant-baseline Gaussian fit pulls the
# centroid by about -(continuum/depth) * sigma**2 / c.
RV_TOL = 1e-3


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_l4(
    monkeypatch,
    sci_obj="Target",
    sky_obj="Sky",
    cal_obj="None",
    berv=0.0,
    bjd=0.0,
    sysrv=0.0,
    perform_fibers=None,
    v_green=V_TRUE,
    v_red=V_TRUE,
):
    """Synthetic KPF2 (absorption at MASK_CENTERS, relativistically shifted by
    ``v_green``/``v_red`` on the respective chip) run through CrossCorrelation
    (mask stubbed, narrow grid) into a KPF4."""
    kpf2 = KPF2()
    seed_sci2_cards(kpf2, sci_obj=sci_obj, sky_obj=sky_obj, cal_obj=cal_obj)
    kpf2.headers["PRIMARY"]["CRV3"] = sysrv  # the stellar CCF grid center

    wave_1d = np.linspace(5000.0, 5050.0, NCOL)
    for chip, n, v in [("GREEN", NORDER_GREEN, v_green), ("RED", NORDER_RED, v_red)]:
        lam_obs = MASK_CENTERS * compute_doppler_factor(v * u.km / u.s)
        flux_1d = absorption_spectrum(wave_1d, lam_obs)
        for fiber in FIBERS:
            kpf2.set_data(
                f"{chip}_{fiber}_WAVE", np.tile(wave_1d, (n, 1)).astype(np.float64)
            )
            kpf2.set_data(
                f"{chip}_{fiber}_FLUX", np.tile(flux_1d, (n, 1)).astype(np.float32)
            )
            kpf2.set_data(
                f"{chip}_{fiber}_VAR", np.tile(flux_1d, (n, 1)).astype(np.float32)
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


# ------------------------------------------------------------------
# _compute_rv_1d / _ccf_noise_corr_length (staticmethods)
# ------------------------------------------------------------------


def _gaussian_ccf(v0=0.0, sigma=4.0, baseline=100.0, depth=30.0):
    vel = np.arange(-402, 403) * 0.25
    ccf = baseline - depth * np.exp(-0.5 * ((vel - v0) / sigma) ** 2)
    ccf_var = ccf.copy()  # photon-noise proxy at the CCF count scale
    vps = RadialVelocity._pixel_velocity_scale(5000.0, 5050.0, 4000)
    return vel, ccf, ccf_var, vps


class TestComputeRV:
    def test_recovers_injected_rv(self):
        # Off-grid v0; a Gaussian fit to an exact Gaussian CCF has no model
        # error, so the 1 m/s budget is generous.
        vel, ccf, ccf_var, vps = _gaussian_ccf(v0=2.37)
        rv, _ = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        assert rv == pytest.approx(2.37, abs=RV_TOL)

    def test_error_finite_and_positive(self):
        vel, ccf, ccf_var, vps = _gaussian_ccf(v0=0.0)
        _, rv_err = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        assert np.isfinite(rv_err) and rv_err > 0

    def test_even_window_pts_allowed(self):
        # min_npts is a minimum point count, not a centered odd window.
        vel, ccf, ccf_var, vps = _gaussian_ccf(v0=1.13)
        rv, _ = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 8
        )
        assert rv == pytest.approx(1.13, abs=RV_TOL)

    def test_nonphysical_variance_raises(self):
        # A non-positive CCF variance in the fit window is corrupt flux, not a
        # recoverable "no error" case -> fail loudly rather than return NaN.
        vel, ccf, ccf_var, vps = _gaussian_ccf(v0=0.0)
        ccf_var[402] = -1.0  # index of v = 0, inside the +/-3 sigma fit window
        with pytest.raises(ValueError, match="non-physical"):
            RadialVelocity._compute_rv_1d(
                vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
            )

    def test_negative_pixels_with_positive_sum_allowed(self):
        # Low-illumination (e.g. SKY) CCFs dip below zero; only a negative window
        # sum is non-physical.
        vel, ccf, _, vps = _gaussian_ccf(baseline=20.0)
        assert np.any(ccf < 0)
        ccf_var = np.full_like(ccf, 100.0)
        rv, rv_err = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        assert rv == pytest.approx(0.0, abs=RV_TOL)
        assert np.isfinite(rv_err) and rv_err > 0

    def test_negative_sum_raises(self):
        # fit_nsigma=3 keeps the window on the dip, so the window sum is < 0.
        vel, ccf, _, vps = _gaussian_ccf(baseline=5.0)
        ccf_var = np.full_like(ccf, 100.0)
        with pytest.raises(ValueError, match="non-physical"):
            RadialVelocity._compute_rv_1d(
                vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 3.0
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
        vel, ccf, ccf_var, vps = _gaussian_ccf(v0=1.0)
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
        assert rv == pytest.approx(v0, abs=RV_TOL) and np.isnan(rv_err)

    def test_narrow_window_returns_nan(self):
        # A first-pass window narrower than min_npts grid points -> NaN, NaN.
        vel, ccf, ccf_var, vps = _gaussian_ccf(v0=0.0)
        rv, rv_err = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-0.1, 0.1], 11
        )
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_first_pass_fit_failure_returns_nan(self, monkeypatch):
        # optimize_lsq raising on the first pass degrades to NaN, not a crash.
        vel, ccf, ccf_var, vps = _gaussian_ccf(v0=0.0)

        def boom(*args, **kwargs):
            raise RuntimeError("singular matrix")

        monkeypatch.setattr("kpfpipe.modules.radial_velocity.optimize_lsq", boom)
        rv, rv_err = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_nonfinite_fit_params_return_nan(self, monkeypatch):
        vel, ccf, ccf_var, vps = _gaussian_ccf(v0=0.0)

        def bad_fit(*args, **kwargs):
            return np.array([100.0, 30.0, np.nan, 4.0]), None

        monkeypatch.setattr("kpfpipe.modules.radial_velocity.optimize_lsq", bad_fit)
        rv, rv_err = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_second_pass_fit_failure_keeps_first_pass_rv(self, monkeypatch):
        vel, ccf, ccf_var, vps = _gaussian_ccf(v0=0.0)
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
        vel, ccf, ccf_var, vps = _gaussian_ccf(v0=0.0)
        _, err_a = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        _, err_b = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 1.0, [-50.0, 50.0], 11
        )
        la = RadialVelocity._ccf_noise_corr_length(vps, 0.5)
        lb = RadialVelocity._ccf_noise_corr_length(vps, 1.0)
        assert err_b / err_a == pytest.approx((lb / la) ** 0.5, rel=1e-6)


class TestRVErrorCalibration:
    """RV_ERR must predict the actual scatter of RVs under photon noise."""

    def test_rv_err_matches_monte_carlo_scatter(self):
        # 200 seeded Poisson-like realizations of one order through the real
        # CCF + fit. The std of a 200-sample estimate is uncertain by
        # 1/sqrt(2*200) = 5%, so [0.8, 1.2] is a 4-sigma band: a mis-scaled error
        # (e.g. a missing Boisse N_scale, sqrt(2) ~ 1.41 or worse) fails it.
        rng = np.random.default_rng(20261031)
        wave = np.linspace(5000.0, 5050.0, 6000)  # ~0.5 km/s per pixel
        centers = np.linspace(5008.0, 5042.0, 40)
        mask_width = 0.5
        mask = make_mask(centers, width=mask_width)
        vel = np.arange(-60, 61) * 0.25
        flux0 = 1e4 * absorption_spectrum(wave, centers, sigma_kms=3.0)
        vps = RadialVelocity._pixel_velocity_scale(wave[0], wave[-1], wave.size)
        rvs, errs = [], []
        for _ in range(200):
            flux = flux0 + rng.normal(0.0, np.sqrt(flux0))
            ccf, ccf_var = CrossCorrelation._compute_ccf_1d(
                wave, flux, flux0, mask, vel, 0.0
            )
            rv, rv_err = RadialVelocity._compute_rv_1d(
                vel, ccf, ccf_var, vps, mask_width, [-10.0, 10.0]
            )
            rvs.append(rv)
            errs.append(rv_err)
        ratio = np.std(rvs) / np.median(errs)
        assert 0.8 <= ratio <= 1.2


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


# ------------------------------------------------------------------
# compute_order_by_order_rvs / compute_weighted_rvs (L4 CCFs loaded)
# ------------------------------------------------------------------


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
    rv_module._load_ccfs(CHIPS, FIBERS)
    return rv_module


class TestComputeRVPublic:
    def test_returns_rv_dict(self, rv_loaded):
        res = rv_loaded.compute_order_by_order_rvs("GREEN", "SCI2")
        assert set(res) == {"rv", "rv_err"}
        assert res["rv"].shape == (NORDER_GREEN,)
        assert res["rv_err"].shape == (NORDER_GREEN,)

    def test_without_loaded_ccf_raises(self, rv_module):
        with pytest.raises(RuntimeError, match="not loaded"):
            rv_module.compute_order_by_order_rvs("GREEN", "SCI2")

    def test_recovers_injected_rv(self, rv_loaded):
        rv = rv_loaded.compute_order_by_order_rvs("GREEN", "SCI2")["rv"]
        np.testing.assert_allclose(rv, V_TRUE, atol=RV_TOL)

    def test_per_ccd_rv_recovers_injected(self, rv_loaded):
        # The per-CCD error comes from the unweighted-summed CCF.
        ccd_rv, ccd_rv_err = rv_loaded.compute_weighted_rvs(
            ["GREEN"], "SCI2", combine_fibers=False, combine_ccds=False
        )["GREEN"]
        assert ccd_rv == pytest.approx(V_TRUE, abs=RV_TOL)
        assert np.isfinite(ccd_rv_err) and ccd_rv_err > 0

    def test_per_ccd_rv_sums_science_fibers(self, rv_loaded):
        # combine_fibers=True sums the three science fibers' cached CCFs before
        # fitting, giving the SCI-combined per-CCD RV.
        ccd_rv, ccd_rv_err = rv_loaded.compute_weighted_rvs(
            ["GREEN"], ["SCI1", "SCI2", "SCI3"], combine_fibers=True, combine_ccds=False
        )["GREEN"]
        assert ccd_rv == pytest.approx(V_TRUE, abs=RV_TOL)
        assert np.isfinite(ccd_rv_err) and ccd_rv_err > 0

    def test_combine_ccds_returns_tuple_and_recovers_injected(self, rv_loaded):
        # combine_ccds=True combines at the RV level, not the CCF level.
        out = rv_loaded.compute_weighted_rvs(
            CHIPS, "SCI2", combine_fibers=False, combine_ccds=True
        )
        assert isinstance(out, tuple) and len(out) == 2
        assert out[0] == pytest.approx(V_TRUE, abs=RV_TOL)
        assert np.isfinite(out[1]) and out[1] > 0

    def test_combine_fibers_without_three_sci_raises(self, rv_loaded):
        with pytest.raises(ValueError, match="three science fibers"):
            rv_loaded.compute_weighted_rvs(
                ["GREEN"], "SCI2", combine_fibers=True, combine_ccds=False
            )

    def test_no_combine_fibers_without_single_raises(self, rv_loaded):
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

    def _raise_nonphysical(self, *args, **kwargs):
        raise ValueError("non-physical CCF or CCF variance in the RV fit window")

    def test_weighted_sky_nonphysical_warns_and_degrades_to_nan(
        self, rv_loaded, monkeypatch, caplog
    ):
        monkeypatch.setattr(rv_loaded, "_compute_rv_1d", self._raise_nonphysical)
        with caplog.at_level(logging.WARNING, logger="kpfpipe"):
            out = rv_loaded.compute_weighted_rvs(
                ["GREEN"], "SKY", combine_fibers=False, combine_ccds=False
            )
        assert np.isnan(out["GREEN"][0]) and np.isnan(out["GREEN"][1])
        assert "GREEN_SKY: non-physical CCF window" in caplog.text

    @pytest.mark.parametrize(
        "fibers, combine_fibers",
        [("SCI2", False), (["SCI1", "SCI2", "SCI3"], True)],
    )
    def test_weighted_science_nonphysical_raises(
        self, rv_loaded, monkeypatch, fibers, combine_fibers
    ):
        monkeypatch.setattr(rv_loaded, "_compute_rv_1d", self._raise_nonphysical)
        with pytest.raises(ValueError, match="non-physical"):
            rv_loaded.compute_weighted_rvs(
                ["GREEN"], fibers, combine_fibers=combine_fibers, combine_ccds=False
            )

    def test_order_weights_read_from_rvn(self, rv_loaded):
        # _get_order_weights slices the WEIGHT column of the L4 RVn table.
        w = rv_loaded._get_order_weights("GREEN", "SCI2")
        assert w.shape == (NORDER_GREEN,) and np.all(w >= 0)
        expected = np.asarray(
            rv_loaded.l4_obj.data["GREEN_SCI2_RV"]["WEIGHT"], dtype=float
        )
        np.testing.assert_array_equal(w, expected)


class TestDeterminism:
    def test_ccf_and_rv_bitwise_reproducible(self, monkeypatch, rv_l4):
        # The same KPF2 must give bit-identical CCFs, CCF variances and RVs on a
        # rebuild: RV stability work compares frames years apart, so any hidden
        # state or order dependence in the chain is a science defect.
        rebuilt = _make_l4(monkeypatch)
        for fiber in ("SCI2", "SKY"):
            np.testing.assert_array_equal(
                rebuilt.data[f"{fiber}_CCF"], rv_l4.data[f"{fiber}_CCF"]
            )
            np.testing.assert_array_equal(
                rebuilt.data[f"{fiber}_CCF_VAR"], rv_l4.data[f"{fiber}_CCF_VAR"]
            )
        config = {"rv_window": RANGE_KMS}
        first = RadialVelocity(rebuilt, config=config).perform()
        second = RadialVelocity(copy.deepcopy(rv_l4), config=config).perform()
        for fiber in ("SCI2", "SKY"):
            for column in ("RV", "RV_ERR"):
                np.testing.assert_array_equal(
                    np.asarray(first.data[f"{fiber}_RV"][column]),
                    np.asarray(second.data[f"{fiber}_RV"][column]),
                )
        for card in ("RV", "RVERR", "RVGREEN", "RVRED"):
            assert first.headers["PRIMARY"][card] == second.headers["PRIMARY"][card]


# ------------------------------------------------------------------
# perform (KPF4 -> KPF4)
# ------------------------------------------------------------------


class TestPerform:
    """perform() on the shared default run, plus the option and failure paths.

    Not marked ``slow``: the class runs on synthetic CCFs in well under a
    second, and it holds the only coverage of the combined-RV contract, the RV
    header keywords and the L4 serialization -- which belong in the default
    pre-commit loop.
    """

    _ILLUMINATED = ["SCI1", "SCI2", "SCI3", "SKY"]  # CLSRC5='None' -> skipped

    @pytest.fixture(scope="module")
    def performed(self, rv_l4):
        """One default-argument perform(), shared read-only.

        Seven tests inspect different facets of the same result and each was
        recomputing it. Tests that pass explicit chips/fibers, or that mutate
        the L4 first, keep taking the per-test ``rv_module``.
        """
        module = RadialVelocity(copy.deepcopy(rv_l4), config={"rv_window": RANGE_KMS})
        return module, module.perform()

    def test_fills_rv_columns_preserving_metadata(self, performed):
        _, l4 = performed
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

    def test_unilluminated_fiber_skipped(self, performed):
        # CLSRC5='None' -> CrossCorrelation wrote no CAL CCF or RV table.
        _, l4 = performed
        assert l4.data["CAL_CCF"].size == 0
        assert len(l4.data["CAL_RV"]) == 0

    def test_recovers_injected_rv_illuminated_orderlets(self, performed):
        _, l4 = performed
        for fiber in self._ILLUMINATED:
            rv = np.asarray(l4.data[f"{fiber}_RV"]["RV"])
            np.testing.assert_allclose(rv, V_TRUE, atol=RV_TOL)

    def test_rv_headers(self, performed):
        _, l4 = performed
        rv_hdr = l4.headers["SCI2_RV"]
        assert rv_hdr["RVMETHOD"] == "CCF"
        assert rv_hdr["SKYRMVD"] is False
        assert rv_hdr["TELLRMVD"] is False
        assert l4.headers["PRIMARY"]["RVMETHOD"] == "CCF"

    def test_reported_rvs_are_absolute(self, monkeypatch):
        # SYSVEL is what was subtracted from the RVs, not the systemic velocity
        # itself. Centering the CCF grid on the injected velocity does not move
        # the answer: the grid stays in absolute velocity, so the fitted RV
        # carries the systemic RV and nothing takes it back out.
        l4 = _make_l4(monkeypatch, sysrv=V_TRUE)
        module = RadialVelocity(l4, config={"rv_window": RANGE_KMS})
        prim = module.perform().headers["PRIMARY"]
        assert prim["RV"] == pytest.approx(V_TRUE, abs=RV_TOL)
        assert prim["SYSVEL"] == 0.0
        assert prim["SYSACC"] == 0.0

    def test_per_ccd_rv_keywords(self, performed):
        # Per-orderlet RVs are written as RV{chip}/ERV{chip} onto that fiber's own
        # RV# table header (SCI2 lives on RV3), not onto PRIMARY.
        _, l4 = performed
        rv_hdr = l4.headers["RV3"]
        assert rv_hdr["RVGREEN"] == pytest.approx(V_TRUE, abs=RV_TOL)
        assert rv_hdr["RVRED"] == pytest.approx(V_TRUE, abs=RV_TOL)
        assert rv_hdr["ERVGREEN"] > 0 and rv_hdr["ERVRED"] > 0
        # RVGREEN is homed on PRIMARY and on every RV# table: the PRIMARY card is
        # the SCI-combined value, a separate write from this per-orderlet one.
        # Not asserted as an inequality -- every fiber in this fixture carries the
        # same synthetic spectrum, so the two writes agree to the bit.
        assert "RVGREEN" in l4.headers["PRIMARY"]

    def test_combined_rv_populated(self, performed):
        # PRIMARY: EPRV RV/RVERR plus the KPF SCI-combined per-CCD RVGREEN/RVRED.
        _, l4 = performed
        prim = l4.headers["PRIMARY"]
        assert prim["RVGREEN"] == pytest.approx(V_TRUE, abs=RV_TOL)
        assert prim["RVRED"] == pytest.approx(V_TRUE, abs=RV_TOL)
        assert prim["ERVGREEN"] > 0 and prim["ERVRED"] > 0
        assert prim["RV"] == pytest.approx(V_TRUE, abs=RV_TOL)
        assert prim["RVERR"] > 0
        assert prim["RVMETHOD"] == "CCF"
        # RVGREEN is homed on PRIMARY and on every RV# table, so RV3 carries its
        # own per-orderlet value; the EPRV combined RV is PRIMARY-only.
        assert "RVGREEN" in l4.headers["RV3"]
        assert "RV" not in l4.headers["RV3"]

    def test_combined_rv_is_weighted_ccd_combine(self, monkeypatch):
        # PRIMARY RV = (v_g*W_g + v_r*W_r)/(W_g + W_r), W the chip's summed CCF
        # order weights. Distinct off-grid per-chip velocities make the weighting
        # observable (equal velocities pass under any weights). The sums are an
        # oracle hand-pinned from reference/ccf_order_weights.csv (G2_espresso:
        # BP-RP 0.823 -> 5770 K): GREEN 12.2085, RED 2.9594. Only the RV value is
        # pinned; the RVERR estimator is an open design question.
        v_green, v_red = 1.12, 2.07
        w_green, w_red = 12.2085, 2.9594
        l4 = _make_l4(monkeypatch, v_green=v_green, v_red=v_red)
        prim = (
            RadialVelocity(l4, config={"rv_window": RANGE_KMS})
            .perform()
            .headers["PRIMARY"]
        )
        assert prim["RVGREEN"] == pytest.approx(v_green, abs=RV_TOL)
        assert prim["RVRED"] == pytest.approx(v_red, abs=RV_TOL)
        expect_rv = (v_green * w_green + v_red * w_red) / (w_green + w_red)
        assert prim["RV"] == pytest.approx(expect_rv, abs=RV_TOL)

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
        l4 = _make_l4(monkeypatch, sci_obj="None", sky_obj="None", cal_obj="ThAr_daily")
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
        assert prim["RV"] == pytest.approx(prim["RVGREEN"], abs=1e-9)
        assert "only chip GREEN present" in caplog.text

    def test_l4_serializes_to_fits(self, rv_module, tmp_path):
        # SCI2's per-orderlet RVs land on the RV3 table.
        l4 = rv_module.perform(fibers=["SCI1", "SCI2", "SCI3"])
        path = tmp_path / "kpf_SL4_20240405T000000.fits"
        l4.to_fits(str(path))
        with fits.open(path) as hdul:
            rv = hdul["RV3"].header
            assert rv["RVMETHOD"] == "CCF"
            assert rv["RVGREEN"] == pytest.approx(V_TRUE, abs=RV_TOL)
            # Only the EPRV combined RV belongs on PRIMARY.
            assert hdul["PRIMARY"].header["RV"] == pytest.approx(V_TRUE, abs=RV_TOL)
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
        assert "RVGREEN" in rv_hdr and rv_hdr["RVGREEN"] is None
        assert "RVRED" in rv_hdr and rv_hdr["RVRED"] is None

    def test_explicit_chips_and_fibers(self, rv_module):
        l4 = rv_module.perform(chips=["GREEN"], fibers=["SCI1", "SCI2", "SCI3"])
        # GREEN-only: science green RVs filled, red rows left NaN.
        rv = np.asarray(l4.data["SCI1_RV"]["RV"])
        assert np.all(np.isfinite(rv[:NORDER_GREEN]))
        assert np.all(np.isnan(rv[NORDER_GREEN:]))
        # SKY not requested -> its RV column stays NaN (CrossCorrelation seeded it).
        assert np.all(np.isnan(np.asarray(l4.data["SKY_RV"]["RV"])))


# ------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------


class TestConstructor:
    def test_invalid_config_type_raises(self):
        with pytest.raises(TypeError, match="None, dict, or ConfigHandler"):
            RadialVelocity(KPF4(), config="not-a-config")

    def test_rv_window_default(self):
        assert RadialVelocity(KPF4()).rv_window == [-25.0, 25.0]

    def test_dict_config_overrides_default(self):
        rv = RadialVelocity(KPF4(), config={"rv_window": [-10.0, 10.0]})
        assert rv.rv_window == [-10.0, 10.0]


# ------------------------------------------------------------------
# Dtype provenance
# ------------------------------------------------------------------


class TestDtypeProvenance:
    """RV and RV_ERR come out at the declared RV-column precision."""

    def test_returns_float64(self):
        vel, ccf, ccf_var, vps = _gaussian_ccf(v0=1.0)
        rv, rv_err = RadialVelocity._compute_rv_1d(
            vel, ccf, ccf_var, vps, 0.5, [-50.0, 50.0], 11
        )
        assert_dtype(np.asarray(rv), RV_FLOAT, "RV scalar")
        assert_dtype(np.asarray(rv_err), RV_FLOAT, "RV_ERR scalar")
