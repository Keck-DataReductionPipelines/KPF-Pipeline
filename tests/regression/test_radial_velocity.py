"""
Tests for the RadialVelocity module (KPF2 -> KPF4: per-orderlet CCFs and RVs).

Static-method unit tests (_compute_ccf_1d, _compute_rv_1d) build synthetic spectra
and CCFs with no fixtures. Build-helper tests use a header-only KPF2 and read
the real on-disk line masks. Integration tests
(compute_ccfs/compute_order_by_order_rvs/perform) use a synthetic KPF2 with
absorption injected at a monkeypatched line mask, and a narrow velocity grid
for speed.
"""

import numpy as np
import pytest
from astropy.constants import c
from astropy.io import fits

from kpfpipe.data_models.level2 import KPF2, NORDER_GREEN, NORDER_RED
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.modules.radial_velocity import RadialVelocity

from ._dtype_policy import CCF, RV_FLOAT, assert_dtype

NORDER = NORDER_GREEN + NORDER_RED
SPEED_OF_LIGHT_KMS = np.float64(c.to("km/s").value)
_FIBERS = ["CAL", "SCI1", "SCI2", "SCI3", "SKY"]  # all orderlets

# Narrow CCF grid for fast integration tests; wide enough for the second-pass
# +/-3 sigma window (sigma ~ 4 km/s) to stay on-grid: +/-15 km/s at 0.25 km/s.
_RANGE_KMS = [-15.0, 15.0]
_STEP_KMS = 0.25  # matches the module default
_NVEL = round((_RANGE_KMS[1] - _RANGE_KMS[0]) / _STEP_KMS) + 1
_V_INJECT = 1.5  # injected RV [km/s], on the grid
_MASK_CENTERS = np.linspace(5015.0, 5035.0, 30)  # vacuum line centers [Å]
# Wide enough that the default compute_ccfs clip (clip_edge_pixels=(500, 500))
# trims the order edges but leaves the 5015-5035 Å mask lines well inside.
NCOL = 2000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mask(centers, weights=None, width=0.5):
    """Build a line-mask dict matching _build_line_mask's structure."""
    centers = np.asarray(centers, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(centers)
    half_width = centers * (width / SPEED_OF_LIGHT_KMS)
    return {
        "center": centers,
        "weight": np.asarray(weights, dtype=np.float64),
        "start": centers - half_width,
        "end": centers + half_width,
    }


def _absorption_spectrum(wave, centers, weights=None, depth=0.6, sigma_kms=4.0):
    """Unit continuum with Gaussian absorption lines at `centers`."""
    if weights is None:
        weights = np.ones_like(centers)
    flux = np.ones_like(wave)
    for center, weight in zip(centers, weights, strict=False):
        sigma_a = center * sigma_kms / SPEED_OF_LIGHT_KMS
        flux -= (
            depth
            * (weight / np.max(weights))
            * np.exp(-0.5 * ((wave - center) / sigma_a) ** 2)
        )
    return flux


# ---------------------------------------------------------------------------
# _compute_ccf_1d (staticmethod)
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
        ccf = RadialVelocity._compute_ccf_1d(wave, flux, mask, vel, 0.0)
        assert vel[np.argmin(ccf)] == pytest.approx(3.0, abs=0.3)

    def test_zero_velocity_dip(self):
        wave, flux, mask, vel = self._order(v_dip=0.0, z=0.0)
        ccf = RadialVelocity._compute_ccf_1d(wave, flux, mask, vel, 0.0)
        assert vel[np.argmin(ccf)] == pytest.approx(0.0, abs=0.3)

    def test_barycorr_z_folds_in(self):
        z = 5.0 / SPEED_OF_LIGHT_KMS
        wave, flux, mask, vel = self._order(v_dip=2.0, z=z)
        ccf = RadialVelocity._compute_ccf_1d(wave, flux, mask, vel, z)
        assert vel[np.argmin(ccf)] == pytest.approx(2.0, abs=0.3)

    def test_descending_wave_raises(self):
        wave, flux, mask, vel = self._order(v_dip=0.0)
        with pytest.raises(ValueError, match="descending"):
            RadialVelocity._compute_ccf_1d(wave[::-1], flux[::-1], mask, vel, 0.0)

    def test_constant_wave_returns_zeros(self):
        wave, flux, mask, vel = self._order()
        ccf = RadialVelocity._compute_ccf_1d(
            np.full_like(wave, 5000.0), flux, mask, vel, 0.0
        )
        assert not np.any(ccf)

    def test_nan_wave_returns_zeros(self):
        wave, flux, mask, vel = self._order()
        wave = wave.copy()
        wave[1000] = np.nan
        ccf = RadialVelocity._compute_ccf_1d(wave, flux, mask, vel, 0.0)
        assert not np.any(ccf)

    def test_no_lines_in_order_returns_zeros(self):
        wave = np.linspace(6000.0, 6050.0, 2000)
        flux = np.ones_like(wave)
        mask = _make_mask(np.linspace(5008.0, 5042.0, 20))  # all outside the order
        vel = np.arange(-402, 403) * 0.25
        ccf = RadialVelocity._compute_ccf_1d(wave, flux, mask, vel, 0.0)
        assert not np.any(ccf)


# ---------------------------------------------------------------------------
# Dtype provenance (see tests/regression/_dtype_policy.py)
# ---------------------------------------------------------------------------


class TestDtypeProvenance:
    """CCF cubes and RV-table floats are float64; a float64 CCF from float32
    flux is the intended deliberate upcast, governed by the result's dtype."""

    def _order(self, v_dip=0.0):
        wave = np.linspace(5000.0, 5050.0, 2000)
        centers = np.linspace(5008.0, 5042.0, 20)
        mask = _make_mask(centers)
        flux = _absorption_spectrum(wave, centers)
        vel = np.arange(-402, 403) * 0.25
        return wave, flux, mask, vel

    def test_ccf_1d_is_float64_from_float32_flux(self):
        wave, flux, mask, vel = self._order()
        ccf = RadialVelocity._compute_ccf_1d(
            wave, flux.astype(np.float32), mask, vel, 0.0
        )
        assert_dtype(ccf, CCF, "CCF (_compute_ccf_1d, float32 flux in)")

    def test_compute_rv_1d_returns_float64(self):
        wave, flux, mask, vel = self._order()
        ccf = RadialVelocity._compute_ccf_1d(wave, flux, mask, vel, 0.0)
        rv, rv_err = RadialVelocity._compute_rv_1d(vel, ccf, wave, [-50.0, 50.0], 11)
        assert_dtype(np.asarray(rv), RV_FLOAT, "RV scalar")
        assert_dtype(np.asarray(rv_err), RV_FLOAT, "RV_ERR scalar")


# ---------------------------------------------------------------------------
# _compute_rv_1d (staticmethod)
# ---------------------------------------------------------------------------


class TestComputeRV:
    def _ccf(self, v0=0.0, sigma=4.0, baseline=100.0, depth=30.0):
        vel = np.arange(-402, 403) * 0.25
        ccf = baseline - depth * np.exp(-0.5 * ((vel - v0) / sigma) ** 2)
        wave = np.linspace(5000.0, 5050.0, 4000)
        return vel, ccf, wave

    def test_recovers_injected_rv(self):
        vel, ccf, wave = self._ccf(v0=2.5)
        rv, rv_err = RadialVelocity._compute_rv_1d(vel, ccf, wave, [-50.0, 50.0], 11)
        assert rv == pytest.approx(2.5, abs=0.05)

    def test_error_finite_and_positive(self):
        vel, ccf, wave = self._ccf(v0=0.0)
        _, rv_err = RadialVelocity._compute_rv_1d(vel, ccf, wave, [-50.0, 50.0], 11)
        assert np.isfinite(rv_err) and rv_err > 0

    def test_even_window_pts_allowed(self):
        # min_npts is a minimum point count, not a centered odd window.
        vel, ccf, wave = self._ccf(v0=1.0)
        rv, _ = RadialVelocity._compute_rv_1d(vel, ccf, wave, [-50.0, 50.0], 8)
        assert rv == pytest.approx(1.0, abs=0.05)

    def test_flat_ccf_returns_nan(self):
        vel = np.arange(-402, 403) * 0.25
        ccf = np.full_like(vel, 100.0)
        wave = np.linspace(5000.0, 5050.0, 4000)
        rv, rv_err = RadialVelocity._compute_rv_1d(vel, ccf, wave, [-50.0, 50.0], 11)
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_nonfinite_ccf_returns_nan(self):
        # Non-finite CCF values fail loudly (NaN) rather than being masked out.
        vel, ccf, wave = self._ccf(v0=1.0)
        ccf[10] = np.nan
        rv, rv_err = RadialVelocity._compute_rv_1d(vel, ccf, wave, [-50.0, 50.0], 11)
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_second_pass_off_grid_returns_first_pass_rv(self):
        # Dip near the low edge: the first-pass fit succeeds, but the +/-3 sigma
        # second-pass window runs off the grid -> first-pass RV, NaN error.
        vel = np.arange(-402, 403) * 0.25
        v0 = vel[20]
        ccf = 100.0 - 30.0 * np.exp(-0.5 * ((vel - v0) / 4.0) ** 2)
        wave = np.linspace(5000.0, 5050.0, 4000)
        rv, rv_err = RadialVelocity._compute_rv_1d(vel, ccf, wave, [-50.0, 50.0], 11)
        assert rv == pytest.approx(v0, abs=0.1) and np.isnan(rv_err)

    def test_min_points_floor_does_not_change_result(self):
        # A larger min-points floor below the +/-3 sigma window doesn't shift the fit.
        vel, ccf, wave = self._ccf(v0=1.0)
        rv9, _ = RadialVelocity._compute_rv_1d(vel, ccf, wave, [-50.0, 50.0], 9)
        rv21, _ = RadialVelocity._compute_rv_1d(vel, ccf, wave, [-50.0, 50.0], 21)
        assert rv9 == pytest.approx(1.0, abs=0.05)
        assert rv21 == pytest.approx(1.0, abs=0.05)

    def test_narrow_window_returns_nan(self):
        # A first-pass window narrower than min_npts grid points -> NaN, NaN.
        vel, ccf, wave = self._ccf(v0=0.0)
        rv, rv_err = RadialVelocity._compute_rv_1d(vel, ccf, wave, [-0.1, 0.1], 11)
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_first_pass_fit_failure_returns_nan(self, monkeypatch):
        # optimize_lsq raising on the first pass fails loudly as NaN, not a crash.
        vel, ccf, wave = self._ccf(v0=0.0)

        def boom(*args, **kwargs):
            raise RuntimeError("singular matrix")

        monkeypatch.setattr("kpfpipe.modules.radial_velocity.optimize_lsq", boom)
        rv, rv_err = RadialVelocity._compute_rv_1d(vel, ccf, wave, [-50.0, 50.0], 11)
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_nonfinite_fit_params_return_nan(self, monkeypatch):
        # A fit returning a non-finite mean/sigma is rejected as NaN.
        vel, ccf, wave = self._ccf(v0=0.0)

        def bad_fit(*args, **kwargs):
            return np.array([100.0, 30.0, np.nan, 4.0]), None

        monkeypatch.setattr("kpfpipe.modules.radial_velocity.optimize_lsq", bad_fit)
        rv, rv_err = RadialVelocity._compute_rv_1d(vel, ccf, wave, [-50.0, 50.0], 11)
        assert np.isnan(rv) and np.isnan(rv_err)

    def test_second_pass_fit_failure_keeps_first_pass_rv(self, monkeypatch):
        # If the refinement (second) fit raises, the first-pass mean is retained.
        vel, ccf, wave = self._ccf(v0=0.0)
        calls = []

        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return np.array([100.0, 30.0, 0.0, 4.0]), None
            raise RuntimeError("refinement failed")

        monkeypatch.setattr("kpfpipe.modules.radial_velocity.optimize_lsq", flaky)
        rv, _ = RadialVelocity._compute_rv_1d(vel, ccf, wave, [-50.0, 50.0], 11)
        assert np.isfinite(rv) and rv == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# _build_line_mask / _build_velocity_grid  (header-only fixture)
# ---------------------------------------------------------------------------


@pytest.fixture
def header_kpf2():
    """KPF2 with only the header keywords the build/dispatch helpers need."""
    kpf2 = KPF2()
    kpf2.headers["INSTRUMENT_HEADER"]["TARGTEFF"] = 5772.0
    kpf2.headers["INSTRUMENT_HEADER"]["TARGRADV"] = 0.0
    kpf2.headers["INSTRUMENT_HEADER"]["SCI-OBJ"] = "Target"
    kpf2.headers["INSTRUMENT_HEADER"]["SKY-OBJ"] = "Sky"
    kpf2.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = "None"
    return kpf2


class TestDispatch:
    """Illumination-source resolution and per-source CCF configuration."""

    @pytest.mark.parametrize(
        "raw, obj",
        [
            ("Target", "target"),
            ("Sky", "sky"),
            ("Th_gold", "thar"),
            ("Th_daily", "thar"),
            ("None", "none"),
            ("TARGET", "target"),
        ],
    )
    def test_raw_value_normalizes_to_object(self, header_kpf2, raw, obj):
        header_kpf2.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = raw
        assert (
            RadialVelocity(header_kpf2)._resolve_illumination_source("GREEN", "CAL")[
                "object"
            ]
            == obj
        )

    def test_unrecognized_source_raises(self, header_kpf2):
        header_kpf2.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = "Frobnicator"
        with pytest.raises(ValueError, match="unrecognized illumination"):
            RadialVelocity(header_kpf2)._resolve_illumination_source("GREEN", "CAL")

    def test_resolve_illumination_per_fiber(self, header_kpf2):
        rv = RadialVelocity(header_kpf2)
        assert rv._resolve_illumination_source("GREEN", "SCI2")["object"] == "target"
        assert rv._resolve_illumination_source("RED", "SKY")["object"] == "sky"
        assert rv._resolve_illumination_source("GREEN", "CAL")["object"] == "none"

    def test_resolve_unknown_fiber_raises(self, header_kpf2):
        with pytest.raises(ValueError, match="unknown fiber"):
            RadialVelocity(header_kpf2)._resolve_illumination_source("GREEN", "BOGUS")

    def test_resolve_missing_keyword_raises(self, header_kpf2):
        del header_kpf2.headers["INSTRUMENT_HEADER"]["CAL-OBJ"]
        with pytest.raises(ValueError, match="CAL-OBJ"):
            RadialVelocity(header_kpf2)._resolve_illumination_source("GREEN", "CAL")

    def test_settings_target(self, header_kpf2):
        header_kpf2.headers["INSTRUMENT_HEADER"]["TARGRADV"] = 11.1  # SCI-OBJ='Target'
        s = RadialVelocity(header_kpf2)._resolve_illumination_source("GREEN", "SCI2")
        assert s == {
            "object": "target",
            "mask_name": "G2_espresso",
            "apply_barycorr": True,
            "vel_grid_center": 11.1,
        }

    def test_settings_sky(self, header_kpf2):
        s = RadialVelocity(header_kpf2)._resolve_illumination_source("GREEN", "SKY")
        assert s == {
            "object": "sky",
            "mask_name": "G2_espresso",
            "apply_barycorr": True,
            "vel_grid_center": 0.0,
        }

    def test_settings_thar(self, header_kpf2):
        header_kpf2.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = "Th_gold"
        s = RadialVelocity(header_kpf2)._resolve_illumination_source("GREEN", "CAL")
        assert s == {
            "object": "thar",
            "mask_name": "thar",
            "apply_barycorr": False,
            "vel_grid_center": 0.0,
        }

    def test_settings_none(self, header_kpf2):
        s = RadialVelocity(header_kpf2)._resolve_illumination_source(
            "GREEN", "CAL"
        )  # CAL-OBJ='None'
        assert s == {
            "object": "none",
            "mask_name": None,
            "apply_barycorr": None,
            "vel_grid_center": None,
        }

    @pytest.mark.parametrize(
        "raw, obj", [("EtalonFiber", "etalon"), ("LFCFiber", "lfc")]
    )
    def test_unimplemented_source_warns_and_skips(self, header_kpf2, raw, obj):
        header_kpf2.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = raw
        with pytest.warns(UserWarning, match=f"{obj}.*not implemented"):
            source = RadialVelocity(header_kpf2)._resolve_illumination_source(
                "GREEN", "CAL"
            )
        assert source == {
            "object": obj,
            "mask_name": None,
            "apply_barycorr": None,
            "vel_grid_center": None,
        }


class TestStellarMaskName:
    def test_targteff_selects_mask(self, header_kpf2):
        rv = RadialVelocity(header_kpf2)
        assert rv._resolve_stellar_mask() == "G2_espresso"  # 5772 K -> G2
        header_kpf2.headers["INSTRUMENT_HEADER"]["TARGTEFF"] = 4000.0
        assert RadialVelocity(header_kpf2)._resolve_stellar_mask() == "K6_espresso"

    @pytest.mark.parametrize("teff", [0.0, -100.0, "nan"])
    def test_invalid_targteff_raises(self, header_kpf2, teff):
        header_kpf2.headers["INSTRUMENT_HEADER"]["TARGTEFF"] = teff
        with pytest.raises(ValueError, match="TARGTEFF"):
            RadialVelocity(header_kpf2)._resolve_stellar_mask()

    def test_missing_targteff_raises(self, header_kpf2):
        del header_kpf2.headers["INSTRUMENT_HEADER"]["TARGTEFF"]
        with pytest.raises(ValueError, match="TARGTEFF"):
            RadialVelocity(header_kpf2)._resolve_stellar_mask()

    def test_missing_targradv_raises(self, header_kpf2):
        del header_kpf2.headers["INSTRUMENT_HEADER"]["TARGRADV"]
        with pytest.raises(ValueError, match="TARGRADV"):
            RadialVelocity(header_kpf2)._get_systemic_rv()


class TestBuildLineMask:
    def test_keys_and_shapes(self, header_kpf2):
        mask = RadialVelocity(header_kpf2)._build_line_mask(
            "GREEN", "SCI2"
        )  # SCI-OBJ='Target' -> G2
        assert set(mask) == {"center", "weight", "start", "end"}
        n = mask["center"].size
        assert all(mask[k].shape == (n,) for k in mask)

    def test_top_hat_edges_bracket_center(self, header_kpf2):
        mask = RadialVelocity(header_kpf2)._build_line_mask("GREEN", "SCI2")
        assert np.all(mask["start"] < mask["center"])
        assert np.all(mask["center"] < mask["end"])

    def test_cached(self, header_kpf2):
        rv = RadialVelocity(header_kpf2)
        assert rv._build_line_mask("GREEN", "SCI2") is rv._build_line_mask(
            "GREEN", "SCI2"
        )

    def test_thar_mask_uniform_weights(self, header_kpf2):
        header_kpf2.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = "Th_gold"  # -> thar mask
        mask = RadialVelocity(header_kpf2)._build_line_mask("GREEN", "CAL")
        assert np.all(mask["weight"] == 1.0)
        # ThAr centers are deduped and sorted (lines recur across overlapping orders).
        assert np.all(np.diff(mask["center"]) > 0)


class TestBuildVelocityGrid:
    def test_centered_on_systemic_rv(self, header_kpf2):
        header_kpf2.headers["INSTRUMENT_HEADER"]["TARGRADV"] = (
            10.0  # SCI2 grid center = TARGRADV
        )
        grid = RadialVelocity(header_kpf2)._build_velocity_grid("GREEN", "SCI2")
        assert grid.mean() == pytest.approx(10.0)

    def test_default_size_and_step(self, header_kpf2):
        grid = RadialVelocity(header_kpf2)._build_velocity_grid(
            "GREEN", "SKY"
        )  # SKY center = 0
        assert grid.size == 801  # [-100, 100] km/s at 0.25 -> arange(-400, 401)
        np.testing.assert_allclose(np.diff(grid), 0.25)

    def test_symmetric_about_zero_center(self, header_kpf2):
        grid = RadialVelocity(header_kpf2)._build_velocity_grid(
            "GREEN", "SKY"
        )  # center 0
        np.testing.assert_allclose(grid.min(), -grid.max())

    def test_cached(self, header_kpf2):
        rv = RadialVelocity(header_kpf2)
        assert rv._build_velocity_grid("GREEN", "SCI2") is rv._build_velocity_grid(
            "GREEN", "SCI2"
        )


# ---------------------------------------------------------------------------
# compute_ccfs / compute_order_by_order_rvs / perform
# (synthetic KPF2 + monkeypatched mask)
# ---------------------------------------------------------------------------


@pytest.fixture
def rv_kpf2():
    """KPF2 with identical per-order synthetic spectra (absorption at _MASK_CENTERS
    shifted by _V_INJECT) for every orderlet and zero barycentric correction."""
    kpf2 = KPF2()
    kpf2.headers["INSTRUMENT_HEADER"]["TARGTEFF"] = 5772.0
    kpf2.headers["INSTRUMENT_HEADER"]["TARGRADV"] = 0.0
    # Illumination sources: SCI on a star, SKY on sky, CAL dark (skipped).
    kpf2.headers["INSTRUMENT_HEADER"]["SCI-OBJ"] = "Target"
    kpf2.headers["INSTRUMENT_HEADER"]["SKY-OBJ"] = "Sky"
    kpf2.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = "None"

    wave_1d = np.linspace(5000.0, 5050.0, NCOL)
    lam_obs = _MASK_CENTERS * (1.0 + _V_INJECT / SPEED_OF_LIGHT_KMS)  # z = 0
    flux_1d = _absorption_spectrum(wave_1d, lam_obs)

    for chip, n in [("GREEN", NORDER_GREEN), ("RED", NORDER_RED)]:
        for fiber in _FIBERS:
            kpf2.set_data(
                f"{chip}_{fiber}_WAVE", np.tile(wave_1d, (n, 1)).astype(np.float64)
            )
            kpf2.set_data(
                f"{chip}_{fiber}_FLUX", np.tile(flux_1d, (n, 1)).astype(np.float64)
            )
    # Per-order barycentric extensions (populated together by BarycentricCorrection).
    kpf2.set_data("BARYCORR_Z", np.zeros(NORDER))
    kpf2.set_data("BARYCORR_KMS", np.zeros(NORDER))
    kpf2.set_data("BJD_TDB", np.zeros(NORDER))
    return kpf2


@pytest.fixture
def rv_module(rv_kpf2, monkeypatch):
    """RadialVelocity on a narrow grid with the line mask stubbed to _MASK_CENTERS."""
    mask = _make_mask(_MASK_CENTERS)
    monkeypatch.setattr(
        RadialVelocity, "_build_line_mask", lambda self, chip, fiber, width=None: mask
    )
    return RadialVelocity(rv_kpf2, config={"ccf_window": _RANGE_KMS})


class TestComputeCCFPublic:
    def test_returns_velocity_and_ccf(self, rv_module):
        res = rv_module.compute_ccfs("GREEN", "SCI2")
        assert set(res) == {"velocity", "ccf"}
        assert res["velocity"].shape == (_NVEL,)
        assert res["ccf"].shape == (NORDER_GREEN, _NVEL)

    def test_red_chip_shape(self, rv_module):
        res = rv_module.compute_ccfs("RED", "SCI1")
        assert res["ccf"].shape == (NORDER_RED, _NVEL)

    def test_dip_at_injected_velocity(self, rv_module):
        res = rv_module.compute_ccfs("GREEN", "SCI2")
        vel, ccf = res["velocity"], res["ccf"]
        assert vel[np.argmin(ccf[0])] == pytest.approx(_V_INJECT, abs=0.3)

    def test_caches_ccf(self, rv_module):
        res = rv_module.compute_ccfs("GREEN", "SCI2")
        assert rv_module._ccf["GREEN_SCI2"] is res["ccf"]

    def test_lowercase_chip_accepted(self, rv_module):
        res = rv_module.compute_ccfs("green", "sci2")
        assert res["ccf"].shape == (NORDER_GREEN, _NVEL)

    def test_missing_barycorr_z_raises(self, rv_kpf2, monkeypatch):
        monkeypatch.setattr(
            RadialVelocity,
            "_build_line_mask",
            lambda self, chip, fiber, width=None: _make_mask(_MASK_CENTERS),
        )
        rv_kpf2.set_data("BARYCORR_Z", np.array([]))
        rv = RadialVelocity(rv_kpf2, config={"ccf_window": _RANGE_KMS})
        with pytest.raises(ValueError, match="BARYCORR_Z"):
            rv.compute_ccfs("GREEN", "SCI2")

    def test_all_zero_ccf_raises(self, rv_module):
        # No usable signal across the whole orderlet -> fail loudly instead of
        # silently returning an all-zero CCF cube.
        flux = np.asarray(rv_module.l2_obj.data["GREEN_SCI2_FLUX"])
        rv_module.l2_obj.set_data("GREEN_SCI2_FLUX", np.zeros_like(flux))
        with pytest.raises(RuntimeError, match="identically zero"):
            rv_module.compute_ccfs("GREEN", "SCI2")

    def test_clip_edge_pixels_zero_keeps_all(self, rv_module):
        # clip_edge_pixels=[0, 0] is a no-op (no pixels removed).
        full = rv_module.compute_ccfs("GREEN", "SCI2", clip_edge_pixels=[0, 0])["ccf"]
        assert np.any(full)

    def test_clip_edge_pixels_too_large_raises(self, rv_module):
        # Clipping more pixels than the order has fails loudly.
        with pytest.raises(ValueError, match="removes all"):
            rv_module.compute_ccfs("GREEN", "SCI2", clip_edge_pixels=[NCOL, NCOL])


class TestComputeRVPublic:
    def test_returns_rv_dict(self, rv_module):
        rv_module.compute_ccfs("GREEN", "SCI2")
        res = rv_module.compute_order_by_order_rvs("GREEN", "SCI2")
        assert set(res) == {"rv", "rv_err"}
        assert res["rv"].shape == (NORDER_GREEN,)
        assert res["rv_err"].shape == (NORDER_GREEN,)

    def test_raises_without_ccf(self, rv_module):
        with pytest.raises(RuntimeError, match="compute_ccfs"):
            rv_module.compute_order_by_order_rvs("GREEN", "SCI2")

    def test_recovers_injected_rv(self, rv_module):
        rv_module.compute_ccfs("GREEN", "SCI2")
        rv = rv_module.compute_order_by_order_rvs("GREEN", "SCI2")["rv"]
        np.testing.assert_allclose(rv, _V_INJECT, atol=0.1)

    def test_per_ccd_rv_recovers_injected(self, rv_module):
        # The weighted-combined per-CCD RV recovers the injected velocity, with a
        # finite positive error from the unweighted-summed CCF. combine_ccds=False
        # returns a dict keyed by chip.
        rv_module.compute_ccfs("GREEN", "SCI2")
        ccd_rv, ccd_rv_err = rv_module.compute_weighted_rvs(
            ["GREEN"], "SCI2", combine_fibers=False, combine_ccds=False
        )["GREEN"]
        assert ccd_rv == pytest.approx(_V_INJECT, abs=0.1)
        assert np.isfinite(ccd_rv_err) and ccd_rv_err > 0

    def test_per_ccd_rv_sums_science_fibers(self, rv_module):
        # combine_fibers=True sums the three science fibers' cached CCFs before
        # fitting (the SCI-combined per-CCD RV); still recovers the injected velocity.
        for f in ("SCI1", "SCI2", "SCI3"):
            rv_module.compute_ccfs("GREEN", f)
        ccd_rv, ccd_rv_err = rv_module.compute_weighted_rvs(
            ["GREEN"], ["SCI1", "SCI2", "SCI3"], combine_fibers=True, combine_ccds=False
        )["GREEN"]
        assert ccd_rv == pytest.approx(_V_INJECT, abs=0.1)
        assert np.isfinite(ccd_rv_err) and ccd_rv_err > 0

    def test_combine_ccds_returns_tuple_and_recovers_injected(self, rv_module):
        # combine_ccds=True returns a single (rv, rv_err) tuple from the RV-level
        # cross-chip combine.
        for chip in ("GREEN", "RED"):
            rv_module.compute_ccfs(chip, "SCI2")
        out = rv_module.compute_weighted_rvs(
            ["GREEN", "RED"], "SCI2", combine_fibers=False, combine_ccds=True
        )
        assert isinstance(out, tuple) and len(out) == 2
        assert out[0] == pytest.approx(_V_INJECT, abs=0.1)
        assert np.isfinite(out[1]) and out[1] > 0

    def test_combine_fibers_requires_three_sci(self, rv_module):
        with pytest.raises(ValueError, match="three science fibers"):
            rv_module.compute_weighted_rvs(
                ["GREEN"], "SCI2", combine_fibers=True, combine_ccds=False
            )

    def test_no_combine_fibers_requires_single(self, rv_module):
        with pytest.raises(ValueError, match="single fiber"):
            rv_module.compute_weighted_rvs(
                ["GREEN"],
                ["SCI1", "SCI2", "SCI3"],
                combine_fibers=False,
                combine_ccds=False,
            )

    def test_combine_ccfs_invalid_fibers_raises(self, rv_module):
        # _combine_ccfs accepts a single fiber or exactly the three SCI fibers.
        with pytest.raises(ValueError, match="single fiber or exactly"):
            rv_module._combine_ccfs("GREEN", ["SCI1", "SCI2"])

    def test_per_ccd_raises_without_ccf(self, rv_module):
        with pytest.raises(RuntimeError, match="compute_ccfs"):
            rv_module.compute_weighted_rvs(
                ["GREEN"], "SCI2", combine_fibers=False, combine_ccds=False
            )

    def test_errors_finite_and_positive(self, rv_module):
        rv_module.compute_ccfs("GREEN", "SCI2")
        rv_err = rv_module.compute_order_by_order_rvs("GREEN", "SCI2")["rv_err"]
        assert np.all(np.isfinite(rv_err)) and np.all(rv_err > 0)

    def test_order_weights_table(self, rv_module):
        # Weights load per orderlet; the column follows the orderlet's mask.
        w = rv_module._get_order_weights(
            "GREEN", "SCI2"
        )  # Target -> G2_espresso column
        assert w.shape == (NORDER_GREEN,) and np.all(w >= 0)
        rv_module.l2_obj.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = "Th_gold"
        assert rv_module._get_order_weights("RED", "CAL").shape == (
            NORDER_RED,
        )  # thar column

    def test_order_weights_missing_column_raises(self, rv_module):
        # Defensive guard: a mask with no weight column fails loud. Seed the
        # source cache with a bogus mask (no resolvable fiber yields one).
        rv_module._illumination_source["GREEN_CAL"] = {
            "object": "x",
            "mask_name": "NOPE_mask",
            "apply_barycorr": None,
            "vel_grid_center": None,
        }
        with pytest.raises(KeyError, match="order-weight column"):
            rv_module._get_order_weights("GREEN", "CAL")


@pytest.mark.slow
class TestPerform:
    _ILLUMINATED = ["SCI1", "SCI2", "SCI3", "SKY"]  # CAL-OBJ='None' -> skipped

    def test_returns_kpf4_with_per_orderlet_extensions(self, rv_module):
        l4 = rv_module.perform()
        assert isinstance(l4, KPF4)
        for fiber in self._ILLUMINATED:
            assert l4.data[f"{fiber}_CCF"].shape == (NORDER, _NVEL)
            assert_dtype(l4.data[f"{fiber}_CCF"], CCF, f"{fiber}_CCF")
            table = l4.data[f"{fiber}_RV"]
            assert len(table) == NORDER
            assert set(table.columns) >= {
                "ORDER_INDEX",
                "BJD_TDB",
                "BERV",
                "WAVE_START",
                "WAVE_END",
                "RV",
                "RV_ERR",
            }
            # EPRV L4: time/wavelength columns are 64-bit; order index is integer.
            assert table["BJD_TDB"].dtype == np.float64
            assert table["WAVE_START"].dtype == np.float64
            assert table["WAVE_END"].dtype == np.float64
            assert np.issubdtype(table["ORDER_INDEX"].dtype, np.integer)

    def test_unilluminated_fiber_skipped(self, rv_module):
        # CAL-OBJ='None' -> no CCF cube or RV table written.
        l4 = rv_module.perform()
        assert l4.data["CAL_CCF"].size == 0
        assert len(l4.data["CAL_RV"]) == 0

    def test_ccf_chip_halves_populated(self, rv_module):
        l4 = rv_module.perform()
        assert l4.data["GREEN_SCI2_CCF"].shape == (NORDER_GREEN, _NVEL)
        assert l4.data["RED_SCI2_CCF"].shape == (NORDER_RED, _NVEL)
        assert np.any(l4.data["GREEN_SCI2_CCF"])
        assert np.any(l4.data["RED_SCI2_CCF"])

    def test_recovers_injected_rv_illuminated_orderlets(self, rv_module):
        l4 = rv_module.perform()
        for fiber in self._ILLUMINATED:
            rv = np.asarray(l4.data[f"{fiber}_RV"]["RV"])
            np.testing.assert_allclose(rv, _V_INJECT, atol=0.1)

    def test_per_fiber_ccf_and_rv_headers(self, rv_module):
        # EPRV L4 keywords on each orderlet's CCF/RV extension; RVMETHOD on PRIMARY.
        l4 = rv_module.perform()
        # Every header (PRIMARY and the CCF/RV extensions) is a fits.Header, so
        # keyword access returns the scalar value.
        ccf_hdr = l4.headers["SCI2_CCF"]
        assert ccf_hdr["VELNSTEP"] == _NVEL
        assert ccf_hdr["VELSTEP"] == pytest.approx(0.25)
        assert ccf_hdr["VELSTART"] == pytest.approx(
            _RANGE_KMS[0]
        )  # center 0 (TARGRADV=0)
        assert ccf_hdr["CCFMASK"] == "G2_espresso"  # 5772 K -> G2
        rv_hdr = l4.headers["SCI2_RV"]
        assert rv_hdr["RVMETHOD"] == "CCF"
        assert rv_hdr["SKYRMVD"] is False
        assert rv_hdr["TELLRMVD"] is False
        assert l4.headers["PRIMARY"]["RVMETHOD"] == "CCF"

    def test_per_ccd_rv_keywords(self, rv_module):
        # Per-orderlet legacy RVs are registered KPF keywords routed to their RV#
        # table header (legacy scheme CCD<n>RV<sfx>/CCD<n>ERV<sfx>; CCD1=GREEN,
        # CCD2=RED; SCI2 per-fiber suffix '2' -> RV3, aliased SCI2_RV).
        l4 = rv_module.perform()
        rv_hdr = l4.headers["RV3"]
        assert rv_hdr["CCD1RV2"] == pytest.approx(_V_INJECT, abs=0.1)
        assert rv_hdr["CCD2RV2"] == pytest.approx(_V_INJECT, abs=0.1)
        assert rv_hdr["CCD1ERV2"] > 0 and rv_hdr["CCD2ERV2"] > 0
        # The per-orderlet keywords do not leak onto PRIMARY; they live on the RV#
        # tables (the SCI-combined CCD<n>RV/CCFRV land on RV3).
        assert "CCD1RV2" not in l4.headers["PRIMARY"]

    def test_combined_rv_populated(self, rv_module):
        # The science combine: SCI-combined CCD1RV/CCD2RV and CCFRV/CCFERV on the
        # RV3 table (registered KPF keywords), alongside the EPRV RV/RVERR on
        # PRIMARY. RV ~ injected value.
        l4 = rv_module.perform()
        rv3 = l4.headers["RV3"]
        assert rv3["CCD1RV"] == pytest.approx(_V_INJECT, abs=0.1)
        assert rv3["CCD2RV"] == pytest.approx(_V_INJECT, abs=0.1)
        assert rv3["CCD1ERV"] > 0 and rv3["CCD2ERV"] > 0
        assert rv3["CCFRV"] == pytest.approx(_V_INJECT, abs=0.1)
        assert rv3["CCFERV"] > 0

        prim = l4.headers["PRIMARY"]
        assert prim["RV"] == pytest.approx(_V_INJECT, abs=0.1)
        assert prim["RVERR"] > 0
        assert prim["RVMETHOD"] == "CCF"
        assert prim["SYSVEL"] is None  # absolute RVs; nothing removed

    def test_ccfrv_is_weighted_ccd_combine(self, rv_module):
        # CCFRV = (CCD1RV*Wg + CCD2RV*Wr)/(Wg+Wr), Wg/Wr the summed order weights;
        # CCFERV = inverse-variance combination of the per-CCD errors.
        l4 = rv_module.perform()
        inst = l4.headers["RV3"]
        wg = np.nansum(rv_module._get_order_weights("GREEN", "SCI1"))
        wr = np.nansum(rv_module._get_order_weights("RED", "SCI1"))
        expect_rv = (inst["CCD1RV"] * wg + inst["CCD2RV"] * wr) / (wg + wr)
        expect_err = (1.0 / inst["CCD1ERV"] ** 2 + 1.0 / inst["CCD2ERV"] ** 2) ** -0.5
        assert inst["CCFRV"] == pytest.approx(expect_rv, abs=1e-9)
        assert inst["CCFERV"] == pytest.approx(expect_err, rel=1e-9)

    def test_primary_berv_bjdtdb_from_per_ccd(self, rv_module):
        # PRIMARY BERV/BJDTDB are the chip-weighted mean of the per-CCD bary
        # summaries (CCD<n>BKMS on BARYCORR_KMS, CCD<n>BJD on BJD_TDB, from
        # BarycentricCorrection). Equal per-CCD values -> the weighted mean is that
        # value, regardless of weights.
        bkms = rv_module.l2_obj.headers["BARYCORR_KMS"]
        bjd = rv_module.l2_obj.headers["BJD_TDB"]
        bkms["CCD1BKMS"] = bkms["CCD2BKMS"] = -12.3
        bjd["CCD1BJD"] = bjd["CCD2BJD"] = 2460123.5
        prim = rv_module.perform().headers["PRIMARY"]
        assert prim["BERV"] == pytest.approx(-12.3)
        assert prim["BJDTDB"] == pytest.approx(2460123.5)

    def test_primary_berv_undefined_without_per_ccd(self, rv_module):
        # No per-CCD bary summaries on PRIMARY -> BERV/BJDTDB UNDEFINED,
        # but the combined RV is still populated.
        prim = rv_module.perform().headers["PRIMARY"]
        assert prim["BERV"] is None and prim["BJDTDB"] is None
        assert prim["RV"] == pytest.approx(_V_INJECT, abs=0.1)

    def test_no_science_illuminated_raises(self, rv_module):
        # SCI requested (default fibers) but all dark -> fail loudly.
        rv_module.l2_obj.headers["INSTRUMENT_HEADER"]["SCI-OBJ"] = "None"
        with pytest.raises(ValueError, match="none illuminated"):
            rv_module.perform()

    def test_cal_only_run_skips_combine(self, rv_module, capsys):
        # A calibration-only run (no SCI requested) does not raise; PRIMARY RV
        # is left UNDEFINED and a note is printed.
        rv_module.l2_obj.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = "Th_gold"
        prim = rv_module.perform(fibers=["CAL"]).headers["PRIMARY"]
        assert prim["RV"] is None  # combine skipped, stays UNDEFINED
        assert "no science orderlet requested" in capsys.readouterr().out

    def test_single_chip_combine_warns(self, rv_module, capsys):
        # One chip present: CCFRV uses it alone (== CCD1RV) and a warning prints.
        l4 = rv_module.perform(chips=["GREEN"])
        inst = l4.headers["RV3"]
        assert inst["CCFRV"] == pytest.approx(inst["CCD1RV"], abs=1e-9)
        assert "only chip GREEN present" in capsys.readouterr().out

    def test_l4_serializes_to_fits(self, rv_module, tmp_path):
        # The CCF/RV extension headers must survive to_fits with comments intact.
        # SCI2 -> CCF3 / RV3.
        l4 = rv_module.perform(fibers=["SCI1", "SCI2", "SCI3"])
        path = tmp_path / "kpf_SL4_20240405T000000.fits"
        l4.to_fits(str(path))
        with fits.open(path) as hdul:
            ccf = hdul["CCF3"].header
            assert ccf["VELNSTEP"] == _NVEL
            assert ccf["VELSTART"] == pytest.approx(_RANGE_KMS[0])
            assert ccf.comments["VELSTART"]  # comment preserved
            rv = hdul["RV3"].header
            assert rv["RVMETHOD"] == "CCF"
            # Per-orderlet legacy RV lives on its RV# table header (SCI2 -> RV3),
            # not on PRIMARY.
            assert rv["CCD1RV2"] == pytest.approx(_V_INJECT, abs=0.1)
            assert "CCD1RV2" not in hdul["PRIMARY"].header
            # The EPRV combined RV survives on PRIMARY.
            assert hdul["PRIMARY"].header["RV"] == pytest.approx(_V_INJECT, abs=0.1)
            assert hdul["PRIMARY"].header["RVERR"] > 0

    def test_failed_combined_fit_written_as_undefined(self, rv_module, monkeypatch):
        # A non-finite fit (failed fit) is written as a FITS UNDEFINED card
        # (present, value None), never a bare NaN. Force every fit non-finite at
        # the lowest seam so the dual-return shape of compute_weighted_rvs is moot.
        monkeypatch.setattr(
            rv_module, "_compute_rv_1d", lambda *args, **kwargs: (np.nan, np.nan)
        )

        l4 = rv_module.perform(fibers=["SCI1", "SCI2", "SCI3"])
        rv_hdr = l4.headers["RV3"]  # SCI2 per-orderlet RVs -> RV3
        assert "CCD1RV2" in rv_hdr and rv_hdr["CCD1RV2"] is None
        assert "CCD2RV2" in rv_hdr and rv_hdr["CCD2RV2"] is None

    def test_thar_mask_recorded_for_cal(self, rv_module):
        # CAL on a ThAr lamp -> CCFMASK 'thar', instrument frame (no barycorr).
        rv_module.l2_obj.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = "Th_gold"
        l4 = rv_module.perform(fibers=["CAL"])
        assert l4.headers["CAL_CCF"]["CCFMASK"] == "thar"
        rv = np.asarray(l4.data["CAL_RV"]["RV"])
        np.testing.assert_allclose(rv, _V_INJECT, atol=0.1)

    @pytest.mark.parametrize(
        "raw, obj", [("EtalonFiber", "etalon"), ("LFCFiber", "lfc")]
    )
    def test_unimplemented_fiber_skipped(self, rv_module, raw, obj):
        rv_module.l2_obj.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = raw
        with pytest.warns(UserWarning, match=f"{obj}.*not implemented"):
            l4 = rv_module.perform(fibers=["CAL"])
        assert l4.data["CAL_CCF"].size == 0
        assert len(l4.data["CAL_RV"]) == 0

    def test_explicit_chips_and_fibers(self, rv_module):
        l4 = rv_module.perform(chips=["GREEN"], fibers=["SCI1", "SCI2", "SCI3"])
        # GREEN-only: science green halves populated, red halves left zero.
        assert np.any(l4.data["GREEN_SCI1_CCF"])
        assert not np.any(l4.data["RED_SCI1_CCF"])
        # SKY/CAL not requested -> untouched (empty extensions).
        assert l4.data["SKY_CCF"].size == 0
        assert len(l4.data["SKY_RV"]) == 0
        rv = np.asarray(l4.data["SCI1_RV"]["RV"])
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
        rv = RadialVelocity(header_kpf2, config={"ccf_mask_width": 1.0})
        assert rv.ccf_mask_width == 1.0

    def test_defaults_applied(self, header_kpf2):
        rv = RadialVelocity(header_kpf2)
        assert rv.ccf_mask_width == 0.5
        assert rv.ccf_step_size == 0.25
