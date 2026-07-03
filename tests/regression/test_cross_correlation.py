"""
Tests for the CrossCorrelation module (KPF2 -> KPF4: per-orderlet CCFs).

Mirrors the CCF-side coverage of test_radial_velocity.py against the standalone
CrossCorrelation module. Static-method unit tests (_compute_ccf_1d) build
synthetic spectra with no fixtures. Build-helper tests use a header-only KPF2 and
read the real on-disk line masks. Integration tests (compute_ccfs/perform) use a
synthetic KPF2 with absorption injected at a monkeypatched line mask, and a
narrow velocity grid for speed. CrossCorrelation writes the CCF cubes, the per-bin
CCF variance cubes, and the metadata-seeded RV tables (RV/RV_ERR left NaN for
RadialVelocity); it does not fit RVs or write any PRIMARY/combined-RV keywords.
"""

import numpy as np
import pytest
from astropy.constants import c
from astropy.io import fits

from kpfpipe.data_models.level2 import KPF2, NORDER_GREEN, NORDER_RED
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.modules.cross_correlation import CrossCorrelation

from ._dtype_policy import CCF, assert_dtype

NORDER = NORDER_GREEN + NORDER_RED
SPEED_OF_LIGHT_KMS = np.float64(c.to("km/s").value)
_FIBERS = ["CAL", "SCI1", "SCI2", "SCI3", "SKY"]  # all orderlets

# Narrow CCF grid for fast integration tests.
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


def _make_mask(centers, weights=None, width=1.0):
    """Build a line-mask dict matching _build_line_mask's structure."""
    centers = np.asarray(centers, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(centers)
    half_width = centers * (width / 2.0 / SPEED_OF_LIGHT_KMS)
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
        var = flux.copy()  # photon-noise proxy
        vel = np.arange(-402, 403) * 0.25
        return wave, flux, var, mask, vel

    def test_dip_at_injected_velocity(self):
        wave, flux, var, mask, vel = self._order(v_dip=3.0, z=0.0)
        ccf, _ = CrossCorrelation._compute_ccf_1d(wave, flux, var, mask, vel, 0.0)
        assert vel[np.argmin(ccf)] == pytest.approx(3.0, abs=0.3)

    def test_zero_velocity_dip(self):
        wave, flux, var, mask, vel = self._order(v_dip=0.0, z=0.0)
        ccf, _ = CrossCorrelation._compute_ccf_1d(wave, flux, var, mask, vel, 0.0)
        assert vel[np.argmin(ccf)] == pytest.approx(0.0, abs=0.3)

    def test_barycorr_z_folds_in(self):
        z = 5.0 / SPEED_OF_LIGHT_KMS
        wave, flux, var, mask, vel = self._order(v_dip=2.0, z=z)
        ccf, _ = CrossCorrelation._compute_ccf_1d(wave, flux, var, mask, vel, z)
        assert vel[np.argmin(ccf)] == pytest.approx(2.0, abs=0.3)

    def test_descending_wave_raises(self):
        wave, flux, var, mask, vel = self._order(v_dip=0.0)
        with pytest.raises(ValueError, match="descending"):
            CrossCorrelation._compute_ccf_1d(
                wave[::-1], flux[::-1], var[::-1], mask, vel, 0.0
            )

    def test_constant_wave_returns_zeros(self):
        wave, flux, var, mask, vel = self._order()
        ccf, _ = CrossCorrelation._compute_ccf_1d(
            np.full_like(wave, 5000.0), flux, var, mask, vel, 0.0
        )
        assert not np.any(ccf)

    def test_nan_wave_returns_zeros(self):
        wave, flux, var, mask, vel = self._order()
        wave = wave.copy()
        wave[1000] = np.nan
        ccf, _ = CrossCorrelation._compute_ccf_1d(wave, flux, var, mask, vel, 0.0)
        assert not np.any(ccf)

    def test_no_lines_in_order_returns_zeros(self):
        wave = np.linspace(6000.0, 6050.0, 2000)
        flux = np.ones_like(wave)
        var = flux.copy()  # photon-noise proxy
        mask = _make_mask(np.linspace(5008.0, 5042.0, 20))  # all outside the order
        vel = np.arange(-402, 403) * 0.25
        ccf, _ = CrossCorrelation._compute_ccf_1d(wave, flux, var, mask, vel, 0.0)
        assert not np.any(ccf)

    def test_ccf_var_propagates_var_not_flux(self):
        # ccf_var carries the per-pixel VARIANCE, not the flux: scaling var
        # scales ccf_var but leaves the CCF value unchanged.
        wave, flux, var, mask, vel = self._order(v_dip=0.0)
        ccf1, var1 = CrossCorrelation._compute_ccf_1d(wave, flux, var, mask, vel, 0.0)
        ccf2, var2 = CrossCorrelation._compute_ccf_1d(
            wave, flux, 4.0 * var, mask, vel, 0.0
        )
        np.testing.assert_allclose(ccf1, ccf2)
        np.testing.assert_allclose(var2, 4.0 * var1)

    def test_ccf_1d_is_float64_from_float32_flux(self):
        # A float64 CCF from float32 flux is the intended deliberate upcast.
        wave, flux, var, mask, vel = self._order(v_dip=0.0)
        ccf, ccf_var = CrossCorrelation._compute_ccf_1d(
            wave, flux.astype(np.float32), var.astype(np.float32), mask, vel, 0.0
        )
        assert_dtype(ccf, CCF, "ccf")
        assert_dtype(ccf_var, CCF, "ccf_var")


# ---------------------------------------------------------------------------
# Dispatch / build helpers (header-only fixture)
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
            CrossCorrelation(header_kpf2)._resolve_illumination_source("GREEN", "CAL")[
                "object"
            ]
            == obj
        )

    def test_unrecognized_source_raises(self, header_kpf2):
        header_kpf2.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = "Frobnicator"
        with pytest.raises(ValueError, match="unrecognized illumination"):
            CrossCorrelation(header_kpf2)._resolve_illumination_source("GREEN", "CAL")

    def test_resolve_illumination_per_fiber(self, header_kpf2):
        cc = CrossCorrelation(header_kpf2)
        assert cc._resolve_illumination_source("GREEN", "SCI2")["object"] == "target"
        assert cc._resolve_illumination_source("RED", "SKY")["object"] == "sky"
        assert cc._resolve_illumination_source("GREEN", "CAL")["object"] == "none"

    def test_resolve_unknown_fiber_raises(self, header_kpf2):
        with pytest.raises(ValueError, match="unknown fiber"):
            CrossCorrelation(header_kpf2)._resolve_illumination_source("GREEN", "BOGUS")

    def test_resolve_missing_keyword_raises(self, header_kpf2):
        del header_kpf2.headers["INSTRUMENT_HEADER"]["CAL-OBJ"]
        with pytest.raises(ValueError, match="CAL-OBJ"):
            CrossCorrelation(header_kpf2)._resolve_illumination_source("GREEN", "CAL")

    def test_settings_target(self, header_kpf2):
        header_kpf2.headers["INSTRUMENT_HEADER"]["TARGRADV"] = 11.1  # SCI-OBJ='Target'
        s = CrossCorrelation(header_kpf2)._resolve_illumination_source("GREEN", "SCI2")
        assert s == {
            "object": "target",
            "mask_name": "G2_espresso",
            "apply_barycorr": True,
            "vel_grid_center": 11.1,
        }

    def test_settings_sky(self, header_kpf2):
        s = CrossCorrelation(header_kpf2)._resolve_illumination_source("GREEN", "SKY")
        assert s == {
            "object": "sky",
            "mask_name": "G2_espresso",
            "apply_barycorr": True,
            "vel_grid_center": 0.0,
        }

    def test_settings_thar(self, header_kpf2):
        header_kpf2.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = "Th_gold"
        s = CrossCorrelation(header_kpf2)._resolve_illumination_source("GREEN", "CAL")
        assert s == {
            "object": "thar",
            "mask_name": "thar",
            "apply_barycorr": False,
            "vel_grid_center": 0.0,
        }

    def test_settings_none(self, header_kpf2):
        s = CrossCorrelation(header_kpf2)._resolve_illumination_source(
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
            source = CrossCorrelation(header_kpf2)._resolve_illumination_source(
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
        cc = CrossCorrelation(header_kpf2)
        assert cc._resolve_stellar_mask() == "G2_espresso"  # 5772 K -> G2
        header_kpf2.headers["INSTRUMENT_HEADER"]["TARGTEFF"] = 4000.0
        assert CrossCorrelation(header_kpf2)._resolve_stellar_mask() == "K6_espresso"

    @pytest.mark.parametrize("teff", [0.0, -100.0, "nan"])
    def test_invalid_targteff_raises(self, header_kpf2, teff):
        header_kpf2.headers["INSTRUMENT_HEADER"]["TARGTEFF"] = teff
        with pytest.raises(ValueError, match="TARGTEFF"):
            CrossCorrelation(header_kpf2)._resolve_stellar_mask()

    def test_missing_targteff_raises(self, header_kpf2):
        del header_kpf2.headers["INSTRUMENT_HEADER"]["TARGTEFF"]
        with pytest.raises(ValueError, match="TARGTEFF"):
            CrossCorrelation(header_kpf2)._resolve_stellar_mask()

    def test_missing_targradv_raises(self, header_kpf2):
        del header_kpf2.headers["INSTRUMENT_HEADER"]["TARGRADV"]
        with pytest.raises(ValueError, match="TARGRADV"):
            CrossCorrelation(header_kpf2)._get_systemic_rv()


class TestBuildLineMask:
    def test_keys_and_shapes(self, header_kpf2):
        mask = CrossCorrelation(header_kpf2)._build_line_mask(
            "GREEN", "SCI2"
        )  # SCI-OBJ='Target' -> G2
        assert set(mask) == {"center", "weight", "start", "end"}
        n = mask["center"].size
        assert all(mask[k].shape == (n,) for k in mask)

    def test_top_hat_edges_bracket_center(self, header_kpf2):
        mask = CrossCorrelation(header_kpf2)._build_line_mask("GREEN", "SCI2")
        assert np.all(mask["start"] < mask["center"])
        assert np.all(mask["center"] < mask["end"])

    def test_cached(self, header_kpf2):
        cc = CrossCorrelation(header_kpf2)
        assert cc._build_line_mask("GREEN", "SCI2") is cc._build_line_mask(
            "GREEN", "SCI2"
        )

    def test_thar_mask_uniform_weights(self, header_kpf2):
        header_kpf2.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = "Th_gold"  # -> thar mask
        mask = CrossCorrelation(header_kpf2)._build_line_mask("GREEN", "CAL")
        assert np.all(mask["weight"] == 1.0)
        # ThAr centers are deduped and sorted (lines recur across overlapping orders).
        assert np.all(np.diff(mask["center"]) > 0)


class TestBuildVelocityGrid:
    def test_centered_on_systemic_rv(self, header_kpf2):
        header_kpf2.headers["INSTRUMENT_HEADER"]["TARGRADV"] = (
            10.0  # SCI2 grid center = TARGRADV
        )
        grid = CrossCorrelation(header_kpf2)._build_velocity_grid("GREEN", "SCI2")
        assert grid.mean() == pytest.approx(10.0)

    def test_default_size_and_step(self, header_kpf2):
        grid = CrossCorrelation(header_kpf2)._build_velocity_grid(
            "GREEN", "SKY"
        )  # SKY center = 0
        assert grid.size == 801  # [-100, 100] km/s at 0.25 -> arange(-400, 401)
        np.testing.assert_allclose(np.diff(grid), 0.25)

    def test_symmetric_about_zero_center(self, header_kpf2):
        grid = CrossCorrelation(header_kpf2)._build_velocity_grid(
            "GREEN", "SKY"
        )  # center 0
        np.testing.assert_allclose(grid.min(), -grid.max())

    def test_cached(self, header_kpf2):
        cc = CrossCorrelation(header_kpf2)
        assert cc._build_velocity_grid("GREEN", "SCI2") is cc._build_velocity_grid(
            "GREEN", "SCI2"
        )


# ---------------------------------------------------------------------------
# compute_ccfs / perform  (synthetic KPF2 + monkeypatched mask)
# ---------------------------------------------------------------------------


@pytest.fixture
def cc_kpf2():
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
            kpf2.set_data(
                f"{chip}_{fiber}_VAR", np.tile(flux_1d, (n, 1)).astype(np.float64)
            )
    # Per-order barycentric extensions (populated together by BarycentricCorrection).
    kpf2.set_data("BARYCORR_Z", np.zeros(NORDER))
    kpf2.set_data("BARYCORR_KMS", np.zeros(NORDER))
    kpf2.set_data("BJD_TDB", np.zeros(NORDER))
    return kpf2


@pytest.fixture
def cc_module(cc_kpf2, monkeypatch):
    """CrossCorrelation on a narrow grid with the line mask stubbed to _MASK_CENTERS."""
    mask = _make_mask(_MASK_CENTERS)
    monkeypatch.setattr(
        CrossCorrelation,
        "_build_line_mask",
        lambda self, chip, fiber, mask_width=None: mask,
    )
    return CrossCorrelation(cc_kpf2, config={"ccf_window": _RANGE_KMS})


class TestComputeCCFPublic:
    def test_returns_velocity_and_ccf(self, cc_module):
        res = cc_module.compute_ccfs("GREEN", "SCI2")
        assert set(res) == {"velocity", "ccf"}
        assert res["velocity"].shape == (_NVEL,)
        assert res["ccf"].shape == (NORDER_GREEN, _NVEL)

    def test_red_chip_shape(self, cc_module):
        res = cc_module.compute_ccfs("RED", "SCI1")
        assert res["ccf"].shape == (NORDER_RED, _NVEL)

    def test_dip_at_injected_velocity(self, cc_module):
        res = cc_module.compute_ccfs("GREEN", "SCI2")
        vel, ccf = res["velocity"], res["ccf"]
        assert vel[np.argmin(ccf[0])] == pytest.approx(_V_INJECT, abs=0.3)

    def test_caches_ccf(self, cc_module):
        res = cc_module.compute_ccfs("GREEN", "SCI2")
        assert cc_module._ccf["GREEN_SCI2"] is res["ccf"]

    def test_lowercase_chip_accepted(self, cc_module):
        res = cc_module.compute_ccfs("green", "sci2")
        assert res["ccf"].shape == (NORDER_GREEN, _NVEL)

    def test_missing_barycorr_z_raises(self, cc_kpf2, monkeypatch):
        monkeypatch.setattr(
            CrossCorrelation,
            "_build_line_mask",
            lambda self, chip, fiber, mask_width=None: _make_mask(_MASK_CENTERS),
        )
        cc_kpf2.set_data("BARYCORR_Z", np.array([]))
        cc = CrossCorrelation(cc_kpf2, config={"ccf_window": _RANGE_KMS})
        with pytest.raises(ValueError, match="BARYCORR_Z"):
            cc.compute_ccfs("GREEN", "SCI2")

    def test_all_zero_ccf_raises(self, cc_module):
        # No usable signal across the whole orderlet -> fail loudly instead of
        # silently returning an all-zero CCF cube.
        flux = np.asarray(cc_module.l2_obj.data["GREEN_SCI2_FLUX"])
        cc_module.l2_obj.set_data("GREEN_SCI2_FLUX", np.zeros_like(flux))
        with pytest.raises(RuntimeError, match="identically zero"):
            cc_module.compute_ccfs("GREEN", "SCI2")

    def test_clip_edge_pixels_zero_keeps_all(self, cc_module):
        # clip_edge_pixels=[0, 0] is a no-op (no pixels removed).
        full = cc_module.compute_ccfs("GREEN", "SCI2", clip_edge_pixels=[0, 0])["ccf"]
        assert np.any(full)

    def test_clip_edge_pixels_too_large_raises(self, cc_module):
        # Clipping more pixels than the order has fails loudly.
        with pytest.raises(ValueError, match="removes all"):
            cc_module.compute_ccfs("GREEN", "SCI2", clip_edge_pixels=[NCOL, NCOL])


class TestPerform:
    _ILLUMINATED = ["SCI1", "SCI2", "SCI3", "SKY"]  # CAL-OBJ='None' -> skipped

    def test_returns_kpf4_with_per_orderlet_extensions(self, cc_module):
        l4 = cc_module.perform()
        assert isinstance(l4, KPF4)
        for fiber in self._ILLUMINATED:
            assert l4.data[f"{fiber}_CCF"].shape == (NORDER, _NVEL)
            assert_dtype(l4.data[f"{fiber}_CCF"], CCF, f"{fiber}_CCF")
            assert l4.data[f"{fiber}_CCF_VAR"].shape == (NORDER, _NVEL)
            table = l4.data[f"{fiber}_RV"]
            assert len(table) == NORDER
            assert set(table.columns) >= {
                "ORDER_INDEX",
                "ORDER_ID",
                "ECHELLE_ORDER",
                "BJD_TDB",
                "BERV",
                "WAVE_START",
                "WAVE_END",
                "RV",
                "RV_ERR",
                "WEIGHT",
            }
            assert table["BJD_TDB"].dtype == np.float64
            assert table["WAVE_START"].dtype == np.float64
            assert table["WAVE_END"].dtype == np.float64
            assert np.issubdtype(table["ORDER_INDEX"].dtype, np.integer)
            assert np.issubdtype(table["ECHELLE_ORDER"].dtype, np.integer)

    def test_rv_columns_are_nan_placeholders(self, cc_module):
        # CrossCorrelation seeds the RV table metadata but leaves RV/RV_ERR NaN
        # for RadialVelocity to fill.
        l4 = cc_module.perform()
        for fiber in self._ILLUMINATED:
            table = l4.data[f"{fiber}_RV"]
            assert np.all(np.isnan(np.asarray(table["RV"], dtype=float)))
            assert np.all(np.isnan(np.asarray(table["RV_ERR"], dtype=float)))

    def test_unilluminated_fiber_skipped(self, cc_module):
        # CAL-OBJ='None' -> no CCF cube, CCF variance, or RV table written.
        l4 = cc_module.perform()
        assert l4.data["CAL_CCF"].size == 0
        assert l4.data["CAL_CCF_VAR"].size == 0
        assert len(l4.data["CAL_RV"]) == 0

    def test_ccf_chip_halves_populated(self, cc_module):
        l4 = cc_module.perform()
        assert l4.data["GREEN_SCI2_CCF"].shape == (NORDER_GREEN, _NVEL)
        assert l4.data["RED_SCI2_CCF"].shape == (NORDER_RED, _NVEL)
        assert np.any(l4.data["GREEN_SCI2_CCF"])
        assert np.any(l4.data["RED_SCI2_CCF"])

    def test_ccf_var_persisted(self, cc_module):
        # The per-bin CCF variance cube is written alongside the CCF and matches
        # the variance compute_ccfs cached.
        l4 = cc_module.perform()
        for chip, norder in (("GREEN", NORDER_GREEN), ("RED", NORDER_RED)):
            var = l4.data[f"{chip}_SCI2_CCF_VAR"]
            assert var.shape == (norder, _NVEL)
            np.testing.assert_array_equal(var, cc_module._ccf_var[f"{chip}_SCI2"])
        assert np.any(l4.data["GREEN_SCI2_CCF_VAR"])

    def test_dip_at_injected_velocity_illuminated_orderlets(self, cc_module):
        # The CCFs dip at the injected velocity (no RV fit here, just the CCF).
        l4 = cc_module.perform()
        vel = cc_module._velocity_grid["RED_SCI2"]
        for fiber in self._ILLUMINATED:
            ccf0 = np.asarray(l4.data[f"{fiber}_CCF"])[0]
            assert vel[np.argmin(ccf0)] == pytest.approx(_V_INJECT, abs=0.3)

    def test_rv_table_weight_column_matches_order_weights(self, cc_module):
        # The per-order CCF-combination weights (ccf_order_weights.csv, column
        # by the orderlet's mask) are persisted as the WEIGHT column, green
        # orders then red, matching _get_order_weights.
        l4 = cc_module.perform()
        for fiber in self._ILLUMINATED:
            weight = np.asarray(l4.data[f"{fiber}_RV"]["WEIGHT"], dtype=float)
            expected = np.concatenate(
                [
                    cc_module._get_order_weights("GREEN", fiber),
                    cc_module._get_order_weights("RED", fiber),
                ]
            )
            assert weight.shape == (NORDER,)
            np.testing.assert_array_equal(weight, expected)

    def test_rv_table_order_id_and_echelle_columns(self, cc_module):
        # ORDER_ID is the KPF chip/fiber/order name, 1-based per chip (green then
        # red). ECHELLE_ORDER is the physical grating order (detector.toml),
        # blue->red: GREEN 137..103, RED 102..71.
        l4 = cc_module.perform()
        for fiber in self._ILLUMINATED:
            table = l4.data[f"{fiber}_RV"]
            order_id = np.asarray(table["ORDER_ID"])
            echelle = np.asarray(table["ECHELLE_ORDER"])
            assert order_id[0] == f"GREEN_{fiber}_1"
            assert order_id[NORDER_GREEN - 1] == f"GREEN_{fiber}_{NORDER_GREEN}"
            assert order_id[NORDER_GREEN] == f"RED_{fiber}_1"
            assert order_id[-1] == f"RED_{fiber}_{NORDER_RED}"
            assert echelle[0] == 137
            assert echelle[NORDER_GREEN - 1] == 103
            assert echelle[NORDER_GREEN] == 102
            assert echelle[-1] == 71

    def test_ccf_and_rv_headers(self, cc_module):
        # CrossCorrelation stamps the CCF EPRV keywords and the RV table-structure
        # CTYPE cards, but NOT the RV-processing descriptors (those are
        # RadialVelocity's, Phase 2) or any PRIMARY combined RV.
        l4 = cc_module.perform()
        ccf_hdr = l4.headers["SCI2_CCF"]
        assert ccf_hdr["CTYPE1"] == "Velocity"
        assert ccf_hdr["CTYPE2"] == "Order-N"
        assert ccf_hdr["VELNSTEP"] == _NVEL
        assert ccf_hdr["VELSTEP"] == pytest.approx(0.25)
        assert ccf_hdr["VELSTART"] == pytest.approx(_RANGE_KMS[0])  # center 0
        assert ccf_hdr["CCFMASK"] == "G2_espresso"  # 5772 K -> G2
        assert ccf_hdr["VELMASK"] == pytest.approx(cc_module.ccf_mask_width)
        rv_hdr = l4.headers["SCI2_RV"]
        assert rv_hdr["CTYPE1"] == "Columns"
        assert rv_hdr["CTYPE2"] == "Order-N"
        assert "RVMETHOD" not in rv_hdr
        assert "SKYRMVD" not in rv_hdr
        assert "TELLRMVD" not in rv_hdr
        # No combined-RV product is written by CrossCorrelation: PRIMARY RV stays
        # UNDEFINED and RVMETHOD stays at the seeded skeleton default (not 'CCF').
        assert l4.headers["PRIMARY"].get("RV") is None
        assert l4.headers["PRIMARY"].get("RVMETHOD") != "CCF"

    def test_thar_mask_recorded_for_cal(self, cc_module):
        # A ThAr-illuminated CAL fiber records CCFMASK='thar'.
        cc_module.l2_obj.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = "Th_gold"
        l4 = cc_module.perform(fibers=["CAL"])
        assert l4.headers["CAL_CCF"]["CCFMASK"] == "thar"

    def test_unimplemented_fiber_skipped(self, cc_module):
        # An etalon/lfc fiber has no CCF path yet -> skipped, empty extensions.
        cc_module.l2_obj.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = "EtalonFiber"
        with pytest.warns(UserWarning, match="etalon.*not implemented"):
            l4 = cc_module.perform(fibers=["CAL"])
        assert l4.data["CAL_CCF"].size == 0
        assert len(l4.data["CAL_RV"]) == 0

    def test_explicit_chips_and_fibers(self, cc_module):
        # A single chip / single fiber writes only that chip's CCF rows.
        l4 = cc_module.perform(chips=["GREEN"], fibers=["SCI2"])
        assert np.any(l4.data["GREEN_SCI2_CCF"])
        assert not np.any(l4.data["RED_SCI2_CCF"])
        assert l4.data["SKY_CCF"].size == 0

    def test_l4_serializes_to_fits(self, cc_module, tmp_path):
        # The CCF/RV extension headers and RV metadata columns must survive
        # to_fits. SCI2 -> CCF3 / RV3.
        l4 = cc_module.perform(fibers=["SCI1", "SCI2", "SCI3"])
        path = tmp_path / "kpf_SL4_20240405T000000.fits"
        l4.to_fits(str(path))
        with fits.open(path) as hdul:
            ccf = hdul["CCF3"].header
            assert ccf["VELNSTEP"] == _NVEL
            assert ccf["VELSTART"] == pytest.approx(_RANGE_KMS[0])
            assert ccf.comments["VELSTART"]  # comment preserved
            rv = hdul["RV3"].header
            assert rv["CTYPE1"] == "Columns"
            assert "RVMETHOD" not in rv
            rv_table = hdul["RV3"].data
            assert rv_table["ORDER_ID"][0] == "GREEN_SCI2_1"
            assert rv_table["ECHELLE_ORDER"][0] == 137
            # The per-bin CCF variance cube round-trips as an image extension.
            assert hdul["CCF_VAR3"].data.shape == (NORDER, _NVEL)


class TestConstructor:
    def test_config_dict_overrides_defaults(self, cc_kpf2):
        cc = CrossCorrelation(cc_kpf2, config={"ccf_step_size": 0.5})
        assert cc.ccf_step_size == 0.5
        assert cc.ccf_mask_width == 1.0  # untouched default

    def test_invalid_config_type_raises(self, cc_kpf2):
        with pytest.raises(TypeError, match="config must be"):
            CrossCorrelation(cc_kpf2, config=42)

    def test_info_before_perform(self, cc_module, capsys):
        cc_module.info()
        assert "perform() has not been called" in capsys.readouterr().out

    def test_info_after_perform(self, cc_module, capsys):
        cc_module.perform()
        cc_module.info()
        out = capsys.readouterr().out
        assert "CrossCorrelation" in out
        assert "SCI2" in out
