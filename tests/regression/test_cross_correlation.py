"""Tests for the CrossCorrelation module (KPF2 -> KPF4: per-orderlet CCFs).

_compute_ccf_1d unit tests build synthetic spectra; build-helper tests use a
header-only KPF2 and the real on-disk line masks; integration tests
(compute_ccfs/perform) use a synthetic KPF2 with absorption injected at a
monkeypatched line mask and a narrow velocity grid for speed. CrossCorrelation
writes the CCF cubes, the per-bin CCF variance cubes, and the metadata-seeded RV
tables (RV/RV_ERR left NaN for RadialVelocity); it does not fit RVs or write any
PRIMARY/combined-RV keywords.
"""

import logging
import re

import numpy as np
import pytest
from astropy.io import fits

from kpfpipe import DETECTOR
from kpfpipe.data_models.level2 import KPF2, NORDER_GREEN
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.modules.cross_correlation import CrossCorrelation

from ._catalog import seed_sci2_cards
from ._dtype_policy import CCF, RV_FLOAT, assert_dtype
from ._science import (
    MASK_CENTERS,
    NCOL,
    NVEL,
    RANGE_KMS,
    SPEED_OF_LIGHT_KMS,
    V_INJECT,
    absorption_spectrum,
    make_mask,
)

NORDER = DETECTOR["numorder"]
NORDER_RED = DETECTOR["norder"]["RED"]
# Fiber order is the module's own config-overridable default, not the canonical
# slicer order -- spelled out so a reordering in production shows up here.
_FIBERS = ["CAL", "SCI1", "SCI2", "SCI3", "SKY"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _compute_ccf_1d (staticmethod)
# ---------------------------------------------------------------------------


class TestComputeCCF:
    def _order(self, v_dip=0.0, z=0.0):
        wave = np.linspace(5000.0, 5050.0, 2000)
        centers = np.linspace(5008.0, 5042.0, 20)
        mask = make_mask(centers)
        # Observed absorption that the CCF should align at velocity step v_dip.
        lam_obs = centers * (1.0 + v_dip / SPEED_OF_LIGHT_KMS) / (1.0 + z)
        flux = absorption_spectrum(wave, lam_obs)
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
        mask = make_mask(np.linspace(5008.0, 5042.0, 20))  # all outside the order
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
    seed_sci2_cards(kpf2)
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
        header_kpf2.headers["PRIMARY"]["CRV3"] = 11.1  # SCI-OBJ='Target'
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
    def test_unimplemented_source_warns_and_skips(self, caplog, header_kpf2, raw, obj):
        header_kpf2.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = raw
        with caplog.at_level(logging.WARNING):
            source = CrossCorrelation(header_kpf2)._resolve_illumination_source(
                "GREEN", "CAL"
            )
        assert re.search(rf"{obj}.*not implemented", caplog.text)
        assert source == {
            "object": obj,
            "mask_name": None,
            "apply_barycorr": None,
            "vel_grid_center": None,
        }


class TestStellarMaskName:
    @pytest.mark.parametrize(
        ("color", "color_name", "mask"),
        [
            (0.823, "Gaia BP-RP", "G2_espresso"),  # G2V, 5770 K
            (0.650, "B-V", "G2_espresso"),  # the same star, SIMBAD's colour
            (1.035, "G-J", "G2_espresso"),  # and WMKO's
            (1.75, "Gaia BP-RP", "K6_espresso"),  # ~4000 K
        ],
    )
    def test_color_selects_mask(self, header_kpf2, color, color_name, mask):
        header_kpf2.headers["PRIMARY"]["CCLR3"] = color
        header_kpf2.headers["PRIMARY"]["CCLRN3"] = color_name
        assert CrossCorrelation(header_kpf2)._resolve_stellar_mask() == mask

    @pytest.mark.parametrize("card", ["CCLR3", "CCLRN3"])
    def test_missing_color_card_raises(self, header_kpf2, card):
        del header_kpf2.headers["PRIMARY"][card]
        with pytest.raises(ValueError, match="CCLR3/CCLRN3"):
            CrossCorrelation(header_kpf2)._resolve_stellar_mask()

    def test_blank_color_name_raises(self, header_kpf2):
        header_kpf2.headers["PRIMARY"]["CCLRN3"] = "   "
        with pytest.raises(ValueError, match="CCLR3/CCLRN3"):
            CrossCorrelation(header_kpf2)._resolve_stellar_mask()

    def test_unrecognized_color_name_raises(self, header_kpf2):
        header_kpf2.headers["PRIMARY"]["CCLRN3"] = "V-Ks"
        with pytest.raises(ValueError, match="unrecognized colour index"):
            CrossCorrelation(header_kpf2)._resolve_stellar_mask()

    def test_ignores_instrument_header_targteff(self, header_kpf2):
        # The catalog colour wins; the raw DCS temperature is not consulted.
        header_kpf2.headers["INSTRUMENT_HEADER"]["TARGTEFF"] = 4000.0
        assert CrossCorrelation(header_kpf2)._resolve_stellar_mask() == "G2_espresso"

    def test_missing_crv_warns_and_centers_on_zero(self, header_kpf2, caplog):
        # Many targets have no catalog rv; center on 0, but say so.
        del header_kpf2.headers["PRIMARY"]["CRV3"]
        with caplog.at_level(logging.WARNING):
            assert CrossCorrelation(header_kpf2)._get_systemic_rv() == 0.0
        assert "CRV3" in caplog.text

    def test_ignores_instrument_header_targradv(self, header_kpf2):
        # The canonical catalog rv wins; the raw DCS value is not consulted.
        header_kpf2.headers["INSTRUMENT_HEADER"]["TARGRADV"] = -42.0
        header_kpf2.headers["PRIMARY"]["CRV3"] = 7.5
        assert CrossCorrelation(header_kpf2)._get_systemic_rv() == pytest.approx(7.5)


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
        header_kpf2.headers["PRIMARY"]["CRV3"] = 10.0  # SCI2 grid center = CRV3
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


def _build_cc_kpf2(berv=None, wave_offsets=None, bjd=None):
    """KPF2 with per-order synthetic spectra (absorption at MASK_CENTERS shifted
    by V_INJECT) for every orderlet.

    ``berv`` (length NORDER, km/s) puts each order in its own barycentric frame:
    BARYCORR_Z/_KMS carry it and the observed lines are placed at
    ``MASK_CENTERS * (1 + V_INJECT/c) / (1 + z)``, the convention
    ``_compute_ccf_1d`` implements (it divides the mask shift by ``1 + z``).
    Recovering V_INJECT then requires the module to have removed the right
    order's z. ``wave_offsets`` (length NORDER, Angstrom) gives each order its
    own wavelength span, and ``bjd`` its own timestamp. All default to the
    degenerate all-orders-identical, zero-barycorr case.
    """
    kpf2 = KPF2()
    # Illumination sources: SCI on a star, SKY on sky, CAL dark (skipped).
    seed_sci2_cards(kpf2)

    berv = np.zeros(NORDER) if berv is None else np.asarray(berv, dtype=float)
    offsets = (
        np.zeros(NORDER)
        if wave_offsets is None
        else np.asarray(wave_offsets, dtype=float)
    )
    bjd = np.zeros(NORDER) if bjd is None else np.asarray(bjd, dtype=float)
    z = berv / SPEED_OF_LIGHT_KMS

    wave_1d = np.linspace(5000.0, 5050.0, NCOL)
    rows = {}
    for order in range(NORDER):
        lam_obs = (
            MASK_CENTERS * (1.0 + V_INJECT / SPEED_OF_LIGHT_KMS) / (1.0 + z[order])
            + offsets[order]
        )
        wave = wave_1d + offsets[order]
        rows[order] = (wave, absorption_spectrum(wave, lam_obs))

    for chip, start, n in [
        ("GREEN", 0, NORDER_GREEN),
        ("RED", NORDER_GREEN, NORDER_RED),
    ]:
        wave = np.stack([rows[start + i][0] for i in range(n)]).astype(np.float64)
        flux = np.stack([rows[start + i][1] for i in range(n)]).astype(np.float32)
        for fiber in _FIBERS:
            kpf2.set_data(f"{chip}_{fiber}_WAVE", wave.copy())
            kpf2.set_data(f"{chip}_{fiber}_FLUX", flux.copy())
            kpf2.set_data(f"{chip}_{fiber}_VAR", flux.copy())
    # Per-order barycentric extensions (populated together by BarycentricCorrection).
    kpf2.set_data("BARYCORR_Z", z)
    kpf2.set_data("BARYCORR_KMS", berv)
    kpf2.set_data("BJD_TDB", bjd)
    return kpf2


def _stub_line_mask(mp):
    """Patch _build_line_mask to the fixed MASK_CENTERS top-hat mask."""
    mask = make_mask(MASK_CENTERS)
    mp.setattr(
        CrossCorrelation,
        "_build_line_mask",
        lambda self, chip, fiber, mask_width=None: mask,
    )


@pytest.fixture
def cc_kpf2():
    return _build_cc_kpf2()


@pytest.fixture
def cc_module(cc_kpf2, monkeypatch):
    """CrossCorrelation on a narrow grid with the line mask stubbed to MASK_CENTERS."""
    _stub_line_mask(monkeypatch)
    return CrossCorrelation(cc_kpf2, config={"ccf_window": RANGE_KMS})


@pytest.fixture(scope="module")
def performed():
    """Run the default full perform() once and share the (module, L4) read-only.

    Every TestPerform test inspects a different facet of the same default-args L4,
    and recomputing it per test cost ~0.9s each. A MonkeyPatch context wraps the
    build because the function-scoped monkeypatch fixture cannot reach module
    scope. No consuming test mutates the module or the L4."""
    with pytest.MonkeyPatch.context() as mp:
        _stub_line_mask(mp)
        cc = CrossCorrelation(_build_cc_kpf2(), config={"ccf_window": RANGE_KMS})
        return cc, cc.perform()


class TestComputeCCFPublic:
    def test_returns_velocity_and_ccf(self, cc_module):
        res = cc_module.compute_ccfs("GREEN", "SCI2")
        assert set(res) == {"velocity", "ccf"}
        assert res["velocity"].shape == (NVEL,)
        assert res["ccf"].shape == (NORDER_GREEN, NVEL)

    def test_red_chip_shape(self, cc_module):
        res = cc_module.compute_ccfs("RED", "SCI1")
        assert res["ccf"].shape == (NORDER_RED, NVEL)

    def test_dip_at_injected_velocity(self, cc_module):
        res = cc_module.compute_ccfs("GREEN", "SCI2")
        vel, ccf = res["velocity"], res["ccf"]
        assert vel[np.argmin(ccf[0])] == pytest.approx(V_INJECT, abs=0.3)

    def test_caches_ccf(self, cc_module):
        res = cc_module.compute_ccfs("GREEN", "SCI2")
        assert cc_module._ccf["GREEN_SCI2"] is res["ccf"]

    def test_lowercase_chip_accepted(self, cc_module):
        res = cc_module.compute_ccfs("green", "sci2")
        assert res["ccf"].shape == (NORDER_GREEN, NVEL)

    def test_missing_barycorr_z_raises(self, cc_kpf2, monkeypatch):
        monkeypatch.setattr(
            CrossCorrelation,
            "_build_line_mask",
            lambda self, chip, fiber, mask_width=None: make_mask(MASK_CENTERS),
        )
        cc_kpf2.set_data("BARYCORR_Z", np.array([]))
        cc = CrossCorrelation(cc_kpf2, config={"ccf_window": RANGE_KMS})
        with pytest.raises(ValueError, match="BARYCORR_Z"):
            cc.compute_ccfs("GREEN", "SCI2")

    def test_barycorr_z_removed_per_order(self, monkeypatch):
        # The plumbing under test is L2 BARYCORR_Z -> {chip}_BARYCORR_Z slice ->
        # barycorr_z[order]. A single constant z would survive a green/red slice
        # swap or an order off-by-one, so every order gets its own BERV.
        _stub_line_mask(monkeypatch)
        berv = -5.0 + 0.05 * np.arange(NORDER)
        cc = CrossCorrelation(
            _build_cc_kpf2(berv=berv), config={"ccf_window": RANGE_KMS}
        )
        for chip in ("GREEN", "RED"):
            res = cc.compute_ccfs(chip, "SCI2")
            vel, ccf = res["velocity"], res["ccf"]
            recovered = vel[np.argmin(ccf, axis=1)]
            np.testing.assert_allclose(recovered, V_INJECT, atol=0.3)

    def test_barycorr_z_sign_is_pinned(self, monkeypatch):
        # Negating only BARYCORR_Z (leaving the spectra alone) must displace the
        # recovered velocity by about -2*BERV. A sign flip, or BARYCORR_KMS used
        # where the dimensionless z belongs, lands here.
        _stub_line_mask(monkeypatch)
        berv = np.full(NORDER, -5.0)
        kpf2 = _build_cc_kpf2(berv=berv)
        kpf2.set_data("BARYCORR_Z", -np.asarray(kpf2.data["BARYCORR_Z"]))
        cc = CrossCorrelation(kpf2, config={"ccf_window": RANGE_KMS})
        res = cc.compute_ccfs("GREEN", "SCI2")
        recovered = res["velocity"][np.argmin(res["ccf"][0])]
        assert recovered == pytest.approx(V_INJECT - 2 * berv[0], abs=0.3)

    def test_all_zero_ccf_raises(self, cc_module):
        # No usable signal across the whole orderlet -> fail loudly instead of
        # silently returning an all-zero CCF cube.
        flux = np.asarray(cc_module.l2_obj.data["GREEN_SCI2_FLUX"])
        cc_module.l2_obj.set_data("GREEN_SCI2_FLUX", np.zeros_like(flux))
        with pytest.raises(RuntimeError, match="identically zero"):
            cc_module.compute_ccfs("GREEN", "SCI2")

    def test_clip_edge_pixels_zero_keeps_all(self, cc_module):
        # "Not all zero" would also hold for an implementation that ignored the
        # argument; the unclipped CCF must differ from a heavily clipped one.
        full = cc_module.compute_ccfs("GREEN", "SCI2", clip_edge_pixels=[0, 0])["ccf"]
        # 800 of the 2000 pixels is 20 A of the 50 A order, enough to cut past
        # the bluest mask lines; a smaller clip leaves every line inside and the
        # CCF genuinely unchanged.
        clipped = cc_module.compute_ccfs("GREEN", "SCI2", clip_edge_pixels=[800, 0])[
            "ccf"
        ]
        assert np.any(full)
        assert not np.allclose(full, clipped)

    def test_clip_edge_pixels_ends_are_distinguished(self, cc_module):
        # clip_edge_pixels is [short_wavelength_end, long_wavelength_end]; WAVE is
        # ascending, so pixel 0 is the short end and clipping one end must not
        # produce the same CCF as clipping the other.
        blue = cc_module.compute_ccfs("GREEN", "SCI2", clip_edge_pixels=[800, 0])["ccf"]
        red = cc_module.compute_ccfs("GREEN", "SCI2", clip_edge_pixels=[0, 800])["ccf"]
        assert not np.allclose(blue, red)

    def test_clip_edge_pixels_too_large_raises(self, cc_module):
        with pytest.raises(ValueError, match="removes all"):
            cc_module.compute_ccfs("GREEN", "SCI2", clip_edge_pixels=[NCOL, NCOL])


class TestPerform:
    _ILLUMINATED = ["SCI1", "SCI2", "SCI3", "SKY"]  # CAL-OBJ='None' -> skipped

    def test_returns_kpf4_with_per_orderlet_extensions(self, performed):
        cc_module, l4 = performed
        assert isinstance(l4, KPF4)
        for fiber in self._ILLUMINATED:
            assert l4.data[f"{fiber}_CCF"].shape == (NORDER, NVEL)
            assert_dtype(l4.data[f"{fiber}_CCF"], CCF, f"{fiber}_CCF")
            assert l4.data[f"{fiber}_CCF_VAR"].shape == (NORDER, NVEL)
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
            for column in ("BJD_TDB", "WAVE_START", "WAVE_END"):
                assert_dtype(table[column], RV_FLOAT, column)
            assert np.issubdtype(table["ORDER_INDEX"].dtype, np.integer)
            assert np.issubdtype(table["ECHELLE_ORDER"].dtype, np.integer)

    def test_rv_columns_are_nan_placeholders(self, performed):
        # CrossCorrelation seeds the RV table metadata but leaves RV/RV_ERR NaN
        # for RadialVelocity to fill.
        cc_module, l4 = performed
        for fiber in self._ILLUMINATED:
            table = l4.data[f"{fiber}_RV"]
            assert np.all(np.isnan(np.asarray(table["RV"], dtype=float)))
            assert np.all(np.isnan(np.asarray(table["RV_ERR"], dtype=float)))

    def test_unilluminated_fiber_skipped(self, performed):
        # CAL-OBJ='None' -> no CCF cube, CCF variance, or RV table written.
        cc_module, l4 = performed
        assert l4.data["CAL_CCF"].size == 0
        assert l4.data["CAL_CCF_VAR"].size == 0
        assert len(l4.data["CAL_RV"]) == 0

    def test_ccf_chip_halves_populated(self, performed):
        cc_module, l4 = performed
        assert l4.data["GREEN_SCI2_CCF"].shape == (NORDER_GREEN, NVEL)
        assert l4.data["RED_SCI2_CCF"].shape == (NORDER_RED, NVEL)
        assert np.any(l4.data["GREEN_SCI2_CCF"])
        assert np.any(l4.data["RED_SCI2_CCF"])

    def test_ccf_var_persisted(self, performed):
        # The per-bin CCF variance cube is written alongside the CCF and matches
        # the variance compute_ccfs cached.
        cc_module, l4 = performed
        for chip, norder in (("GREEN", NORDER_GREEN), ("RED", NORDER_RED)):
            var = l4.data[f"{chip}_SCI2_CCF_VAR"]
            assert var.shape == (norder, NVEL)
            np.testing.assert_array_equal(var, cc_module._ccf_var[f"{chip}_SCI2"])
        assert np.any(l4.data["GREEN_SCI2_CCF_VAR"])

    def test_dip_at_injected_velocity_illuminated_orderlets(self, performed):
        # No RV fit is involved here -- only the CCF minimum.
        cc_module, l4 = performed
        vel = cc_module._velocity_grid["RED_SCI2"]
        for fiber in self._ILLUMINATED:
            ccf0 = np.asarray(l4.data[f"{fiber}_CCF"])[0]
            assert vel[np.argmin(ccf0)] == pytest.approx(V_INJECT, abs=0.3)

    def test_rv_table_weight_column_matches_order_weights(self, performed):
        # The per-order CCF-combination weights (ccf_order_weights.csv, column
        # by the orderlet's mask) are persisted as the WEIGHT column, green
        # orders then red, matching _get_order_weights.
        cc_module, l4 = performed
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

    def test_rv_table_order_id_and_echelle_columns(self, performed):
        # ORDER_ID is the KPF chip/fiber/order name, 0-based per chip (green then
        # red). ECHELLE_ORDER is the physical grating order (detector.toml),
        # blue->red: GREEN 137..103, RED 102..71.
        cc_module, l4 = performed
        for fiber in self._ILLUMINATED:
            table = l4.data[f"{fiber}_RV"]
            order_id = np.asarray(table["ORDER_ID"])
            echelle = np.asarray(table["ECHELLE_ORDER"])
            assert order_id[0] == f"GREEN_{fiber}_0"
            assert order_id[NORDER_GREEN - 1] == f"GREEN_{fiber}_{NORDER_GREEN - 1}"
            assert order_id[NORDER_GREEN] == f"RED_{fiber}_0"
            assert order_id[-1] == f"RED_{fiber}_{NORDER_RED - 1}"
            assert echelle[0] == 137
            assert echelle[NORDER_GREEN - 1] == 103
            assert echelle[NORDER_GREEN] == 102
            assert echelle[-1] == 71

    def test_rv_table_metadata_columns_match_the_l2(self, monkeypatch):
        # BJD_TDB/BERV are copied from the L2 barycentric extensions and
        # WAVE_START/WAVE_END are the per-order first/last wavelength. With the
        # shared fixture every order carries the same wave and a zero barycorr,
        # so a wave[:, 0]/wave[:, -1] swap or a green/red row slip is invisible;
        # here each order gets its own span and timestamp.
        _stub_line_mask(monkeypatch)
        berv = -5.0 + 0.05 * np.arange(NORDER)
        bjd = 2460000.0 + np.arange(NORDER)
        offsets = 5.0 * np.arange(NORDER)
        cc = CrossCorrelation(
            _build_cc_kpf2(berv=berv, wave_offsets=offsets, bjd=bjd),
            config={"ccf_window": RANGE_KMS},
        )
        l4 = cc.perform(chips=["GREEN"], fibers=["SCI2"])
        table = l4.data["GREEN_SCI2_RV"]
        wave = np.asarray(cc.l2_obj.data["GREEN_SCI2_WAVE"])
        np.testing.assert_allclose(table["WAVE_START"], wave[:, 0])
        np.testing.assert_allclose(table["WAVE_END"], wave[:, -1])
        np.testing.assert_allclose(table["BJD_TDB"], bjd[:NORDER_GREEN])
        np.testing.assert_allclose(table["BERV"], berv[:NORDER_GREEN])

    def test_ccf_and_rv_headers(self, performed):
        # CrossCorrelation stamps the CCF EPRV keywords and the RV table-structure
        # CTYPE cards, but not the RV-processing descriptors (RadialVelocity's)
        # or any PRIMARY combined RV.
        cc_module, l4 = performed
        ccf_hdr = l4.headers["SCI2_CCF"]
        assert ccf_hdr["CTYPE1"] == "Velocity"
        assert ccf_hdr["CTYPE2"] == "Order-N"
        assert ccf_hdr["VELNSTEP"] == NVEL
        assert ccf_hdr["VELSTEP"] == pytest.approx(0.25)
        assert ccf_hdr["VELSTART"] == pytest.approx(RANGE_KMS[0])  # center 0
        assert ccf_hdr["CCFMASK"] == "G2_espresso"  # BP-RP 0.823 -> 5770 K -> G2
        assert ccf_hdr["VELWIDTH"] == pytest.approx(cc_module.ccf_mask_width)
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
        cc_module.l2_obj.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = "Th_gold"
        l4 = cc_module.perform(fibers=["CAL"])
        assert l4.headers["CAL_CCF"]["CCFMASK"] == "thar"

    def test_unimplemented_fiber_skipped(self, caplog, cc_module):
        # An etalon/lfc fiber has no CCF path yet -> skipped, empty extensions.
        cc_module.l2_obj.headers["INSTRUMENT_HEADER"]["CAL-OBJ"] = "EtalonFiber"
        with caplog.at_level(logging.WARNING):
            l4 = cc_module.perform(fibers=["CAL"])
        assert re.search(r"etalon.*not implemented", caplog.text)
        assert l4.data["CAL_CCF"].size == 0
        assert len(l4.data["CAL_RV"]) == 0

    def test_explicit_chips_and_fibers(self, cc_module):
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
            assert ccf["VELNSTEP"] == NVEL
            assert ccf["VELSTART"] == pytest.approx(RANGE_KMS[0])
            assert ccf.comments["VELSTART"]  # comment preserved
            rv = hdul["RV3"].header
            assert rv["CTYPE1"] == "Columns"
            assert "RVMETHOD" not in rv
            rv_table = hdul["RV3"].data
            assert rv_table["ORDER_ID"][0] == "GREEN_SCI2_0"
            assert rv_table["ECHELLE_ORDER"][0] == 137
            # The per-bin CCF variance cube round-trips as an image extension.
            assert hdul["CCF3_VAR"].data.shape == (NORDER, NVEL)


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

    def test_info_after_perform(self, performed, capsys):
        cc_module, _ = performed
        cc_module.info()
        out = capsys.readouterr().out
        assert "CrossCorrelation" in out
        assert "SCI2" in out
