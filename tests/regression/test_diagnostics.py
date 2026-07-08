"""Tests for the Diagnostics framework and per-level subclasses."""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from kpfpipe import DETECTOR
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.quality_control.diagnostics import (
    DiagL0,
    DiagL1,
    DiagL2,
    DiagL4,
    Diagnostics,
)

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
NCOL = DETECTOR["ccd"]["ncol"]

_FIBERS = ("SCI1", "SCI2", "SCI3", "SKY", "CAL")
_NAN_KEYS = ("NANSCI1", "NANSCI2", "NANSCI3", "NANSKY", "NANCAL")


# ---------------------------------------------------------------------------
# Diagnostics base class
# ---------------------------------------------------------------------------


class TestDiagnosticsBase:
    def _make_obj(self):
        class _FakeObj:
            headers = {"PRIMARY": {}}
            data = {}

            def set_keyword(self, key, value):
                # Mirror the real routing: set_keyword writes the value only
                # (the comment comes from the registry). The base test keys are
                # not in any registry, so the stub just lands them on PRIMARY.
                self.headers["PRIMARY"][key] = value

        return _FakeObj()

    def test_writes_returned_keys_to_primary(self):
        obj = self._make_obj()

        class MyDiag(Diagnostics):
            def metric_a(self):
                return {"KEYA": (3.14, "metric a")}

            metric_a._diag_name = "metric_a"

        results = MyDiag(obj).run()
        assert obj.headers["PRIMARY"]["KEYA"] == 3.14
        assert results["KEYA"] == (3.14, "metric a")

    def test_method_can_emit_multiple_keys(self):
        obj = self._make_obj()

        class MyDiag(Diagnostics):
            def multi(self):
                return {"K1": (1, "one"), "K2": (2, "two")}

            multi._diag_name = "multi"

        MyDiag(obj).run()
        assert obj.headers["PRIMARY"]["K1"] == 1
        assert obj.headers["PRIMARY"]["K2"] == 2

    def test_empty_dict_writes_nothing(self):
        obj = self._make_obj()

        class MyDiag(Diagnostics):
            def skipped(self):
                return {}

            skipped._diag_name = "skipped"

        results = MyDiag(obj).run()
        assert results == {}
        assert obj.headers["PRIMARY"] == {}

    def test_raising_method_propagates_runtime_error(self):
        obj = self._make_obj()

        class MyDiag(Diagnostics):
            def boom(self):
                raise ValueError("boom!")

            boom._diag_name = "boom"

        with pytest.raises(RuntimeError, match="Diagnostic 'boom' raised"):
            MyDiag(obj).run()

    def test_repeated_run_resets_results(self):
        obj = self._make_obj()
        obj.value = 1

        class MyDiag(Diagnostics):
            def metric(self):
                return {"VAL": (self.kpf_obj.value, "value")}

            metric._diag_name = "metric"

        d = MyDiag(obj)
        d.run()
        assert d.results == {"VAL": (1, "value")}

        obj.value = 99
        d.run()
        assert d.results == {"VAL": (99, "value")}

    def test_empty_subclass_runs_cleanly(self):
        obj = self._make_obj()

        class EmptyDiag(Diagnostics):
            pass

        results = EmptyDiag(obj).run()
        assert results == {}


# ---------------------------------------------------------------------------
# DiagL0 with no pointing (e.g. a calibration frame) and DiagL1 with no
# calibrations are each a clean no-op
# ---------------------------------------------------------------------------


class TestEmptyLevels:
    def _make_obj(self):
        class _FakeObj:
            headers = {"PRIMARY": {}}
            data = {}

        return _FakeObj()

    def test_diag_l0_runs_cleanly(self):
        # No RA/DEC/GAIAID -> GAIAOFF/TARGOFF both skip (fail-soft), no crash.
        results = DiagL0(self._make_obj()).run()
        assert results == {}

    def test_diag_l1_runs_cleanly(self):
        # No RECEIPT/INSTRUMENT_HEADER -> calibration_ages returns {} (no crash).
        results = DiagL1(self._make_obj()).run()
        assert results == {}


# ---------------------------------------------------------------------------
# DiagL0 — pointing / identity offsets (GAIAOFF, TARGOFF)
# ---------------------------------------------------------------------------

_PT_RA, _PT_DEC = "01:44:01.30", "-15:55:54.0"


def _make_l0_pointing(**overrides):
    """A KPF0 with L0 PRIMARY pointing + DCS target + GAIAID natives.

    Pass ``KEY=None`` to omit a native (used to exercise the skip paths).
    """
    prim = {
        "RA": _PT_RA,
        "DEC": _PT_DEC,
        "MJD-OBS": 60540.6,
        "GAIAID": "DR3 12345",
        "TARGRA": _PT_RA,
        "TARGDEC": _PT_DEC,
        "TARGPMRA": 0.0,
        "TARGPMDC": 0.0,
        "TARGPLAX": 100.0,
        "TARGFRAM": "FK5",
        "TARGEPOC": 2000.0,
    }
    prim.update(overrides)
    l0 = KPF0()
    for k, v in prim.items():
        if v is not None:
            l0.headers["PRIMARY"][k] = v
    return l0


def _fake_gaia_job(ra_deg, dec_deg):
    """Stand-in for Gaia.launch_job: a one-row results Table, no network.

    Zero proper motion so ``apply_space_motion`` leaves the position at
    (ra_deg, dec_deg).
    """
    tbl = Table(
        {
            "ra": [ra_deg],
            "dec": [dec_deg],
            "pmra": [0.0],
            "pmdec": [0.0],
            "parallax": [100.0],
            "ref_epoch": [2016.0],
        }
    )

    class _Job:
        def get_results(self):
            return tbl

    return _Job()


class TestDiagL0Pointing:
    def _gaia_at_pointing(self):
        # Gaia source placed exactly at the pointing -> GAIAOFF ~ 0.
        pt = SkyCoord(_PT_RA, _PT_DEC, unit=(u.hourangle, u.deg))
        return _fake_gaia_job(pt.ra.deg, pt.dec.deg)

    def test_offsets_written_to_quality_control(self):
        l0 = _make_l0_pointing()
        with patch(
            "kpfpipe.quality_control.diagnostics.level0.Gaia.launch_job",
            return_value=self._gaia_at_pointing(),
        ):
            results = DiagL0(l0).run()
        # Pointing == target == Gaia position, so both offsets are ~0 (TARGOFF
        # carries the ~23 mas ICRS<->FK5 frame bias); well under the 1" budget.
        assert results["GAIAOFF"][0] < 0.1
        assert results["TARGOFF"][0] < 0.1
        # set_keyword routed both to QUALITY_CONTROL.
        assert l0.headers["QUALITY_CONTROL"]["GAIAOFF"] == results["GAIAOFF"][0]
        assert l0.headers["QUALITY_CONTROL"]["TARGOFF"] == results["TARGOFF"][0]

    def test_skip_when_no_pointing(self):
        # A calibration-like frame with no RA/DEC/target -> both metrics N/A.
        l0 = _make_l0_pointing(RA=None, DEC=None, TARGRA=None, GAIAID=None)
        results = DiagL0(l0).run()
        assert "GAIAOFF" not in results
        assert "TARGOFF" not in results

    def test_skip_gaiaoff_when_no_gaiaid(self):
        # Pointing + target present, GAIAID absent -> TARGOFF only, no warning.
        l0 = _make_l0_pointing(GAIAID=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = DiagL0(l0).run()
        assert "GAIAOFF" not in results
        assert "TARGOFF" in results
        assert not any("GAIAOFF skipped" in str(c.message) for c in caught)

    def test_gaia_failure_warns_and_skips(self):
        # Gaia unreachable -> GAIAOFF warns and skips; the network-free TARGOFF
        # still computes (fail-soft, so the L0 checkpoint does not fail).
        l0 = _make_l0_pointing()
        with (
            patch(
                "kpfpipe.quality_control.diagnostics.level0.Gaia.launch_job",
                side_effect=ConnectionError("gaia down"),
            ),
            pytest.warns(UserWarning, match="GAIAOFF skipped"),
        ):
            results = DiagL0(l0).run()
        assert "GAIAOFF" not in results
        assert "TARGOFF" in results


# ---------------------------------------------------------------------------
# DiagL0 — OBJECT-name offset via SIMBAD (OBJOFF)
# ---------------------------------------------------------------------------


def _fake_simbad(ra_deg=None, dec_deg=None, no_match=False):
    """Stand-in for the Simbad class: Simbad() -> instance whose query_object
    returns a one-row Table (astroquery 0.4.11 schema), or None for no match."""
    instance = MagicMock()
    if no_match:
        instance.query_object.return_value = None
    else:
        instance.query_object.return_value = Table(
            {
                "ra": [ra_deg],
                "dec": [dec_deg],
                "pmra": [0.0],
                "pmdec": [0.0],
                "plx_value": [100.0],
            }
        )
    return instance


class TestDiagL0Object:
    def test_object_offset_routed_to_quality_control(self):
        # GAIAID absent -> GAIAOFF skips silently (no Gaia network); OBJECT set,
        # SIMBAD placed at the pointing -> OBJOFF ~ 0, routed to QUALITY_CONTROL.
        l0 = _make_l0_pointing(GAIAID=None)
        l0.headers["PRIMARY"]["OBJECT"] = "10700"
        pt = SkyCoord(_PT_RA, _PT_DEC, unit=(u.hourangle, u.deg))
        with patch(
            "kpfpipe.quality_control.diagnostics.level0.Simbad",
            return_value=_fake_simbad(pt.ra.deg, pt.dec.deg),
        ):
            results = DiagL0(l0).run()
        assert results["OBJOFF"][0] < 0.1
        assert l0.headers["QUALITY_CONTROL"]["OBJOFF"] == results["OBJOFF"][0]

    def test_object_name_hd_prefix(self):
        # Bare-numeric OBJECT gets an 'HD ' prefix; named targets pass through.
        l0 = _make_l0_pointing()
        l0.headers["PRIMARY"]["OBJECT"] = "10700"
        assert DiagL0(l0)._object_name() == "HD 10700"
        l0.headers["PRIMARY"]["OBJECT"] = "tau Cet"
        assert DiagL0(l0)._object_name() == "tau Cet"

    def test_skip_when_no_object(self):
        # No OBJECT native -> metric N/A, skip silently (no SIMBAD call).
        l0 = _make_l0_pointing()  # helper sets no OBJECT
        assert DiagL0(l0).object_ra_dec_offset() == {}

    def test_simbad_no_match_warns_and_skips(self):
        l0 = _make_l0_pointing()
        l0.headers["PRIMARY"]["OBJECT"] = "NotARealStar"
        with (
            patch(
                "kpfpipe.quality_control.diagnostics.level0.Simbad",
                return_value=_fake_simbad(no_match=True),
            ),
            pytest.warns(UserWarning, match="OBJOFF skipped"),
        ):
            assert DiagL0(l0).object_ra_dec_offset() == {}


# ---------------------------------------------------------------------------
# DiagL1 — master calibration ages
# ---------------------------------------------------------------------------


def _make_kpf1_with_calibrations(date_obs="2024-04-05T11:08:33", files=None):
    """A KPF1 carrying a PRIMARY DATE-OBS and RECEIPT master paths.

    Mirrors the finished-L1 state DiagL1 reads: CalibrationAssociation has
    written each ``{PREFIX}FILE`` to RECEIPT (via set_keyword) and to_kpf1 has
    populated the EPRV PRIMARY (DATE-OBS).
    """
    l1 = KPF1()
    l1.headers["PRIMARY"]["DATE-OBS"] = date_obs
    for kw, path in (files or {}).items():
        l1.set_keyword(kw, path)  # *FILE routes to RECEIPT
    return l1


class TestDiagL1CalibrationAges:
    def test_signed_age_same_day(self):
        # Master at 2024-04-05 01:00:37 UTC vs obs 11:08:33 UTC -> -0.422176 d.
        l1 = _make_kpf1_with_calibrations(
            files={"BIASFILE": "/m/KP.20240405.03637.74_master_bias_L1.fits"}
        )
        results = DiagL1(l1).run()
        assert results["BIASAGE"][0] == pytest.approx(-0.422176, abs=1e-5)
        # Routed to QUALITY_CONTROL with the registry comment.
        qc = l1.headers["QUALITY_CONTROL"]
        assert qc["BIASAGE"] == pytest.approx(-0.422176, abs=1e-5)
        assert qc.comments["BIASAGE"] == "Master bias age [days]"

    def test_signed_age_previous_day(self):
        # Master 2024-04-04 22:00:00 UTC vs obs 2024-04-05 11:08:33 UTC.
        l1 = _make_kpf1_with_calibrations(
            files={"BIASFILE": "/m/KP.20240404.79200.00_master_bias_L1.fits"}
        )
        results = DiagL1(l1).run()
        assert results["BIASAGE"][0] == pytest.approx(-0.547604, abs=1e-5)

    def test_all_cal_types(self):
        l1 = _make_kpf1_with_calibrations(
            files={
                "BIASFILE": "/m/KP.20240405.03637.74_master_bias_L1.fits",
                "DARKFILE": "/m/KP.20240405.03637.74_master_dark_L1.fits",
                "FLATFILE": "/m/KP.20240405.03637.74_master_flat_L1.fits",
                "WLSFILE": "/m/KP.20240405.03637.74_master_thar_L2.fits",
            }
        )
        results = DiagL1(l1).run()
        assert set(results) == {"BIASAGE", "DARKAGE", "FLATAGE", "WLSAGE"}
        for kw in results:
            assert l1.headers["QUALITY_CONTROL"][kw] == pytest.approx(
                -0.422176, abs=1e-5
            )

    def test_missing_cal_type_skipped(self):
        # Only a bias path present -> only BIASAGE written.
        l1 = _make_kpf1_with_calibrations(
            files={"BIASFILE": "/m/KP.20240405.03637.74_master_bias_L1.fits"}
        )
        results = DiagL1(l1).run()
        assert set(results) == {"BIASAGE"}
        assert "DARKAGE" not in l1.headers["QUALITY_CONTROL"]

    def test_no_date_obs_skips_all(self):
        l1 = _make_kpf1_with_calibrations(
            files={"BIASFILE": "/m/KP.20240405.03637.74_master_bias_L1.fits"}
        )
        del l1.headers["PRIMARY"]["DATE-OBS"]
        results = DiagL1(l1).run()
        assert results == {}


# ---------------------------------------------------------------------------
# DiagL2 — NaN counts + zero-flux fraction
# ---------------------------------------------------------------------------


def _make_kpf2_with_flux(nan_frac=0.0, zero_frac=0.0):
    """Build a minimal KPF1, convert to KPF2, populate FLUX/VAR extensions
    with controllable NaN and zero fractions across all (chip, fiber) pairs.

    Each FLUX extension has shape (norder[chip], NCOL). Each is initialized
    to ones, then a fraction is replaced with NaN, then a fraction with 0.0.
    """
    from io import BytesIO

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-01-01T00:00:00"
    green_ccd = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="GREEN_CCD")
    green_var = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="GREEN_VAR")
    red_ccd = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_CCD")
    red_var = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_VAR")

    # Round-trip via in-memory FITS to produce a valid KPF1 → to_kpf2().
    buf = BytesIO()
    fits.HDUList([primary, green_ccd, green_var, red_ccd, red_var]).writeto(buf)
    buf.seek(0)
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
        tmp.write(buf.read())
        tmp_path = tmp.name
    try:
        l1 = KPF1.from_fits(tmp_path)
    finally:
        os.unlink(tmp_path)
    kpf2 = l1.to_kpf2()

    norder = {"GREEN": NORDER_GREEN, "RED": NORDER_RED}
    rng = np.random.default_rng(42)
    for chip in ("GREEN", "RED"):
        for fiber in _FIBERS:
            n = norder[chip]
            arr = np.ones((n, NCOL), dtype=np.float32)
            mask = rng.random(arr.shape)
            if nan_frac > 0:
                arr[mask < nan_frac] = np.nan
            if zero_frac > 0:
                # Zeros fall in the band [nan_frac, nan_frac+zero_frac)
                arr[(mask >= nan_frac) & (mask < nan_frac + zero_frac)] = 0.0
            kpf2.set_data(f"{chip}_{fiber}_FLUX", arr)
    return kpf2


class TestDiagL2NanCounts:
    def test_writes_all_five_keys_with_zero_when_clean(self):
        kpf2 = _make_kpf2_with_flux(nan_frac=0.0)
        DiagL2(kpf2).run()
        for key in _NAN_KEYS:
            assert key in kpf2.headers["QUALITY_CONTROL"], f"missing {key}"
            assert kpf2.headers["QUALITY_CONTROL"].get(key) == 0

    def test_counts_injected_nans_per_fiber(self):
        kpf2 = _make_kpf2_with_flux(nan_frac=0.0)
        # Inject one NaN into GREEN_SCI1_FLUX; expect NANSCI1==1, others==0.
        kpf2.data["GREEN_SCI1_FLUX"][0, 0] = np.nan
        DiagL2(kpf2).run()
        assert kpf2.headers["QUALITY_CONTROL"].get("NANSCI1") == 1
        for key in ("NANSCI2", "NANSCI3", "NANSKY", "NANCAL"):
            assert kpf2.headers["QUALITY_CONTROL"].get(key) == 0

    def test_writes_keys_even_when_no_data(self):
        """KPF2 with no FLUX extensions populated should still write all 5
        keys with value 0 (consistent header schema)."""
        # Build a KPF2 without populating any FLUX arrays.
        from io import BytesIO

        primary = fits.PrimaryHDU()
        primary.header["INSTRUME"] = "KPF"
        primary.header["DATE-OBS"] = "2024-01-01T00:00:00"
        hdul = fits.HDUList(
            [
                primary,
                fits.ImageHDU(
                    data=np.zeros((4, 4), dtype=np.float32), name="GREEN_CCD"
                ),
                fits.ImageHDU(
                    data=np.zeros((4, 4), dtype=np.float32), name="GREEN_VAR"
                ),
                fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_CCD"),
                fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_VAR"),
            ]
        )
        buf = BytesIO()
        hdul.writeto(buf)
        buf.seek(0)
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            tmp.write(buf.read())
            tmp_path = tmp.name
        try:
            l1 = KPF1.from_fits(tmp_path)
        finally:
            os.unlink(tmp_path)
        kpf2 = l1.to_kpf2()
        DiagL2(kpf2).run()
        for key in _NAN_KEYS:
            assert kpf2.headers["QUALITY_CONTROL"].get(key) == 0


class TestDiagL2ZeroFlux:
    def test_zerofrac_written_when_data_present(self):
        kpf2 = _make_kpf2_with_flux(zero_frac=0.0)  # all ones
        DiagL2(kpf2).run()
        assert "ZEROFRAC" in kpf2.headers["QUALITY_CONTROL"]
        assert kpf2.headers["QUALITY_CONTROL"].get("ZEROFRAC") == pytest.approx(0.0)

    def test_zerofrac_one_when_all_zero(self):
        kpf2 = _make_kpf2_with_flux(zero_frac=1.0)
        DiagL2(kpf2).run()
        assert kpf2.headers["QUALITY_CONTROL"].get("ZEROFRAC") == pytest.approx(1.0)

    def test_zerofrac_approximate_when_partial(self):
        """50% zeros sprinkled randomly → ZEROFRAC ≈ 0.5 within sampling error."""
        kpf2 = _make_kpf2_with_flux(zero_frac=0.5)
        DiagL2(kpf2).run()
        assert kpf2.headers["QUALITY_CONTROL"].get("ZEROFRAC") == pytest.approx(
            0.5, abs=0.01
        )

    def test_zerofrac_skipped_when_no_data(self):
        """KPF2 with no populated FLUX extensions → no ZEROFRAC key written."""
        from io import BytesIO

        primary = fits.PrimaryHDU()
        primary.header["DATE-OBS"] = "2024-01-01T00:00:00"
        hdul = fits.HDUList(
            [
                primary,
                fits.ImageHDU(
                    data=np.zeros((4, 4), dtype=np.float32), name="GREEN_CCD"
                ),
                fits.ImageHDU(
                    data=np.zeros((4, 4), dtype=np.float32), name="GREEN_VAR"
                ),
                fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_CCD"),
                fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_VAR"),
            ]
        )
        buf = BytesIO()
        hdul.writeto(buf)
        buf.seek(0)
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            tmp.write(buf.read())
            tmp_path = tmp.name
        try:
            l1 = KPF1.from_fits(tmp_path)
        finally:
            os.unlink(tmp_path)
        kpf2 = l1.to_kpf2()
        DiagL2(kpf2).run()
        assert "ZEROFRAC" not in kpf2.headers["QUALITY_CONTROL"]


def _set_fiber_arrays(kpf2, suffix, value, chips=("GREEN", "RED"), fibers=_FIBERS):
    """Populate {chip}_{fiber}_{suffix} with a constant for the given fibers."""
    norder = {"GREEN": NORDER_GREEN, "RED": NORDER_RED}
    for chip in chips:
        for fiber in fibers:
            kpf2.set_data(
                f"{chip}_{fiber}_{suffix}",
                np.full((norder[chip], NCOL), value, dtype=np.float32),
            )


# ---------------------------------------------------------------------------
# DiagL2 — per-fiber SNR
# ---------------------------------------------------------------------------


class TestDiagL2Snr:
    _SNR_KEYS = ("GSNRSCI", "GSNRSKY", "GSNRCAL", "RSNRSCI", "RSNRSKY", "RSNRCAL")

    def test_keys_written_when_flux_and_var_present(self):
        kpf2 = _make_kpf2_with_flux()  # all FLUX ones
        _set_fiber_arrays(kpf2, "VAR", 0.25)
        DiagL2(kpf2).run()
        for key in self._SNR_KEYS:
            assert key in kpf2.headers["QUALITY_CONTROL"], f"missing {key}"
            assert kpf2.headers["QUALITY_CONTROL"].get(key) > 0

    def test_single_fiber_snr_value(self):
        # SKY flux=2, var=0.04 -> SNR = 2/sqrt(0.04) = 10.0 in every pixel.
        kpf2 = _make_kpf2_with_flux()
        _set_fiber_arrays(kpf2, "FLUX", 2.0, fibers=("SKY",))
        _set_fiber_arrays(kpf2, "VAR", 0.04, fibers=("SKY",))
        DiagL2(kpf2).run()
        assert kpf2.headers["QUALITY_CONTROL"].get("GSNRSKY") == pytest.approx(
            10.0, abs=0.01
        )
        assert kpf2.headers["QUALITY_CONTROL"].get("RSNRSKY") == pytest.approx(
            10.0, abs=0.01
        )

    def test_summed_sci_snr_value(self):
        # Each SCI fiber flux=2, var=0.04 -> summed flux=6, var=0.12;
        # SNR = 6/sqrt(0.12) ~= 17.32.
        kpf2 = _make_kpf2_with_flux()
        _set_fiber_arrays(kpf2, "FLUX", 2.0, fibers=("SCI1", "SCI2", "SCI3"))
        _set_fiber_arrays(kpf2, "VAR", 0.04, fibers=("SCI1", "SCI2", "SCI3"))
        DiagL2(kpf2).run()
        assert kpf2.headers["QUALITY_CONTROL"].get("GSNRSCI") == pytest.approx(
            17.32, abs=0.05
        )

    def test_summed_sci_skipped_when_a_sci_var_missing(self):
        # VAR for SCI1/SCI2 only (SCI3 var stays empty) -> summed-SCI skipped,
        # but SKY (var present) is still computed.
        kpf2 = _make_kpf2_with_flux()
        _set_fiber_arrays(kpf2, "VAR", 0.25, fibers=("SCI1", "SCI2", "SKY", "CAL"))
        DiagL2(kpf2).run()
        assert "GSNRSCI" not in kpf2.headers["QUALITY_CONTROL"]
        assert "GSNRSKY" in kpf2.headers["QUALITY_CONTROL"]

    def test_skipped_without_var(self):
        # Default fixture leaves VAR empty -> no SNR keys at all.
        kpf2 = _make_kpf2_with_flux()
        DiagL2(kpf2).run()
        for key in self._SNR_KEYS:
            assert key not in kpf2.headers["QUALITY_CONTROL"]


# ---------------------------------------------------------------------------
# DiagL2 — orderlet flux ratios
# ---------------------------------------------------------------------------


class TestDiagL2OrderletFluxRatios:
    _RATIO_KEYS = (
        "GFR12",
        "GFR32",
        "GFRS2",
        "GFRC2",
        "RFR12",
        "RFR32",
        "RFRS2",
        "RFRC2",
    )

    def test_keys_written_and_unity_when_uniform(self):
        # All fibers flux=1 (default) -> every inter-fiber ratio == 1.0.
        kpf2 = _make_kpf2_with_flux()
        DiagL2(kpf2).run()
        for key in self._RATIO_KEYS:
            assert key in kpf2.headers["QUALITY_CONTROL"], f"missing {key}"
            assert kpf2.headers["QUALITY_CONTROL"].get(key) == pytest.approx(1.0)

    def test_ratio_value(self):
        # GREEN SCI1 flux=2 over SCI2 flux=1 -> GFR12 == 2.0.
        kpf2 = _make_kpf2_with_flux()
        _set_fiber_arrays(kpf2, "FLUX", 2.0, chips=("GREEN",), fibers=("SCI1",))
        DiagL2(kpf2).run()
        assert kpf2.headers["QUALITY_CONTROL"].get("GFR12") == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# DiagL4 — per-order BJD / barycentric-RV dispersion (ported from v2.12)
# ---------------------------------------------------------------------------


def _l4_with_sci2_rv(bjd, berv, weight):
    """KPF4 carrying a SCI2 per-order RV table with BJD_TDB/BERV/WEIGHT."""
    l4 = KPF4()
    l4.set_data(
        "SCI2_RV",
        Table(
            {
                "ORDER_INDEX": np.arange(len(bjd), dtype=np.int64),
                "BJD_TDB": np.asarray(bjd, dtype=float),
                "BERV": np.asarray(berv, dtype=float),
                "WEIGHT": np.asarray(weight, dtype=float),
            }
        ),
    )
    return l4


class TestDiagL4:
    # Two equal-weight orders with toy values give exact, hand-checkable stats:
    #   BJD  [10, 20] -> mean 15, std 5 d (=432000 s), range 10 d (=864000 s)
    #   BERV [0.1, 0.3] -> mean 0.2, std 0.1 km/s (=100 m/s), range 0.2 (=200 m/s)
    #   per-order BERV %dev = [-50, +50] of the 0.2 mean
    def test_bjd_berv_dispersion_values(self):
        l4 = _l4_with_sci2_rv([10.0, 20.0], [0.1, 0.3], [1.0, 1.0])
        DiagL4(l4).run()
        qc = l4.headers["QUALITY_CONTROL"]
        assert qc["BJDMEAN"] == pytest.approx(15.0)
        assert qc["BJDSTD"] == pytest.approx(432000.0)
        assert qc["BJDRNG"] == pytest.approx(864000.0)
        assert qc["BERVMEAN"] == pytest.approx(0.2)
        assert qc["BERVSTD"] == pytest.approx(100.0)
        assert qc["BERVRNG"] == pytest.approx(200.0)
        assert qc["BERVMAXP"] == pytest.approx(50.0)
        assert qc["BERVMINP"] == pytest.approx(-50.0)

    def test_zero_weight_orders_excluded(self):
        # Third order has zero weight: excluded from the weighted mean and the
        # nonzero-weight range, matching v2.12.
        l4 = _l4_with_sci2_rv([10.0, 20.0, 999.0], [0.1, 0.3, 9.0], [1.0, 1.0, 0.0])
        DiagL4(l4).run()
        qc = l4.headers["QUALITY_CONTROL"]
        assert qc["BJDMEAN"] == pytest.approx(15.0)
        assert qc["BJDRNG"] == pytest.approx(864000.0)
        assert qc["BERVMEAN"] == pytest.approx(0.2)

    def test_skips_without_sci2_rv_table(self):
        # No SCI2 RV table (e.g. unilluminated science) -> no metrics written.
        results = DiagL4(KPF4()).run()
        assert results == {}

    def test_skips_without_weight_column(self):
        # WEIGHT column absent (pre-weights L4) -> skip rather than guess.
        l4 = KPF4()
        l4.set_data(
            "SCI2_RV",
            Table({"ORDER_INDEX": [0, 1], "BJD_TDB": [10.0, 20.0], "BERV": [0.1, 0.3]}),
        )
        assert DiagL4(l4).run() == {}
