"""Tests for the Diagnostics framework and per-level subclasses."""

import logging

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from kpfpipe import DETECTOR
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.modules.astro_query import AstroQuery
from kpfpipe.quality_control.diagnostics import (
    DiagL0,
    DiagL1,
    DiagL2,
    DiagL4,
    Diagnostics,
)

from ._data_models import set_fiber_arrays, set_wave_bands

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
_NCOL_TEST = 8  # DiagL2 metrics are pixel aggregates, so the real detector width
# is moot

_FIBERS = ("SCI1", "SCI2", "SCI3", "SKY", "CAL")
_NAN_KEYS = ("NANSCI1", "NANSCI2", "NANSCI3", "NANSKY", "NANCAL")
_ZERO_KEYS = ("ZEROSCI1", "ZEROSCI2", "ZEROSCI3", "ZEROSKY", "ZEROCAL")


# ---------------------------------------------------------------------------
# Diagnostics base class
# ---------------------------------------------------------------------------


class TestDiagnosticsBase:
    def _make_obj(self):
        class _FakeObj:
            headers = {"PRIMARY": {}}
            data = {}

            def set_keyword(self, key, value):
                # The real set_keyword writes the value only (the comment comes
                # from the registry); these test keys are in no registry, so the
                # stub just lands them on PRIMARY.
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

    def test_none_return_logs_and_writes_nothing(self, caplog):
        obj = self._make_obj()

        class MyDiag(Diagnostics):
            LEVEL = "L0"

            def nothing(self):
                return None

            nothing._diag_name = "nothing"

        with caplog.at_level(logging.ERROR):
            results = MyDiag(obj).run()
        assert results == {}
        assert obj.headers["PRIMARY"] == {}
        assert "diagnostic 'nothing' raised" in caplog.text

    def test_raising_method_logs_and_continues(self, caplog):
        obj = self._make_obj()

        class MyDiag(Diagnostics):
            LEVEL = "L0"

            def boom(self):
                raise ValueError("boom!")

            boom._diag_name = "boom"

            def after(self):
                return {"KEYA": (3.14, "metric a")}

            after._diag_name = "after"

        # Informational layer: run() logs the offender at ERROR and keeps going, so
        # a later metric is still computed. Halting is the checkpoint layer's job.
        with caplog.at_level(logging.ERROR):
            results = MyDiag(obj).run()
        assert "diagnostic 'boom' raised" in caplog.text
        assert "boom!" in caplog.text
        assert results == {"KEYA": (3.14, "metric a")}
        assert obj.headers["PRIMARY"]["KEYA"] == 3.14

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
# DiagL0 -- pointing offsets from CATALOG_RECORD (GAIAOFF, TCSOFF, OBJOFF)
# ---------------------------------------------------------------------------

_PT_RA, _PT_DEC = "01:44:01.30", "-15:55:54.0"

# The flag-carrying sources; 'kpf-drp' is the merged row and has no flag.
_CATALOG_SOURCES = ("wmko", "gaia", "simbad")


def _record_at(coord, **overrides):
    """A catalog record at ``coord`` (zero PM, finite plx) in EPRV C*# format:
    RA/Dec sexagesimal strings, PM arcsec/yr."""
    rec = {
        "object": "test",
        "ra": coord.ra.to_string(unit=u.hourangle, sep=":", pad=True, precision=4),
        "dec": coord.dec.to_string(
            unit=u.deg, sep=":", pad=True, alwayssign=True, precision=3
        ),
        "pmra": 0.0,
        "pmdec": 0.0,
        "parallax": 100.0,
        "rv": 0.0,
        "frame": "icrs",
        "epoch": 2016.0,
        "equinox": 2000.0,
    }
    rec.update(overrides)
    return rec


def _set_catalog_record(l0, records):
    """Write l0's CATALOG_RECORD rows and presence flags from a
    {source: record-dict-or-None} mapping, the way perform does. A source left out
    of the mapping gets flag 0, exactly as a gated-off one does in production."""
    aq = AstroQuery(l0)
    for source, record in records.items():
        aq._write_catalog_record(source, record)
        if source in _CATALOG_SOURCES:
            setattr(aq, f"_{source}", record)
    aq._set_headers(l0)


def _make_l0_pointing():
    """A KPF0 with just an L0 PRIMARY pointing (RA/DEC/MJD-OBS), no catalog yet.
    IMTYPE 'Object' so AstroQuery accepts it."""
    l0 = KPF0()
    l0.headers["PRIMARY"]["IMTYPE"] = "Object"
    l0.headers["PRIMARY"]["RA"] = _PT_RA
    l0.headers["PRIMARY"]["DEC"] = _PT_DEC
    l0.headers["PRIMARY"]["MJD-OBS"] = 60540.6
    return l0


def _make_l0_with_catalog():
    """A KPF0 with L0 PRIMARY pointing and a fully-populated CATALOG_RECORD whose
    gaia/simbad/wmko records all sit at the pointing (all three offsets ~ 0)."""
    l0 = _make_l0_pointing()
    pt = SkyCoord(_PT_RA, _PT_DEC, unit=(u.hourangle, u.deg))
    _set_catalog_record(l0, {src: _record_at(pt) for src in ("gaia", "simbad", "wmko")})
    return l0


class TestDiagL0Offsets:
    def test_offsets_written_to_quality_control(self):
        # All three sources sit at the pointing -> every offset ~0, routed to QC.
        l0 = _make_l0_with_catalog()
        results = DiagL0(l0).run()
        for key in ("GAIAOFF", "TCSOFF", "OBJOFF"):
            assert results[key][0] < 0.1
            assert l0.headers["QUALITY_CONTROL"][key] == results[key][0]

    def test_offset_reflects_catalog_separation(self):
        # Move the Gaia record 10" north of the pointing -> GAIAOFF ~ 10", while
        # the still-at-pointing wmko record keeps TCSOFF ~ 0.
        l0 = _make_l0_pointing()
        pt = SkyCoord(_PT_RA, _PT_DEC, unit=(u.hourangle, u.deg))
        _set_catalog_record(
            l0,
            {
                "gaia": _record_at(pt.directional_offset_by(0 * u.deg, 10 * u.arcsec)),
                "simbad": _record_at(pt),
                "wmko": _record_at(pt),
            },
        )
        results = DiagL0(l0).run()
        assert results["GAIAOFF"][0] == pytest.approx(10.0, abs=0.1)
        assert results["TCSOFF"][0] < 0.1


class TestDiagL0Contingency:
    """Unusable astrometry raises; an unmatched optional source emits no key."""

    def test_no_catalog_record_raises(self):
        # AstroQuery not run: CATALOG_RECORD auto-created but no presence flags.
        with pytest.raises(KeyError, match="GAIACR"):
            DiagL0(_make_l0_pointing()).gaia_ra_dec_offset()

    def test_unmatched_optional_source_emits_no_key(self):
        # Gaia lookup disabled/failed -> GAIACR=0 -> no GAIAOFF; others compute.
        l0 = _make_l0_pointing()
        pt = SkyCoord(_PT_RA, _PT_DEC, unit=(u.hourangle, u.deg))
        _set_catalog_record(
            l0, {"gaia": None, "simbad": _record_at(pt), "wmko": _record_at(pt)}
        )
        results = DiagL0(l0).run()
        assert "GAIAOFF" not in results
        assert "GAIAOFF" not in l0.headers["QUALITY_CONTROL"]
        assert results["TCSOFF"][0] < 0.1
        assert results["OBJOFF"][0] < 0.1

    def test_unmatched_required_source_raises(self):
        # The DCS target offset is required: no wmko row means no TCSOFF to emit.
        l0 = _make_l0_pointing()
        pt = SkyCoord(_PT_RA, _PT_DEC, unit=(u.hourangle, u.deg))
        _set_catalog_record(
            l0, {"gaia": _record_at(pt), "simbad": _record_at(pt), "wmko": None}
        )
        with pytest.raises(IndexError):
            DiagL0(l0).target_ra_dec_offset()

    def test_incomplete_record_raises(self):
        # A record present (flag 1) but missing epoch, the propagation baseline.
        # PM/parallax instead fall back to zero (below).
        l0 = _make_l0_pointing()
        pt = SkyCoord(_PT_RA, _PT_DEC, unit=(u.hourangle, u.deg))
        _set_catalog_record(
            l0,
            {
                "gaia": _record_at(pt),
                "simbad": _record_at(pt),
                "wmko": _record_at(pt, epoch=None),
            },
        )
        with pytest.raises((TypeError, ValueError)):
            DiagL0(l0).target_ra_dec_offset()

    # No usable parallax means no distance on the SkyCoord, and ERFA warns that it
    # overrode the distance while propagating -- the documented consequence of the
    # PM=0/parallax=0 fallback under test, not a fault.
    @pytest.mark.filterwarnings('ignore:ERFA function "pmsafe":erfa.ErfaWarning')
    @pytest.mark.parametrize("bad_plx", [None, 0.0, -5.0])
    def test_missing_or_nonpositive_parallax_falls_back(self, bad_plx, caplog):
        # Gaia DR3 reports parallax <= 0 (or none) for faint sources; the offset falls
        # back to parallax=0 rather than emitting empty, so it stays finite.
        l0 = _make_l0_pointing()
        pt = SkyCoord(_PT_RA, _PT_DEC, unit=(u.hourangle, u.deg))
        _set_catalog_record(
            l0,
            {
                "gaia": _record_at(pt, parallax=bad_plx),
                "simbad": _record_at(pt),
                "wmko": _record_at(pt),
            },
        )
        with caplog.at_level(logging.DEBUG):
            results = DiagL0(l0).run()
        assert results["GAIAOFF"][0] < 0.1  # at the pointing -> ~0, not empty
        assert "using PM=0, parallax=0" in caplog.text
        # The fall-back is offset-local: CATALOG_RECORD keeps the original value.
        tbl = l0.data["CATALOG_RECORD"]
        stored = float(tbl[tbl["source"] == "gaia"]["parallax"][0])
        if bad_plx is None:
            assert np.isnan(stored)
        else:
            assert stored == pytest.approx(bad_plx)

    def test_missing_pm_falls_back(self, caplog):
        # A record with position + epoch but no proper motion still yields a finite
        # offset (PM falls back to zero), not an empty one.
        l0 = _make_l0_pointing()
        pt = SkyCoord(_PT_RA, _PT_DEC, unit=(u.hourangle, u.deg))
        _set_catalog_record(
            l0,
            {
                "gaia": _record_at(pt, pmra=None, pmdec=None),
                "simbad": _record_at(pt),
                "wmko": _record_at(pt),
            },
        )
        with caplog.at_level(logging.DEBUG):
            results = DiagL0(l0).run()
        assert results["GAIAOFF"][0] < 0.1
        assert "using PM=0, parallax=0" in caplog.text

    def test_malformed_astrometry_raises(self):
        # An unparseable RA is a malformed record, not a missing one.
        l0 = _make_l0_pointing()
        pt = SkyCoord(_PT_RA, _PT_DEC, unit=(u.hourangle, u.deg))
        _set_catalog_record(
            l0,
            {
                "gaia": _record_at(pt),
                "simbad": _record_at(pt),
                "wmko": _record_at(pt, ra="garbage"),
            },
        )
        with pytest.raises(ValueError, match="garbage"):
            DiagL0(l0).target_ra_dec_offset()


# ---------------------------------------------------------------------------
# DiagL1 -- master calibration ages
# ---------------------------------------------------------------------------


_CAL_AGE_KEYS = ("BIASAGE", "DARKAGE", "FLATAGE", "WLSAGE")


def _make_kpf1_with_calibrations(date_obs="2024-04-05T11:08:33", files=None):
    """A KPF1 carrying a PRIMARY DATE-OBS, RECEIPT master paths, and assembled CCDs.

    Mirrors the finished-L1 state DiagL1 reads: CalibrationAssociation has written
    each ``{PREFIX}FILE`` to RECEIPT, to_kpf1 has populated the EPRV PRIMARY, and
    ImageAssembly has filled both CCDs (flux_percentiles reads them on every run).
    """
    l1 = KPF1()
    l1.headers["PRIMARY"]["DATE-OBS"] = date_obs
    for chip in ("GREEN", "RED"):
        l1.data[f"{chip}_CCD"] = np.ones((4, 4), dtype=float)
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
        assert set(results) >= set(_CAL_AGE_KEYS)
        for kw in _CAL_AGE_KEYS:
            assert l1.headers["QUALITY_CONTROL"][kw] == pytest.approx(
                -0.422176, abs=1e-5
            )

    def test_missing_cal_type_skipped(self):
        # Only a bias path present -> only BIASAGE written.
        l1 = _make_kpf1_with_calibrations(
            files={"BIASFILE": "/m/KP.20240405.03637.74_master_bias_L1.fits"}
        )
        results = DiagL1(l1).run()
        assert set(results) & set(_CAL_AGE_KEYS) == {"BIASAGE"}
        assert "DARKAGE" not in l1.headers["QUALITY_CONTROL"]

    def test_no_date_obs_raises(self):
        # DATE-OBS is guaranteed by the L1 checkpoint's KWRDPRL1 raise gate; if it
        # is missing anyway, calibration_ages fails loud (a broken upstream
        # invariant).
        l1 = _make_kpf1_with_calibrations(
            files={"BIASFILE": "/m/KP.20240405.03637.74_master_bias_L1.fits"}
        )
        del l1.headers["PRIMARY"]["DATE-OBS"]
        with pytest.raises(KeyError, match="DATE-OBS"):
            DiagL1(l1).calibration_ages()


class TestDiagL1FluxPercentiles:
    # 0..100 makes every percentile land exactly on its own value.
    _RAMP = np.arange(101, dtype=float).reshape(1, 101)

    def _l1(self, **ccds):
        l1 = _make_kpf1_with_calibrations()  # real DATE-OBS, no cal paths
        for ext, arr in ccds.items():
            l1.data[ext] = arr
        return l1

    def test_values_written_to_quality_control(self):
        l1 = self._l1(GREEN_CCD=self._RAMP, RED_CCD=self._RAMP * 2)
        DiagL1(l1).run()
        qc = l1.headers["QUALITY_CONTROL"]
        for pct in (99, 90, 50, 10):
            assert qc[f"GCCD{pct}P"] == pytest.approx(pct)
            assert qc[f"RCCD{pct}P"] == pytest.approx(2 * pct)
        assert (
            qc.comments["GCCD99P"] == "99th percentile flux in the GREEN CCD image [e-]"
        )

    def test_nans_ignored(self):
        ramp = self._RAMP.copy()
        ramp[0, ::10] = np.nan
        l1 = self._l1(GREEN_CCD=ramp)
        results = DiagL1(l1).run()
        assert np.isfinite(results["GCCD50P"][0])

    def test_empty_chip_raises(self):
        l1 = self._l1(GREEN_CCD=self._RAMP, RED_CCD=np.array([], dtype=float))
        with pytest.raises(RuntimeWarning, match="Mean of empty slice"):
            DiagL1(l1).flux_percentiles()


# ---------------------------------------------------------------------------
# DiagL2 -- NaN counts + non-positive counts
# ---------------------------------------------------------------------------


def _make_kpf2_nan_pixels(nan_frac=0.0, zero_frac=0.0, populate=True, var=0.25):
    """Minimal KPF2 with FLUX at controllable NaN and zero fractions.

    Injects real NaN/zero PIXELS, because DiagL2 measures them. Not the same as
    test_qc_flags.py's ``_make_kpf2_nan_headers``, which writes NaN/non-positive
    count HEADERS over clean arrays for QCL2 to read -- the opposite mechanism. Do
    not merge them; each would destroy the other's test.

    A bare KPF2() already exposes the FLUX extensions DiagL2 reads, so no FITS
    round-trip is needed. Each extension is (norder[chip], _NCOL_TEST) ones, then
    nan_frac of the pixels replaced with NaN and zero_frac with 0.0. VAR is filled
    with ``var`` (None leaves it empty) since every DiagL2 method now reads it.
    populate=False sets no arrays at all -- the "no data populated" schema case.
    """
    kpf2 = KPF2()
    if not populate:
        return kpf2

    norder = {"GREEN": NORDER_GREEN, "RED": NORDER_RED}
    rng = np.random.default_rng(42)
    for chip in ("GREEN", "RED"):
        for fiber in _FIBERS:
            n = norder[chip]
            arr = np.ones((n, _NCOL_TEST), dtype=np.float32)
            mask = rng.random(arr.shape)
            if nan_frac > 0:
                arr[mask < nan_frac] = np.nan
            if zero_frac > 0:
                # Zeros fall in the band [nan_frac, nan_frac+zero_frac)
                arr[(mask >= nan_frac) & (mask < nan_frac + zero_frac)] = 0.0
            kpf2.set_data(f"{chip}_{fiber}_FLUX", arr)
    set_wave_bands(kpf2, ncol=_NCOL_TEST)
    if var is not None:
        set_fiber_arrays(kpf2, "VAR", var, ncol=_NCOL_TEST)
    return kpf2


class TestDiagL2NanCounts:
    def test_writes_all_five_keys_with_zero_when_clean(self):
        kpf2 = _make_kpf2_nan_pixels(nan_frac=0.0)
        DiagL2(kpf2).run()
        for key in _NAN_KEYS:
            assert key in kpf2.headers["QUALITY_CONTROL"], f"missing {key}"
            assert kpf2.headers["QUALITY_CONTROL"].get(key) == 0

    def test_counts_injected_nans_per_fiber(self):
        kpf2 = _make_kpf2_nan_pixels(nan_frac=0.0)
        # Inject one NaN into GREEN_SCI1_FLUX; expect NANSCI1==1, others==0.
        kpf2.data["GREEN_SCI1_FLUX"][0, 0] = np.nan
        DiagL2(kpf2).run()
        assert kpf2.headers["QUALITY_CONTROL"].get("NANSCI1") == 1
        for key in ("NANSCI2", "NANSCI3", "NANSKY", "NANCAL"):
            assert kpf2.headers["QUALITY_CONTROL"].get(key) == 0


class TestDiagL2ZeroCounts:
    def test_writes_all_five_keys_with_zero_when_clean(self):
        kpf2 = _make_kpf2_nan_pixels(zero_frac=0.0)  # all ones
        DiagL2(kpf2).run()
        for key in _ZERO_KEYS:
            assert key in kpf2.headers["QUALITY_CONTROL"], f"missing {key}"
            assert kpf2.headers["QUALITY_CONTROL"].get(key) == 0

    def test_counts_injected_non_positive_pixels_per_fiber(self):
        # Negative flux counts alongside exact zeros, and both chips contribute.
        kpf2 = _make_kpf2_nan_pixels(zero_frac=0.0)
        kpf2.data["GREEN_SCI1_FLUX"][0, 0] = 0.0
        kpf2.data["RED_SCI1_FLUX"][0, 0] = -1.0
        DiagL2(kpf2).run()
        assert kpf2.headers["QUALITY_CONTROL"].get("ZEROSCI1") == 2
        for key in ("ZEROSCI2", "ZEROSCI3", "ZEROSKY", "ZEROCAL"):
            assert kpf2.headers["QUALITY_CONTROL"].get(key) == 0


# ---------------------------------------------------------------------------
# DiagL2 -- per-wavelength SNR
# ---------------------------------------------------------------------------


class TestDiagL2Snr:
    def test_keys_written_when_flux_and_var_present(self):
        kpf2 = _make_kpf2_nan_pixels()  # all FLUX ones, all VAR 0.25
        DiagL2(kpf2).run()
        # Per fiber SNR = 1/sqrt(0.25) = 2.0; summed SCI is 3/sqrt(0.75) = 3.464.
        qc = kpf2.headers["QUALITY_CONTROL"]
        for wavelength in (452, 548, 652, 747, 852):
            for code, expected in (("SC", 3.464), ("SK", 2.0), ("CL", 2.0)):
                key = f"SNR{code}{wavelength}"
                assert key in qc, f"missing {key}"
                assert qc.get(key) == pytest.approx(expected, abs=0.01), key

    def test_each_wavelength_reads_its_own_chip(self):
        # GREEN SKY flux=2 (SNR=4) carries 452/548; RED keeps the default SNR=2.
        kpf2 = _make_kpf2_nan_pixels()
        set_fiber_arrays(
            kpf2, "FLUX", 2.0, chips=("GREEN",), fibers=("SKY",), ncol=_NCOL_TEST
        )
        DiagL2(kpf2).run()
        qc = kpf2.headers["QUALITY_CONTROL"]
        assert qc.get("SNRSK452") == pytest.approx(4.0, abs=0.01)
        assert qc.get("SNRSK548") == pytest.approx(4.0, abs=0.01)
        assert qc.get("SNRSK652") == pytest.approx(2.0, abs=0.01)

    def test_summed_sci_uses_all_three_orderlets(self):
        # SCI2 flux=4 against SCI1/SCI3's 1 -> summed flux 6 over sqrt(0.75).
        kpf2 = _make_kpf2_nan_pixels()
        set_fiber_arrays(kpf2, "FLUX", 4.0, fibers=("SCI2",), ncol=_NCOL_TEST)
        DiagL2(kpf2).run()
        assert kpf2.headers["QUALITY_CONTROL"].get("SNRSC652") == pytest.approx(
            6.0 / np.sqrt(0.75), abs=0.01
        )

    def test_raises_without_var(self):
        kpf2 = _make_kpf2_nan_pixels(var=None)
        with pytest.raises(IndexError):
            DiagL2(kpf2).snr()


# ---------------------------------------------------------------------------
# DiagL2 -- orderlet flux ratios
# ---------------------------------------------------------------------------


class TestDiagL2OrderFluxRatios:
    _RATIO_KEYS = ("FR452652", "FR548652", "FR747652", "FR852652")

    def test_keys_written_and_unity_when_uniform(self):
        # SCI2 flux=1 (default) at every wavelength -> every ratio == 1.0.
        kpf2 = _make_kpf2_nan_pixels()
        DiagL2(kpf2).run()
        for key in self._RATIO_KEYS:
            assert key in kpf2.headers["QUALITY_CONTROL"], f"missing {key}"
            assert kpf2.headers["QUALITY_CONTROL"].get(key) == pytest.approx(1.0)

    def test_ratio_value(self):
        # GREEN SCI2 flux=2 against RED's 1: the two green wavelengths double,
        # the two red ones (652 is itself red) stay at unity.
        kpf2 = _make_kpf2_nan_pixels()
        set_fiber_arrays(
            kpf2, "FLUX", 2.0, chips=("GREEN",), fibers=("SCI2",), ncol=_NCOL_TEST
        )
        DiagL2(kpf2).run()
        qc = kpf2.headers["QUALITY_CONTROL"]
        assert qc.get("FR452652") == pytest.approx(2.0)
        assert qc.get("FR548652") == pytest.approx(2.0)
        assert qc.get("FR747652") == pytest.approx(1.0)

    def test_raises_when_wavelength_uncovered(self):
        # The WAVE arrays are the order map; without one no order can be found.
        kpf2 = _make_kpf2_nan_pixels()
        set_fiber_arrays(kpf2, "WAVE", 1.0, ncol=_NCOL_TEST, dtype=np.float64)
        with pytest.raises(LookupError, match="452 nm"):
            DiagL2(kpf2).order_flux_ratios()


class TestDiagL2OrderletFluxRatios:
    def test_keys_written_and_unity_when_uniform(self):
        # All fibers flux=1 (default) -> every median ratio == 1.0 with no scatter.
        kpf2 = _make_kpf2_nan_pixels()
        DiagL2(kpf2).run()
        qc = kpf2.headers["QUALITY_CONTROL"]
        for code in ("FR12", "FR32", "FRS2", "FRC2"):
            for wavelength in (452, 548, 652, 747, 852):
                assert qc.get(f"{code}M{wavelength}") == pytest.approx(1.0)
                assert qc.get(f"{code}U{wavelength}") == pytest.approx(0.0)

    def test_ratio_value(self):
        # SCI1 flux=2 over SCI2 flux=1 on both chips -> FR12M* == 2.0 everywhere.
        kpf2 = _make_kpf2_nan_pixels()
        set_fiber_arrays(kpf2, "FLUX", 2.0, fibers=("SCI1",), ncol=_NCOL_TEST)
        DiagL2(kpf2).run()
        qc = kpf2.headers["QUALITY_CONTROL"]
        for wavelength in (452, 548, 652, 747, 852):
            assert qc.get(f"FR12M{wavelength}") == pytest.approx(2.0)
            assert qc.get(f"FR32M{wavelength}") == pytest.approx(1.0)

    def test_descending_wavelength_grid_interpolates(self):
        # A fiber whose WAVE runs the other way carries the same spectrum, so the
        # ratio must survive the interpolation onto SCI2's grid.
        kpf2 = _make_kpf2_nan_pixels()
        for chip in ("GREEN", "RED"):
            wave = np.asarray(kpf2.data[f"{chip}_SCI1_WAVE"])
            kpf2.set_data(f"{chip}_SCI1_WAVE", wave[:, ::-1].copy())
            flux = np.asarray(kpf2.data[f"{chip}_SCI1_FLUX"])
            kpf2.set_data(f"{chip}_SCI1_FLUX", flux[:, ::-1].copy())
        DiagL2(kpf2).run()
        assert kpf2.headers["QUALITY_CONTROL"].get("FR12M652") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DiagL4 -- per-order BJD / barycentric-RV dispersion (ported from v2.12)
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

    def test_zero_weight_orders_excluded(self):
        # Third order has zero weight: excluded from the weighted mean and the
        # nonzero-weight range, matching v2.12.
        l4 = _l4_with_sci2_rv([10.0, 20.0, 999.0], [0.1, 0.3, 9.0], [1.0, 1.0, 0.0])
        DiagL4(l4).run()
        qc = l4.headers["QUALITY_CONTROL"]
        assert qc["BJDMEAN"] == pytest.approx(15.0)
        assert qc["BJDRNG"] == pytest.approx(864000.0)
        assert qc["BERVMEAN"] == pytest.approx(0.2)

    def test_all_zero_weights_raises(self):
        # No positive total weight: the photon-weighted mean is undefined.
        l4 = _l4_with_sci2_rv([10.0, 20.0], [0.1, 0.3], [0.0, 0.0])
        with pytest.raises(RuntimeWarning, match="invalid value"):
            DiagL4(l4).bjd_dispersion()

    def test_non_finite_samples_dropped(self):
        # A NaN BJD is masked out rather than poisoning the mean; the surviving
        # order is the answer.
        l4 = _l4_with_sci2_rv([10.0, np.nan], [0.1, 0.3], [1.0, 1.0])
        DiagL4(l4).run()
        assert l4.headers["QUALITY_CONTROL"]["BJDMEAN"] == pytest.approx(10.0)

    def test_raises_without_sci2_rv_table(self):
        with pytest.raises(KeyError, match="BJD_TDB"):
            DiagL4(KPF4()).bjd_dispersion()

    def test_raises_without_weight_column(self):
        # WEIGHT column absent (pre-weights L4) -> fail loud rather than guess.
        l4 = KPF4()
        l4.set_data(
            "SCI2_RV",
            Table({"ORDER_INDEX": [0, 1], "BJD_TDB": [10.0, 20.0], "BERV": [0.1, 0.3]}),
        )
        with pytest.raises(KeyError, match="WEIGHT"):
            DiagL4(l4).bjd_dispersion()
