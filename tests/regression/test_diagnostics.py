"""Tests for the Diagnostics framework and per-level subclasses."""

import logging

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
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

from ._data_models import set_fiber_arrays, set_wave_bands, write_amp_l0

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
_NCOL_TEST = 8  # DiagL2 metrics are pixel aggregates, so the real detector width
# is moot

_FIBERS = ("SCI1", "SCI2", "SCI3", "SKY", "CAL")
_NAN_KEYS = ("NANSCI1", "NANSCI2", "NANSCI3", "NANSKY", "NANCAL")
_ZERO_KEYS = ("ZEROSCI1", "ZEROSCI2", "ZEROSCI3", "ZEROSKY", "ZEROCAL")
# Clean exposure-meter flux: 4 readings x 25 wavelength channels per fiber.
_EM_CLEAN_FLUX = np.full((4, 25), 1000.0)


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
    """Write l0's CATALOG_RECORD rows from a {source: record-dict-or-None} mapping,
    the way perform does. A None record leaves the source with no row, exactly as a
    gated-off one does in production."""
    aq = AstroQuery(l0)
    for source, record in records.items():
        aq._write_catalog_record(source, record)


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
        # AstroQuery not run: CATALOG_RECORD auto-created but empty, so it has no
        # 'source' column to look a row up in.
        with pytest.raises(KeyError, match="source"):
            DiagL0(_make_l0_pointing()).gaia_ra_dec_offset()

    def test_unmatched_optional_source_emits_no_key(self):
        # Gaia lookup disabled/failed -> no gaia row -> no GAIAOFF; others compute.
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


class TestDiagL0SolarLunarGeometry:
    """Sun altitude and target-Moon separation at mid-exposure."""

    def _make_l0(self, date_mid, ra=_PT_RA, dec=_PT_DEC):
        l0 = _make_l0_pointing()
        l0.headers["PRIMARY"]["RA"] = ra
        l0.headers["PRIMARY"]["DEC"] = dec
        l0.headers["PRIMARY"]["DATE-MID"] = date_mid
        return l0

    def test_matches_legacy_2d_product(self):
        # KP.20240405.40113.57, whose legacy 2D carries TCSSUN = -61.60211 and
        # TCSMOON = 54.2. TCSSUN differs in the 4th decimal: v2.12 computed sun
        # and moon geometry from a site 647 m from the one its own barycentric
        # correction used, which is the lineage of KECK_LOCATION.
        l0 = self._make_l0(
            "2024-04-05T11:09:11.082", ra="10:59:27.50", dec="+40:25:50.0"
        )
        results = DiagL0(l0).solar_lunar_geometry()
        assert results["TCSSUN"][0] == pytest.approx(-61.60211, abs=1e-3)
        assert results["TCSMOON"][0] == pytest.approx(54.2, abs=0.01)

    def test_sun_above_horizon_at_local_noon(self):
        # Maunakea noon is 22:00 UT; the Sun clears the horizon by a wide margin.
        results = DiagL0(
            self._make_l0("2024-04-05T22:00:00.000")
        ).solar_lunar_geometry()
        assert results["TCSSUN"][0] > 30

    def test_moon_separation_at_the_moon(self):
        # Pointing at the Moon's own 2024-04-05T11:09 position.
        l0 = self._make_l0(
            "2024-04-05T11:09:11.082", ra="12:58:57.79", dec="-06:17:27.7"
        )
        assert DiagL0(l0).solar_lunar_geometry()["TCSMOON"][0] < 1.0

    def test_written_to_quality_control(self):
        l0 = _make_l0_with_catalog()
        l0.headers["PRIMARY"]["DATE-MID"] = "2024-04-05T11:09:11.082"
        results = DiagL0(l0).run()
        for key in ("TCSSUN", "TCSMOON"):
            assert l0.headers["QUALITY_CONTROL"][key] == results[key][0]

    def test_missing_date_mid_raises(self):
        with pytest.raises(KeyError, match="DATE-MID"):
            DiagL0(_make_l0_pointing()).solar_lunar_geometry()

    def test_diag_name_correct(self):
        assert (
            DiagL0.__dict__["solar_lunar_geometry"]._diag_name == "solar_lunar_geometry"
        )


class TestDiagL0PixelFractions:
    """Worst-amp dead/saturated pixel fractions, one pair per chip.

    ``write_amp_l0`` fills every amp with a flat 1e6 D.N., clearing both
    thresholds, so each test drives a chosen pixel count past one bound. Each amp
    here is 10x10 = 100 pixels, so a count is also a percentage.
    """

    def _make_amp_l0(self, tmp_path, namps=4):
        fn = write_amp_l0(
            tmp_path / "KP.20240405.00001.00.fits", namps=namps, shape=(10, 10)
        )
        return KPF0.from_fits(fn)

    def test_clean_frame_is_zero(self, tmp_path):
        l0 = self._make_amp_l0(tmp_path)
        assert DiagL0(l0).dead_pixel_fractions()["DEADPXFG"][0] == 0.0
        assert DiagL0(l0).saturated_pixel_fractions()["SATPXFR"][0] == 0.0

    def test_written_to_quality_control(self, tmp_path):
        l0 = self._make_amp_l0(tmp_path)
        l0.data["GREEN_AMP3"].flat[:6] = 0.0
        results = DiagL0(l0).run()
        for key in ("DEADPXFG", "DEADPXFR", "SATPXFG", "SATPXFR"):
            assert l0.headers["QUALITY_CONTROL"][key] == results[key][0]
        assert results["DEADPXFG"][0] == 0.06

    def test_dead_counts_pixels_below_threshold(self, tmp_path):
        # Strictly below 1.0e4 D.N. counts; a pixel exactly at it does not.
        l0 = self._make_amp_l0(tmp_path)
        l0.data["GREEN_AMP3"].flat[:50] = 1.0e4
        l0.data["GREEN_AMP3"].flat[:5] = 0.0
        assert DiagL0(l0).dead_pixel_fractions()["DEADPXFG"][0] == 0.05

    def test_saturated_counts_pixels_above_threshold(self, tmp_path):
        # Strictly above 5.0e8 D.N. counts; a pixel exactly at it does not.
        l0 = self._make_amp_l0(tmp_path)
        l0.data["RED_AMP2"].flat[:50] = 5.0e8
        l0.data["RED_AMP2"].flat[:15] = 6.0e8
        assert DiagL0(l0).saturated_pixel_fractions()["SATPXFR"][0] == 0.15

    def test_chips_measured_separately(self, tmp_path):
        # A dead GREEN amp leaves the RED fraction at zero, and vice versa.
        l0 = self._make_amp_l0(tmp_path)
        l0.data["GREEN_AMP1"].flat[:50] = 0.0
        results = DiagL0(l0).dead_pixel_fractions()
        assert results["DEADPXFG"][0] == 0.5
        assert results["DEADPXFR"][0] == 0.0

    def test_worst_amp_decides(self, tmp_path):
        # One bad amp sets its chip's fraction even though the other three are clean.
        l0 = self._make_amp_l0(tmp_path)
        l0.data["RED_AMP4"].flat[:50] = 0.0
        assert DiagL0(l0).dead_pixel_fractions()["DEADPXFR"][0] == 0.5

    def test_two_amp_readout(self, tmp_path):
        # Absent amps are skipped, so a 2-amp frame is measured on the amps it has.
        l0 = self._make_amp_l0(tmp_path, namps=2)
        assert DiagL0(l0).dead_pixel_fractions()["DEADPXFG"][0] == 0.0

    def test_no_amp_data_raises(self, tmp_path):
        l0 = self._make_amp_l0(tmp_path, namps=0)
        with pytest.raises(ValueError):
            DiagL0(l0).dead_pixel_fractions()

    def test_diag_names_correct(self):
        assert DiagL0.__dict__["dead_pixel_fractions"]._diag_name == (
            "dead_pixel_fractions"
        )
        assert DiagL0.__dict__["saturated_pixel_fractions"]._diag_name == (
            "saturated_pixel_fractions"
        )


class TestDiagL0AmpPercentiles:
    """Per-amplifier raw D.N. percentiles: 2 chips x 3 percentiles x 4 amps."""

    def _make_amp_l0(self, tmp_path, namps=4):
        fn = write_amp_l0(
            tmp_path / "KP.20240405.00001.00.fits", namps=namps, shape=(10, 10)
        )
        return KPF0.from_fits(fn)

    def test_all_24_keywords_emitted(self, tmp_path):
        results = DiagL0(self._make_amp_l0(tmp_path)).amp_percentiles()
        expected = {
            f"P{pct}{letter}AMP{i}"
            for letter in ("G", "R")
            for pct in (16, 50, 84)
            for i in range(1, 5)
        }
        assert set(results) == expected

    def test_percentiles_match_the_amp_image(self, tmp_path):
        # A 0..99 ramp over the amp: the percentiles are of the raw image as
        # stored, prescan and overscan included.
        l0 = self._make_amp_l0(tmp_path)
        l0.data["GREEN_AMP2"].flat[:] = np.arange(100.0)
        results = DiagL0(l0).amp_percentiles()
        for pct in (16, 50, 84):
            assert results[f"P{pct}GAMP2"][0] == pytest.approx(
                np.percentile(np.arange(100.0), pct)
            )

    def test_nans_excluded(self, tmp_path):
        # NaN pixels are dropped rather than poisoning the whole amp.
        l0 = self._make_amp_l0(tmp_path)
        l0.data["RED_AMP1"].flat[:10] = np.nan
        assert DiagL0(l0).amp_percentiles()["P50RAMP1"][0] == 1.0e6

    def test_written_to_quality_control(self, tmp_path):
        l0 = self._make_amp_l0(tmp_path)
        results = DiagL0(l0).run()
        assert l0.headers["QUALITY_CONTROL"]["P84GAMP3"] == results["P84GAMP3"][0]

    def test_absent_amps_emit_no_keyword(self, tmp_path):
        # A 2-amp readout writes only the amps it has.
        results = DiagL0(self._make_amp_l0(tmp_path, namps=2)).amp_percentiles()
        assert "P50GAMP1" in results
        assert "P50GAMP3" not in results

    def test_diag_name_correct(self):
        assert DiagL0.__dict__["amp_percentiles"]._diag_name == "amp_percentiles"


class TestDiagL0ExpmeterChannels:
    """Per-fiber EM channel metrics: saturation rate and negative/non-finite runs.

    Each fiber here is 4 readings x 25 wavelength channels, so the two interior
    readings carry the saturation count.
    """

    def _make_l0_with_expmeter(self, tmp_path, sci, sky=None):
        def table(values):
            columns = {"Date-Beg": ["2024-09-23T09:12:09.484"] * len(values)}
            columns.update(
                {str(5000.0 + i): values[:, i] for i in range(values.shape[1])}
            )
            return Table(columns)

        fn = write_amp_l0(
            tmp_path / "KP.20240405.00005.00.fits",
            shape=(10, 10),
            extra_hdus=[
                fits.BinTableHDU(table(sci), name="EXPMETER_SCI"),
                fits.BinTableHDU(
                    table(sci if sky is None else sky), name="EXPMETER_SKY"
                ),
            ],
        )
        return KPF0.from_fits(fn)

    def test_clean_flux_is_zero(self, tmp_path):
        results = DiagL0(self._make_l0_with_expmeter(tmp_path, _EM_CLEAN_FLUX)).run()
        for fiber in ("SCI", "SKY"):
            for metric in ("SAT", "NEG", "INF"):
                assert results[f"EM{fiber}{metric}"][0] == 0

    def test_saturation_is_per_reading(self, tmp_path):
        # 3 saturated elements over the 2 interior readings -> 1.5 per reading.
        flux = _EM_CLEAN_FLUX.copy()
        flux[1, :2] = 0.95 * 1.93e6
        flux[2, 0] = 0.95 * 1.93e6
        results = DiagL0(
            self._make_l0_with_expmeter(tmp_path, flux)
        ).expmeter_channel_metrics()
        assert results["EMSCISAT"][0] == 1.5

    def test_saturation_drops_edge_readings(self, tmp_path):
        # The first and last readings are partial, so saturation there is dropped.
        flux = _EM_CLEAN_FLUX.copy()
        flux[[0, -1], :] = 0.95 * 1.93e6
        results = DiagL0(
            self._make_l0_with_expmeter(tmp_path, flux)
        ).expmeter_channel_metrics()
        assert results["EMSCISAT"][0] == 0.0

    def test_negative_run_length(self, tmp_path):
        # A channel counts as negative when its time-summed flux is negative.
        flux = _EM_CLEAN_FLUX.copy()
        flux[:, 5:25] = -1000.0
        results = DiagL0(
            self._make_l0_with_expmeter(tmp_path, flux)
        ).expmeter_channel_metrics()
        assert results["EMSCINEG"][0] == 20

    def test_negative_run_counts_adjacent_only(self, tmp_path):
        # Two separated blocks of 3 and 5: the longest run is 5, not their sum.
        flux = _EM_CLEAN_FLUX.copy()
        flux[:, 2:5] = -1000.0
        flux[:, 10:15] = -1000.0
        results = DiagL0(
            self._make_l0_with_expmeter(tmp_path, flux)
        ).expmeter_channel_metrics()
        assert results["EMSCINEG"][0] == 5

    def test_negative_needs_the_time_sum(self, tmp_path):
        # One negative reading a channel does not make: the sum stays positive.
        flux = _EM_CLEAN_FLUX.copy()
        flux[0, 5:25] = -1000.0
        results = DiagL0(
            self._make_l0_with_expmeter(tmp_path, flux)
        ).expmeter_channel_metrics()
        assert results["EMSCINEG"][0] == 0

    def test_non_finite_run_length(self, tmp_path):
        # A channel with any non-finite reading counts, NaN or inf.
        flux = _EM_CLEAN_FLUX.copy()
        flux[0, 5:9] = np.nan
        flux[2, 9] = np.inf
        results = DiagL0(
            self._make_l0_with_expmeter(tmp_path, flux)
        ).expmeter_channel_metrics()
        assert results["EMSCIINF"][0] == 5

    def test_fibers_measured_separately(self, tmp_path):
        # A clean SCI and a negative SKY: only the SKY keyword moves.
        sky = _EM_CLEAN_FLUX.copy()
        sky[:, 5:25] = -1000.0
        results = DiagL0(
            self._make_l0_with_expmeter(tmp_path, _EM_CLEAN_FLUX, sky=sky)
        ).expmeter_channel_metrics()
        assert results["EMSCINEG"][0] == 0
        assert results["EMSKYNEG"][0] == 20

    def test_written_to_quality_control(self, tmp_path):
        l0 = self._make_l0_with_expmeter(tmp_path, _EM_CLEAN_FLUX)
        results = DiagL0(l0).run()
        assert l0.headers["QUALITY_CONTROL"]["EMSCISAT"] == results["EMSCISAT"][0]

    def test_no_em_data_emits_no_keyword(self, tmp_path):
        # A frame with no EM extension (e.g. a calibration): the metrics are skipped.
        fn = write_amp_l0(tmp_path / "KP.20240405.00006.00.fits", shape=(10, 10))
        assert "EMSCISAT" not in DiagL0(KPF0.from_fits(fn)).run()

    def test_diag_name_correct(self):
        assert DiagL0.__dict__["expmeter_channel_metrics"]._diag_name == (
            "expmeter_channel_metrics"
        )


class TestDiagL0ExpmeterCounts:
    """Cumulative EM counts per wavelength band, and the SKY/SCI ratio.

    One channel per band plus a 900 nm channel outside the 445-870 nm range, so
    an out-of-band channel is seen to be dropped everywhere. Column labels are nm:
    ImageAssembly converts them to Angstroms only at L0 -> L1.
    """

    _WAVES = [500.0, 600.0, 700.0, 800.0, 900.0]

    def _make_l0_with_expmeter(self, tmp_path, sci_counts, sky_counts):
        def table(counts):
            columns = {
                "Date-Beg": ["2024-09-23T09:12:09.484", "2024-09-23T09:12:12.484"]
            }
            columns.update(
                {
                    str(w): [float(c), float(c)]
                    for w, c in zip(self._WAVES, counts, strict=True)
                }
            )
            return Table(columns)

        fn = write_amp_l0(
            tmp_path / "KP.20240405.00008.00.fits",
            shape=(10, 10),
            extra_hdus=[
                fits.BinTableHDU(table(sci_counts), name="EXPMETER_SCI"),
                fits.BinTableHDU(table(sky_counts), name="EXPMETER_SKY"),
            ],
        )
        return KPF0.from_fits(fn)

    def test_counts_summed_over_readings_and_bands(self, tmp_path):
        # Two readings of each channel, so every band doubles its per-reading count.
        l0 = self._make_l0_with_expmeter(tmp_path, [1, 2, 4, 8, 16], [1, 1, 1, 1, 1])
        results = DiagL0(l0).expmeter_counts()
        assert results["EMSCCT45"][0] == 2  # 500 nm
        assert results["EMSCCT56"][0] == 4  # 600 nm
        assert results["EMSCCT67"][0] == 8  # 700 nm
        assert results["EMSCCT78"][0] == 16  # 800 nm; the 900 nm channel is dropped
        assert results["EMSCCT48"][0] == 30  # 445-870 nm: the four sub-bands

    def test_sky_fiber_counted_separately(self, tmp_path):
        l0 = self._make_l0_with_expmeter(tmp_path, [1, 1, 1, 1, 1], [3, 0, 0, 0, 0])
        results = DiagL0(l0).expmeter_counts()
        assert results["EMSKCT45"][0] == 6
        assert results["EMSCCT45"][0] == 2

    def test_nans_excluded(self, tmp_path):
        l0 = self._make_l0_with_expmeter(tmp_path, [1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
        l0.data["EXPMETER_SCI"]["500.0"][0] = np.nan
        results = DiagL0(l0).expmeter_counts()
        assert results["EMSCCT45"][0] == 1

    def test_sky_sci_ratio_scaled_by_throughput(self, tmp_path):
        # Equal SKY and SCI totals -> the ratio is 1/14.1, the twilight throughput.
        l0 = self._make_l0_with_expmeter(tmp_path, [1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
        results = DiagL0(l0).sky_sci_flux_ratio()
        assert results["SKYSCIMS"][0] == round(1 / 14.1, 6)

    def test_sky_sci_ratio_uses_every_channel(self, tmp_path):
        # The ratio is over all channels, the out-of-band 900 nm one included.
        l0 = self._make_l0_with_expmeter(tmp_path, [1, 1, 1, 1, 1], [2, 2, 2, 2, 2])
        results = DiagL0(l0).sky_sci_flux_ratio()
        assert results["SKYSCIMS"][0] == round(2 / 14.1, 6)

    def test_written_to_quality_control(self, tmp_path):
        l0 = self._make_l0_with_expmeter(tmp_path, [1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
        results = DiagL0(l0).run()
        for key in ("SKYSCIMS", "EMSCCT48", "EMSKCT78"):
            assert l0.headers["QUALITY_CONTROL"][key] == results[key][0]

    def test_diag_names_correct(self):
        assert DiagL0.__dict__["expmeter_counts"]._diag_name == "expmeter_counts"
        assert DiagL0.__dict__["sky_sci_flux_ratio"]._diag_name == "sky_sci_flux_ratio"


class TestDiagL0CcdTemperatures:
    """GREEN/RED CCD offsets from the -100 C setpoint, read off TELEMETRY."""

    def _make_l0_with_telemetry(self, tmp_path, green, red):
        table = Table(
            {
                "keyword": ["kpfgreen.STA_CCD_T", "kpfred.STA_CCD_T"],
                "average": [green, red],
            }
        )
        fn = write_amp_l0(
            tmp_path / "KP.20240405.00008.00.fits",
            shape=(10, 10),
            extra_hdus=[fits.BinTableHDU(table, name="TELEMETRY")],
        )
        return KPF0.from_fits(fn)

    def test_offset_is_signed_millikelvin(self, tmp_path):
        l0 = self._make_l0_with_telemetry(tmp_path, -100.004, -99.993)
        results = DiagL0(l0).ccd_temperature_offsets()
        assert results["GTEMPOFF"][0] == pytest.approx(-4.0, abs=1e-3)
        assert results["RTEMPOFF"][0] == pytest.approx(7.0, abs=1e-3)

    def test_at_setpoint_is_zero(self, tmp_path):
        l0 = self._make_l0_with_telemetry(tmp_path, -100.0, -100.0)
        results = DiagL0(l0).ccd_temperature_offsets()
        assert results["GTEMPOFF"][0] == 0.0
        assert results["RTEMPOFF"][0] == 0.0

    def test_written_to_quality_control(self, tmp_path):
        l0 = self._make_l0_with_telemetry(tmp_path, -100.004, -99.993)
        results = DiagL0(l0).run()
        for key in ("GTEMPOFF", "RTEMPOFF"):
            assert l0.headers["QUALITY_CONTROL"][key] == results[key][0]

    def test_diag_name_correct(self):
        name = DiagL0.__dict__["ccd_temperature_offsets"]._diag_name
        assert name == "ccd_temperature_offsets"


class TestDiagL0EtalonTemperature:
    """Etalon chamber offset from setpoint, off the PRIMARY temperature cards."""

    def _make_l0_with_etalon(self, **cards):
        l0 = KPF0()
        l0.headers["PRIMARY"].update({"ETAV1C3T": 23.6, "ETAV1C4T": 23.9})
        l0.headers["PRIMARY"].update(cards)
        return l0

    def test_at_design_setpoints_is_zero(self):
        # No ETAV1C3S/ETAV1C4S recorded, so the design values apply.
        l0 = self._make_l0_with_etalon()
        assert DiagL0(l0).etalon_temperature_offset()["ETATOFF"][0] == 0.0

    def test_offset_is_signed_millikelvin(self):
        l0 = self._make_l0_with_etalon(ETAV1C3T=23.6004)
        results = DiagL0(l0).etalon_temperature_offset()
        assert results["ETATOFF"][0] == pytest.approx(0.4, abs=1e-3)

    def test_recorded_setpoint_wins_over_design(self):
        l0 = self._make_l0_with_etalon(ETAV1C3T=24.0, ETAV1C3S=24.0)
        assert DiagL0(l0).etalon_temperature_offset()["ETATOFF"][0] == 0.0

    def test_worst_chamber_reported(self):
        # The outer chamber is further off, so it is the one that survives.
        l0 = self._make_l0_with_etalon(ETAV1C3T=23.6002, ETAV1C4T=23.8993)
        results = DiagL0(l0).etalon_temperature_offset()
        assert results["ETATOFF"][0] == pytest.approx(-0.7, abs=1e-3)

    def test_written_to_quality_control(self):
        l0 = self._make_l0_with_etalon(ETAV1C3T=23.6004)
        results = DiagL0(l0).run()
        assert l0.headers["QUALITY_CONTROL"]["ETATOFF"] == results["ETATOFF"][0]

    def test_diag_name_correct(self):
        name = DiagL0.__dict__["etalon_temperature_offset"]._diag_name
        assert name == "etalon_temperature_offset"


class TestDiagL0Guider:
    """Guiding error statistics and guide-camera saturation.

    Twelve frames, so the centroids clear the 11-distinct-position floor below
    which v2.12 declares the guide camera untracked.
    """

    _NFRAMES = 12

    def _make_l0_with_guider(
        self,
        tmp_path,
        *,
        dx=0.0,
        dy=0.0,
        peak=0.0,
        avg_level=0.0,
        nbright=0,
        rows=None,
        axes=(0.0, 0.0),
        flux=100.0,
    ):
        offsets = np.arange(self._NFRAMES, dtype=float)
        columns = {
            "timestamp": offsets + 1.0,
            "target_x": offsets + dx,
            "target_y": offsets * 2.0 + dy,
            "object1_x": offsets,
            "object1_y": offsets * 2.0,
            "object1_flux": np.full(self._NFRAMES, flux),
            "object1_peak": np.full(self._NFRAMES, peak),
            "object1_a": np.full(self._NFRAMES, axes[0]),
            "object1_b": np.full(self._NFRAMES, axes[1]),
        }
        if rows is not None:
            columns = {k: v[:rows] for k, v in columns.items()}
        avg = np.full((512, 640), avg_level)
        avg[255, 270 : 270 + nbright] = 1e5
        fn = write_amp_l0(
            tmp_path / "KP.20240405.00008.00.fits",
            shape=(10, 10),
            extra_hdus=[
                fits.ImageHDU(avg, name="GUIDER_AVG"),
                fits.BinTableHDU(Table(columns), name="GUIDER_CUBE_ORIGINS"),
            ],
        )
        return KPF0.from_fits(fn)

    def test_constant_offset_gives_rms_and_bias(self, tmp_path):
        # A 0.5 pixel offset in x on every frame: 0.056"/pix -> 28 mas, and with
        # no scatter the RMS equals the bias.
        l0 = self._make_l0_with_guider(tmp_path, dx=0.5)
        results = DiagL0(l0).guider_errors()
        assert results["GDRXRMS"][0] == pytest.approx(28.0)
        assert results["GDRXBIAS"][0] == pytest.approx(28.0)
        assert results["GDRYRMS"][0] == 0.0
        assert results["GDRYBIAS"][0] == 0.0

    def test_bias_keeps_its_sign(self, tmp_path):
        l0 = self._make_l0_with_guider(tmp_path, dy=-0.5)
        results = DiagL0(l0).guider_errors()
        assert results["GDRYBIAS"][0] == pytest.approx(-28.0)
        assert results["GDRYRMS"][0] == pytest.approx(28.0)

    def test_untracked_camera_emits_no_keyword(self, tmp_path):
        l0 = self._make_l0_with_guider(tmp_path, dx=0.5, rows=8)
        assert DiagL0(l0).guider_errors() == {}

    def test_unwritten_rows_dropped(self, tmp_path):
        l0 = self._make_l0_with_guider(tmp_path, peak=1.0)
        l0.data["GUIDER_CUBE_ORIGINS"]["object1_flux"][:2] = 0.0
        l0.data["GUIDER_CUBE_ORIGINS"]["object1_peak"][:2] = 1e5
        # The two zero-flux frames are unwritten, so their peaks do not count.
        assert DiagL0(l0).guider_saturation()["GDRFRSAT"][0] == 0.0

    def test_saturated_pixels_counted_in_central_box(self, tmp_path):
        l0 = self._make_l0_with_guider(tmp_path, nbright=4)
        assert DiagL0(l0).guider_saturation()["GDRNSAT"][0] == 4

    def test_pixels_outside_the_box_ignored(self, tmp_path):
        l0 = self._make_l0_with_guider(tmp_path)
        l0.data["GUIDER_AVG"][0, 0] = 1e5
        assert DiagL0(l0).guider_saturation()["GDRNSAT"][0] == 0

    def test_saturated_frame_fraction(self, tmp_path):
        l0 = self._make_l0_with_guider(tmp_path)
        l0.data["GUIDER_CUBE_ORIGINS"]["object1_peak"][:3] = 1e5
        assert DiagL0(l0).guider_saturation()["GDRFRSAT"][0] == 0.25

    def test_below_saturation_threshold_not_counted(self, tmp_path):
        # 90% of the 15830 ADU CRED-2 saturation level is the threshold.
        l0 = self._make_l0_with_guider(tmp_path, peak=14000.0, avg_level=14000.0)
        results = DiagL0(l0).guider_saturation()
        assert results["GDRFRSAT"][0] == 0.0
        assert results["GDRNSAT"][0] == 0

    def test_radial_rms_combines_both_axes(self, tmp_path):
        # A 0.5 pixel offset on each axis: R is their quadrature sum.
        l0 = self._make_l0_with_guider(tmp_path, dx=0.5, dy=0.5)
        results = DiagL0(l0).guider_errors()
        assert results["GDRRRMS"][0] == pytest.approx(28.0 * 2**0.5)

    def test_fwhm_from_the_fitted_gaussian_axes(self, tmp_path):
        # sigma=(3,4) pixels -> 5 px in quadrature, x2.3548 to FWHM, x56 mas/pix.
        l0 = self._make_l0_with_guider(tmp_path, axes=(3.0, 4.0))
        results = DiagL0(l0).guider_image_stats()
        assert results["GDRFWMD"][0] == pytest.approx(5 * 2.3548 * 56.0, rel=1e-4)
        assert results["GDRFWSTD"][0] == 0.0

    def test_flux_and_peak_statistics(self, tmp_path):
        l0 = self._make_l0_with_guider(tmp_path, flux=200.0, peak=50.0)
        l0.data["GUIDER_CUBE_ORIGINS"]["object1_flux"][0] = 100.0
        results = DiagL0(l0).guider_image_stats()
        assert results["GDRFXMD"][0] == 200.0
        assert results["GDRFXSTD"][0] > 0.0
        assert results["GDRPKMD"][0] == 50.0
        assert results["GDRPKSTD"][0] == 0.0

    def test_written_to_quality_control(self, tmp_path):
        l0 = self._make_l0_with_guider(tmp_path, dx=0.5, nbright=2, axes=(3.0, 4.0))
        results = DiagL0(l0).run()
        for key in ("GDRXRMS", "GDRRRMS", "GDRYBIAS", "GDRNSAT", "GDRFRSAT", "GDRFWMD"):
            assert l0.headers["QUALITY_CONTROL"][key] == results[key][0]

    def test_diag_names_correct(self):
        for name in ("guider_errors", "guider_saturation", "guider_image_stats"):
            assert DiagL0.__dict__[name]._diag_name == name


class TestDiagL0GuiderSeeing:
    """J+Z-band and V-band seeing from the Moffat fit to the co-added guider image."""

    def _make_l0_with_moffat(self, tmp_path, alpha, *, corrupt=False):
        y, x = np.indices((81, 81))
        image = 100.0 * (1 + ((x - 40.0) ** 2 + (y - 40.0) ** 2) / alpha**2) ** -2.5
        if corrupt:
            image = np.full_like(image, np.nan)
        fn = write_amp_l0(
            tmp_path / "KP.20240405.00008.00.fits",
            shape=(10, 10),
            primary_cards={"GCCRPIX1": 40.0, "GCCRPIX2": 40.0},
            extra_hdus=[fits.ImageHDU(image, name="GUIDER_AVG")],
        )
        return KPF0.from_fits(fn)

    def test_seeing_is_alpha_in_arcsec(self, tmp_path):
        # alpha = 8 px at the 0.056"/pix CRED-2 scale -> 0.448" seeing.
        l0 = self._make_l0_with_moffat(tmp_path, 8.0)
        results = DiagL0(l0).guider_seeing()
        assert results["GDRSEEJZ"][0] == pytest.approx(8.0 * 0.056, rel=0.05)

    def test_v_band_seeing_is_the_scaled_jz_seeing(self, tmp_path):
        # Kolmogorov lambda^(1/5) from the 950-1200 nm band midpoint to 550 nm.
        results = DiagL0(self._make_l0_with_moffat(tmp_path, 8.0)).guider_seeing()
        assert results["GDRSEEV"][0] == pytest.approx(
            results["GDRSEEJZ"][0] * 1.1434288742094985, rel=1e-5
        )

    def test_wider_profile_gives_larger_seeing(self, tmp_path):
        narrow = DiagL0(self._make_l0_with_moffat(tmp_path, 5.0)).guider_seeing()
        wide = DiagL0(self._make_l0_with_moffat(tmp_path, 15.0)).guider_seeing()
        assert wide["GDRSEEJZ"][0] > narrow["GDRSEEJZ"][0]

    def test_unfittable_image_emits_no_keyword(self, tmp_path):
        l0 = self._make_l0_with_moffat(tmp_path, 8.0, corrupt=True)
        assert DiagL0(l0).guider_seeing() == {}

    def test_written_to_quality_control(self, tmp_path):
        l0 = self._make_l0_with_moffat(tmp_path, 8.0)
        results = DiagL0(l0).run()
        for key in ("GDRSEEJZ", "GDRSEEV"):
            assert l0.headers["QUALITY_CONTROL"][key] == results[key][0]

    def test_diag_name_correct(self):
        assert DiagL0.__dict__["guider_seeing"]._diag_name == "guider_seeing"


def _make_kpf1(date_obs="2024-04-05T11:08:33"):
    """A KPF1 carrying a PRIMARY DATE-OBS and both assembled CCDs.

    Mirrors the finished-L1 state DiagL1 reads: to_kpf1 has populated the EPRV
    PRIMARY and ImageAssembly has filled both CCDs.
    """
    l1 = KPF1()
    l1.headers["PRIMARY"]["DATE-OBS"] = date_obs
    for chip in ("GREEN", "RED"):
        l1.data[f"{chip}_CCD"] = np.ones((4, 4), dtype=float)
    return l1


class TestDiagL1FluxPercentiles:
    # 0..100 makes every percentile land exactly on its own value.
    _RAMP = np.arange(101, dtype=float).reshape(1, 101)

    def _l1(self, **ccds):
        l1 = _make_kpf1()
        for ext, arr in ccds.items():
            l1.data[ext] = arr
        return l1

    def test_values_written_to_quality_control(self):
        l1 = self._l1(GREEN_CCD=self._RAMP, RED_CCD=self._RAMP * 2)
        DiagL1(l1).run()
        qc = l1.headers["QUALITY_CONTROL"]
        for pct in (99, 90, 50, 10):
            assert qc[f"FFIG{pct}P"] == pytest.approx(pct)
            assert qc[f"FFIR{pct}P"] == pytest.approx(2 * pct)
        assert (
            qc.comments["FFIG99P"] == "99th percentile flux in the GREEN CCD image [e-]"
        )

    def test_nans_ignored(self):
        ramp = self._RAMP.copy()
        ramp[0, ::10] = np.nan
        l1 = self._l1(GREEN_CCD=ramp)
        results = DiagL1(l1).run()
        assert np.isfinite(results["FFIG50P"][0])

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

    def test_summed_sci_mirrored_to_primary(self):
        kpf2 = _make_kpf2_nan_pixels()
        DiagL2(kpf2).run()
        primary, qc = kpf2.headers["PRIMARY"], kpf2.headers["QUALITY_CONTROL"]
        for index, wavelength in enumerate((452, 548, 652, 747, 852), start=1):
            assert primary.get(f"EXSNR{index}") == qc.get(f"SNRSC{wavelength}")
            assert primary.get(f"EXSNRW{index}") == wavelength * 10.0

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
