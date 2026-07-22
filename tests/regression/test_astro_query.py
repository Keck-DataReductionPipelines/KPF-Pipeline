"""Regression tests for the AstroQuery module.

Focus: ``merge_catalog_records`` -- the merge of the gaia/simbad/wmko CATALOG_RECORD
rows into the canonical ``kpf-drp`` row -- and ``read_wmko_header``. The external
Gaia/SIMBAD queries are not exercised here (no network); rows are written directly via
``AstroQuery._write_catalog_record`` and the merge is driven off them.
"""

import logging

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Angle

from kpfpipe.data_models import KPF0
from kpfpipe.modules.astro_query import AstroQuery


def _ra_str(deg):
    """RA in deg -> the EPRV C*# sexagesimal hour-angle string the schema stores."""
    return Angle(deg, u.deg).to_string(unit=u.hourangle, sep=":", pad=True, precision=4)


def _dec_str(deg):
    """Dec in deg -> the EPRV C*# sexagesimal string the schema stores."""
    return Angle(deg, u.deg).to_string(
        unit=u.deg, sep=":", pad=True, alwayssign=True, precision=3
    )


def _record(obj, ra=180.0, **overrides):
    """A complete canonical record at ``ra`` (zero PM, finite plx/rv), in the EPRV
    C*# format: RA/Dec sexagesimal strings, PM arcsec/yr. Pass a field as None (via
    overrides) to mark it missing."""
    rec = {
        "object": obj,
        "ra": _ra_str(ra),
        "dec": _dec_str(40.0),
        "pmra": 1.0,
        "pmdec": 2.0,
        "parallax": 50.0,
        "rv": 10.0,
        "frame": "icrs",
        "epoch": 2016.0,
        "equinox": 2000.0,
    }
    rec.update(overrides)
    return rec


def _make_l0(records, imtype="object"):
    """A science KPF0 (IMTYPE) with the given {source: record} rows written directly
    via AstroQuery's writer (no network)."""
    l0 = KPF0()
    l0.headers["PRIMARY"]["IMTYPE"] = imtype
    aq = AstroQuery(l0)
    for source, record in records.items():
        aq._write_catalog_record(source, record)
    return l0


def _merged_row(l0):
    table = l0.data["CATALOG_RECORD"]
    return table[table["source"] == "kpf-drp"][0]


class TestMergeCatalogRecords:
    def test_all_gaia_complete(self):
        l0 = _make_l0({"gaia": _record("G123")})
        AstroQuery(l0).merge_catalog_records()

        row = _merged_row(l0)
        assert row["radec_src"] == "gaia"
        assert row["object"] == "G123"
        assert row["ra"] == _ra_str(180.0)
        assert row["parallax"] == pytest.approx(50.0)
        assert row["rv"] == pytest.approx(10.0)
        # Four rows now: gaia + the merged kpf-drp, and no CATCR presence flag.
        assert len(l0.data["CATALOG_RECORD"]) == 2
        assert "CATCR" not in l0.headers["CATALOG_RECORD"]

    def test_position_follows_priority(self):
        # All three complete but at different RAs -> position from Gaia (highest).
        l0 = _make_l0(
            {
                "gaia": _record("G", ra=10.0),
                "simbad": _record("S", ra=20.0),
                "wmko": _record("W", ra=30.0),
            }
        )
        AstroQuery(l0).merge_catalog_records()

        row = _merged_row(l0)
        assert row["radec_src"] == "gaia"
        assert row["ra"] == _ra_str(10.0)

    def test_rv_filled_from_lower_priority_logs_mix(self, caplog):
        # Gaia supplies position + parallax but lacks rv -> rv borrowed from SIMBAD.
        l0 = _make_l0(
            {
                "gaia": _record("G", rv=None),
                "simbad": _record("S", rv=99.0),
                "wmko": _record("W"),
            }
        )
        with caplog.at_level(logging.DEBUG, logger="kpfpipe.modules.astro_query"):
            AstroQuery(l0).merge_catalog_records()

        row = _merged_row(l0)
        assert row["radec_src"] == "gaia"  # position still from Gaia
        assert row["rv"] == pytest.approx(99.0)  # rv from SIMBAD
        assert row["rv_src"] == "simbad"
        assert row["parallax"] == pytest.approx(50.0)  # parallax stayed with Gaia
        assert row["plx_src"] == "gaia"
        assert "mixes sources" in caplog.text

    def test_parallax_priority_over_wmko(self):
        # Gaia lacks parallax; SIMBAD (higher priority than WMKO) supplies it.
        l0 = _make_l0(
            {
                "gaia": _record("G", parallax=None),
                "simbad": _record("S", parallax=7.0),
                "wmko": _record("W", parallax=8.0),
            }
        )
        AstroQuery(l0).merge_catalog_records()
        row = _merged_row(l0)
        assert row["parallax"] == pytest.approx(7.0)
        assert row["plx_src"] == "simbad"

    def test_wmko_only(self, caplog):
        l0 = _make_l0({"wmko": _record("W")})
        with caplog.at_level(logging.DEBUG, logger="kpfpipe.modules.astro_query"):
            AstroQuery(l0).merge_catalog_records()

        row = _merged_row(l0)
        assert row["radec_src"] == "wmko"
        assert row["object"] == "W"
        assert "mixes sources" not in caplog.text  # single source, no mixing

    def test_missing_position_raises(self):
        # Only WMKO has a position, but it is excluded from the merge -> no position.
        l0 = _make_l0({"wmko": _record("W")})
        aq = AstroQuery(l0, {"use_wmko": False})
        with pytest.raises(ValueError, match="canonical astrometry position"):
            aq.merge_catalog_records()

    def test_incomplete_position_raises(self):
        # The only candidate lacks a coherent position block (pmra missing).
        l0 = _make_l0({"gaia": _record("G", pmra=None)})
        with pytest.raises(ValueError, match="canonical astrometry position"):
            AstroQuery(l0).merge_catalog_records()

    def test_optional_rv_missing_everywhere_left_nan(self):
        # Position present but no source supplies rv -> merged row builds, rv is NaN.
        l0 = _make_l0(
            {
                "gaia": _record("G", rv=None),
                "simbad": _record("S", rv=None),
            }
        )
        AstroQuery(l0).merge_catalog_records()

        row = _merged_row(l0)
        assert row["radec_src"] == "gaia"
        assert np.isnan(row["rv"])
        assert row["rv_src"] == ""  # nothing supplied rv -> empty provenance
        assert row["parallax"] == pytest.approx(50.0)  # parallax still filled

    def test_use_wmko_gates_merge_only(self):
        # Gaia complete; WMKO present but excluded from the merge. Merge still
        # succeeds off Gaia, and the wmko row itself is untouched.
        l0 = _make_l0({"gaia": _record("G"), "wmko": _record("W")})
        AstroQuery(l0, {"use_wmko": False}).merge_catalog_records()

        assert _merged_row(l0)["radec_src"] == "gaia"
        table = l0.data["CATALOG_RECORD"]
        assert len(table[table["source"] == "wmko"]) == 1  # wmko row preserved
        assert l0.headers["CATALOG_RECORD"]["WMKOCR"] == 1


class TestSingleSourceProvenance:
    def test_source_row_provenance_defaults_to_source(self):
        # A plain source row's provenance labels are all its own label (its values
        # are its own).
        l0 = _make_l0({"gaia": _record("G"), "wmko": _record("W")})
        table = l0.data["CATALOG_RECORD"]
        gaia = table[table["source"] == "gaia"][0]
        wmko = table[table["source"] == "wmko"][0]
        for col in ("radec_src", "plx_src", "rv_src"):
            assert gaia[col] == "gaia"
            assert wmko[col] == "wmko"


class TestReadWmkoHeader:
    """read_wmko_header builds the native wmko record from L0 PRIMARY TARG* (moved
    here from KPF0 read-time population); fail-soft on absent/malformed astrometry."""

    _GOOD_TARG = {
        "TARGRA": "12:00:00.00",
        "TARGDEC": "+40:00:00.0",
        "TARGFRAM": "FK5",
        "TARGEQUI": 2000.0,
        "TARGPMRA": 0.0,
        "TARGPMDC": 0.0,
        "TARGPLAX": 100.0,
        "TARGEPOC": 2000.0,
    }

    @staticmethod
    def _l0_targ(**targ):
        l0 = KPF0()
        p = l0.headers["PRIMARY"]
        p["IMTYPE"] = "object"
        p["OBJECT"] = "testtarget"
        for key, value in targ.items():
            p[key] = value
        return l0

    def test_good_targ_builds_row_and_flag(self):
        # Well-formed TARG* -> a wmko record sanitized to the EPRV C*# format;
        # writing it sets WMKOCR=1.
        l0 = self._l0_targ(**self._GOOD_TARG)
        aq = AstroQuery(l0)
        aq._write_catalog_record("wmko", aq.read_wmko_header())
        assert l0.headers["CATALOG_RECORD"]["WMKOCR"] == 1
        table = l0.data["CATALOG_RECORD"]
        wmko = table[table["source"] == "wmko"][0]
        assert wmko["ra"] == "12:00:00.0000"  # RA hour-angle sexagesimal
        assert wmko["dec"] == "+40:00:00.000"
        assert wmko["object"] == "testtarget"

    def test_pmra_time_to_angle_conversion(self):
        # TARGPMRA is DCS seconds-of-time/yr: read_wmko_header must convert it to
        # on-sky arcsec/yr via x15 x cos(dec). TARGPMDC is already arcsec/yr and
        # passes through. Uses a non-zero TARGPMRA so the factor is exercised (a
        # zero would pass regardless).
        dec_deg = 40.0
        rec = AstroQuery(
            self._l0_targ(**{**self._GOOD_TARG, "TARGPMRA": 0.5, "TARGPMDC": 2.0})
        ).read_wmko_header()

        expected_pmra = 0.5 * 15.0 * np.cos(np.deg2rad(dec_deg))
        assert rec["pmra"] == pytest.approx(expected_pmra)
        assert expected_pmra != pytest.approx(0.5)  # factor actually applied
        assert rec["pmdec"] == pytest.approx(2.0)  # dec PM unchanged

    def test_no_targ_returns_none_no_warning(self, caplog):
        # No TARGRA (e.g. a science frame with no pointing) -> None, silently.
        with caplog.at_level(logging.WARNING):
            assert AstroQuery(self._l0_targ()).read_wmko_header() is None
        assert "CATALOG_RECORD" not in caplog.text

    def test_malformed_targ_warns_returns_none(self, caplog):
        # Unparseable TARG* astrometry -> warns and returns None (never raises).
        l0 = self._l0_targ(**{**self._GOOD_TARG, "TARGDEC": "not-a-coordinate"})
        with caplog.at_level(logging.WARNING):
            assert AstroQuery(l0).read_wmko_header() is None
        assert "could not build wmko CATALOG_RECORD" in caplog.text
