"""Regression tests for the AstroQuery module.

Focus: ``merge_catalog_records`` -- the merge of the gaia/simbad/wmko records into
the canonical ``kpf-drp`` row -- and ``read_wmko_header``. The external Gaia/SIMBAD
queries are not exercised here (no network); merge is driven off in-memory source
records preset on the instance (as perform's build steps leave them), and
read_wmko_header off synthetic L0 PRIMARY TARG*.
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


def _merge(records, targradv=None):
    """Merge the given {source: in-memory record} the way perform does: preset each
    source's just-built record on a fresh AstroQuery, then merge. Returns the
    canonical kpf-drp record dict (missing cells are None, as merge emits them).
    ``targradv`` seeds L0 PRIMARY TARGRADV so the rv fallback can be exercised."""
    l0 = KPF0()
    l0.headers["PRIMARY"]["IMTYPE"] = "object"
    if targradv is not None:
        l0.headers["PRIMARY"]["TARGRADV"] = targradv
    aq = AstroQuery(l0)
    for source, record in records.items():
        setattr(aq, f"_{source}", record)
    return aq.merge_catalog_records()


class TestMergeCatalogRecords:
    def test_all_gaia_complete(self):
        row = _merge({"gaia": _record("G123")})
        assert row["radec_src"] == "gaia"
        assert row["object"] == "G123"
        assert row["ra"] == _ra_str(180.0)
        assert row["parallax"] == pytest.approx(50.0)
        assert row["rv"] == pytest.approx(10.0)

    def test_position_follows_priority(self):
        # All three complete but at different RAs -> position from Gaia (highest).
        row = _merge(
            {
                "gaia": _record("G", ra=10.0),
                "simbad": _record("S", ra=20.0),
                "wmko": _record("W", ra=30.0),
            }
        )
        assert row["radec_src"] == "gaia"
        assert row["ra"] == _ra_str(10.0)

    def test_rv_rides_with_astrometric_base(self):
        # rv comes from the astrometric base source, not borrowed across catalogs:
        # Gaia is the base and supplies rv, so SIMBAD's rv is ignored.
        row = _merge(
            {
                "gaia": _record("G", rv=11.0),
                "simbad": _record("S", rv=99.0),
            }
        )
        assert row["radec_src"] == "gaia"
        assert row["rv"] == pytest.approx(11.0)
        assert row["rv_src"] == "gaia"

    def test_rv_not_borrowed_from_lower_priority(self):
        # rv is no longer borrowed across catalogs: Gaia is the base and lacks rv;
        # SIMBAD's rv is NOT pulled in, and with no TARGRADV rv is left missing.
        row = _merge(
            {
                "gaia": _record("G", rv=None),
                "simbad": _record("S", rv=99.0),
            }
        )
        assert row["radec_src"] == "gaia"
        assert row["rv"] is None
        assert row["rv_src"] == ""

    def test_rv_missing_falls_back_to_targradv(self, caplog):
        # Base (Gaia) lacks rv -> rv comes from the telescope TARGRADV on PRIMARY (km/s,
        # no conversion), tagged rv_src='wmko', with a WARNING.
        with caplog.at_level(logging.WARNING, logger="kpfpipe.modules.astro_query"):
            row = _merge({"gaia": _record("G", rv=None)}, targradv=-4.5)
        assert row["radec_src"] == "gaia"
        assert row["rv"] == pytest.approx(-4.5)
        assert row["rv_src"] == "wmko"
        assert "falling back to the telescope TARGRADV" in caplog.text

    def test_missing_parallax_disqualifies_astrometric_base(self):
        # parallax is part of the astrometric block: Gaia lacking it is disqualified as
        # the base entirely, so the whole position (and parallax) comes from SIMBAD,
        # the highest-priority source that supplies a complete solution.
        row = _merge(
            {
                "gaia": _record("G", ra=10.0, parallax=None),
                "simbad": _record("S", ra=20.0, parallax=7.0),
                "wmko": _record("W", ra=30.0, parallax=8.0),
            }
        )
        assert row["radec_src"] == "simbad"
        assert row["ra"] == _ra_str(20.0)  # position from SIMBAD, not Gaia
        assert row["parallax"] == pytest.approx(7.0)
        assert row["plx_src"] == "simbad"

    def test_lone_source_missing_parallax_raises(self):
        # parallax is part of the astrometric block, so a sole source lacking it cannot
        # anchor the canonical position.
        with pytest.raises(ValueError, match="position"):
            _merge({"gaia": _record("G", parallax=None)})

    def test_wmko_only(self):
        row = _merge({"wmko": _record("W")})
        assert row["radec_src"] == "wmko"
        assert row["object"] == "W"
        assert row["rv_src"] == "wmko"  # rv rides with the sole source

    def test_no_source_raises(self):
        # No source built a record -> nothing to correct -> raise.
        with pytest.raises(ValueError, match="position"):
            _merge({})

    def test_incomplete_position_raises(self):
        # The only candidate lacks a coherent position block (pmra missing).
        with pytest.raises(ValueError, match="position"):
            _merge({"gaia": _record("G", pmra=None)})

    def test_optional_rv_missing_everywhere_left_missing(self):
        # Position present but no source supplies rv -> merged row builds, rv is None.
        row = _merge(
            {
                "gaia": _record("G", rv=None),
                "simbad": _record("S", rv=None),
            }
        )
        assert row["radec_src"] == "gaia"
        assert row["rv"] is None
        assert row["rv_src"] == ""  # nothing supplied rv -> empty provenance
        assert row["parallax"] == pytest.approx(50.0)  # parallax still filled


class TestSingleSourceProvenance:
    def test_source_row_provenance_defaults_to_source(self):
        # A plain source row's provenance labels are all its own label (its values
        # are its own).
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "object"
        aq = AstroQuery(l0)
        aq._write_catalog_record("gaia", _record("G"))
        aq._write_catalog_record("wmko", _record("W"))
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
        # Well-formed FK5 TARG* -> a wmko record rotated to ICRS (so all sources
        # share one frame), sanitized to the EPRV C*# format; writing sets WMKOCR=1.
        l0 = self._l0_targ(**self._GOOD_TARG)
        aq = AstroQuery(l0)
        aq.read_wmko_header()  # builds the wmko row and writes it in one go
        assert l0.headers["CATALOG_RECORD"]["WMKOCR"] == 1
        table = l0.data["CATALOG_RECORD"]
        wmko = table[table["source"] == "wmko"][0]
        assert wmko["object"] == "testtarget"
        assert wmko["frame"] == "icrs"  # native FK5 rotated to ICRS, not relabeled
        # ICRS position sits within the ~tens-of-mas FK5->ICRS frame bias of the
        # FK5 input, and is shifted from it (a real rotation, not a copy).
        ra = Angle(wmko["ra"], unit=u.hourangle)
        dec = Angle(wmko["dec"], unit=u.deg)
        fk5_ra = Angle(self._GOOD_TARG["TARGRA"], unit=u.hourangle)
        fk5_dec = Angle(self._GOOD_TARG["TARGDEC"], unit=u.deg)
        assert abs((ra - fk5_ra).arcsec) < 1.0
        assert abs((dec - fk5_dec).arcsec) < 1.0
        assert wmko["ra"] != "12:00:00.0000"  # rotated, not copied

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

    def test_non_fk5_frame_raises(self):
        # KPF pointing is always FK5; any other TARGFRAM (e.g. galactic) raises
        # rather than being coerced onto a frame it does not have.
        l0 = self._l0_targ(**{**self._GOOD_TARG, "TARGFRAM": "galactic"})
        with pytest.raises(ValueError, match="TARGFRAM"):
            AstroQuery(l0).read_wmko_header()

    def test_absent_frame_raises(self):
        # An absent TARGFRAM cannot be verified as FK5 -> raise (never guess a frame).
        targ = {k: v for k, v in self._GOOD_TARG.items() if k != "TARGFRAM"}
        with pytest.raises(ValueError, match="TARGFRAM"):
            AstroQuery(self._l0_targ(**targ)).read_wmko_header()

    def test_use_wmko_tcs_false_skips_build(self):
        # use_wmko_tcs gates the wmko build in perform, like do_gaia_query/
        # do_simbad_query gate their queries: off -> read_wmko_header is not called,
        # so no wmko record and (catalogs off too) no position -> merge raises.
        aq = AstroQuery(
            self._l0_targ(**self._GOOD_TARG),
            {"do_gaia_query": False, "do_simbad_query": False, "use_wmko_tcs": False},
        )
        with pytest.raises(ValueError, match="position"):
            aq.perform()
        assert aq._wmko is None
