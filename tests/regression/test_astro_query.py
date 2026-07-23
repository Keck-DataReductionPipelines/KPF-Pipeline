"""Regression tests for the AstroQuery module.

Covers ``merge_catalog_records`` -- the merge of the gaia/simbad/wmko records into
the canonical ``kpf-drp`` row -- ``read_wmko_header`` (off synthetic L0 PRIMARY
TARG*), and the external Gaia/SIMBAD query contract (``TestExternalQueries``). The
network is never touched: ``Gaia.launch_job``/``Simbad`` are mocked with one-row
result Tables, so query parsing, each fail-soft None+warning path, and the
``_verify_units`` schema-drift guard are exercised offline.
"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Column, Table

from kpfpipe.data_models import KPF0
from kpfpipe.modules.astro_query import _GAIA_UNITS, _SIMBAD_UNITS, AstroQuery
from kpfpipe.utils.astro import compute_redshift


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
        "color": 0.8,
        "color_name": "Gaia BP-RP",
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

    def test_color_rides_with_astrometric_base(self):
        # color/color_name come from the astrometric base source, like the position:
        # Gaia is the base, so its color wins over SIMBAD's.
        row = _merge(
            {
                "gaia": _record("G", color=1.1, color_name="Gaia BP-RP"),
                "simbad": _record("S", color=0.6, color_name="B-V"),
            }
        )
        assert row["radec_src"] == "gaia"
        assert row["color"] == pytest.approx(1.1)
        assert row["color_name"] == "Gaia BP-RP"

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

    def test_demoted_source_warns_naming_missing_field(self, caplog):
        # A present-but-incomplete higher-priority source is demoted to a lower one;
        # the fallback must be auditable, not silent -- warn naming the dropped
        # source and the field(s) it lacks (contrast the rv fallback, which warns).
        with caplog.at_level(logging.WARNING, logger="kpfpipe.modules.astro_query"):
            row = _merge(
                {
                    "gaia": _record("G", parallax=None),
                    "simbad": _record("S"),
                }
            )
        assert row["radec_src"] == "simbad"
        assert "gaia astrometry incomplete (missing parallax)" in caplog.text

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


class TestRedshift:
    """z (CZ#) is derived from rv at write time, stored on every CATALOG_RECORD row."""

    def test_redshift_helper_variants(self):
        assert AstroQuery._redshift(None) is None
        assert AstroQuery._redshift(0.0) == pytest.approx(0.0)
        assert AstroQuery._redshift(10.0) == pytest.approx(
            compute_redshift(10.0 * u.km / u.s)
        )

    def test_written_row_carries_redshift(self):
        # A written row's z column is the redshift derived from its rv.
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "object"
        aq = AstroQuery(l0)
        aq._write_catalog_record("gaia", _record("G", rv=11.0))
        row = l0.data["CATALOG_RECORD"][0]
        assert row["z"] == pytest.approx(compute_redshift(11.0 * u.km / u.s))

    def test_missing_rv_leaves_redshift_nan(self):
        # rv absent -> z is NaN (blank CZ# downstream), not an error.
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "object"
        aq = AstroQuery(l0)
        aq._write_catalog_record("gaia", _record("G", rv=None))
        row = l0.data["CATALOG_RECORD"][0]
        assert np.isnan(row["z"])


class TestColor:
    """_color pairs a bluer-minus-redder magnitude difference with its label."""

    def test_color_variants(self):
        assert AstroQuery._color(9.5, 8.5, "Gaia BP-RP") == (
            pytest.approx(1.0),
            "Gaia BP-RP",
        )
        assert AstroQuery._color(None, 8.5, "B-V") == (None, None)
        assert AstroQuery._color(9.5, None, "B-V") == (None, None)


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

    def test_builds_g_minus_j_color(self):
        # The G-J color comes straight off PRIMARY: GAIAMAG - 2MASSMAG.
        rec = AstroQuery(
            self._l0_targ(**{**self._GOOD_TARG, "GAIAMAG": 7.25, "2MASSMAG": 5.5})
        ).read_wmko_header()
        assert rec["color"] == pytest.approx(1.75)
        assert rec["color_name"] == "G-J"

    def test_absent_magnitude_leaves_color_none(self):
        # A missing GAIAMAG/2MASSMAG -> no color (both fields None), not an error.
        rec = AstroQuery(self._l0_targ(**self._GOOD_TARG)).read_wmko_header()
        assert rec["color"] is None and rec["color_name"] is None

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

    def test_nonnumeric_numeric_fields_laundered_to_none(self):
        # A non-numeric TARG* numeric card must be laundered to None via _scalar,
        # matching the gaia/simbad paths -- otherwise the stray value survives the
        # merge's is-not-None completeness gate as a valid-looking astrometric
        # solution and corrupts the float CATALOG_RECORD. (A NaN cannot occur --
        # astropy rejects NaN in FITS headers -- and a valueless card already reads
        # back as None; a stray string is the residual reachable case.)
        rec = AstroQuery(
            self._l0_targ(**{**self._GOOD_TARG, "TARGPLAX": "UNKNOWN"})
        ).read_wmko_header()
        assert rec["parallax"] is None

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


# ---------------------------------------------------------------------------
# External Gaia / SIMBAD queries -- mocked (no network)
# ---------------------------------------------------------------------------

# Default one-row result values, keyed to each catalog's canonical column schema.
# pmra/pmdec are 500 / -250 mas/yr so the mas->arcsec/yr conversion (/1e3) is
# visible (0.5 / -0.25); ra/dec 180/40 deg drive the sexagesimal formatting.
_GAIA_VALUES = {
    "ra": 180.0,
    "dec": 40.0,
    "pmra": 500.0,
    "pmdec": -250.0,
    "parallax": 100.0,
    "radial_velocity": 12.3,
    "ref_epoch": 2016.0,
    "phot_bp_mean_mag": 9.5,
    "phot_rp_mean_mag": 8.5,
}
_SIMBAD_VALUES = {
    "ra": 180.0,
    "dec": 40.0,
    "pmra": 500.0,
    "pmdec": -250.0,
    "plx_value": 100.0,
    "rvz_radvel": 12.3,
    "B": 9.5,
    "V": 8.5,
}


def _units_table(unit_map, value_map, unit_overrides=None):
    """One-row astropy Table with canonical column units.

    A plain Table (not QTable), so ``_verify_units`` sees ``column.unit`` while row
    access yields plain floats for ``_scalar``. ``unit_overrides`` swaps a column's
    unit to drive the schema-drift guard.
    """
    units = dict(unit_map)
    if unit_overrides:
        units.update(unit_overrides)
    table = Table()
    for col, unit in units.items():
        table[col] = Column([value_map[col]], unit=unit)
    return table


def _gaia_table(values=None, units=None):
    vals = {**_GAIA_VALUES, **(values or {})}
    return _units_table(_GAIA_UNITS, vals, units)


def _simbad_table(values=None, units=None):
    vals = {**_SIMBAD_VALUES, **(values or {})}
    return _units_table(_SIMBAD_UNITS, vals, units)


def _gaia_job(table):
    """Stand-in for Gaia.launch_job(query): an object whose get_results() -> table."""
    job = MagicMock()
    job.get_results.return_value = table
    return job


def _simbad_instance(table):
    """Stand-in for Simbad(): an instance whose query_object() -> table (or None)."""
    inst = MagicMock()
    inst.query_object.return_value = table
    return inst


def _l0_for_query(**primary):
    """A fresh science L0 whose PRIMARY carries the given cards (e.g. GAIAID/OBJECT)."""
    l0 = KPF0()
    l0.headers["PRIMARY"]["IMTYPE"] = "object"
    for key, value in primary.items():
        l0.headers["PRIMARY"][key] = value
    return l0


def _patch_gaia(job_or_exc):
    target = "kpfpipe.modules.astro_query.Gaia.launch_job"
    if isinstance(job_or_exc, Exception):
        return patch(target, side_effect=job_or_exc)
    return patch(target, return_value=job_or_exc)


class TestExternalQueries:
    """Gaia/SIMBAD query parsing, fail-soft None+warning, and the unit guard."""

    # -- pure helpers ------------------------------------------------------

    def test_scalar_variants(self):
        # astroquery hands back masked/NaN cells for unmeasured quantities; all the
        # missing/unusable forms must coerce to a clean None, real values to float.
        assert AstroQuery._scalar(None) is None
        assert AstroQuery._scalar(np.ma.masked) is None
        assert AstroQuery._scalar(float("nan")) is None
        assert AstroQuery._scalar("not-a-number") is None
        assert AstroQuery._scalar(3.5) == 3.5
        assert AstroQuery._scalar("2.5") == 2.5

    def test_gaia_source_id(self):
        assert AstroQuery(_l0_for_query(GAIAID="Gaia DR3 12345"))._gaia_source_id() == (
            "12345"
        )
        assert AstroQuery(_l0_for_query(GAIAID="12345"))._gaia_source_id() == "12345"
        assert AstroQuery(_l0_for_query())._gaia_source_id() is None
        assert (
            AstroQuery(_l0_for_query(GAIAID="Gaia DR3 abc"))._gaia_source_id() is None
        )
        assert AstroQuery(_l0_for_query(GAIAID=""))._gaia_source_id() is None

    def test_simbad_resolvable_name(self):
        assert (
            AstroQuery(_l0_for_query(OBJECT="10700"))._simbad_resolvable_name()
            == "HD 10700"
        )
        assert (
            AstroQuery(_l0_for_query(OBJECT="tau Cet"))._simbad_resolvable_name()
            == "tau Cet"
        )
        assert AstroQuery(_l0_for_query())._simbad_resolvable_name() is None
        assert AstroQuery(_l0_for_query(OBJECT=""))._simbad_resolvable_name() is None

    def test_verify_units_missing_and_mismatch_raise(self):
        AstroQuery._verify_units(_gaia_table(), _GAIA_UNITS, "Gaia DR3")  # no raise
        dropped = _gaia_table()
        dropped.remove_column("parallax")
        with pytest.raises(ValueError, match="unexpected column units"):
            AstroQuery._verify_units(dropped, _GAIA_UNITS, "Gaia DR3")
        bad = _gaia_table(units={"parallax": u.arcsec})
        with pytest.raises(ValueError, match="unexpected column units"):
            AstroQuery._verify_units(bad, _GAIA_UNITS, "Gaia DR3")

    # -- Gaia --------------------------------------------------------------

    def test_query_gaia_parses_and_writes(self):
        l0 = _l0_for_query(GAIAID="Gaia DR3 12345")
        aq = AstroQuery(l0)
        with _patch_gaia(_gaia_job(_gaia_table())):
            rec = aq.query_gaia()
        assert rec is not None
        exp_ra, exp_dec = AstroQuery._sexagesimal_radec(
            SkyCoord(180.0, 40.0, unit=u.deg, frame="icrs")
        )
        assert rec["ra"] == exp_ra and rec["dec"] == exp_dec
        assert rec["pmra"] == pytest.approx(0.5)  # 500 mas/yr -> arcsec/yr
        assert rec["pmdec"] == pytest.approx(-0.25)
        assert rec["parallax"] == pytest.approx(100.0)
        assert rec["rv"] == pytest.approx(12.3)
        assert rec["epoch"] == pytest.approx(2016.0)
        assert rec["frame"] == "icrs"
        assert rec["equinox"] == pytest.approx(2000.0)
        assert rec["object"] == "12345"
        assert rec["color"] == pytest.approx(1.0)  # G_BP - G_RP
        assert rec["color_name"] == "Gaia BP-RP"
        # Row + presence flag written to CATALOG_RECORD.
        assert l0.headers["CATALOG_RECORD"]["GAIACR"] == 1
        assert "gaia" in [str(s) for s in l0.data["CATALOG_RECORD"]["source"]]

    def test_query_gaia_missing_rv_becomes_none(self):
        aq = AstroQuery(_l0_for_query(GAIAID="12345"))
        with _patch_gaia(_gaia_job(_gaia_table({"radial_velocity": float("nan")}))):
            rec = aq.query_gaia()
        assert rec["rv"] is None

    def test_query_gaia_missing_photometry_leaves_color_none(self):
        # A color needs both magnitudes; one unmeasured (masked/NaN) -> no color.
        aq = AstroQuery(_l0_for_query(GAIAID="12345"))
        with _patch_gaia(_gaia_job(_gaia_table({"phot_rp_mean_mag": float("nan")}))):
            rec = aq.query_gaia()
        assert rec["color"] is None and rec["color_name"] is None

    def test_record_without_color_writes_blank(self):
        # A record that omits color/color_name writes NaN / "" for them, not an error.
        record = _record("K")
        record.pop("color")
        record.pop("color_name")
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "object"
        aq = AstroQuery(l0)
        aq._write_catalog_record("kpf-drp", record)
        row = l0.data["CATALOG_RECORD"][0]
        assert np.isnan(row["color"]) and row["color_name"] == ""

    def test_query_gaia_no_gaiaid_returns_none(self, caplog):
        aq = AstroQuery(_l0_for_query())  # no GAIAID
        with caplog.at_level(logging.WARNING):
            assert aq.query_gaia() is None
        assert "no usable GAIAID" in caplog.text

    def test_query_gaia_lookup_failure_returns_none(self, caplog):
        aq = AstroQuery(_l0_for_query(GAIAID="12345"))
        with (
            _patch_gaia(ConnectionError("gaia down")),
            caplog.at_level(logging.WARNING),
        ):
            assert aq.query_gaia() is None
        assert "Gaia query failed" in caplog.text

    def test_query_gaia_no_match_returns_none(self, caplog):
        aq = AstroQuery(_l0_for_query(GAIAID="12345"))
        with _patch_gaia(_gaia_job(Table())), caplog.at_level(logging.WARNING):
            assert aq.query_gaia() is None
        assert "no match" in caplog.text

    def test_query_gaia_unit_mismatch_raises(self):
        aq = AstroQuery(_l0_for_query(GAIAID="12345"))
        with _patch_gaia(_gaia_job(_gaia_table(units={"parallax": u.arcsec}))):
            with pytest.raises(ValueError, match="unexpected column units"):
                aq.query_gaia()

    # -- SIMBAD ------------------------------------------------------------

    def test_query_simbad_parses_and_writes(self):
        l0 = _l0_for_query(OBJECT="10700")
        aq = AstroQuery(l0)
        with patch(
            "kpfpipe.modules.astro_query.Simbad",
            return_value=_simbad_instance(_simbad_table()),
        ):
            rec = aq.query_simbad()
        assert rec is not None
        exp_ra, exp_dec = AstroQuery._sexagesimal_radec(
            SkyCoord(180.0, 40.0, unit=u.deg, frame="icrs")
        )
        assert rec["ra"] == exp_ra and rec["dec"] == exp_dec
        assert rec["pmra"] == pytest.approx(0.5)
        assert rec["pmdec"] == pytest.approx(-0.25)
        assert rec["parallax"] == pytest.approx(100.0)
        assert rec["rv"] == pytest.approx(12.3)
        assert rec["object"] == "HD 10700"  # bare-numeric OBJECT -> HD prefix
        assert rec["frame"] == "icrs"
        assert rec["epoch"] == pytest.approx(2000.0)
        assert rec["color"] == pytest.approx(1.0)  # Johnson B - V
        assert rec["color_name"] == "B-V"
        assert l0.headers["CATALOG_RECORD"]["SIMBADCR"] == 1

    def test_query_simbad_missing_photometry_leaves_color_none(self):
        # A color needs both magnitudes; one unmeasured -> no color.
        l0 = _l0_for_query(OBJECT="10700")
        aq = AstroQuery(l0)
        with patch(
            "kpfpipe.modules.astro_query.Simbad",
            return_value=_simbad_instance(_simbad_table({"V": float("nan")})),
        ):
            rec = aq.query_simbad()
        assert rec["color"] is None and rec["color_name"] is None

    def test_query_simbad_no_object_returns_none(self, caplog):
        aq = AstroQuery(_l0_for_query())
        with caplog.at_level(logging.WARNING):
            assert aq.query_simbad() is None
        assert "no OBJECT name" in caplog.text

    def test_query_simbad_lookup_failure_returns_none(self, caplog):
        aq = AstroQuery(_l0_for_query(OBJECT="tau Cet"))
        inst = MagicMock()
        inst.query_object.side_effect = ConnectionError("simbad down")
        with (
            patch("kpfpipe.modules.astro_query.Simbad", return_value=inst),
            caplog.at_level(logging.WARNING),
        ):
            assert aq.query_simbad() is None
        assert "SIMBAD query failed" in caplog.text

    def test_query_simbad_no_match_returns_none(self, caplog):
        aq = AstroQuery(_l0_for_query(OBJECT="NotARealStar"))
        with (
            patch(
                "kpfpipe.modules.astro_query.Simbad",
                return_value=_simbad_instance(None),
            ),
            caplog.at_level(logging.WARNING),
        ):
            assert aq.query_simbad() is None
        assert "no match" in caplog.text

    def test_query_simbad_unit_mismatch_raises(self):
        aq = AstroQuery(_l0_for_query(OBJECT="tau Cet"))
        with patch(
            "kpfpipe.modules.astro_query.Simbad",
            return_value=_simbad_instance(_simbad_table(units={"plx_value": u.arcsec})),
        ):
            with pytest.raises(ValueError, match="unexpected column units"):
                aq.query_simbad()
