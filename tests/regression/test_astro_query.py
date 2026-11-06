"""Regression tests for the AstroQuery module.

Covers ``merge_catalog_records`` (gaia/simbad/wmko -> the canonical ``kpf-drp``
row), ``read_wmko_header`` off synthetic L0 PRIMARY TARG*, and the external query
contract. The network is never touched: the ``gaia_client``/``simbad_client``
factories are mocked to hand back one-row result Tables, so parsing, the fail-soft
None+warning paths, and the ``_verify_units`` schema-drift guard all run offline.
That is enforced, not just intended -- tests/conftest.py blocks connects to the
catalog hosts for the whole suite.
"""

import json
import logging
import re
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import FK5, Angle, SkyCoord
from astropy.table import Column, Table
from astropy.time import Time

from kpfpipe.data_models import KPF0
from kpfpipe.modules.astro_query import AstroQuery
from kpfpipe.utils.astro import compute_redshift
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.header import HeaderError

from ._scripts import REPO_ROOT


def _ra_str(deg):
    """RA in deg -> the sexagesimal hour-angle string the CRA# schema stores."""
    return Angle(deg, u.deg).to_string(unit=u.hourangle, sep=":", pad=True, precision=4)


def _dec_str(deg):
    """Dec in deg -> the sexagesimal string the CDEC# schema stores."""
    return Angle(deg, u.deg).to_string(
        unit=u.deg, sep=":", pad=True, alwayssign=True, precision=3
    )


def _record(obj, ra=180.0, **overrides):
    """A complete canonical record at ``ra``: sexagesimal RA/Dec, PM in arcsec/yr.
    Override a field with None to mark it missing."""
    rec = {
        "object": obj,
        "catalog": "Gaia DR3",
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


def _merge(records, targradv=None, priority=("gaia", "simbad", "wmko")):
    """Merge {source: record} the way perform does: preset each source's record on
    a fresh AstroQuery, then merge. Returns the canonical kpf-drp record dict
    (missing cells None). ``targradv`` seeds TARGRADV for the rv fallback. The
    default ``priority`` admits all three sources so these exercise the merge
    itself; the shipped policy is pinned in TestAstrometryPriority."""
    l0 = KPF0()
    l0.headers["PRIMARY"]["OBSTYPE"] = "Target"
    if targradv is not None:
        l0.headers["INSTRUMENT_HEADER"]["TARGRADV"] = targradv
    aq = AstroQuery(l0, {"astrometry_priority": priority})
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
        # All three complete but at different RAs, so the winner is visible.
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
        # rv comes from the astrometric base (Gaia), not borrowed from SIMBAD.
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
        row = _merge(
            {
                "gaia": _record("G", color=1.1, color_name="Gaia BP-RP"),
                "simbad": _record("S", color=0.6, color_name="B-V"),
            }
        )
        assert row["radec_src"] == "gaia"
        assert row["color"] == pytest.approx(1.1)
        assert row["color_name"] == "Gaia BP-RP"

    def test_color_borrowed_from_lower_priority_when_base_lacks(self, caplog):
        # A color index is independent of the astrometry (unlike rv), so borrowing
        # one from a lower-priority catalog is acceptable -- with a WARNING.
        with caplog.at_level(logging.WARNING, logger="kpfpipe.modules.astro_query"):
            row = _merge(
                {
                    "gaia": _record("G", color=None, color_name=None),
                    "simbad": _record("S", color=0.6, color_name="B-V"),
                }
            )
        assert row["radec_src"] == "gaia"
        assert row["color"] == pytest.approx(0.6)
        assert row["color_name"] == "B-V"
        assert "astrometric base has no color" in caplog.text

    def test_color_missing_everywhere_left_blank(self):
        row = _merge(
            {
                "gaia": _record("G", color=None, color_name=None),
                "simbad": _record("S", color=None, color_name=None),
            }
        )
        assert row["color"] is None
        assert row["color_name"] is None

    def test_rv_not_borrowed_from_lower_priority(self):
        # rv must ride with the astrometric base: SIMBAD's is not borrowed, and
        # with no TARGRADV the rv stays missing.
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
        # TARGRADV is already km/s, so the fallback needs no conversion.
        with caplog.at_level(logging.WARNING, logger="kpfpipe.modules.astro_query"):
            row = _merge({"gaia": _record("G", rv=None)}, targradv=-4.5)
        assert row["radec_src"] == "gaia"
        assert row["rv"] == pytest.approx(-4.5)
        assert row["rv_src"] == "wmko"
        assert "falling back to the telescope TARGRADV" in caplog.text

    def test_missing_parallax_disqualifies_astrometric_base(self):
        # parallax is part of the astrometric block, so Gaia lacking it is
        # disqualified as the base and the whole position comes from SIMBAD.
        row = _merge(
            {
                "gaia": _record("G", ra=10.0, parallax=None),
                "simbad": _record("S", ra=20.0, parallax=7.0),
                "wmko": _record("W", ra=30.0, parallax=8.0),
            }
        )
        assert row["radec_src"] == "simbad"
        assert row["ra"] == _ra_str(20.0)  # SIMBAD's RA, not Gaia's
        assert row["parallax"] == pytest.approx(7.0)
        assert row["plx_src"] == "simbad"

    def test_demoted_source_warns_naming_missing_field(self, caplog):
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
        # parallax is part of the astrometric block, so a sole source lacking it
        # cannot anchor the position.
        with pytest.raises(ValueError, match="position"):
            _merge({"gaia": _record("G", parallax=None)})

    def test_wmko_only(self):
        row = _merge({"wmko": _record("W")})
        assert row["radec_src"] == "wmko"
        assert row["object"] == "W"
        assert row["rv_src"] == "wmko"  # rv rides with the sole source

    def test_no_source_raises(self):
        with pytest.raises(ValueError, match="position"):
            _merge({})

    def test_incomplete_position_raises(self):
        with pytest.raises(ValueError, match="position"):
            _merge({"gaia": _record("G", pmra=None)})

    def test_optional_rv_missing_everywhere_left_missing(self):
        # rv is optional: the merged row still builds with the position intact.
        row = _merge(
            {
                "gaia": _record("G", rv=None),
                "simbad": _record("S", rv=None),
            }
        )
        assert row["radec_src"] == "gaia"
        assert row["rv"] is None
        assert row["rv_src"] == ""  # nothing supplied rv -> empty provenance
        assert row["parallax"] == pytest.approx(50.0)


class TestAstrometryPriority:
    """Which sources may anchor the merged position.

    The knob exists so a catalog outage cannot silently substitute lower-precision
    telescope astrometry into the RV path: a source outside the priority still gets
    its row (DiagL0 needs the wmko row for TCSOFF) but can never become the base.
    """

    def test_shipped_default_bars_wmko_from_anchoring(self):
        # A Gaia+SIMBAD outage must not fall through to TCS values.
        aq = AstroQuery(_l0_for_query())
        assert "wmko" not in aq.astrometry_priority

    def test_source_outside_priority_anchoring_raises(self):
        with pytest.raises(ValueError, match="astrometry_priority"):
            _merge({"wmko": _record("W")}, priority=("gaia", "simbad"))

    def test_incomplete_astrometry_message_names_each_gap(self):
        # A Gaia two-parameter solution, no SIMBAD record: a DRP limitation.
        gaia = _record("G", pmra=None, pmdec=None, parallax=None)
        with pytest.raises(ValueError, match="no complete astrometric solution") as e:
            _merge({"gaia": gaia}, priority=("gaia", "simbad"))
        message = str(e.value)
        assert "gaia: missing pmra, pmdec, parallax; simbad: no record" in message
        assert "not an L0 header problem" in message

    def test_priority_order_is_honored(self):
        records = {"gaia": _record("G", ra=10.0), "simbad": _record("S", ra=20.0)}
        assert _merge(records, priority=("gaia", "simbad"))["radec_src"] == "gaia"
        assert _merge(records, priority=("simbad", "gaia"))["radec_src"] == "simbad"

    def test_excluded_source_still_supplies_rv_and_color(self):
        # Barring wmko from the position does not bar the TARGRADV rv fallback.
        row = _merge(
            {"gaia": _record("G", rv=None)},
            targradv=-4.5,
            priority=("gaia",),
        )
        assert row["radec_src"] == "gaia"
        assert row["rv"] == pytest.approx(-4.5)
        assert row["rv_src"] == "wmko"

    @pytest.mark.parametrize("bad", [(), ("gaia", "nope"), ("tcs",)])
    def test_invalid_priority_raises_at_construction(self, bad):
        # An unknown name would otherwise read as "that source had no record" and
        # silently demote the position, so it must be caught at construction.
        with pytest.raises(ValueError, match="astrometry_priority"):
            AstroQuery(_l0_for_query(), {"astrometry_priority": bad})


class TestSingleSourceProvenance:
    def test_source_row_provenance_defaults_to_source(self):
        # A source row holds only its own values, so all three labels are its own.
        l0 = KPF0()
        l0.headers["PRIMARY"]["OBSTYPE"] = "Target"
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
        l0 = KPF0()
        l0.headers["PRIMARY"]["OBSTYPE"] = "Target"
        aq = AstroQuery(l0)
        aq._write_catalog_record("gaia", _record("G", rv=11.0))
        row = l0.data["CATALOG_RECORD"][0]
        assert row["z"] == pytest.approx(compute_redshift(11.0 * u.km / u.s))

    def test_missing_rv_leaves_redshift_nan(self):
        # rv absent -> z is NaN (blank CZ# downstream), not an error.
        l0 = KPF0()
        l0.headers["PRIMARY"]["OBSTYPE"] = "Target"
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
    """read_wmko_header builds the wmko record from L0 PRIMARY TARG*, fail-soft on
    absent or malformed astrometry."""

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
        l0.headers["PRIMARY"]["OBSTYPE"] = "Target"
        l0.headers["PRIMARY"]["OBJECT"] = "testtarget"
        p = l0.headers["INSTRUMENT_HEADER"]
        for key, value in targ.items():
            p[key] = value
        return l0

    def test_good_targ_builds_row(self):
        # TARG* is FK5 but the record is rotated to ICRS, so all sources share
        # one frame.
        l0 = self._l0_targ(**self._GOOD_TARG)
        aq = AstroQuery(l0)
        aq.read_wmko_header()  # builds the wmko row and writes it in one go
        table = l0.data["CATALOG_RECORD"]
        wmko = table[table["source"] == "wmko"][0]
        assert wmko["object"] == "testtarget"
        assert wmko["frame"] == "icrs"
        # The FK5->ICRS bias is tens of mas, so the result stays within 1 arcsec of
        # the input yet must not equal it -- a real rotation, not a relabel.
        ra = Angle(wmko["ra"], unit=u.hourangle)
        dec = Angle(wmko["dec"], unit=u.deg)
        fk5_ra = Angle(self._GOOD_TARG["TARGRA"], unit=u.hourangle)
        fk5_dec = Angle(self._GOOD_TARG["TARGDEC"], unit=u.deg)
        assert abs((ra - fk5_ra).arcsec) < 1.0
        assert abs((dec - fk5_dec).arcsec) < 1.0
        assert wmko["ra"] != "12:00:00.0000"  # rotated, not copied

    def test_pmra_time_to_angle_conversion(self):
        # TARGPMRA is DCS seconds-of-time/yr: on-sky arcsec/yr needs x15 x cos(dec).
        # TARGPMDC is already arcsec/yr and passes through.
        dec_deg = 40.0
        rec = AstroQuery(
            self._l0_targ(**{**self._GOOD_TARG, "TARGPMRA": 0.5, "TARGPMDC": 2.0})
        ).read_wmko_header()

        expected_pmra = 0.5 * 15.0 * np.cos(np.deg2rad(dec_deg))
        assert rec["pmra"] == pytest.approx(expected_pmra)
        assert expected_pmra != pytest.approx(0.5)  # the factor really applies
        assert rec["pmdec"] == pytest.approx(2.0)

    def test_builds_g_minus_j_color(self):
        # The G-J color comes straight off PRIMARY: GAIAMAG - 2MASSMAG.
        rec = AstroQuery(
            self._l0_targ(**{**self._GOOD_TARG, "GAIAMAG": 7.25, "2MASSMAG": 5.5})
        ).read_wmko_header()
        assert rec["color"] == pytest.approx(1.75)
        assert rec["color_name"] == "G-J"

    def test_absent_magnitude_leaves_color_none(self):
        # A missing magnitude -> no color, not an error.
        rec = AstroQuery(self._l0_targ(**self._GOOD_TARG)).read_wmko_header()
        assert rec["color"] is None and rec["color_name"] is None

    def test_no_targ_returns_none_no_warning(self, caplog):
        # No TARGRA (e.g. a frame with no pointing) -> None, and no warning.
        with caplog.at_level(logging.WARNING):
            assert AstroQuery(self._l0_targ()).read_wmko_header() is None
        assert "CATALOG_RECORD" not in caplog.text

    def test_malformed_targ_warns_returns_none(self, caplog):
        # Unparseable TARG* astrometry warns rather than raising.
        l0 = self._l0_targ(**{**self._GOOD_TARG, "TARGDEC": "not-a-coordinate"})
        with caplog.at_level(logging.WARNING):
            assert AstroQuery(l0).read_wmko_header() is None
        assert "could not build wmko CATALOG_RECORD" in caplog.text

    def test_nonnumeric_numeric_fields_laundered_to_none(self):
        # A non-numeric numeric card must launder to None, or it passes the merge's
        # completeness gate as a valid-looking solution. A string is the only
        # reachable case: astropy rejects NaN in headers, and a valueless card
        # already reads as None.
        rec = AstroQuery(
            self._l0_targ(**{**self._GOOD_TARG, "TARGPLAX": "UNKNOWN"})
        ).read_wmko_header()
        assert rec["parallax"] is None

    @staticmethod
    def _sep_from_fk5_reading(rec):
        # Arcsec from the record (ICRS) to _GOOD_TARG's coordinates read as FK5 J2000.
        fk5 = SkyCoord(
            "12:00:00.00",
            "+40:00:00.0",
            unit=(u.hourangle, u.deg),
            frame=FK5(equinox="J2000"),
        )
        return (
            SkyCoord(rec["ra"], rec["dec"], unit=(u.hourangle, u.deg))
            .separation(fk5)
            .arcsec
        )

    def test_fk4_frame_rotated_with_besselian_equinox(self):
        # FK4 at equinox 2000 (Keck planet targets) is Besselian B2000: under 1" from
        # the FK5 J2000 reading, but a distinct rotation, not a relabel.
        rec = AstroQuery(
            self._l0_targ(**{**self._GOOD_TARG, "TARGFRAM": "FK4"})
        ).read_wmko_header()
        assert rec["frame"] == "icrs"
        assert 0.1 < self._sep_from_fk5_reading(rec) < 1.5

    def test_fk4_absent_equinox_defaults_to_b1950(self):
        # FK4 with no TARGEQUI is B1950: precession puts it arcmin from J2000.
        targ = {k: v for k, v in self._GOOD_TARG.items() if k != "TARGEQUI"}
        rec = AstroQuery(
            self._l0_targ(**{**targ, "TARGFRAM": "FK4"})
        ).read_wmko_header()
        assert self._sep_from_fk5_reading(rec) > 600

    def test_unsupported_frame_raises(self):
        # KPF pointing is FK5 or FK4; another TARGFRAM must not be coerced onto a
        # frame the data does not have.
        l0 = self._l0_targ(**{**self._GOOD_TARG, "TARGFRAM": "galactic"})
        with pytest.raises(ValueError, match="TARGFRAM"):
            AstroQuery(l0).read_wmko_header()

    def test_absent_frame_raises(self):
        # An absent TARGFRAM cannot be verified, and a frame is never guessed.
        targ = {k: v for k, v in self._GOOD_TARG.items() if k != "TARGFRAM"}
        with pytest.raises(ValueError, match="TARGFRAM"):
            AstroQuery(self._l0_targ(**targ)).read_wmko_header()

    def test_wmko_row_built_even_when_barred_from_anchoring(self):
        # The wmko row is unconditional (TCSOFF needs it), but with wmko outside
        # astrometry_priority and both catalogs off nothing may anchor, so the merge
        # raises rather than quietly falling back to telescope astrometry.
        aq = AstroQuery(
            self._l0_targ(**self._GOOD_TARG),
            {"do_gaia_query": False, "do_simbad_query": False},
        )
        with pytest.raises(ValueError, match="astrometry_priority"):
            aq.perform()
        assert aq._wmko is not None
        table = aq.l0_obj.data["CATALOG_RECORD"]
        assert "wmko" in [str(s) for s in table["source"]]


# ------------------------------------------------------------------
# External Gaia / SIMBAD queries -- mocked (no network)
# ------------------------------------------------------------------

# Default one-row result values, keyed to each catalog's column schema. pmra/pmdec
# are 500 / -250 mas/yr so the mas -> arcsec/yr conversion is visible (0.5 / -0.25).
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
    "main_id": "* tau Cet",
    "ids": "HD  10700|* tau Cet|Gaia DR3 2452378776434477184",
    "coo_bibcode": "2020yCat.1350....0G",  # Gaia EDR3
    "otype": "PM*",
}


def _units_table(unit_map, value_map, unit_overrides=None):
    """One-row astropy Table with canonical column units.

    A plain Table (not QTable), so ``_verify_units`` sees ``column.unit`` while row
    access yields plain floats for ``_scalar``. ``unit_overrides`` swaps a unit to
    drive the schema-drift guard.
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
    return _units_table(AstroQuery._GAIA_UNITS, vals, units)


def _simbad_table(values=None, units=None):
    vals = {**_SIMBAD_VALUES, **(values or {})}
    return _units_table(AstroQuery._SIMBAD_UNITS, vals, units)


def _gaia_job(table):
    """Stand-in for TAPService.run_sync(query): an object whose to_table() -> table."""
    job = MagicMock()
    job.to_table.return_value = table
    return job


def _simbad_instance(table):
    """Stand-in for Simbad(): an instance whose query_object() -> table (or None)."""
    inst = MagicMock()
    inst.query_object.return_value = table
    return inst


def _l0_for_query(**primary):
    """A fresh science L0 whose PRIMARY carries the given cards (e.g. GAIAID, in
    HeaderParser's canonical form; OBJECT Unknown and GAIAID blank unless given)."""
    l0 = KPF0()
    l0.headers["PRIMARY"]["OBSTYPE"] = "Target"
    l0.headers["PRIMARY"]["OBJECT"] = "Unknown"
    l0.headers["PRIMARY"]["GAIAID"] = None
    for key, value in primary.items():
        header = "PRIMARY" if key in ("OBJECT", "GAIAID") else "INSTRUMENT_HEADER"
        l0.headers[header][key] = value
    return l0


@contextmanager
def _patch_gaia(job_or_exc):
    """Patch the Gaia client factory, yielding the ``run_sync`` mock to assert on.

    Production builds a client per attempt (``gaia_client(url, ...).run_sync(...)``);
    one mock client answers every mirror, so call_count accumulates across attempts.
    """
    client = MagicMock()
    if isinstance(job_or_exc, Exception):
        client.run_sync.side_effect = job_or_exc
    else:
        client.run_sync.return_value = job_or_exc
    with patch("kpfpipe.modules.astro_query.gaia_client", return_value=client):
        yield client.run_sync


def _gaia_mirror_client(down=()):
    """gaia_client stand-in: a mirror whose URL is in ``down`` refuses the connection.

    The client built for each URL is kept on ``.clients`` to assert its query on.
    """

    def client(url, timeout):
        mock = client.clients[url] = MagicMock()
        if url in down:
            mock.run_sync.side_effect = ConnectionError(f"{url} down")
        else:
            mock.run_sync.return_value = _gaia_job(_gaia_table())
        return mock

    client.clients = {}
    return client


def _patch_simbad(instance):
    """Patch the SIMBAD client factory to return ``instance``."""
    return patch("kpfpipe.modules.astro_query.simbad_client", return_value=instance)


class TestExternalQueries:
    """Gaia/SIMBAD query parsing, fail-soft None+warning, and the unit guard."""

    # -- pure helpers ------------------------------------------------------

    def test_scalar_variants(self):
        # astroquery hands back masked/NaN cells for unmeasured quantities; every
        # unusable form must coerce to None, real values to float.
        assert AstroQuery._scalar(None) is None
        assert AstroQuery._scalar(np.ma.masked) is None
        assert AstroQuery._scalar(float("nan")) is None
        assert AstroQuery._scalar("not-a-number") is None
        assert AstroQuery._scalar(3.5) == 3.5
        assert AstroQuery._scalar("2.5") == 2.5

    @pytest.mark.parametrize(
        ("main_id", "ids", "expected"),
        [
            ("* tau Cet", "HD  10700|* tau Cet", "tau Cet"),  # Bayer, from main_id
            ("* alf Lyr", "* alf Lyr|NAME Vega", "Vega"),  # proper name wins
            ("HD  219134", "HD  219134|HIP 114622", "HD 219134"),  # no common name
        ],
    )
    def test_simbad_common_name(self, main_id, ids, expected):
        aq = AstroQuery(_l0_for_query(OBJECT="HD 10700"))
        table = _simbad_table({"main_id": main_id, "ids": ids})
        with _patch_simbad(_simbad_instance(table)):
            aq.query_simbad()
        assert aq._common_name == expected

    def test_verify_units_missing_and_mismatch_raises(self):
        units = AstroQuery._GAIA_UNITS
        AstroQuery._verify_units(_gaia_table(), units, "Gaia DR3")  # no raise
        dropped = _gaia_table()
        dropped.remove_column("parallax")
        with pytest.raises(ValueError, match="unexpected column units"):
            AstroQuery._verify_units(dropped, units, "Gaia DR3")
        bad = _gaia_table(units={"parallax": u.arcsec})
        with pytest.raises(ValueError, match="unexpected column units"):
            AstroQuery._verify_units(bad, units, "Gaia DR3")

    # -- Gaia --------------------------------------------------------------

    def test_query_gaia_parses_and_writes(self):
        l0 = _l0_for_query(GAIAID="Gaia DR3 1234567890")
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
        assert rec["object"] == "Gaia DR3 1234567890"  # full designation -> CID#
        assert rec["color"] == pytest.approx(1.0)  # G_BP - G_RP
        assert rec["color_name"] == "Gaia BP-RP"
        assert "gaia" in [str(s) for s in l0.data["CATALOG_RECORD"]["source"]]

    def test_query_gaia_missing_rv_becomes_none(self):
        aq = AstroQuery(_l0_for_query(GAIAID="Gaia DR3 1234567890"))
        with _patch_gaia(_gaia_job(_gaia_table({"radial_velocity": float("nan")}))):
            rec = aq.query_gaia()
        assert rec["rv"] is None

    def test_query_gaia_missing_photometry_leaves_color_none(self):
        # A color needs both magnitudes; one unmeasured (masked/NaN) -> no color.
        aq = AstroQuery(_l0_for_query(GAIAID="Gaia DR3 1234567890"))
        with _patch_gaia(_gaia_job(_gaia_table({"phot_rp_mean_mag": float("nan")}))):
            rec = aq.query_gaia()
        assert rec["color"] is None and rec["color_name"] is None

    def test_record_without_color_writes_blank(self):
        # An omitted color writes NaN / "" rather than raising.
        record = _record("K")
        record.pop("color")
        record.pop("color_name")
        l0 = KPF0()
        l0.headers["PRIMARY"]["OBSTYPE"] = "Target"
        aq = AstroQuery(l0)
        aq._write_catalog_record("kpf-drp", record)
        row = l0.data["CATALOG_RECORD"][0]
        assert np.isnan(row["color"]) and row["color_name"] == ""

    def test_query_gaia_no_gaiaid_returns_none(self, caplog):
        aq = AstroQuery(_l0_for_query())  # no GAIAID
        with caplog.at_level(logging.WARNING):
            assert aq.query_gaia() is None
        assert "no GAIAID on L0 PRIMARY" in caplog.text

    def test_query_gaia_lookup_failure_returns_none(self, caplog):
        # A dropped connection is transient, so every mirror is tried twice before
        # the lookup is given up on; sleep is patched so the pause is not waited out.
        aq = AstroQuery(_l0_for_query(GAIAID="Gaia DR3 1234567890"))
        with (
            _patch_gaia(ConnectionError("gaia down")) as run_sync,
            patch("kpfpipe.utils.network.time.sleep"),
            caplog.at_level(logging.WARNING),
        ):
            assert aq.query_gaia() is None
            assert run_sync.call_count == 2 * len(AstroQuery._GAIA_MIRRORS["DR3"])
        assert "Gaia query failed" in caplog.text

    def test_query_gaia_no_match_returns_none(self, caplog):
        aq = AstroQuery(_l0_for_query(GAIAID="Gaia DR3 1234567890"))
        with _patch_gaia(_gaia_job(Table())), caplog.at_level(logging.WARNING):
            assert aq.query_gaia() is None
        assert "no match" in caplog.text

    def test_query_gaia_unit_mismatch_raises(self):
        aq = AstroQuery(_l0_for_query(GAIAID="Gaia DR3 1234567890"))
        with _patch_gaia(_gaia_job(_gaia_table(units={"parallax": u.arcsec}))):
            with pytest.raises(ValueError, match="unexpected column units"):
                aq.query_gaia()

    # -- SIMBAD ------------------------------------------------------------

    def test_query_simbad_parses_and_writes(self):
        l0 = _l0_for_query(OBJECT="HD 10700")
        aq = AstroQuery(l0)
        with _patch_simbad(_simbad_instance(_simbad_table())):
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
        # The coordinates cite Gaia EDR3, so CID/CSRC take its designation and label.
        assert rec["object"] == "Gaia DR3 2452378776434477184"
        assert rec["catalog"] == "Gaia EDR3"
        assert rec["frame"] == "icrs"
        assert rec["epoch"] == pytest.approx(2000.0)
        assert rec["color"] == pytest.approx(1.0)  # Johnson B - V
        assert rec["color_name"] == "B-V"
        assert "simbad" in [str(s) for s in l0.data["CATALOG_RECORD"]["source"]]

    def test_query_simbad_uncited_designation_keeps_name(self):
        # Hipparcos coordinates, but SIMBAD lists no HIP id: CID falls back to the name.
        aq = AstroQuery(_l0_for_query(OBJECT="HD 10700"))
        table = _simbad_table({"coo_bibcode": "2007A&A...474..653V"})
        with _patch_simbad(_simbad_instance(table)):
            rec = aq.query_simbad()
        assert (rec["catalog"], rec["object"]) == ("Hipparcos (2007)", "HD 10700")

    @pytest.mark.parametrize(
        "ids, expected",
        [
            (
                "HD  10700|Gaia DR2 2452378776434477184|Gaia DR3 2452378776434477184",
                "Gaia DR3 2452378776434477184",
            ),
            ("HD  10700|Gaia DR2 2452378776434477184", "Gaia DR2 2452378776434477184"),
            ("HD  10700|* tau Cet", None),
        ],
    )
    def test_query_simbad_fills_blank_gaiaid(self, ids, expected):
        # DR3 preferred over DR2, whatever catalog the coordinates cite.
        l0 = _l0_for_query(OBJECT="HD 10700")
        table = _simbad_table({"ids": ids, "coo_bibcode": "2007A&A...474..653V"})
        with _patch_simbad(_simbad_instance(table)):
            AstroQuery(l0).query_simbad()
        assert l0.headers["PRIMARY"]["GAIAID"] == expected

    def test_query_simbad_malformed_gaia_id_not_filled(self, caplog):
        l0 = _l0_for_query(OBJECT="HD 10700")
        table = _simbad_table({"ids": "HD  10700|Gaia DR3 333"})
        with _patch_simbad(_simbad_instance(table)), caplog.at_level(logging.WARNING):
            AstroQuery(l0).query_simbad()
        assert l0.headers["PRIMARY"]["GAIAID"] is None
        assert "SIMBAD Gaia id 'Gaia DR3 333' is not a Gaia DR2/DR3" in caplog.text

    def test_query_simbad_keeps_header_gaiaid(self):
        l0 = _l0_for_query(OBJECT="HD 10700", GAIAID="Gaia DR3 1234567890")
        with _patch_simbad(_simbad_instance(_simbad_table())):
            AstroQuery(l0).query_simbad()
        assert l0.headers["PRIMARY"]["GAIAID"] == "Gaia DR3 1234567890"

    def test_query_simbad_unknown_catalog_warns(self, caplog):
        aq = AstroQuery(_l0_for_query(OBJECT="HD 10700"))
        table = _simbad_table({"coo_bibcode": "1999XYZ.....1....Q"})
        with _patch_simbad(_simbad_instance(table)), caplog.at_level(logging.WARNING):
            rec = aq.query_simbad()
        assert rec["catalog"] == "Simbad Query (catalog unknown)"
        assert rec["object"] == "HD 10700"
        assert "unrecognized catalog" in caplog.text

    def test_query_simbad_missing_photometry_leaves_color_none(self):
        # A color needs both magnitudes; one unmeasured -> no color.
        l0 = _l0_for_query(OBJECT="HD 10700")
        aq = AstroQuery(l0)
        with _patch_simbad(_simbad_instance(_simbad_table({"V": float("nan")}))):
            rec = aq.query_simbad()
        assert rec["color"] is None and rec["color_name"] is None

    @pytest.mark.parametrize("obj", ["Unknown", "Gaia 2452378776434477184"])
    def test_query_simbad_skips_non_simbad_object(self, obj, caplog):
        # HeaderParser marks an untrusted native OBJECT Unknown; SIMBAD resolves a
        # Gaia designation only with its release. Neither, nor the native card
        # behind it, is queried.
        l0 = _l0_for_query(OBJECT=obj)
        l0.headers["INSTRUMENT_HEADER"]["OBJECT"] = "10700"
        aq = AstroQuery(l0)
        with _patch_simbad(_simbad_instance(_simbad_table())) as simbad:
            with caplog.at_level(logging.WARNING):
                assert aq.query_simbad() is None
        assert "is not a SIMBAD name" in caplog.text
        simbad.assert_not_called()

    def test_query_simbad_keeps_gaia_object(self):
        # OBJECT is never rewritten; the common name goes only to ALIASES.
        l0 = _l0_for_query(OBJECT="Gaia DR3 2452378776434477184")
        aq = AstroQuery(l0)
        table = _simbad_table({"main_id": "* tau Cet", "ids": "HD  10700|* tau Cet"})
        with _patch_simbad(_simbad_instance(table)) as simbad:
            assert aq.query_simbad() is not None
        assert simbad.return_value.query_object.call_args.args == (
            "Gaia DR3 2452378776434477184",
        )
        assert l0.headers["PRIMARY"]["OBJECT"] == "Gaia DR3 2452378776434477184"
        assert aq._common_name == "tau Cet"

    @pytest.mark.parametrize(
        "main_id, ids, expected",
        [
            ("HD  10700", "HD  10700|HIP  8102", "HD 10700"),
            ("WASP-173B", "WASP-173B|HIP  8102", "WASP-173B"),
        ],
    )
    def test_query_simbad_canonicalizes_returned_names(self, main_id, ids, expected):
        # Common name (main_id here) and the cited catalog's designation.
        aq = AstroQuery(_l0_for_query(OBJECT="HD 10700"))
        table = _simbad_table(
            {"main_id": main_id, "ids": ids, "coo_bibcode": "2007A&A...474..653V"}
        )
        with _patch_simbad(_simbad_instance(table)):
            rec = aq.query_simbad()
        assert (aq._common_name, rec["object"]) == (expected, "HIP 8102")

    def test_query_simbad_planet_returns_none(self, caplog):
        # SIMBAD is case-insensitive: 'TOI-1338B' can resolve to the planet.
        aq = AstroQuery(_l0_for_query(OBJECT="TOI-1338B"))
        inst = _simbad_instance(_simbad_table({"main_id": "TOI-1338b", "otype": "Pl"}))
        with _patch_simbad(inst), caplog.at_level(logging.WARNING):
            assert aq.query_simbad() is None
        assert inst.query_object.call_count == 1
        assert "SIMBAD resolved 'TOI-1338B' to planet 'TOI-1338b'" in (caplog.text)

    def test_query_simbad_accepts_non_stellar_object(self):
        aq = AstroQuery(_l0_for_query(OBJECT="M 31"))
        inst = _simbad_instance(_simbad_table({"main_id": "M  31", "otype": "AGN"}))
        with _patch_simbad(inst):
            assert aq.query_simbad() is not None
        assert inst.query_object.call_count == 1

    def test_query_simbad_lookup_failure_returns_none(self, caplog):
        # Failed over like the Gaia failure above; sleep is patched for the same reason.
        aq = AstroQuery(_l0_for_query(OBJECT="tau Cet"))
        inst = MagicMock()
        inst.query_object.side_effect = ConnectionError("simbad down")
        with (
            _patch_simbad(inst),
            patch("kpfpipe.utils.network.time.sleep"),
            caplog.at_level(logging.WARNING),
        ):
            assert aq.query_simbad() is None
            assert inst.query_object.call_count == 2 * len(AstroQuery._SIMBAD_MIRRORS)
        assert "SIMBAD query failed" in caplog.text

    def test_query_gaia_fails_over_to_the_next_mirror(self):
        # ESA and GAVO down: MAST answers, queried under its own table name, with a
        # record identical to the one ESA would have given.
        (esa, _), (gavo, _), (mast, _) = AstroQuery._GAIA_MIRRORS["DR3"]
        with patch(
            "kpfpipe.modules.astro_query.gaia_client", side_effect=_gaia_mirror_client()
        ):
            expected = AstroQuery(
                _l0_for_query(GAIAID="Gaia DR3 1234567890")
            ).query_gaia()
        aq = AstroQuery(_l0_for_query(GAIAID="Gaia DR3 1234567890"))
        stand_in = _gaia_mirror_client(down={esa, gavo})
        with patch(
            "kpfpipe.modules.astro_query.gaia_client", side_effect=stand_in
        ) as factory:
            assert aq.query_gaia() == expected
        assert [c.args[0] for c in factory.call_args_list] == [esa, gavo, mast]
        assert (
            "FROM dbo.gaia_source" in stand_in.clients[mast].run_sync.call_args.args[0]
        )

    def test_query_gaia_dr2_never_asks_a_mirror_without_dr2(self):
        aq = AstroQuery(_l0_for_query(GAIAID="Gaia DR2 1234567890"))
        everything = {
            url for mirrors in AstroQuery._GAIA_MIRRORS.values() for url, _ in mirrors
        }
        with (
            patch(
                "kpfpipe.modules.astro_query.gaia_client",
                side_effect=_gaia_mirror_client(down=everything),
            ) as factory,
            patch("kpfpipe.utils.network.time.sleep"),
        ):
            assert aq.query_gaia() is None
        asked = {c.args[0] for c in factory.call_args_list}
        assert asked == {url for url, _ in AstroQuery._GAIA_MIRRORS["DR2"]}

    def test_query_simbad_fails_over_to_the_next_mirror(self):
        cds, cfa = AstroQuery._SIMBAD_MIRRORS

        def client(server, fields, timeout):
            if server == cds:
                raise ConnectionError("cds down")
            return _simbad_instance(_simbad_table())

        aq = AstroQuery(_l0_for_query(OBJECT="tau Cet"))
        with patch(
            "kpfpipe.modules.astro_query.simbad_client", side_effect=client
        ) as factory:
            assert aq.query_simbad() is not None
        assert [c.args[0] for c in factory.call_args_list] == [cds, cfa]

    def test_query_simbad_no_match_returns_none(self, caplog):
        aq = AstroQuery(_l0_for_query(OBJECT="NotARealStar"))
        with (
            _patch_simbad(_simbad_instance(None)),
            caplog.at_level(logging.WARNING),
        ):
            assert aq.query_simbad() is None
        assert "no match" in caplog.text

    def test_query_simbad_unit_mismatch_raises(self):
        aq = AstroQuery(_l0_for_query(OBJECT="tau Cet"))
        with _patch_simbad(
            _simbad_instance(_simbad_table(units={"plx_value": u.arcsec}))
        ):
            with pytest.raises(ValueError, match="unexpected column units"):
                aq.query_simbad()


# ------------------------------------------------------------------
# Request vs parse -- the fields asked for are the fields read back
# ------------------------------------------------------------------


def _adql_select_columns(query):
    """The column names in an ADQL SELECT list, as a set."""
    select = re.search(r"SELECT\s+(.*?)\s+FROM", query, re.DOTALL | re.IGNORECASE)
    return {col.strip() for col in select.group(1).split(",")}


class TestRequestMatchesParse:
    """Each query asks for exactly the columns its parser reads.

    The mocked tables are built from the parser's own unit schemas, so they answer
    whatever was asked and a drifted request (asking SIMBAD for the deprecated
    'plx' while reading 'plx_value') leaves every other test green.
    """

    def test_gaia_select_list_matches_parsed_columns(self):
        aq = AstroQuery(_l0_for_query(GAIAID="Gaia DR3 1234567890"))
        with _patch_gaia(_gaia_job(_gaia_table())) as run_sync:
            aq.query_gaia()
        assert _adql_select_columns(run_sync.call_args.args[0]) == set(
            AstroQuery._GAIA_UNITS
        )

    @pytest.mark.parametrize(
        ("gaiaid", "table", "designation"),
        [
            ("Gaia DR3 1234567890", "gaiadr3.gaia_source", "Gaia DR3 1234567890"),
            ("Gaia DR2 1234567890", "gaiadr2.gaia_source", "Gaia DR2 1234567890"),
        ],
    )
    def test_gaia_release_selects_table_and_designation(
        self, gaiaid, table, designation
    ):
        # A source_id denotes different stars in different releases, so the table
        # GAIAID names is what keeps the astrometry on the right star.
        aq = AstroQuery(_l0_for_query(GAIAID=gaiaid))
        with _patch_gaia(_gaia_job(_gaia_table())) as run_sync:
            rec = aq.query_gaia()
        assert f"FROM {table}" in run_sync.call_args.args[0]
        assert rec["object"] == designation
        # A named release is taken at its word -- no probe queries.
        assert run_sync.call_count == 1

    def test_simbad_votable_fields_match_parsed_columns(self):
        # ra/dec/main_id arrive in SIMBAD's default basic set, so are not requested.
        aq = AstroQuery(_l0_for_query(OBJECT="tau Cet"))
        with _patch_simbad(_simbad_instance(_simbad_table())) as simbad_client:
            aq.query_simbad()
        # Spelled out rather than derived from _SIMBAD_UNITS, which is what the
        # module builds the request from: a derived expectation would follow a
        # renamed field instead of catching it.
        requested = set(simbad_client.call_args.args[1])
        assert requested == {
            "pmra",
            "pmdec",
            "plx_value",
            "rvz_radvel",
            "B",
            "V",
            "ids",
            "coo_bibcode",
            "otype",
        }


def _release_aware_run_sync(present=(), failing=()):
    """TAPService.run_sync stand-in keyed on the release its query names.

    A release probe (``SELECT source_id ...``) answers one row for a release in
    ``present`` and none otherwise; one in ``failing`` errors the way an archive
    outage would. The full record query returns the standard one-row table, so the
    resolved release shows up in its FROM clause.
    """

    def run_sync(query):
        release = re.search(r"FROM gaia(dr\d)\.gaia_source", query).group(1).upper()
        if release in failing:
            raise ValueError(f"gaia{release.lower()}.gaia_source unreachable")
        if query.lstrip().startswith("SELECT source_id "):
            ids = [1234567890] if release in present else []
            return _gaia_job(Table({"source_id": ids}))
        return _gaia_job(_gaia_table())

    return run_sync


class TestBareGaiaIdResolution:
    """A GAIAID with no release prefix is verified against the archive, not assumed.

    The same source_id denotes different stars in different releases, so guessing
    would silently attach another star's astrometry to the frame.
    """

    @staticmethod
    def _query(l0=None, **kwargs):
        # Bare: no release prefix.
        aq = AstroQuery(l0 or _l0_for_query(GAIAID="Gaia 1234567890"))
        client = MagicMock()
        client.run_sync.side_effect = _release_aware_run_sync(**kwargs)
        with patch("kpfpipe.modules.astro_query.gaia_client", return_value=client):
            return aq.query_gaia()

    @pytest.mark.parametrize("obj", ["Gaia 1234567890", "tau Cet"])
    def test_resolved_release_recorded_on_primary(self, obj):
        # GAIAID takes the release; a bare-Gaia OBJECT takes it in _resolve_identity.
        l0 = _l0_for_query(GAIAID="Gaia 1234567890", OBJECT=obj)
        self._query(l0, present={"DR3"})
        assert l0.headers["PRIMARY"]["GAIAID"] == "Gaia DR3 1234567890"
        assert l0.headers["PRIMARY"]["OBJECT"] == obj

    def test_unresolved_release_leaves_primary(self):
        l0 = _l0_for_query(GAIAID="Gaia 1234567890", OBJECT="Gaia 1234567890")
        with pytest.raises(ValueError):
            self._query(l0)
        assert l0.headers["PRIMARY"]["GAIAID"] == "Gaia 1234567890"
        assert l0.headers["PRIMARY"]["OBJECT"] == "Gaia 1234567890"

    def test_newest_matching_release_wins(self, caplog):
        # Present in both -> DR3, the most recent.
        with caplog.at_level(logging.INFO):
            rec = self._query(present={"DR2", "DR3"})
        assert rec["object"] == "Gaia DR3 1234567890"
        assert "resolved bare GAIAID 1234567890 to Gaia DR3" in caplog.text

    def test_older_release_used_when_newer_lacks_the_id(self):
        rec = self._query(present={"DR2"})
        assert rec["object"] == "Gaia DR2 1234567890"

    def test_bare_id_always_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            self._query(present={"DR3"})
        assert "carries no data release" in caplog.text

    def test_failed_probe_does_not_fall_through(self, caplog):
        # Only a clean no-match moves on: an older release may hold a different star.
        l0 = _l0_for_query(GAIAID="Gaia 1234567890")
        with caplog.at_level(logging.WARNING):
            rec = self._query(l0, present={"DR2"}, failing={"DR3"})
        assert rec is None
        assert l0.headers["PRIMARY"]["GAIAID"] == "Gaia 1234567890"
        assert "could not probe Gaia DR3" in caplog.text

    def test_no_matching_release_raises(self):
        # Fail loud: a bare id matching nothing must not fall back to a guess.
        with pytest.raises(
            HeaderError, match="in no Gaia release.*L0 header must be fixed:"
        ):
            self._query()


# ------------------------------------------------------------------
# perform() -- the real entry point, with both catalogs mocked (no network)
# ------------------------------------------------------------------


def _perform(**config):
    """Run perform() on a science L0 with both catalogs mocked at distinct RAs.

    Gaia answers at RA 10 deg, SIMBAD at 20 deg, so the merged row shows which query
    landed on which source attribute. Returns (AstroQuery, CATALOG_RECORD table).
    """
    l0 = _l0_for_query(GAIAID="Gaia DR3 1234567890", OBJECT="tau Cet")
    aq = AstroQuery(l0, config or None)
    with (
        _patch_gaia(_gaia_job(_gaia_table({"ra": 10.0}))),
        _patch_simbad(_simbad_instance(_simbad_table({"ra": 20.0, "ids": _IDS}))),
    ):
        aq.perform()
    return aq, aq.l0_obj.data["CATALOG_RECORD"]


# SIMBAD identifiers agreeing with _perform's GAIAID.
_IDS = "HD  10700|* tau Cet|Gaia DR3 1234567890"


def _kpf_drp_row(record):
    """The canonical merged row of a CATALOG_RECORD table."""
    return record[record["source"] == "kpf-drp"][0]


class TestPerform:
    """perform() wires each query onto its own source attribute, then merges.

    TestMergeCatalogRecords presets those attributes by hand and so cannot see the
    wiring; these drive the entry point end to end, catching a query routed to the
    wrong attribute -- which would invert the gaia > simbad precedence silently.
    """

    def test_gaia_named_object_takes_high_tier_name(self):
        # A bare Gaia OBJECT/GAIAID: Gaia adds the release, then the low-tier Gaia
        # OBJECT takes SIMBAD's catalogued name; ALIASES lists each name once.
        l0 = _l0_for_query(
            GAIAID="Gaia 2452378776434477184", OBJECT="Gaia 2452378776434477184"
        )
        aq = AstroQuery(l0)
        client = MagicMock()
        client.run_sync.side_effect = _release_aware_run_sync(present={"DR3"})
        with (
            patch("kpfpipe.modules.astro_query.gaia_client", return_value=client),
            _patch_simbad(_simbad_instance(_simbad_table())),
        ):
            aq.perform()
        primary = l0.headers["PRIMARY"]
        designation = "Gaia DR3 2452378776434477184"
        assert (primary["GAIAID"], primary["OBJECT"]) == (designation, "HD 10700")
        assert primary["ALIASES"] == f"HD 10700; {designation}; tau Cet"

    def test_each_query_lands_on_its_own_attribute(self):
        aq, _ = _perform()
        assert (
            aq._gaia["object"] == "Gaia DR3 1234567890"
        )  # from GAIAID, via query_gaia
        # Cited catalog's designation, via query_simbad
        assert aq._simbad["object"] == "Gaia DR3 1234567890"
        assert aq._gaia["ra"] == _ra_str(10.0)
        assert aq._simbad["ra"] == _ra_str(20.0)

    def test_merged_row_takes_gaia_over_simbad(self):
        _, record = _perform()
        row = _kpf_drp_row(record)
        assert row["ra"] == _ra_str(10.0)
        assert row["radec_src"] == "gaia"
        assert row["plx_src"] == "gaia"
        assert row["rv_src"] == "gaia"
        assert row["object"] == "Gaia DR3 1234567890"

    def test_row_written_for_every_resolved_source(self):
        # A source's row is its presence record. No wmko row here: the L0 carries
        # no TARG*.
        _, record = _perform()
        assert set(record["source"]) == {"gaia", "simbad", "kpf-drp"}

    def test_gated_off_source_gets_no_row(self):
        _, record = _perform(do_gaia_query=False)
        assert "gaia" not in record["source"]
        assert "simbad" in record["source"]

    def test_aliases_written_to_primary(self):
        # OBJECT 'tau Cet' is high tier and kept, listed first; the common name
        # 'tau Cet' de-duplicates away. (Order is covered by TestResolveIdentity;
        # the short fixture id is no catalogued Gaia id.)
        aq, _ = _perform()
        primary = aq.l0_obj.headers["PRIMARY"]
        assert primary["OBJECT"] == "tau Cet"
        aliases = primary["ALIASES"].split("; ")
        assert aliases[0] == "tau Cet"
        assert sorted(aliases) == ["Gaia DR3 1234567890", "HD 10700", "tau Cet"]

    def test_gaia_off_falls_through_to_simbad(self):
        # The provenance must follow the merge down rather than stay "gaia".
        aq, record = _perform(do_gaia_query=False)
        assert aq._gaia is None
        row = _kpf_drp_row(record)
        assert row["ra"] == _ra_str(20.0)
        assert row["radec_src"] == "simbad"
        assert row["object"] == "Gaia DR3 1234567890"

    @staticmethod
    def _perform_simbad_gaiaid(gaiaid, gaia_table, caplog, **config):
        """perform() on 'tau Cet' with the given PRIMARY GAIAID; SIMBAD lists Gaia DR3
        2452378776434477184. Returns (AstroQuery, Gaia run_sync mock)."""
        aq = AstroQuery(_l0_for_query(GAIAID=gaiaid, OBJECT="tau Cet"), config or None)
        with (
            _patch_gaia(_gaia_job(gaia_table)) as run_sync,
            _patch_simbad(_simbad_instance(_simbad_table({"ra": 20.0}))),
            caplog.at_level(logging.WARNING),
        ):
            aq.perform()
        return aq, run_sync

    def test_blank_gaiaid_filled_from_simbad(self, caplog):
        aq, run_sync = self._perform_simbad_gaiaid(
            None, _gaia_table({"ra": 10.0}), caplog
        )
        assert aq.l0_obj.headers["PRIMARY"]["GAIAID"] == "Gaia DR3 2452378776434477184"
        assert "filled from SIMBAD" in caplog.text
        assert "2452378776434477184" in run_sync.call_args.args[0]
        row = _kpf_drp_row(aq.l0_obj.data["CATALOG_RECORD"])
        assert (row["radec_src"], row["ra"]) == ("gaia", _ra_str(10.0))

    def test_header_gaiaid_contradicted_by_simbad_raises(self, caplog):
        # SIMBAD's match for OBJECT lists another DR3 id: the cards name different
        # stars. Never replaced, never accepted.
        with pytest.raises(HeaderError, match="name different stars") as err:
            self._perform_simbad_gaiaid("Gaia DR3 1234567890", Table(), caplog)
        assert "L0 header must be fixed:" in str(err.value)
        assert "filled from SIMBAD" not in caplog.text

    @pytest.mark.parametrize(
        "gaiaid, config, expected",
        [
            # wmko may anchor the position, but never for an unidentified target.
            (
                "Gaia DR3 1234567890",
                {"astrometry_priority": ("gaia", "simbad", "wmko")},
                "Gaia and SIMBAD resolved no source for OBJECT 'tau Cet', "
                "GAIAID 'Gaia DR3 1234567890'",
            ),
            (
                None,
                {},
                "Gaia and SIMBAD resolved no source for OBJECT 'tau Cet', GAIAID blank",
            ),
            (
                "Gaia DR3 1234567890",
                {"do_gaia_query": False},
                "SIMBAD resolved no source",
            ),
        ],
    )
    def test_unidentified_target_raises(self, gaiaid, config, expected):
        aq = AstroQuery(_l0_for_query(GAIAID=gaiaid, OBJECT="tau Cet"), config or None)
        with (
            _patch_gaia(_gaia_job(Table())),
            _patch_simbad(_simbad_instance(Table())),
            pytest.raises(ValueError, match="target identity unresolved") as err,
        ):
            aq.perform()
        assert expected in str(err.value)
        assert "L0 header must be fixed:" in str(err.value)

    def test_gaia_off_fills_gaiaid_without_query(self, caplog):
        aq, run_sync = self._perform_simbad_gaiaid(
            None, _gaia_table(), caplog, do_gaia_query=False
        )
        assert aq.l0_obj.headers["PRIMARY"]["GAIAID"] == "Gaia DR3 2452378776434477184"
        assert run_sync.call_count == 0


class TestResolveIdentity:
    """perform() settles OBJECT/ALIASES by one rule (target.resolve_identity): a
    non-numeric header OBJECT is kept; Unknown or a numeric id takes its mission's
    successor (TIC -> TOI); a contradicting SIMBAD match raises."""

    # SIMBAD identifiers of TOI-2473 = HD 287319 (DR2 and DR3 source_ids differ).
    _TOI_IDS = (
        "TIC 399868187|Gaia DR3 3291551614399361152|BD+10   678|HD 287319|"
        "Gaia DR2 3291551614397199360|TOI-2473"
    )

    @staticmethod
    def _perform(obj, gaiaid, simbad_table, caplog=None):
        aq = AstroQuery(_l0_for_query(OBJECT=obj, GAIAID=gaiaid))
        inst = _simbad_instance(simbad_table)
        with (
            _patch_gaia(_gaia_job(_gaia_table())),
            _patch_simbad(inst),
            caplog.at_level(logging.INFO) if caplog else nullcontext(),
        ):
            aq.perform()
        return aq.l0_obj.headers["PRIMARY"], inst

    def test_unknown_resolved_via_gaiaid(self, caplog):
        # Alias cards '201092'/'61 Cyg B': SIMBAD is asked for the GAIAID, and its
        # identifiers name the star.
        gaiaid = "Gaia DR3 1872046574983497216"
        ids = (
            "TIC 165602023|Gaia DR3 1872046574983497216|*  61 Cyg B|HD 201092|"
            "HIP 104217|Gaia DR2 1872046574983497216"
        )
        table = _simbad_table({"main_id": "*  61 Cyg B", "ids": ids})
        primary, inst = self._perform("Unknown", gaiaid, table, caplog)
        assert inst.query_object.call_args.args[0] == gaiaid
        assert primary["OBJECT"] == "HD 201092"
        assert primary["ALIASES"] == (
            "HD 201092; HIP 104217; TIC 165602023; Gaia DR2 1872046574983497216; "
            "Gaia DR3 1872046574983497216; 61 Cyg B"
        )
        assert "OBJECT 'HD 201092' (resolved from 'Unknown'); ALIASES" in caplog.text

    def test_high_tier_object_kept(self):
        # TOI-2473 is not renamed HD 287319; the DR2 id differing from the DR3
        # GAIAID is no conflict.
        table = _simbad_table({"main_id": "HD 287319", "ids": self._TOI_IDS})
        primary, _ = self._perform("TOI-2473", "Gaia DR3 3291551614399361152", table)
        assert primary["OBJECT"] == "TOI-2473"
        assert primary["ALIASES"].startswith(
            "TOI-2473; HD 287319; BD+10 678; TIC 399868187; Gaia DR2 "
        )

    def test_low_tier_object_takes_high_tier_name(self):
        table = _simbad_table({"main_id": "HD 287319", "ids": self._TOI_IDS})
        primary, _ = self._perform(
            "TIC 399868187", "Gaia DR3 3291551614399361152", table
        )
        assert primary["OBJECT"] == "TOI-2473"  # a TIC's mission successor

    def test_object_spelling_variant_takes_simbad_spelling(self):
        # 'V1298Tau' differs from SIMBAD's 'V* V1298 Tau' only by spacing.
        gaiaid = "Gaia DR3 51886335968692480"
        ids = f"V* V1298 Tau|{gaiaid}|TIC 15756231"
        table = _simbad_table({"main_id": "V* V1298 Tau", "ids": ids})
        primary, _ = self._perform("V1298Tau", gaiaid, table)
        assert primary["OBJECT"] == "V1298 Tau"

    def test_object_missing_from_simbad_match_raises(self):
        table = _simbad_table({"main_id": "HD 287319", "ids": self._TOI_IDS})
        with pytest.raises(HeaderError, match="not an identifier") as err:
            self._perform("HD 10700", "Gaia DR3 3291551614399361152", table)
        assert "L0 header must be fixed:" in str(err.value)

    def test_unknown_without_simbad_takes_gaia_designation(self, caplog):
        gaiaid = "Gaia DR3 1872046574983497216"
        primary, _ = self._perform("Unknown", gaiaid, Table(), caplog)
        assert (primary["OBJECT"], primary["ALIASES"]) == (gaiaid, gaiaid)
        assert "SIMBAD gave no identifiers" in caplog.text


# One-row Horizons ephemeris; rates of 3600 / -1800 arcsec/hr make the arcsec/yr
# conversion visible (x 8766 hr per Julian yr).
_HORIZONS_VALUES = {
    "RA": 180.0,
    "DEC": 10.0,
    "RA_rate": 3600.0,
    "DEC_rate": -1800.0,
    "delta": 5.0,
}


def _horizons_table(values=None, units=None):
    vals = {**_HORIZONS_VALUES, **(values or {})}
    return _units_table(AstroQuery._HORIZONS_UNITS, vals, units)


def _patch_horizons(table_or_exc):
    """Patch the Horizons client factory; its ephemerides() -> table (or raises)."""
    client = MagicMock()
    if isinstance(table_or_exc, Exception):
        client.ephemerides.side_effect = table_or_exc
    else:
        client.ephemerides.return_value = table_or_exc
    return patch("kpfpipe.modules.astro_query.horizons_client", return_value=client)


def _l0_solar_system(obj="Jupiter", jd=2461000.5, exptime=600.0, **cards):
    l0 = _l0_for_query(OBJECT=obj, **cards)
    l0.headers["PRIMARY"]["JD_UTC"] = jd
    l0.headers["PRIMARY"]["EXPTIME"] = exptime
    return l0


class TestHorizonsQuery:
    """A solar-system OBJECT is resolved by JPL Horizons in place of Gaia/SIMBAD."""

    def test_record_conversions(self):
        aq = AstroQuery(_l0_solar_system())
        with _patch_horizons(_horizons_table()) as factory:
            record = aq.query_horizons()
        # Looked up by Horizons ID from Mauna Kea at the exposure midpoint.
        target_id, location, epoch, _ = factory.call_args.args
        assert (target_id, location) == ("599", "568")
        midpoint = Time(2461000.5, format="jd") + 300 * u.s
        assert epoch == pytest.approx(midpoint.jd, abs=1e-9)
        assert record["epoch"] == pytest.approx(midpoint.jyear)
        assert (record["object"], record["catalog"]) == ("599", "JPL Horizons")
        expected = AstroQuery._sexagesimal_radec(SkyCoord(180.0, 10.0, unit=u.deg))
        assert (record["ra"], record["dec"]) == expected
        assert record["pmra"] == pytest.approx(3600.0 * 8766.0)
        assert record["pmdec"] == pytest.approx(-1800.0 * 8766.0)
        # A 1 au baseline seen from 5 au: 206264.806 arcsec / 5, in mas.
        assert record["parallax"] == pytest.approx(206264806.2 / 5.0)
        assert record["rv"] is None and record["color"] is None

    def test_perform_skips_gaia_and_simbad(self):
        l0 = _l0_solar_system()
        with (
            _patch_horizons(_horizons_table()),
            _patch_gaia(_gaia_job(_gaia_table())) as gaia,
            _patch_simbad(_simbad_instance(_simbad_table())) as simbad,
        ):
            AstroQuery(l0).perform()
        assert gaia.call_count == 0 and simbad.call_count == 0
        row = _kpf_drp_row(l0.data["CATALOG_RECORD"])
        assert row["radec_src"] == "horizons"
        primary = l0.headers["PRIMARY"]
        assert (primary["CID3"], primary["CSRC3"]) == ("599", "JPL Horizons")
        assert primary["CRA3"] == _ra_str(180.0)
        assert primary["OBJECT"] == "Jupiter"

    def test_failed_query_is_fail_soft_then_unresolved(self, caplog):
        aq = AstroQuery(_l0_solar_system())
        with _patch_horizons(ValueError("Ambiguous target name")):
            with caplog.at_level(logging.WARNING):
                assert aq.query_horizons() is None
            assert "JPL Horizons query failed" in caplog.text
            with pytest.raises(ValueError, match="JPL Horizons resolved no source"):
                aq.perform()

    def test_unit_drift_raises(self):
        aq = AstroQuery(_l0_solar_system())
        with _patch_horizons(_horizons_table(units={"delta": u.km})):
            with pytest.raises(ValueError, match="unexpected column units"):
                aq.query_horizons()

    def test_toggle_off_skips_query(self):
        aq = AstroQuery(_l0_solar_system(), {"do_horizons_query": False})
        with _patch_horizons(_horizons_table()) as factory:
            with pytest.raises(ValueError, match="no complete astrometric solution"):
                aq.perform()
        assert factory.call_count == 0

    def test_header_color_not_borrowed(self):
        # A G-J color off the DCS cards is stellar; a planet's stays missing.
        l0 = _l0_solar_system(
            TARGRA="12:00:00.00",
            TARGDEC="+10:00:00.0",
            TARGFRAM="FK5",
            TARGEPOC=2025.8,
            GAIAMAG=5.0,
            **{"2MASSMAG": 4.0},
        )
        with _patch_horizons(_horizons_table()):
            AstroQuery(l0).perform()
        table = l0.data["CATALOG_RECORD"]
        assert table[table["source"] == "wmko"][0]["color"] == pytest.approx(1.0)
        row = _kpf_drp_row(table)
        assert np.isnan(row["color"]) and not str(row["color_name"]).strip()
        assert "CCLR3" not in l0.headers["PRIMARY"]

    def test_header_rv_not_borrowed(self, caplog):
        # A planet needs no systemic rv: TARGRADV is not used, CRV3 stays blank,
        # and the C*# cards are single-source (no mixed-sources warning).
        l0 = _l0_solar_system(TARGRADV=0.0)
        with _patch_horizons(_horizons_table()):
            with caplog.at_level(logging.WARNING):
                AstroQuery(l0).perform()
        row = _kpf_drp_row(l0.data["CATALOG_RECORD"])
        assert np.isnan(row["rv"]) and not str(row["rv_src"]).strip()
        assert not str(l0.headers["PRIMARY"].get("CRV3", "")).strip()
        assert "TARGRADV" not in caplog.text
        assert "mixed sources" not in caplog.text

    def test_no_simbad_identifier_warning(self, caplog):
        l0 = _l0_solar_system()
        with _patch_horizons(_horizons_table()):
            with caplog.at_level(logging.WARNING):
                AstroQuery(l0).perform()
        assert "SIMBAD gave no identifiers" not in caplog.text
        assert l0.headers["PRIMARY"]["ALIASES"] == "Jupiter"

    def test_l0_qc_flags(self):
        # A Jupiter-like Horizons row passes ASTROMOK; COLOROK does not apply.
        from kpfpipe.quality_control.qc_flags.level0 import QCL0

        l0 = _l0_solar_system()
        with _patch_horizons(_horizons_table({"RA_rate": 20.0, "DEC_rate": -8.0})):
            AstroQuery(l0).perform()
        results = QCL0(l0).run()
        assert results["ASTROMOK"][0] is True
        assert "COLOROK" not in results


class TestConstructor:
    """The constructor's guards. Every other AstroQuery(...) in the suite
    passes OBSTYPE='Target' and either None or a plain dict, so none of these is
    otherwise exercised."""

    @staticmethod
    def _l0(obstype):
        l0 = KPF0()
        if obstype is not None:
            l0.headers["PRIMARY"]["OBSTYPE"] = obstype
        return l0

    @pytest.mark.parametrize("obstype", ["Bias", "Flat", None])
    def test_non_object_frame_raises(self, obstype):
        # The class docstring's headline promise. Without it, AstroQuery would
        # fire SIMBAD lookups on OBJECT='bias' and write a bogus CATALOG_RECORD
        # that to_kpf1 copies onto the EPRV C*# cards.
        with pytest.raises(ValueError, match="science frames"):
            AstroQuery(self._l0(obstype))

    def test_socal_frame_raises(self):
        l0 = self._l0("Target")
        l0.headers["PRIMARY"]["ISSOLAR"] = True
        with pytest.raises(ValueError, match="SoCal"):
            AstroQuery(l0)

    def test_unsupported_config_type_raises(self):
        with pytest.raises(TypeError, match="must be None, dict, or ConfigHandler"):
            AstroQuery(self._l0("Target"), config=["gaia"])

    def test_config_handler_branch_is_read(self, tmp_path):
        # How the recipes construct it. A section-name typo here would be
        # invisible: params would come back empty and _DEFAULTS would silently
        # turn both catalog queries back on.
        cfg = tmp_path / "aq.toml"
        cfg.write_text(
            "[DATA_DIRS]\n[TRACES]\n[MODULE_ASTRO_QUERY]\ndo_gaia_query = false\n"
        )
        aq = AstroQuery(self._l0("Target"), config=ConfigHandler(str(cfg)))
        assert aq.do_gaia_query is False


# ------------------------------------------------------------------
# Import purity -- witnessed in a child, because this suite's conftest hides it
# ------------------------------------------------------------------

# Records every outbound connect, imports the module, and reports what it saw.
# Run through `python -c` so no conftest loads: the suite's own network guard
# would mask exactly the connection this is here to detect.
_IMPORT_PROBE = """
import json, socket, sys
hits = []
socket.socket.connect = lambda self, addr, *a, **k: hits.append(addr)
socket.create_connection = lambda addr, *a, **k: hits.append(addr)
import kpfpipe.modules.astro_query
print(json.dumps({
    "hits": [str(h) for h in hits],
    "astroquery": sorted(m for m in sys.modules if m.startswith("astroquery")),
}))
"""


class TestImportPurity:
    def test_importing_astro_query_touches_no_network(self):
        """Importing the module must not open a connection, nor pull in a client.

        ``astroquery.gaia`` builds a module-level ``GaiaClass()`` whose constructor
        fetches ESA status messages, so importing it *is* a connection. Both client
        imports are deferred into the factories to keep that off every code path
        that never queries a catalog -- this is the only thing that witnesses it.
        """
        proc = subprocess.run(
            [sys.executable, "-c", _IMPORT_PROBE],
            cwd=REPO_ROOT,
            env={"PYTHONPATH": REPO_ROOT, "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        seen = json.loads(proc.stdout.strip().splitlines()[-1])

        assert seen["hits"] == []
        # astroquery.exceptions is imported at module scope for the retryable-error
        # tuple; it is cheap and reaches nothing. The two client modules are what
        # must stay out, so name them rather than the package.
        assert "astroquery.gaia" not in seen["astroquery"]
        assert "astroquery.simbad" not in seen["astroquery"]
