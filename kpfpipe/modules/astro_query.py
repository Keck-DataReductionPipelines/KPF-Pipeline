"""
KPF AstroQuery module.

Consolidates the pipeline's external astronomical-catalog lookups into a single
L0-stage module. Given a raw L0 science frame, it resolves the target's astrometry
from Gaia (by GAIAID, whose release prefix picks the data release queried) and SIMBAD
(by OBJECT), builds the telescope-native ``wmko`` row from the L0 PRIMARY ``TARG*``
astrometry (no query), merges them into a canonical ``kpf-drp`` row, and writes all
four rows to the L0 ``CATALOG_RECORD`` extension with the
``WMKOCR``/``GAIACR``/``SIMBADCR`` presence flags.

CATALOG_RECORD bridges an ordering problem: the results ultimately belong on the EPRV
PRIMARY catalog keywords (``C*#``), but that conversion does not happen until
``KPF0.to_kpf1()`` downstream, which overlays the merged record onto those cards.
AstroQuery never modifies the L0 PRIMARY header. All rows share one schema in the EPRV
C*# PRIMARY format (see ``_CATALOG_COLUMNS``), so ``to_kpf1`` can copy cells straight
onto the catalog cards.
"""

import logging

import astropy.units as u
import numpy as np
from astropy.coordinates import FK5, ICRS, Angle, SkyCoord
from astropy.table import Table
from astropy.time import Time
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad

from kpfpipe import DEFAULTS
from kpfpipe.utils.astro import compute_redshift
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.network import retry_request

logger = logging.getLogger(__name__)

_DEFAULTS = {
    **DEFAULTS,
    "do_gaia_query": True,
    "do_simbad_query": True,
    "astrometry_priority": ("gaia", "simbad"),
}

# Sources that can supply a CATALOG_RECORD row, highest catalog priority first.
_SOURCES = ("gaia", "simbad", "wmko")

# The astrometric block the merge takes whole from one source: proper motion is
# meaningless without the parallax it was measured against, and both need the
# frame/epoch that qualify the coordinates. equinox and rv are handled separately.
_ASTROMETRY = ("ra", "dec", "pmra", "pmdec", "parallax", "epoch", "frame")

# Expected result units, verified by _verify_units so a silent upstream schema
# change fails loudly.
_GAIA_UNITS = {
    "ra": u.deg,
    "dec": u.deg,
    "pmra": u.mas / u.yr,
    "pmdec": u.mas / u.yr,
    "parallax": u.mas,
    "radial_velocity": u.km / u.s,
    "ref_epoch": u.yr,
    "phot_bp_mean_mag": u.mag,
    "phot_rp_mean_mag": u.mag,
}
_SIMBAD_UNITS = {
    "ra": u.deg,
    "dec": u.deg,
    "pmra": u.mas / u.yr,
    "pmdec": u.mas / u.yr,
    "plx_value": u.mas,
    "rvz_radvel": u.km / u.s,
    "B": None,
    "V": None,
}

# Queryable Gaia release -> its gaia_source table, newest last. DR1 and EDR3 are
# excluded (no radial_velocity or BP/RP photometry, and superseded, respectively).
# A release absent here is neither probed nor accepted.
_GAIA_TABLES = {
    "DR2": "gaiadr2.gaia_source",
    "DR3": "gaiadr3.gaia_source",
}

# CATALOG_RECORD write schema (AstroQuery is the sole populator): one row per resolved
# source (wmko/gaia/simbad) plus the merged 'kpf-drp' row. radec_src/plx_src/rv_src
# name the source each value block came from -- its own for a source row, the winner
# (or "" if none) for the merged row. Values are in the EPRV C*# PRIMARY format: RA/Dec
# sexagesimal strings (ICRS), PM [arcsec/yr] (RA incl. cos Dec), parallax [mas], rv
# [km/s], z the redshift derived from rv, epoch/equinox [Julian yr]. color is a color
# index and color_name labels it (Gaia "Gaia BP-RP", SIMBAD "B-V", WMKO "G-J"), both
# blank when a magnitude is missing. Missing floats -> NaN, strings -> "".
_CATALOG_COLUMNS = (
    "source",
    "object",
    "radec_src",
    "plx_src",
    "rv_src",
    "ra",
    "dec",
    "pmra",
    "pmdec",
    "parallax",
    "rv",
    "z",
    "frame",
    "epoch",
    "equinox",
    "color",
    "color_name",
)
_CATALOG_STR_COLUMNS = frozenset(
    {
        "source",
        "object",
        "radec_src",
        "plx_src",
        "rv_src",
        "frame",
        "ra",
        "dec",
        "color_name",
    }
)

# RA / DEC are sexagesimal strings, so not subject to astropy unit conversion
_CATALOG_UNITS = {
    "pmra": u.arcsec / u.yr,
    "pmdec": u.arcsec / u.yr,
    "parallax": u.mas,
    "rv": u.km / u.s,
    "z": u.dimensionless_unscaled,
    "epoch": u.yr,
    "equinox": u.yr,
    "color": u.mag,
}
# Per-source presence flag written to the CATALOG_RECORD header (int 0/1). DiagL0
# keeps its own local mirror rather than importing this schema.
_CATALOG_FLAGS = {"gaia": "GAIACR", "simbad": "SIMBADCR", "wmko": "WMKOCR"}


class AstroQuery:
    """
    Resolve target astrometry from external catalogs and write it to an L0.

    Runs the two external catalog queries (Gaia by GAIAID, SIMBAD by OBJECT), builds
    the native ``wmko`` row from the L0 PRIMARY ``TARG*`` astrometry, and writes all
    three rows to the L0 ``CATALOG_RECORD`` extension for downstream use (EPRV ``C*#``
    catalog keywords, DiagL0 pointing offsets, BarycentricCorrection). Only science
    frames are supported: the constructor raises on a non-``Object`` IMTYPE. Fail-soft
    otherwise -- a missing GAIAID/OBJECT or a failed lookup yields a ``None`` record.

    Parameters
    ----------
    l0_obj : KPF0
        Raw L0 science frame (IMTYPE ``Object``). Its PRIMARY header (IMTYPE, GAIAID,
        OBJECT, TARG*) is read but never modified.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: do_gaia_query, do_simbad_query,
        astrometry_priority.
    """

    def __init__(self, l0_obj, config=None):
        self.l0_obj = l0_obj

        imtype = l0_obj.headers["PRIMARY"].get("IMTYPE")
        if str(imtype).strip().lower() != "object":
            raise ValueError(
                f"AstroQuery runs only on science frames (IMTYPE 'Object'); got "
                f"IMTYPE={imtype!r}. It must not be called on a calibration frame."
            )

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS", "TRACES", "MODULE_ASTRO_QUERY"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in _DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self._validate_priority()

        self._wmko = None  # native WMKO record; set by read_wmko_header()
        self._gaia = None  # Gaia DR3 record; set by query_gaia()
        self._simbad = None  # SIMBAD record; set by query_simbad()
        self._canonical = None  # merged kpf-drp record; set by merge_catalog_records()
        self._info = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_priority(self):
        """Reject an astrometry_priority naming an unknown or no source.

        Caught at construction rather than at the merge, where an unknown name would
        read as "that source had no record" and silently demote the position.

        Raises
        ------
        ValueError
            ``astrometry_priority`` is empty or names a source outside ``_SOURCES``.
        """
        self.astrometry_priority = tuple(self.astrometry_priority)
        unknown = [s for s in self.astrometry_priority if s not in _SOURCES]
        if unknown or not self.astrometry_priority:
            raise ValueError(
                f"astrometry_priority={list(self.astrometry_priority)} must be a "
                f"non-empty ordered subset of {list(_SOURCES)}"
                + (f"; unknown source(s) {unknown}" if unknown else "")
            )

    def _gaia_source(self):
        """``(release, source_id)`` from L0 GAIAID, or None if absent/unusable.

        GAIAID names the release its id belongs to (e.g. 'DR3 12345'), which selects
        the queried table -- a source_id denotes different stars across releases, so the
        prefix is honored rather than discarded. ``release`` is None for a bare id, left
        for ``_resolve_gaia_release``. None when GAIAID is absent or blank, the trailing
        token is not all digits, or the release is not in ``_GAIA_TABLES``.
        """
        raw = self.l0_obj.headers["PRIMARY"].get("GAIAID")
        if raw is None:
            return None
        tokens = str(raw).strip().split()
        if not tokens or not tokens[-1].isdigit():
            return None
        source_id = tokens[-1]
        if len(tokens) == 1:
            return None, source_id
        release = tokens[-2].upper()
        return (release, source_id) if release in _GAIA_TABLES else None

    def _resolve_gaia_release(self, source_id):
        """The newest Gaia release whose gaia_source contains ``source_id``.

        For a bare GAIAID only: rather than assume a release, probe the archive and take
        the newest one holding the id, since the same source_id denotes different stars
        in different releases. Every release in ``_GAIA_TABLES`` is published, so a
        failed probe means something is genuinely wrong: it warns and falls through to
        an older release, which may hold a different star under that id. An id no
        release holds raises rather than being mistaken for a resolved one.

        Parameters
        ----------
        source_id : str
            Digit-only Gaia source_id, with no release prefix.

        Returns
        -------
        str
            The release key into ``_GAIA_TABLES``.

        Raises
        ------
        ValueError
            No queryable release contains ``source_id``.
        """
        for release in sorted(_GAIA_TABLES, reverse=True):
            query = (
                f"SELECT source_id FROM {_GAIA_TABLES[release]} "
                f"WHERE source_id = {source_id}"
            )
            try:
                found = retry_request(
                    lambda q=query: Gaia.launch_job(q).get_results(), f"Gaia {release}"
                )
            except Exception as e:
                logger.warning(
                    "could not probe Gaia %s for source_id %s (%s: %s); falling "
                    "through to an older release, which may denote a different star",
                    release,
                    source_id,
                    type(e).__name__,
                    e,
                )
                continue
            if len(found):
                logger.warning("resolved bare GAIAID %s to Gaia %s", source_id, release)
                return release
        raise ValueError(
            f"GAIAID {source_id} carries no data release and no queryable Gaia release "
            f"({', '.join(sorted(_GAIA_TABLES))}) contains that source_id; refusing to "
            "guess which release it belongs to."
        )

    def _simbad_resolvable_name(self):
        """SIMBAD-resolvable name from L0 PRIMARY OBJECT, or None if absent.

        KPF OBJECT for standard stars is a bare HD number (e.g. '10700') that
        SIMBAD resolves only with an 'HD ' prefix; named targets pass through.
        """
        obj = self.l0_obj.headers["PRIMARY"].get("OBJECT")
        if obj is None:
            return None
        obj = str(obj).strip()
        if not obj:
            return None
        return f"HD {obj}" if obj.isdigit() else obj

    @staticmethod
    def _scalar(value):
        """Coerce a catalog cell to a plain float, or None if missing/masked/NaN.

        astroquery returns masked columns; a star without a measured RV or
        parallax comes back masked, which must become a clean None in the record.
        """
        if value is None or value is np.ma.masked:
            return None
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(f) else f

    @staticmethod
    def _redshift(rv_kms):
        """Relativistic redshift z for a catalog rv [km/s], or None if rv is missing.

        Dimensionless, and carried on the record so ``to_kpf1`` can overlay it onto the
        EPRV ``CZ#`` card without a consumer recomputing it.
        """
        if rv_kms is None:
            return None
        return float(compute_redshift(rv_kms * u.km / u.s))

    @staticmethod
    def _color(blue_mag, red_mag, name):
        """The (color, color_name) pair for a bluer-minus-redder magnitude difference.

        ``(None, None)`` when either magnitude is missing -- a color index needs both,
        and dropping the label with it avoids a blank color carrying an orphan name.
        """
        if blue_mag is None or red_mag is None:
            return None, None
        return blue_mag - red_mag, name

    @staticmethod
    def _sexagesimal_radec(coord):
        """ICRS SkyCoord -> the EPRV C*# sexagesimal (ra, dec) strings.

        The single formatter for the colon-separated 'h:m:s' cards (RA hour-angle, Dec
        deg), so a precision change lands in one place. astropy's hmsdms already pads
        and signs both axes; this only splits them.
        """
        return tuple(coord.to_string("hmsdms", sep=":", precision=4).split())

    @staticmethod
    def _verify_units(table, expected, source):
        """Verify a query result's column units before its values are trusted.

        Raises ``ValueError`` if an expected column is missing or carries a unit other
        than the record schema assumes, so a silent catalog schema change fails loudly.
        """
        mismatched = {}
        for col, want in expected.items():
            if col not in table.colnames:
                mismatched[col] = "MISSING"
            elif table[col].unit != want:
                mismatched[col] = table[col].unit
        if mismatched:
            raise ValueError(
                f"{source} returned unexpected column units {mismatched}; expected "
                f"{expected}. The catalog schema may have changed; AstroQuery's unit "
                "assumptions must be revalidated before use."
            )

    def _write_catalog_record(self, source, record):
        """Upsert one source's row into the L0 CATALOG_RECORD extension.

        The sole writer for CATALOG_RECORD. A ``None`` ``record`` drops the source's
        row, otherwise it is (re)written; other sources' rows are preserved (upsert).
        Provenance labels (``radec_src``/``plx_src``/``rv_src``) default to ``source``,
        since a source row's values are its own; the merged row supplies them
        explicitly. Missing floats become NaN, strings "".
        """
        l0 = self.l0_obj
        table = l0.data["CATALOG_RECORD"]
        rows = {}
        if table.colnames:
            for row in table:
                rows[str(row["source"])] = {
                    name: row[name] for name in _CATALOG_COLUMNS
                }
        if record is None:
            rows.pop(source, None)
        else:
            row = {
                "source": source,
                "radec_src": source,
                "plx_src": source,
                "rv_src": source,
                **record,
            }
            row["z"] = self._redshift(row["rv"])
            rows[source] = row

        ordered = list(rows.values())
        new_table = Table()
        for name in _CATALOG_COLUMNS:
            if name in _CATALOG_STR_COLUMNS:
                new_table[name] = np.array(
                    ["" if r.get(name) is None else r.get(name) for r in ordered],
                    dtype=str,
                )
            else:
                new_table[name] = np.array(
                    [np.nan if r.get(name) is None else r.get(name) for r in ordered],
                    dtype=float,
                )
                new_table[name].unit = _CATALOG_UNITS[name]
        l0.set_data("CATALOG_RECORD", new_table)

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def query_gaia(self):
        """Query Gaia for the target's ICRS astrometry, or None (fail-soft).

        The release named in GAIAID selects the queried table (``_GAIA_TABLES``), so a
        DR2 id is never looked up among DR3 source_ids; a bare id is identified against
        the archive by ``_resolve_gaia_release``. Returns None (warned) when GAIAID
        yields no usable release/source_id, when the lookup fails after the
        ``retry_request`` retries, or when the source is not found. Raises ValueError
        if a bare id matches no queryable release, or if the result's column units
        differ from the schema shared by the supported releases (deg, mas/yr, mas,
        km/s, Julian yr, mag; ICRS). Gated in perform by ``do_gaia_query``.
        """
        source = self._gaia_source()
        if source is None:
            logger.warning(
                "no usable GAIAID on L0 PRIMARY (%r); Gaia astrometry unavailable "
                "(queryable releases: %s)",
                self.l0_obj.headers["PRIMARY"].get("GAIAID"),
                ", ".join(sorted(_GAIA_TABLES)),
            )
            return None
        release, gaia_id = source
        if release is None:
            logger.warning(
                "GAIAID %r carries no data release; querying Gaia to identify it",
                self.l0_obj.headers["PRIMARY"].get("GAIAID"),
            )
            release = self._resolve_gaia_release(gaia_id)
        query = f"""
        SELECT ra, dec, pmra, pmdec, parallax, radial_velocity, ref_epoch,
               phot_bp_mean_mag, phot_rp_mean_mag
        FROM {_GAIA_TABLES[release]}
        WHERE source_id = {gaia_id}
        """
        logger.info("querying Gaia %s for source_id %s", release, gaia_id)
        try:
            results = retry_request(
                lambda: Gaia.launch_job(query).get_results(), f"Gaia {release}"
            )
        except Exception as e:
            logger.warning(
                "Gaia query failed (%s: %s); Gaia astrometry unavailable",
                type(e).__name__,
                e,
            )
            return None
        if len(results) == 0:
            logger.warning(
                "Gaia %s returned no match for source_id %s; Gaia astrometry "
                "unavailable",
                release,
                gaia_id,
            )
            return None
        self._verify_units(results, _GAIA_UNITS, f"Gaia {release}")
        row = results[0]
        ra, dec = self._scalar(row["ra"]), self._scalar(row["dec"])
        pmra, pmdec = self._scalar(row["pmra"]), self._scalar(row["pmdec"])
        # To the EPRV C*# format: deg -> sexagesimal; PM mas/yr -> arcsec/yr.
        ra_str, dec_str = self._sexagesimal_radec(
            SkyCoord(ra, dec, unit=u.deg, frame="icrs")
        )
        color, color_name = self._color(
            self._scalar(row["phot_bp_mean_mag"]),
            self._scalar(row["phot_rp_mean_mag"]),
            "Gaia BP-RP",
        )
        record = {
            "object": f"Gaia {release} {gaia_id}",
            "ra": ra_str,
            "dec": dec_str,
            "pmra": None if pmra is None else pmra / 1e3,
            "pmdec": None if pmdec is None else pmdec / 1e3,
            "parallax": self._scalar(row["parallax"]),
            "rv": self._scalar(row["radial_velocity"]),
            # frame/equinox are definitional, not queried: every Gaia release is ICRS
            # (Gaia-CRF) with no equinox, and 2000.0 satisfies EPRV's required CEQNX#.
            "frame": "icrs",
            "epoch": self._scalar(row["ref_epoch"]),
            "equinox": 2000.0,
            "color": color,
            "color_name": color_name,
        }
        logger.info("successfully built record for gaia")
        self._write_catalog_record("gaia", record)
        return record

    def query_simbad(self):
        """Query SIMBAD for the OBJECT's ICRS J2000 astrometry, or None (fail-soft).

        Returns None (warned, not raised) when L0 has no OBJECT name, or when the
        lookup fails after the ``retry_request`` retries or resolves nothing. Raises
        ValueError if the result's column units differ from the assumed schema: the
        astroquery 0.4.11 lowercase form (ra/dec deg, pmra/pmdec mas/yr, plx_value mas,
        rvz_radvel km/s), where the Johnson B/V magnitudes come back unlabeled (unit
        None) and form the B-V color. Gated in perform by ``do_simbad_query``.
        """
        name = self._simbad_resolvable_name()
        if name is None:
            logger.warning(
                "no OBJECT name on L0 PRIMARY; SIMBAD astrometry unavailable"
            )
            return None
        logger.info("querying SIMBAD for %r", name)
        try:
            simbad = Simbad()
            simbad.add_votable_fields(
                "pmra", "pmdec", "plx_value", "rvz_radvel", "B", "V"
            )
            result = retry_request(lambda: simbad.query_object(name), "SIMBAD")
        except Exception as e:
            logger.warning(
                "SIMBAD query failed (%s: %s); SIMBAD astrometry unavailable",
                type(e).__name__,
                e,
            )
            return None
        if result is None or len(result) == 0:
            logger.warning(
                "SIMBAD returned no match for %r; SIMBAD astrometry unavailable", name
            )
            return None
        self._verify_units(result, _SIMBAD_UNITS, "SIMBAD")
        row = result[0]
        ra, dec = self._scalar(row["ra"]), self._scalar(row["dec"])
        pmra, pmdec = self._scalar(row["pmra"]), self._scalar(row["pmdec"])
        # To the EPRV C*# format: deg -> sexagesimal; PM mas/yr -> arcsec/yr.
        ra_str, dec_str = self._sexagesimal_radec(
            SkyCoord(ra, dec, unit=u.deg, frame="icrs")
        )
        color, color_name = self._color(
            self._scalar(row["B"]), self._scalar(row["V"]), "B-V"
        )
        record = {
            "object": name,
            "ra": ra_str,
            "dec": dec_str,
            "pmra": None if pmra is None else pmra / 1e3,
            "pmdec": None if pmdec is None else pmdec / 1e3,
            "parallax": self._scalar(row["plx_value"]),
            "rv": self._scalar(row["rvz_radvel"]),
            # frame/epoch/equinox are definitional, not queried: astroquery's SIMBAD
            # returns basic ra/dec as ICRS J2000 with no per-object frame/epoch/equinox
            # to read, and 2000.0 satisfies EPRV's required CEQNX#.
            "frame": "icrs",
            "epoch": 2000.0,
            "equinox": 2000.0,
            "color": color,
            "color_name": color_name,
        }
        logger.info("successfully built record for simbad")
        self._write_catalog_record("simbad", record)
        return record

    def read_wmko_header(self):
        """Read the native WMKO/DCS astrometry from L0 PRIMARY TARG*, or None.

        The telescope-side counterpart to query_gaia/query_simbad: no query, just the
        raw TARG* pointing sanitized to the EPRV C*# format and rotated from its native
        FK5 (J2000) to ICRS, so all three sources share one frame. KPF pointing is
        always FK5, so a non-FK5 or absent TARGFRAM raises rather than being coerced --
        a wrong frame would corrupt the barycentric correction. Returns None (WMKOCR=0,
        warned) when TARGRA is absent or the TARG* astrometry cannot be parsed. Always
        run: TARGOFF needs this row even when ``astrometry_priority`` bars wmko from
        anchoring the position.
        """
        primary = self.l0_obj.headers["PRIMARY"]
        if primary.get("TARGRA") is None:
            return None
        targfram = str(primary.get("TARGFRAM") or "").strip()
        if targfram.upper() != "FK5":
            raise ValueError(
                f"unexpected TARGFRAM={primary.get('TARGFRAM')!r}; KPF pointing must "
                "be 'FK5' (J2000). Refusing to guess the frame, which would corrupt "
                "the barycentric correction."
            )
        try:
            ra = Angle(primary["TARGRA"], unit=u.hourangle)
            dec = Angle(primary["TARGDEC"], unit=u.deg)
            pmra, pmdec = (
                self._scalar(primary.get("TARGPMRA")),
                self._scalar(primary.get("TARGPMDC")),
            )
            equinox = self._scalar(primary.get("TARGEQUI"))

            # Rotate the native FK5 pointing to ICRS (no time propagation, so epoch is
            # unchanged); proper motion only when both components are present, since a
            # lone one is meaningless under rotation. TARGPMRA time-s/yr -> arcsec/yr.
            fk5 = FK5(
                equinox="J2000" if equinox is None else Time(equinox, format="jyear")
            )
            components = {"ra": ra, "dec": dec, "frame": fk5}
            has_pm = pmra is not None and pmdec is not None
            if has_pm:
                cosdec = np.cos(dec.radian)
                components["pm_ra_cosdec"] = pmra * 15.0 * cosdec * u.arcsec / u.yr
                components["pm_dec"] = pmdec * u.arcsec / u.yr
            icrs = SkyCoord(**components).transform_to(ICRS())

            ra_str, dec_str = self._sexagesimal_radec(icrs)
            color, color_name = self._color(
                self._scalar(primary.get("GAIAMAG")),
                self._scalar(primary.get("2MASSMAG")),
                "G-J",
            )
            record = {
                "object": primary.get("OBJECT"),
                "ra": ra_str,
                "dec": dec_str,
                "pmra": icrs.pm_ra_cosdec.to_value(u.arcsec / u.yr) if has_pm else None,
                "pmdec": icrs.pm_dec.to_value(u.arcsec / u.yr) if has_pm else None,
                "parallax": self._scalar(primary.get("TARGPLAX")),
                "rv": self._scalar(primary.get("TARGRADV")),
                "frame": "icrs",
                "epoch": self._scalar(primary.get("TARGEPOC")),
                "equinox": 2000.0,
                "color": color,
                "color_name": color_name,
            }
        except Exception as exc:
            logger.warning(
                "could not build wmko CATALOG_RECORD from L0 PRIMARY TARG* "
                "(%s: %s); left empty",
                type(exc).__name__,
                exc,
            )
            return None
        logger.info("successfully built record for wmko-tcs")
        self._write_catalog_record("wmko", record)
        return record

    def merge_catalog_records(self):
        """Merge the built source records into the canonical ``kpf-drp`` row.

        Consumes the in-memory gaia/simbad/wmko records (``None`` for a source toggled
        off or unresolved). Only a source named in ``astrometry_priority`` may anchor
        the position, in that order, and it supplies the astrometric solution --
        ra/dec/pmra/pmdec/parallax plus the frame, epoch and equinox that qualify them
        -- whole, since those fields are one measurement and must not be spliced across
        catalogs. Sources left out of the priority still get their row written and their
        offsets diagnosed; they simply cannot become the base. Its rv rides along; when
        that source has none (Gaia commonly lacks radial_velocity) rv falls back to the
        telescope TARGRADV on PRIMARY, never to another catalog. ``color`` and
        ``color_name`` also ride along, but a base source without a color borrows one
        from any built row, with a mixed-catalog warning. ``radec_src``/``plx_src`` name
        the astrometric source, ``rv_src`` the rv source ("wmko" for the TARGRADV
        fallback, "" when nothing supplied).

        Raises ``ValueError`` when no permitted source supplies a complete block --
        without a position there is nothing to correct, and silently demoting to a
        source the operator excluded is exactly what the priority exists to prevent. rv
        is optional, left missing when neither the astrometric source nor TARGRADV
        supplies it.
        """
        candidates = [
            (source, getattr(self, f"_{source}"))
            for source in _SOURCES
            if getattr(self, f"_{source}") is not None
        ]

        # Only a source named in astrometry_priority may anchor the position, in that
        # order; the first with a complete block wins and an incomplete one ahead of it
        # is demoted with an auditable warning.
        base_source, base_record = None, None
        for source in self.astrometry_priority:
            record = getattr(self, f"_{source}")
            if record is None:
                continue
            missing = [field for field in _ASTROMETRY if record[field] is None]
            if not missing:
                base_source, base_record = source, record
                break
            logger.warning(
                "%s astrometry incomplete (missing %s); skipping it as the "
                "astrometric base",
                source,
                ", ".join(missing),
            )
        if base_record is None:
            available = [source for source, _ in candidates]
            raise ValueError(
                f"cannot build a canonical astrometry position for "
                f"{self.l0_obj.obs_id or 'unknown'}: no source in astrometry_priority "
                f"{list(self.astrometry_priority)} supplies a complete "
                f"ra/dec/pmra/pmdec/parallax/epoch block (rows built: {available})"
            )

        # rv from the base source, else telescope TARGRADV off PRIMARY -- independent of
        # astrometry_priority, which governs only the position.
        rv_value = base_record["rv"]
        rv_source = base_source
        if rv_value is None:
            rv_value = self._scalar(self.l0_obj.headers["PRIMARY"].get("TARGRADV"))
            rv_source = "wmko" if rv_value is not None else ""
            if rv_value is not None:
                logger.warning(
                    "no catalog supplied a radial velocity; falling back to the "
                    "telescope TARGRADV=%s km/s from L0 PRIMARY",
                    rv_value,
                )

        # A color index is independent of the astrometry, so borrowing one from another
        # catalog when the base source has none is acceptable, but flagged.
        color = base_record["color"]
        color_name = base_record["color_name"]
        if color is None:
            for source, candidate in candidates:
                if candidate["color"] is not None:
                    color, color_name = candidate["color"], candidate["color_name"]
                    logger.warning(
                        "%s astrometric base has no color; using the %s %s color",
                        base_source,
                        source,
                        color_name,
                    )
                    break

        record = {
            "object": base_record["object"],
            "radec_src": base_source,
            "plx_src": base_source,
            "rv_src": rv_source,
            "ra": base_record["ra"],
            "dec": base_record["dec"],
            "pmra": base_record["pmra"],
            "pmdec": base_record["pmdec"],
            "parallax": base_record["parallax"],
            "rv": rv_value,
            "frame": base_record["frame"],
            "epoch": base_record["epoch"],
            "equinox": base_record["equinox"],
            "color": color,
            "color_name": color_name,
        }
        self._canonical = record
        self._write_catalog_record("kpf-drp", record)
        return record

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------

    def _track_info(self):
        """Build and cache the info() summary text from instance attributes."""
        gaia = self._gaia["object"] if self._gaia else "n/a"
        simbad = self._simbad["object"] if self._simbad else "n/a"
        canonical = self._canonical["radec_src"] if self._canonical else "n/a"
        lines = [
            "AstroQuery",
            f"  obs_id:  {self.l0_obj.obs_id or 'unknown'}",
            f"  Gaia DR3:  {gaia}",
            f"  SIMBAD:    {simbad}",
            f"  canonical position source:  {canonical}",
        ]
        self._info = "\n\n" + "\n".join(lines) + "\n\n"

    def _set_headers(self, l0_obj):
        """Sole place this module writes headers; reads instance attributes.

        One presence flag per queryable source: 1 when it resolved a record, 0
        otherwise -- absent, gated off and failed are alike, since a consumer cannot
        act on the difference. All three are written together, so a missing flag means
        AstroQuery did not complete (what DiagL0 warns on). The merged ``kpf-drp`` row
        has no flag; merge_catalog_records raises instead.
        """
        for source in _SOURCES:
            record = getattr(self, f"_{source}")
            l0_obj.set_keyword(_CATALOG_FLAGS[source], 1 if record is not None else 0)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(
        self, *, do_gaia_query=None, do_simbad_query=None, astrometry_priority=None
    ):
        """
        Resolve external catalog astrometry and write it to the L0.

        Parameters
        ----------
        do_gaia_query, do_simbad_query : bool, optional
            Override the configured catalog-query toggles for this call.
        astrometry_priority : sequence of str, optional
            Override which sources may anchor the merged position, highest first.

        Returns
        -------
        l0_obj : KPF0
            The input L0 (PRIMARY unchanged), with the ``wmko``/``gaia``/``simbad`` and
            merged ``kpf-drp`` rows written to ``CATALOG_RECORD``, plus an 'astro_query'
            receipt entry. Unusually for a pipeline module this returns an L0, not the
            next level -- AstroQuery runs before assembly.

        Raises
        ------
        ValueError
            No source named in ``astrometry_priority`` supplied a complete astrometric
            block.
        """
        if do_gaia_query is not None:
            self.do_gaia_query = do_gaia_query
        if do_simbad_query is not None:
            self.do_simbad_query = do_simbad_query
        if astrometry_priority is not None:
            self.astrometry_priority = astrometry_priority
            self._validate_priority()

        self._wmko = self.read_wmko_header()
        self._gaia = self.query_gaia() if self.do_gaia_query else None
        self._simbad = self.query_simbad() if self.do_simbad_query else None

        self.merge_catalog_records()

        self._set_headers(self.l0_obj)
        self._track_info()
        self.l0_obj.receipt_add_entry("astro_query", "", "PASS")

        logger.info("%s", self._info)
        return self.l0_obj

    def info(self):
        """Print a summary of the resolved catalog astrometry."""
        if self._info is None:
            print(f"{type(self).__name__}: perform() has not been called")
        else:
            print(self._info)
