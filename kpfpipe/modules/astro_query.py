"""
KPF AstroQuery module.

Consolidates the pipeline's external astronomical-catalog lookups into a single
L0-stage module. Given a raw L0 frame, it resolves the target's astrometry from
Gaia DR3 (by GAIAID) and SIMBAD (by OBJECT), builds the telescope-native ``wmko``
row from the L0 PRIMARY ``TARG*`` astrometry (no query), and writes all three rows
to the L0 ``CATALOG_RECORD`` extension, setting the ``WMKOCR``/``GAIACR``/``SIMBADCR``
presence flags. ``KPF0.read`` leaves ``CATALOG_RECORD`` empty; AstroQuery is its sole
populator.

The extension is the bridge across an ordering problem: the query results
ultimately belong on the EPRV PRIMARY catalog keywords (``C*#``), but the WMKO ->
EPRV PRIMARY conversion does not happen until ``KPF0.to_kpf1()`` downstream. Rather
than write PRIMARY here, AstroQuery records the queried quantities on the L0 and
lets ``to_kpf1()`` overlay them onto the L1 PRIMARY (a follow-up integration).

AstroQuery never modifies the L0 PRIMARY header (a pure pass-through to
INSTRUMENT_HEADER). All three source rows and the merged ``kpf-drp`` row share one
schema in the EPRV C*# PRIMARY format (see ``_CATALOG_COLUMNS``), so a consumer reads
any source identically and ``to_kpf1`` can copy cells straight onto the catalog cards.
Per-fiber fan-out and derived quantities (offsets, redshift) are downstream jobs.
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
from kpfpipe.utils.config import ConfigHandler

logger = logging.getLogger(__name__)

_DEFAULTS = {
    **DEFAULTS,
    "do_gaia_query": True,
    "do_simbad_query": True,
    "use_wmko_tcs": True,
}

# The astrometric block the merge takes whole from one source: ra/dec/pmra/pmdec/
# parallax are one astrometric fit -- proper motion without the parallax it was measured
# against is nearly meaningless -- plus the frame/epoch that qualify the coordinates.
# equinox is not gated (an ICRS formality), so a WMKO base missing TARGEQUI still
# qualifies. rv is handled separately (base source, else TARGRADV). Priority (highest
# first): gaia > simbad > wmko.
_ASTROMETRY = ("ra", "dec", "pmra", "pmdec", "parallax", "epoch", "frame")

# Column units AstroQuery assumes for each catalog result, verified before the
# values are trusted so a silent upstream schema change fails loudly instead of
# corrupting the record.
_GAIA_UNITS = {
    "ra": u.deg,
    "dec": u.deg,
    "pmra": u.mas / u.yr,
    "pmdec": u.mas / u.yr,
    "parallax": u.mas,
    "radial_velocity": u.km / u.s,
    "ref_epoch": u.yr,
}
_SIMBAD_UNITS = {
    "ra": u.deg,
    "dec": u.deg,
    "pmra": u.mas / u.yr,
    "pmdec": u.mas / u.yr,
    "plx_value": u.mas,
    "rvz_radvel": u.km / u.s,
}

# CATALOG_RECORD write schema (AstroQuery is the sole populator): one row per resolved
# source (wmko/gaia/simbad) plus the merged 'kpf-drp' row. 'source' labels the row,
# 'object' is the queried target name, and radec_src/plx_src/rv_src name the source each
# value block (position, parallax, rv) came from -- its own 'source' for a plain source
# row, the winning source (or "" if none) for the merged row. Values are in the EPRV
# C*# PRIMARY format: RA/Dec sexagesimal strings (RA hour-angle, Dec deg, ICRS), PM
# arcsec/yr (RA incl. cos Dec), parallax mas, rv km/s, epoch/equinox Julian years.
# Missing floats -> NaN, missing strings -> "".
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
    "frame",
    "epoch",
    "equinox",
)
_CATALOG_STR_COLUMNS = frozenset(
    {"source", "object", "radec_src", "plx_src", "rv_src", "frame", "ra", "dec"}
)

# RA / DEC are sexagesimal strings, so not subject to astropy unit conversion
_CATALOG_UNITS = {
    "pmra": u.arcsec / u.yr,
    "pmdec": u.arcsec / u.yr,
    "parallax": u.mas,
    "rv": u.km / u.s,
    "epoch": u.yr,
    "equinox": u.yr,
}
# Presence flag written to the CATALOG_RECORD header per source (int 0/1). DiagL0 keeps
# its own local mirror (the DiagL0 convention of not importing this schema).
_CATALOG_FLAGS = {"gaia": "GAIACR", "simbad": "SIMBADCR", "wmko": "WMKOCR"}


class AstroQuery:
    """
    Resolve target astrometry from external catalogs and write it to an L0.

    Runs the two external catalog queries (Gaia DR3 by GAIAID, SIMBAD by OBJECT),
    builds the native ``wmko`` row from the L0 PRIMARY ``TARG*`` astrometry, and writes
    all three rows to the L0 ``CATALOG_RECORD`` extension for downstream use (EPRV
    ``C*#`` catalog keywords, DiagL0 pointing offsets, BarycentricCorrection). Only
    science frames are supported: the constructor raises on a non-``Object`` IMTYPE (a
    calibration). Fail-soft otherwise: a missing GAIAID/OBJECT or a failed network
    lookup yields a ``None`` record rather than an error.

    Parameters
    ----------
    l0_obj : KPF0
        Raw L0 science frame (IMTYPE ``Object``). Its PRIMARY header (IMTYPE, GAIAID,
        OBJECT, TARG*) is read but never modified; the resolved catalog data is written
        to the L0 ``CATALOG_RECORD`` extension.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: do_gaia_query, do_simbad_query,
        use_wmko_tcs.
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

        self._wmko = None  # native WMKO record; set by read_wmko_header()
        self._gaia = None  # Gaia DR3 record; set by query_gaia()
        self._simbad = None  # SIMBAD record; set by query_simbad()
        self._canonical = None  # merged kpf-drp record; set by merge_catalog_records()
        self._info = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _gaia_source_id(self):
        """Digit-only Gaia DR3 id from L0 GAIAID, or None if absent/malformed.

        GAIAID may arrive as a prefixed string (e.g. 'Gaia DR3 12345'); take the
        trailing token and require it to be all digits.
        """
        raw = self.l0_obj.headers["PRIMARY"].get("GAIAID")
        if raw is None:
            return None
        token = str(raw).strip().split()[-1] if str(raw).strip() else ""
        return token if token.isdigit() else None

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
    def _sexagesimal_radec(coord):
        """ICRS SkyCoord -> the EPRV C*# sexagesimal (ra, dec) strings.

        The single formatter for the canonical colon-separated 'h:m:s' cards (RA
        hour-angle, Dec deg), so a precision/padding change lands in one place.
        astropy's combined hmsdms already pads and signs; we just split the two axes.
        """
        return tuple(coord.to_string("hmsdms", sep=":", precision=4).split())

    @staticmethod
    def _verify_units(table, expected, source):
        """Verify a query result's column units before its values are trusted.

        Guards the canonical-unit assumption: raises ``ValueError`` if any
        expected column is missing or carries a unit other than the one the
        record schema assumes, so a silent catalog schema change fails loudly.
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
        """Upsert one source's row into the L0 CATALOG_RECORD extension, set its flag.

        The sole writer for CATALOG_RECORD (``wmko``/``gaia``/``simbad`` and the merged
        ``kpf-drp`` row). ``record`` is a canonical record dict or None; a None record
        drops the source's row and clears its flag, otherwise the row is (re)written and
        the flag set to 1. Provenance labels (``radec_src``/``plx_src``/``rv_src``)
        default to ``source`` when the record omits them (a source row's values are its
        own); the merged row supplies them explicitly. Other sources' rows are preserved
        (upsert). ``kpf-drp`` writes no flag. Missing floats become NaN, strings "".
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
            rows[source] = {
                "source": source,
                "radec_src": source,
                "plx_src": source,
                "rv_src": source,
                **record,
            }

        ordered = list(rows.values())
        new_table = Table()
        for name in _CATALOG_COLUMNS:
            if name in _CATALOG_STR_COLUMNS:
                new_table[name] = np.array(
                    ["" if r[name] is None else r[name] for r in ordered], dtype=str
                )
            else:
                new_table[name] = np.array(
                    [np.nan if r[name] is None else r[name] for r in ordered],
                    dtype=float,
                )
                new_table[name].unit = _CATALOG_UNITS[name]
        l0.set_data("CATALOG_RECORD", new_table)
        flag = _CATALOG_FLAGS.get(source)
        if flag is not None:
            l0.set_keyword(flag, 1 if record is not None else 0)

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def query_gaia(self):
        """Query Gaia DR3 for the target's ICRS astrometry, or None (fail-soft).

        Returns None (warned) when GAIAID yields no usable source_id, when the
        lookup fails, or when the source is not found. Raises ValueError if the
        result's column units differ from the assumed canonical schema (deg,
        mas/yr, mas, km/s, epoch in Julian years; ICRS). Whether it runs at all is
        gated upstream in perform by ``do_gaia_query``.
        """
        gaia_id = self._gaia_source_id()
        if gaia_id is None:
            logger.warning(
                "no usable GAIAID on L0 PRIMARY; Gaia astrometry unavailable"
            )
            return None
        query = f"""
        SELECT ra, dec, pmra, pmdec, parallax, radial_velocity, ref_epoch
        FROM gaiadr3.gaia_source
        WHERE source_id = {gaia_id}
        """
        logger.info("querying Gaia DR3 for source_id %s", gaia_id)
        try:
            results = Gaia.launch_job(query).get_results()
        except Exception as e:
            logger.warning(
                "Gaia query failed (%s: %s); Gaia astrometry unavailable",
                type(e).__name__,
                e,
            )
            return None
        if len(results) == 0:
            logger.warning(
                "Gaia returned no match for source_id %s; Gaia astrometry unavailable",
                gaia_id,
            )
            return None
        self._verify_units(results, _GAIA_UNITS, "Gaia DR3")
        row = results[0]
        ra, dec = self._scalar(row["ra"]), self._scalar(row["dec"])
        pmra, pmdec = self._scalar(row["pmra"]), self._scalar(row["pmdec"])
        # Sanitize to the EPRV C*# format: RA/Dec deg -> sexagesimal (RA hour-angle,
        # Dec deg); proper motion mas/yr -> arcsec/yr.
        ra_str, dec_str = self._sexagesimal_radec(
            SkyCoord(ra, dec, unit=u.deg, frame="icrs")
        )
        record = {
            "object": gaia_id,
            "ra": ra_str,
            "dec": dec_str,
            "pmra": None if pmra is None else pmra / 1e3,
            "pmdec": None if pmdec is None else pmdec / 1e3,
            "parallax": self._scalar(row["parallax"]),
            "rv": self._scalar(row["radial_velocity"]),
            # frame/equinox are definitional, not queried: Gaia DR3 is ICRS (Gaia-CRF3)
            # with no equinox -- 2000.0 is the J2000 convention EPRV's Required CEQNX#
            # demands. Only epoch (ref_epoch, J2016.0 for DR3) is a real query output.
            "frame": "icrs",
            "epoch": self._scalar(row["ref_epoch"]),
            "equinox": 2000.0,
        }
        logger.info("successfully built record for gaia")
        self._write_catalog_record("gaia", record)
        return record

    def query_simbad(self):
        """Query SIMBAD for the OBJECT's ICRS J2000 astrometry, or None (fail-soft).

        Returns None when L0 has no OBJECT name, or when the lookup fails /
        resolves nothing (warned, not raised). Raises ValueError if the result's
        column units differ from the assumed schema. Column schema is the
        astroquery 0.4.11 lowercase form (ra/dec deg, pmra/pmdec mas/yr, plx_value
        mas, rvz_radvel km/s). Whether it runs at all is gated upstream in perform
        by ``do_simbad_query``.
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
            simbad.add_votable_fields("pmra", "pmdec", "plx", "rvz_radvel")
            result = simbad.query_object(name)
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
        # Sanitize to the EPRV C*# format: RA/Dec deg -> sexagesimal (RA hour-angle,
        # Dec deg); proper motion mas/yr -> arcsec/yr.
        ra_str, dec_str = self._sexagesimal_radec(
            SkyCoord(ra, dec, unit=u.deg, frame="icrs")
        )
        record = {
            "object": name,
            "ra": ra_str,
            "dec": dec_str,
            "pmra": None if pmra is None else pmra / 1e3,
            "pmdec": None if pmdec is None else pmdec / 1e3,
            "parallax": self._scalar(row["plx_value"]),
            "rv": self._scalar(row["rvz_radvel"]),
            # frame/epoch/equinox are definitional here, not queried: astroquery's
            # SIMBAD returns basic ra/dec as ICRS J2000, with no per-object
            # frame/epoch/equinox to read (unlike Gaia's real ref_epoch). equinox 2000.0
            # satisfies EPRV's Required CEQNX#.
            "frame": "icrs",
            "epoch": 2000.0,
            "equinox": 2000.0,
        }
        logger.info("successfully built record for simbad")
        self._write_catalog_record("simbad", record)
        return record

    def read_wmko_header(self):
        """Read the native WMKO/DCS astrometry from L0 PRIMARY TARG*, or None.

        The telescope-side counterpart to query_gaia/query_simbad: no query, just the
        raw TARG* pointing sanitized to the EPRV C*# format (TARGPMRA time-s/yr ->
        arcsec/yr via x15 cos Dec; TARGPMDC already arcsec/yr) and rotated from its
        native FK5 (J2000) to ICRS, so all three sources share one frame. KPF pointing
        is always FK5, so a non-FK5 TARGFRAM (absent included) raises rather than being
        coerced -- a wrong frame would corrupt the barycentric correction. Returns None
        (WMKOCR=0) when there is no pointing (TARGRA absent) or the TARG* astrometry
        cannot be parsed (warned, never raised). Gated upstream in perform by
        ``use_wmko_tcs``.
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

            # Rotate the native FK5 pointing to ICRS. Position always; proper motion too
            # when both components are present (TARGPMRA time-s/yr -> arcsec/yr via x15
            # cos Dec first; a lone component is meaningless under rotation, so it is
            # both-or-neither). Rotation carries no time propagation, so epoch is
            # unchanged; parallax/RV are rotation-invariant.
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

        Consumes the in-memory gaia/simbad/wmko records this instance just built
        (``None`` for a source that was toggled off or resolved nothing), in priority
        order gaia > simbad > wmko. The canonical record is the highest-priority source
        that supplies a complete astrometric solution -- ra/dec/pmra/pmdec/parallax plus
        the frame, epoch, and equinox that make them meaningful -- taken whole, since
        those fields are one measurement and must not be spliced across catalogs. Its rv
        rides along; when that source has none (Gaia commonly lacks radial_velocity), rv
        falls back to the telescope TARGRADV on PRIMARY (not borrowed from a lower-
        priority catalog). ``radec_src``/``plx_src`` name the astrometric source;
        ``rv_src`` names the rv source ("wmko" for the TARGRADV fallback, "" when
        nothing supplied it).

        Raises ``ValueError`` when no source supplies a complete ra/dec/pmra/pmdec/
        parallax/epoch block -- without a position there is nothing to correct, so it
        must fail loudly. rv is optional, left missing when neither the astrometric
        source nor TARGRADV supplies it.
        """
        # Our own records, just built and schema-clean; assemble in priority order (a
        # source toggled off or that resolved nothing is already None).
        candidates = []
        for source, record in (
            ("gaia", self._gaia),
            ("simbad", self._simbad),
            ("wmko", self._wmko),
        ):
            if record is not None:
                candidates.append((source, record))

        # Take the whole astrometric block from the first source that has it complete;
        # raise if none does (without a position there is nothing to correct).
        base_source, base_record = None, None
        for source, record in candidates:
            if all(record[field] is not None for field in _ASTROMETRY):
                base_source, base_record = source, record
                break
        if base_record is None:
            available = [source for source, _ in candidates]
            raise ValueError(
                f"cannot build a canonical astrometry position for "
                f"{self.l0_obj.obs_id or 'unknown'}: no source supplies a complete "
                f"ra/dec/pmra/pmdec/parallax/epoch block (have {available})"
            )

        # rv from the base source, else telescope TARGRADV off PRIMARY -- independent of
        # use_wmko_tcs (which gates only the wmko position row) and already km/s, so no
        # conversion.
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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, *, do_gaia_query=None, do_simbad_query=None, use_wmko_tcs=None):
        """
        Resolve external catalog astrometry and write it to the L0.

        Parameters
        ----------
        do_gaia_query, do_simbad_query, use_wmko_tcs : bool, optional
            Override the configured source toggles for this call.

        Returns
        -------
        l0_obj : KPF0
            The input L0 (PRIMARY unchanged), now with its ``wmko``/``gaia``/``simbad``
            rows and the merged ``kpf-drp`` row written to the ``CATALOG_RECORD``
            extension, and an 'astro_query' receipt entry. Unusually for a pipeline
            module this returns an L0, not the next level -- AstroQuery runs before
            assembly.
        """
        if do_gaia_query is not None:
            self.do_gaia_query = do_gaia_query
        if do_simbad_query is not None:
            self.do_simbad_query = do_simbad_query
        if use_wmko_tcs is not None:
            self.use_wmko_tcs = use_wmko_tcs

        # Each source is gated here: a toggled-off source is neither queried nor
        # built, so its row stays absent. Each method that runs looks up its
        # astrometry and writes its own CATALOG_RECORD row in one go.
        self._wmko = self.read_wmko_header() if self.use_wmko_tcs else None
        self._gaia = self.query_gaia() if self.do_gaia_query else None
        self._simbad = self.query_simbad() if self.do_simbad_query else None

        # Merge the source rows (gaia/simbad/wmko) into the canonical kpf-drp row that
        # downstream (to_kpf1 C*#, barycentric correction) consumes. Raises if a
        # complete set cannot be assembled.
        self.merge_catalog_records()
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
