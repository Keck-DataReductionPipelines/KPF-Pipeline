"""
KPF AstroQuery module.

Consolidates every external astronomical-catalog lookup the pipeline needs into
a single L0-stage module. Given a raw L0 frame, it resolves the target's
astrometry from Gaia DR3 (by GAIAID) and SIMBAD (by OBJECT), and snapshots the
DCS/TCS target astrometry already on the raw header, then writes all three to the
L0 ``CATALOG_RECORD`` extension -- a BinTable with one row per resolved source and
``WMKOCR``/``GAIACR``/``SIMBADCR`` presence flags on its header.

The extension is the bridge across an ordering problem: the query results
ultimately belong on the EPRV PRIMARY catalog keywords (``C*#``), but the WMKO ->
EPRV PRIMARY conversion does not happen until ``KPF0.to_kpf1()`` downstream. Rather
than write PRIMARY here, AstroQuery records the queried quantities on the L0 and
lets ``to_kpf1()`` overlay them onto the L1 PRIMARY (a follow-up integration).

AstroQuery never modifies the L0 PRIMARY header -- that header is a pure
pass-through to INSTRUMENT_HEADER. The three records share one schema and one set
of units and reference frame, so a consumer reads any source identically: ICRS,
RA/Dec in degrees, proper motion in mas/yr (RA including cos(Dec)), parallax in
mas, RV in km/s, epoch/equinox in Julian years. Per-fiber fan-out and derived
quantities (offsets, redshift) remain the jobs of downstream consumers.
"""

import logging

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from astropy.table import Table
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad

from kpfpipe import DEFAULTS
from kpfpipe.utils.config import ConfigHandler

logger = logging.getLogger(__name__)

_DEFAULTS = {
    **DEFAULTS,
    "use_gaia": True,
    "use_simbad": True,
}

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

# The DCS TARG* astrometry format the wmko conversion assumes (verified before use).
_WMKO_FRAME = "FK5"
_WMKO_EQUINOX = 2000.0

# Schema of the CATALOG_RECORD BinTable extension: one row per resolved source,
# carrying the canonical record fields plus a leading 'source' label. Float columns
# hold NaN where a value is missing; string columns hold "". Units document the
# canonical schema (deg, mas/yr incl. cos Dec, mas, km/s, Julian years).
_CATALOG_COLUMNS = (
    "source",
    "source_id",
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
_CATALOG_STR_COLUMNS = frozenset({"source", "source_id", "frame"})
_CATALOG_UNITS = {
    "ra": u.deg,
    "dec": u.deg,
    "pmra": u.mas / u.yr,
    "pmdec": u.mas / u.yr,
    "parallax": u.mas,
    "rv": u.km / u.s,
    "epoch": u.yr,
    "equinox": u.yr,
}
# Presence flag written to the CATALOG_RECORD header per source (int 0/1).
_CATALOG_FLAGS = {"wmko": "WMKOCR", "gaia": "GAIACR", "simbad": "SIMBADCR"}


class AstroQuery:
    """
    Resolve target astrometry from external catalogs and attach it to an L0.

    Runs the two external catalog queries (Gaia DR3 by GAIAID, SIMBAD by OBJECT)
    plus a verbatim snapshot of the DCS/TCS ``TARG*`` astrometry, and writes all
    three to the L0 ``CATALOG_RECORD`` extension for downstream use (EPRV ``C*#``
    catalog keywords, DiagL0 pointing offsets, BarycentricCorrection). Only science
    frames
    are supported: the constructor raises on a non-``Object`` IMTYPE (a calibration).
    Fail-soft otherwise: a missing GAIAID/OBJECT or a failed network lookup yields a
    ``None`` record rather than an error.

    Parameters
    ----------
    l0_obj : KPF0
        Raw L0 science frame (IMTYPE ``Object``). Its PRIMARY header (IMTYPE, GAIAID,
        OBJECT, TARG*) is read but never modified; the resolved catalog data is
        written to the L0 ``CATALOG_RECORD`` extension.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: use_gaia, use_simbad.
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

        self._gaia = None  # Gaia DR3 record; set by query_gaia()
        self._simbad = None  # SIMBAD record; set by query_simbad()
        self._wmko = None  # DCS/TCS TARG* record; set by read_wmko_target()
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

    def _verify_wmko_format(self, primary):
        """Verify the DCS TARG* astrometry is in the assumed input format.

        The wmko conversion assumes FK5 J2000 with sexagesimal TARGRA/TARGDEC
        (hourangle/deg), TARGPMRA in s/yr and TARGPMDC in arcsec/yr. The pm units
        are not encoded in the header, but the frame, equinox, and sexagesimal
        coordinate format are -- verify those and raise ``ValueError`` on any
        surprise so a silent DCS format change cannot corrupt the record.
        """
        frame = primary.get("TARGFRAM")
        equinox = primary.get("TARGEQUI")
        ra, dec = primary.get("TARGRA"), primary.get("TARGDEC")
        problems = []
        if not (isinstance(frame, str) and frame.strip().upper() == _WMKO_FRAME):
            problems.append(f"TARGFRAM={frame!r} (expected {_WMKO_FRAME!r})")
        if equinox != _WMKO_EQUINOX:
            problems.append(f"TARGEQUI={equinox!r} (expected {_WMKO_EQUINOX})")
        if not (
            isinstance(ra, str) and ":" in ra and isinstance(dec, str) and ":" in dec
        ):
            problems.append(f"TARGRA/TARGDEC not sexagesimal ({ra!r}, {dec!r})")
        if problems:
            raise ValueError(
                "unexpected WMKO/DCS astrometry format: "
                + "; ".join(problems)
                + "; AstroQuery assumes FK5 J2000 sexagesimal input. Verify the DCS "
                "header format before proceeding."
            )

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def read_wmko_target(self):
        """Read the DCS/TCS ``TARG*`` target astrometry from L0 PRIMARY.

        Converts the native DCS values to the canonical schema (RA/Dec degrees,
        proper motion mas/yr, ICRS): TARGRA/TARGDEC are sexagesimal (hourangle /
        deg); TARGPMRA is time-seconds/yr (-> mas/yr via x15 cos(Dec)) and TARGPMDC
        is arcsec/yr (-> mas/yr). TARGFRAM is FK5 J2000, relabeled ICRS (the ~23 mas
        frame tie is negligible for this fallback record). Returns None when the
        frame carries no target pointing (TARGRA absent -- e.g. a calibration).
        """
        primary = self.l0_obj.headers["PRIMARY"]
        if primary.get("TARGRA") is None:
            logger.warning("no TARGRA on L0 PRIMARY; WMKO/DCS astrometry unavailable")
            return None
        self._verify_wmko_format(primary)
        dec_deg = Angle(primary["TARGDEC"], unit=u.deg).deg
        pmra = primary.get("TARGPMRA")
        pmdec = primary.get("TARGPMDC")
        record = {
            "source_id": primary.get("OBJECT"),
            "ra": Angle(primary["TARGRA"], unit=u.hourangle).to(u.deg).value,
            "dec": dec_deg,
            "pmra": None
            if pmra is None
            else pmra * 15.0 * np.cos(np.radians(dec_deg)) * 1e3,
            "pmdec": None if pmdec is None else pmdec * 1e3,
            "parallax": primary.get("TARGPLAX"),
            "rv": primary.get("TARGRADV"),
            "frame": "icrs",
            "epoch": primary.get("TARGEPOC"),
            "equinox": primary.get("TARGEQUI"),
        }
        logger.info("successfully built record for wmko")
        return record

    def query_gaia(self):
        """Query Gaia DR3 for the target's ICRS astrometry, or None (fail-soft).

        Returns None (warned) when disabled, when GAIAID yields no usable
        source_id, when the lookup fails, or when the source is not found. Raises
        ValueError if the result's column units differ from the assumed canonical
        schema (deg, mas/yr, mas, km/s, epoch in Julian years; ICRS).
        """
        if not self.use_gaia:
            return None
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
        record = {
            "source_id": gaia_id,
            "ra": self._scalar(row["ra"]),
            "dec": self._scalar(row["dec"]),
            "pmra": self._scalar(row["pmra"]),
            "pmdec": self._scalar(row["pmdec"]),
            "parallax": self._scalar(row["parallax"]),
            "rv": self._scalar(row["radial_velocity"]),
            "frame": "icrs",
            "epoch": self._scalar(row["ref_epoch"]),
            "equinox": 2000.0,
        }
        logger.info("successfully built record for gaia")
        return record

    def query_simbad(self):
        """Query SIMBAD for the OBJECT's ICRS J2000 astrometry, or None (fail-soft).

        Returns None when disabled, when L0 has no OBJECT name, or when the lookup
        fails / resolves nothing (warned, not raised). Raises ValueError if the
        result's column units differ from the assumed schema. Column schema is the
        astroquery 0.4.11 lowercase form (ra/dec deg, pmra/pmdec mas/yr, plx_value
        mas, rvz_radvel km/s).
        """
        if not self.use_simbad:
            return None
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
        record = {
            "source_id": name,
            "ra": self._scalar(row["ra"]),
            "dec": self._scalar(row["dec"]),
            "pmra": self._scalar(row["pmra"]),
            "pmdec": self._scalar(row["pmdec"]),
            "parallax": self._scalar(row["plx_value"]),
            "rv": self._scalar(row["rvz_radvel"]),
            "frame": "icrs",
            "epoch": 2000.0,
            "equinox": 2000.0,
        }
        logger.info("successfully built record for simbad")
        return record

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------

    def _track_info(self):
        """Build and cache the info() summary text from instance attributes."""
        gaia = self._gaia["source_id"] if self._gaia else "n/a"
        simbad = self._simbad["source_id"] if self._simbad else "n/a"
        wmko = "resolved" if self._wmko else "n/a"
        lines = [
            "AstroQuery",
            f"  obs_id:  {self.l0_obj.obs_id or 'unknown'}",
            f"  Gaia DR3:  {gaia}",
            f"  SIMBAD:    {simbad}",
            f"  WMKO/DCS:  {wmko}",
        ]
        self._info = "\n\n" + "\n".join(lines) + "\n\n"

    def _attach_catalog_record(self, l0_obj):
        """Write the resolved catalog records to the CATALOG_RECORD extension.

        The module's sole output site (analogous to ``_set_headers`` on a transform
        module). Builds a BinTable with one row per resolved source (canonical ICRS
        schema; missing floats -> NaN, missing source_id -> "") and sets the
        WMKOCR/GAIACR/SIMBADCR presence flags on the extension header. The L0 PRIMARY
        is never touched -- the EPRV ``C*#`` keywords are built later in
        ``KPF0.to_kpf1()``; this extension is the L0-stage bridge to that step.
        """
        records = {"wmko": self._wmko, "gaia": self._gaia, "simbad": self._simbad}
        rows = [
            {"source": src, **rec} for src, rec in records.items() if rec is not None
        ]
        table = Table()
        for name in _CATALOG_COLUMNS:
            if name in _CATALOG_STR_COLUMNS:
                table[name] = np.array(
                    ["" if r[name] is None else r[name] for r in rows], dtype=str
                )
            else:
                table[name] = np.array(
                    [np.nan if r[name] is None else r[name] for r in rows], dtype=float
                )
                table[name].unit = _CATALOG_UNITS[name]
        l0_obj.set_data("CATALOG_RECORD", table)

        for source, keyword in _CATALOG_FLAGS.items():
            l0_obj.set_keyword(keyword, 1 if records[source] is not None else 0)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, *, use_gaia=None, use_simbad=None):
        """
        Resolve external catalog astrometry and attach it to the L0.

        Parameters
        ----------
        use_gaia, use_simbad : bool, optional
            Override the configured query toggles for this call.

        Returns
        -------
        l0_obj : KPF0
            The input L0 (PRIMARY unchanged), now carrying the ``CATALOG_RECORD``
            extension (one row per resolved source, with WMKOCR/GAIACR/SIMBADCR
            presence flags) and an 'astro_query' receipt entry. Unusually for a
            pipeline module this returns an L0, not the next level -- AstroQuery
            runs before assembly.
        """
        if use_gaia is not None:
            self.use_gaia = use_gaia
        if use_simbad is not None:
            self.use_simbad = use_simbad

        self._wmko = self.read_wmko_target()
        self._gaia = self.query_gaia()
        self._simbad = self.query_simbad()

        self._attach_catalog_record(self.l0_obj)
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
