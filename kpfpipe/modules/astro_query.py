"""
KPF AstroQuery module.

Consolidates the pipeline's external astronomical-catalog lookups into a single
L0-stage module. Given a raw L0 frame, it resolves the target's astrometry from
Gaia DR3 (by GAIAID) and SIMBAD (by OBJECT) and writes those two rows to the L0
``CATALOG_RECORD`` extension via ``KPF0.set_catalog_record``, setting the
``GAIACR``/``SIMBADCR`` presence flags. The third (``wmko``) row is telescope-native
(no query) and is populated by ``KPF0`` at read time, not here.

The extension is the bridge across an ordering problem: the query results
ultimately belong on the EPRV PRIMARY catalog keywords (``C*#``), but the WMKO ->
EPRV PRIMARY conversion does not happen until ``KPF0.to_kpf1()`` downstream. Rather
than write PRIMARY here, AstroQuery records the queried quantities on the L0 and
lets ``to_kpf1()`` overlay them onto the L1 PRIMARY (a follow-up integration).

AstroQuery never modifies the L0 PRIMARY header -- that header is a pure
pass-through to INSTRUMENT_HEADER. All three records share one schema and one set
of units and reference frame, so a consumer reads any source identically: ICRS,
RA/Dec in degrees, proper motion in mas/yr (RA including cos(Dec)), parallax in
mas, RV in km/s, epoch/equinox in Julian years. Per-fiber fan-out and derived
quantities (offsets, redshift) remain the jobs of downstream consumers.
"""

import logging

import astropy.units as u
import numpy as np
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad

from kpfpipe import DEFAULTS
from kpfpipe.utils.config import ConfigHandler

logger = logging.getLogger(__name__)

_DEFAULTS = {
    **DEFAULTS,
    "use_gaia": True,
    "use_simbad": True,
    "use_wmko": True,
}

# Source-merge configuration for the canonical ``kpf-drp`` row. Priority order
# (highest first) governs which source supplies the coherent position block and,
# independently, the parallax and RV. _PRESENCE_FLAGS maps each source to its
# CATALOG_RECORD header flag (a local mirror of level0._CATALOG_FLAGS, matching the
# DiagL0 convention of not importing the private schema).
_MERGE_PRIORITY = ("gaia", "simbad", "wmko")
_PRESENCE_FLAGS = {"gaia": "GAIACR", "simbad": "SIMBADCR", "wmko": "WMKOCR"}
# The position block is taken together from one source so ra/dec/PM/epoch stay
# internally consistent; parallax and rv are filled per-column.
_POSITION_FIELDS = ("ra", "dec", "pmra", "pmdec", "epoch")
_MERGE_STR_FIELDS = frozenset({"source_id", "frame"})
_MERGE_READ_FIELDS = (
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


class AstroQuery:
    """
    Resolve target astrometry from external catalogs and write it to an L0.

    Runs the two external catalog queries (Gaia DR3 by GAIAID, SIMBAD by OBJECT) and
    writes their rows to the L0 ``CATALOG_RECORD`` extension for downstream use (EPRV
    ``C*#`` catalog keywords, DiagL0 pointing offsets, BarycentricCorrection). The
    native ``wmko`` row is populated by ``KPF0`` at read time, not here. Only science
    frames are supported: the constructor raises on a non-``Object`` IMTYPE (a
    calibration). Fail-soft otherwise: a missing GAIAID/OBJECT or a failed network
    lookup yields a ``None`` record rather than an error.

    Parameters
    ----------
    l0_obj : KPF0
        Raw L0 science frame (IMTYPE ``Object``). Its PRIMARY header (IMTYPE, GAIAID,
        OBJECT) is read but never modified; the resolved catalog data is written to
        the L0 ``CATALOG_RECORD`` extension via ``set_catalog_record``.
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

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

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

    def _read_catalog_row(self, table, source):
        """Read one CATALOG_RECORD row into a record dict, or None if absent.

        Coerces the missing-value sentinels back to None (NaN floats, "" strings) so
        the merge sees a single "missing" marker per cell regardless of source.
        """
        match = table[table["source"] == source]
        if len(match) == 0:
            return None
        row = match[0]
        record = {}
        for name in _MERGE_READ_FIELDS:
            value = row[name]
            if name in _MERGE_STR_FIELDS:
                record[name] = None if str(value) == "" else str(value)
            else:
                f = float(value)
                record[name] = None if np.isnan(f) else f
        return record

    def merge_catalog_records(self):
        """Merge the source rows into the canonical ``kpf-drp`` CATALOG_RECORD row.

        Combines the ``gaia``/``simbad``/``wmko`` rows (each gated by its ``use_*``
        toggle and presence flag) into one canonical record, in ``_MERGE_PRIORITY``
        order: the coherent position block (ra/dec/pmra/pmdec/epoch, plus frame/equinox)
        is taken together from the highest-priority source that supplies it complete,
        while parallax and rv are each filled independently from the highest-priority
        source that has them. ``astr_src`` records the position source's label; an
        optional parallax/rv borrowed from a different source is a DEBUG-level note
        (Gaia commonly lacks radial_velocity, so this is routine, not an error).

        Raises ``ValueError`` when a coherent position block (ra, dec, pmra, pmdec,
        epoch) cannot be assembled from the enabled sources -- without a position there
        is nothing to correct, so its absence must fail loudly. parallax and rv are
        optional: filled when any source has them, else left missing (NaN) for the
        downstream consumer to default (as BarycentricCorrection already does).
        """
        table = self.l0_obj.data["CATALOG_RECORD"]
        header = self.l0_obj.headers["CATALOG_RECORD"]

        # Candidate (source_label, record) pairs in priority order, gated by toggle
        # and presence flag.
        candidates = []
        for source in _MERGE_PRIORITY:
            if not getattr(self, f"use_{source}"):
                continue
            if not header.get(_PRESENCE_FLAGS[source]):
                continue
            record = self._read_catalog_row(table, source)
            if record is not None:
                candidates.append((source, record))

        base = next(
            (
                (source, record)
                for source, record in candidates
                if all(record[f] is not None for f in _POSITION_FIELDS)
            ),
            None,
        )
        if base is None:
            enabled = [s for s in _MERGE_PRIORITY if getattr(self, f"use_{s}")]
            raise ValueError(
                f"cannot build a canonical astrometry position for "
                f"{self.l0_obj.obs_id or 'unknown'}: missing ra/dec/pmra/pmdec/epoch "
                f"across enabled sources {enabled}"
            )
        base_source, base_record = base

        # parallax and rv are optional; take each from the highest-priority source
        # that has it, or leave missing.
        parallax_source, parallax_value = next(
            (
                (source, record["parallax"])
                for source, record in candidates
                if record["parallax"] is not None
            ),
            (None, None),
        )
        rv_source, rv_value = next(
            (
                (source, record["rv"])
                for source, record in candidates
                if record["rv"] is not None
            ),
            (None, None),
        )

        mixed = [
            f"{field}={src}"
            for field, src in (("parallax", parallax_source), ("rv", rv_source))
            if src is not None and src != base_source
        ]
        if mixed:
            logger.debug(
                "canonical astrometry mixes sources: position=%s, %s",
                base_source,
                ", ".join(mixed),
            )

        record = {
            "source_id": base_record["source_id"],
            "astr_src": base_source,
            "ra": base_record["ra"],
            "dec": base_record["dec"],
            "pmra": base_record["pmra"],
            "pmdec": base_record["pmdec"],
            "parallax": parallax_value,
            "rv": rv_value,
            "frame": base_record["frame"],
            "epoch": base_record["epoch"],
            "equinox": base_record["equinox"],
        }
        self._canonical = record
        self.l0_obj.set_catalog_record("kpf-drp", record)
        return record

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------

    def _track_info(self):
        """Build and cache the info() summary text from instance attributes."""
        gaia = self._gaia["source_id"] if self._gaia else "n/a"
        simbad = self._simbad["source_id"] if self._simbad else "n/a"
        canonical = self._canonical["astr_src"] if self._canonical else "n/a"
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

    def perform(self, *, use_gaia=None, use_simbad=None):
        """
        Resolve external catalog astrometry and write it to the L0.

        Parameters
        ----------
        use_gaia, use_simbad : bool, optional
            Override the configured query toggles for this call.

        Returns
        -------
        l0_obj : KPF0
            The input L0 (PRIMARY unchanged), now with its ``gaia``/``simbad`` rows
            written to the ``CATALOG_RECORD`` extension (the ``wmko`` row is native,
            populated at read) and an 'astro_query' receipt entry. Unusually for a
            pipeline module this returns an L0, not the next level -- AstroQuery
            runs before assembly.
        """
        if use_gaia is not None:
            self.use_gaia = use_gaia
        if use_simbad is not None:
            self.use_simbad = use_simbad

        self._gaia = self.query_gaia()
        self._simbad = self.query_simbad()

        self.l0_obj.set_catalog_record("gaia", self._gaia)
        self.l0_obj.set_catalog_record("simbad", self._simbad)
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
