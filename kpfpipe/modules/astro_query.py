"""
KPF AstroQuery module.

Consolidates every external astronomical-catalog lookup the pipeline needs into
a single L0-stage module. Given a raw L0 frame, it resolves the target's
astrometry from Gaia DR3 (by GAIAID) and SIMBAD (by OBJECT), and snapshots the
DCS/TCS target astrometry already on the raw header, then hands all three back on
a lightweight ``catalog_query`` dict attached to the L0 object.

The dict is the bridge across an ordering problem: the query results ultimately
belong on the EPRV PRIMARY catalog keywords (``C*#``), but the WMKO -> EPRV
PRIMARY conversion does not happen until ``KPF0.to_kpf1()`` downstream. Rather
than write headers here, AstroQuery tracks the queried quantities on the L0 and
lets ``to_kpf1()`` overlay them onto the L1 PRIMARY (a follow-up integration).

AstroQuery never modifies the L0 PRIMARY header -- that header is a pure
pass-through to INSTRUMENT_HEADER. Values are stored raw, in each source's native
units; unit conversion, per-fiber fan-out, and derived quantities (offsets,
redshift) are the jobs of the downstream consumers, not this module.
"""

import logging

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
}

# WMKO/DCS target-astrometry cards on the raw L0 PRIMARY, snapshotted verbatim
# (native units) into the 'wmko' record. Consumers apply the unit conventions
# (e.g. TARGPMRA is time-seconds/yr) exactly as BarycentricCorrection/DiagL0 do.
_WMKO_KEYS = {
    "ra": "TARGRA",
    "dec": "TARGDEC",
    "pmra": "TARGPMRA",
    "pmdec": "TARGPMDC",
    "parallax": "TARGPLAX",
    "rv": "TARGRADV",
    "frame": "TARGFRAM",
    "epoch": "TARGEPOC",
    "equinox": "TARGEQUI",
}


class AstroQuery:
    """
    Resolve target astrometry from external catalogs and attach it to an L0.

    Runs the two external catalog queries (Gaia DR3 by GAIAID, SIMBAD by OBJECT)
    plus a verbatim snapshot of the DCS/TCS ``TARG*`` astrometry, and deposits all
    three on ``l0_obj.catalog_query`` for downstream use (EPRV ``C*#`` catalog
    keywords, DiagL0 pointing offsets, BarycentricCorrection). Fail-soft: a frame
    with no GAIAID/OBJECT (e.g. a calibration) or a failed network lookup yields a
    ``None`` record rather than an error.

    Parameters
    ----------
    l0_obj : KPF0
        Raw L0 frame. Its PRIMARY header (GAIAID, OBJECT, TARG*) is read but never
        modified; the resolved catalog data is attached as ``l0_obj.catalog_query``.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: use_gaia, use_simbad.
    """

    def __init__(self, l0_obj, config=None):
        self.l0_obj = l0_obj

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
        primary = self.l0_obj.headers.get("PRIMARY")
        raw = primary.get("GAIAID") if primary is not None else None
        if raw is None:
            return None
        token = str(raw).strip().split()[-1] if str(raw).strip() else ""
        return token if token.isdigit() else None

    def _object_name(self):
        """SIMBAD-resolvable name from L0 PRIMARY OBJECT, or None if absent.

        KPF OBJECT for standard stars is a bare HD number (e.g. '10700') that
        SIMBAD resolves only with an 'HD ' prefix; named targets pass through.
        """
        primary = self.l0_obj.headers.get("PRIMARY")
        obj = primary.get("OBJECT") if primary is not None else None
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

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def read_wmko_target(self):
        """Snapshot the DCS/TCS ``TARG*`` target astrometry from L0 PRIMARY.

        Returns the raw header values (native units, no conversion) keyed by the
        record schema, or None when the frame carries no target pointing (TARGRA
        absent -- e.g. a calibration frame).
        """
        primary = self.l0_obj.headers.get("PRIMARY")
        if primary is None or primary.get("TARGRA") is None:
            return None
        record = {"source_id": primary.get("OBJECT")}
        record.update({field: primary.get(card) for field, card in _WMKO_KEYS.items()})
        return record

    def query_gaia(self):
        """Query Gaia DR3 for the target's ICRS astrometry, or None (fail-soft).

        Returns None when disabled, when L0 GAIAID yields no usable source_id, or
        when the lookup fails (warned, not raised). Values are Gaia-native: deg,
        mas/yr, mas, km/s, and ref_epoch in Julian years.
        """
        if not self.use_gaia:
            return None
        gaia_id = self._gaia_source_id()
        if gaia_id is None:
            logger.debug("no usable GAIAID on L0 PRIMARY; skipping Gaia query")
            return None
        query = f"""
        SELECT ra, dec, pmra, pmdec, parallax, radial_velocity, ref_epoch
        FROM gaiadr3.gaia_source
        WHERE source_id = {gaia_id}
        """
        logger.info("querying Gaia DR3 for source_id %s", gaia_id)
        try:
            row = Gaia.launch_job(query).get_results()[0]
        except Exception as e:
            logger.warning("Gaia query failed (%s: %s); skipping", type(e).__name__, e)
            return None
        return {
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

    def query_simbad(self):
        """Query SIMBAD for the OBJECT's ICRS J2000 astrometry, or None (fail-soft).

        Returns None when disabled, when L0 has no OBJECT name, or when the lookup
        fails / resolves nothing (warned, not raised). Column schema is the
        astroquery 0.4.11 lowercase form (ra/dec deg, pmra/pmdec mas/yr, plx_value
        mas, rvz_radvel km/s).
        """
        if not self.use_simbad:
            return None
        name = self._object_name()
        if name is None:
            logger.debug("no OBJECT name on L0 PRIMARY; skipping SIMBAD query")
            return None
        logger.info("querying SIMBAD for %r", name)
        try:
            simbad = Simbad()
            simbad.add_votable_fields("pmra", "pmdec", "plx", "rvz_radvel")
            result = simbad.query_object(name)
        except Exception as e:
            logger.warning(
                "SIMBAD query failed (%s: %s); skipping", type(e).__name__, e
            )
            return None
        if result is None or len(result) == 0:
            logger.warning("SIMBAD returned no match for %r; skipping", name)
            return None
        row = result[0]
        return {
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

    def _attach_catalog_query(self, l0_obj):
        """Deposit the resolved catalog records on ``l0_obj.catalog_query``.

        The module's sole output site (analogous to ``_set_headers`` on a
        transform module). No header is written: the L0 PRIMARY is an immutable
        pass-through to INSTRUMENT_HEADER, and the EPRV ``C*#`` keywords live on
        the L1 PRIMARY, which ``KPF0.to_kpf1()`` builds downstream from this dict.
        """
        l0_obj.catalog_query = {
            "gaia": self._gaia,
            "simbad": self._simbad,
            "wmko": self._wmko,
        }

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
            The input L0 (PRIMARY unchanged), now carrying ``catalog_query`` with
            the ``gaia`` / ``simbad`` / ``wmko`` records (each a dict or None), and
            an 'astro_query' receipt entry. Unusually for a pipeline module this
            returns an L0, not the next level -- AstroQuery runs before assembly.
        """
        if use_gaia is not None:
            self.use_gaia = use_gaia
        if use_simbad is not None:
            self.use_simbad = use_simbad

        self._wmko = self.read_wmko_target()
        self._gaia = self.query_gaia()
        self._simbad = self.query_simbad()

        self._attach_catalog_query(self.l0_obj)
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
