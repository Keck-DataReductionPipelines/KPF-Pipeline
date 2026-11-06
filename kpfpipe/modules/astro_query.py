"""
KPF AstroQuery module.

Consolidates the pipeline's external astronomical-catalog lookups into a single
L0-stage module. Given a raw L0 science frame, it resolves the target's astrometry
from Gaia (by GAIAID, whose release prefix picks the data release queried) and SIMBAD
(by OBJECT) -- or, for a solar-system OBJECT, from JPL Horizons at the exposure
midpoint instead -- builds the telescope-native ``wmko`` row from the L0 PRIMARY
``TARG*`` astrometry (no query), merges them into a canonical ``kpf-drp`` row, and
writes all rows to the L0 ``CATALOG_RECORD`` extension.

All rows share one schema in the EPRV C*# PRIMARY format (see ``_CATALOG_SCHEMA``), so
the merged row is copied straight onto the L0 PRIMARY ``C*#`` cards. Every frame's
identity is then settled the same way (``_resolve_identity``): the header OBJECT and
GAIAID are checked against the catalog matches, and OBJECT and ``ALIASES`` are written
from the catalog aliases. AstroQuery writes no other PRIMARY card except to complete
GAIAID: the release Gaia resolves for a bare GAIAID (also on an OBJECT naming it), or
SIMBAD's Gaia designation for a blank one. Target names sent to or received from
SIMBAD go through ``kpf_target_name``.
"""

import logging

import astropy.units as u
import numpy as np
from astropy.coordinates import FK4, FK5, ICRS, Angle, Distance, SkyCoord
from astropy.table import Table
from astropy.time import Time

from kpfpipe import DEFAULT_CFG, DETECTOR, SCI_FIBERS, SOLAR_SYSTEM_OBJECTS
from kpfpipe.utils.astro import compute_redshift
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.header import HeaderError
from kpfpipe.utils.kpf import kpf_target_name
from kpfpipe.utils.network import (
    gaia_client,
    horizons_client,
    query_mirrors,
    simbad_client,
)

logger = logging.getLogger(__name__)

_DEFAULT_CFG = {
    **DEFAULT_CFG,
    "do_gaia_query": True,
    "do_simbad_query": True,
    "do_horizons_query": True,
    "astrometry_priority": ("gaia", "simbad", "horizons"),
    "query_timeout": 15,
}


class AstroQuery:
    """
    Resolve target astrometry from external catalogs and write it to an L0.

    Runs the two external catalog queries (Gaia by GAIAID, SIMBAD by OBJECT) -- or,
    for a solar-system OBJECT, the JPL Horizons query in their place -- builds the
    native ``wmko`` row from the L0 PRIMARY ``TARG*`` astrometry, and writes these
    rows and the merged ``kpf-drp`` row to the L0 ``CATALOG_RECORD`` extension for
    downstream use (EPRV ``C*#`` catalog keywords, DiagL0 pointing offsets,
    BarycentricCorrection). Only science frames are supported: the constructor raises
    on a non-``Target`` OBSTYPE or a SoCal (ISSOLAR) frame. Each query is fail-soft
    -- a missing GAIAID/OBJECT or a failed lookup yields a ``None`` record -- but
    ``perform`` raises if no query identifies the target.

    Parameters
    ----------
    l0_obj : KPF0
        Standardized L0 science frame (OBSTYPE ``Target``). Its PRIMARY header
        (OBSTYPE, ISSOLAR, GAIAID, OBJECT, TARG*) is read; OBJECT, ALIASES and GAIAID
        are written only as the module docstring says.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: do_gaia_query, do_simbad_query,
        do_horizons_query, astrometry_priority, query_timeout.
    """

    def __init__(self, l0_obj, config=None):
        self.l0_obj = l0_obj

        obstype = l0_obj.headers["PRIMARY"].get("OBSTYPE")
        if obstype != "Target":
            raise ValueError(
                f"AstroQuery runs only on science frames (OBSTYPE 'Target'); got "
                f"OBSTYPE={obstype!r}. It must not be called on a calibration frame."
            )
        if l0_obj.headers["PRIMARY"].get("ISSOLAR"):
            raise ValueError(
                "AstroQuery does not run on SoCal frames (ISSOLAR): the Sun has no "
                "catalog entry"
            )

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS", "TRACES", "MODULE_ASTRO_QUERY"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in _DEFAULT_CFG.items():
            setattr(self, k, params.get(k, v))
        self.chips = tuple(self.chips)
        self.fibers = tuple(self.fibers)

        for k, v in DETECTOR.items():
            setattr(self, k, v)
        self.nrow, self.ncol = self.ccd["nrow"], self.ccd["ncol"]

        self._validate_priority()

        self._wmko = None  # native WMKO record; set by read_wmko_header()
        self._gaia = None  # Gaia record; set by query_gaia()
        self._simbad = None  # SIMBAD record; set by query_simbad()
        self._horizons = None  # JPL Horizons record; set by query_horizons()
        self._common_name = None  # SIMBAD common name; set by query_simbad()
        self._simbad_ids = []  # SIMBAD ids (_simbad_name); set by query_simbad()
        self._canonical = None  # merged kpf-drp record; set by merge_catalog_records()
        self._info = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # Sources that can supply a CATALOG_RECORD row, highest catalog priority first.
    _SOURCES = ("gaia", "simbad", "horizons", "wmko")

    # Catalogs of the ids ``_resolve_identity`` names OBJECT from, most preferred
    # first; together the tiers list every catalog in
    # reference/star_catalog_aliases.csv. The low tier is the purely numeric ids,
    # which name OBJECT only when no high-tier id names the star.
    _HIGH_TIER = (
        "HD",
        "HIP",
        "TOI",
        "Kepler",
        "KOI",
        "K2",
        "WASP",
        "HR",
        "GJ",
        "HAT-P",
    )
    _LOW_TIER = ("Gaia", "TIC", "KIC")

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
        unknown = [s for s in self.astrometry_priority if s not in self._SOURCES]
        if unknown or not self.astrometry_priority:
            raise ValueError(
                f"astrometry_priority={list(self.astrometry_priority)} must be a "
                f"non-empty ordered subset of {list(self._SOURCES)}"
                + (f"; unknown source(s) {unknown}" if unknown else "")
            )

    # Queryable Gaia release -> ((TAP endpoint, gaia_source table), ...), in failover
    # order: ESA, then the GAVO and MAST mirrors. Each was checked to return values and
    # units identical to ESA's; IRSA (truncated values) and NOIRLab/AIP (units astropy
    # cannot parse) were not. MAST holds no DR2. DR1 and EDR3 are excluded: a release
    # absent here is neither probed nor accepted.
    _GAIA_MIRRORS = {
        "DR2": (
            ("https://gea.esac.esa.int/tap-server/tap", "gaiadr2.gaia_source"),
            ("https://gaia.ari.uni-heidelberg.de/tap", "gaiadr2.gaia_source"),
        ),
        "DR3": (
            ("https://gea.esac.esa.int/tap-server/tap", "gaiadr3.gaia_source"),
            ("https://gaia.ari.uni-heidelberg.de/tap", "gaiadr3.gaia_source"),
            ("https://mast.stsci.edu/vo-tap/api/v0.1/gaiadr3", "dbo.gaia_source"),
        ),
    }

    # SIMBAD servers in failover order: CDS, then its Harvard (CfA) mirror.
    _SIMBAD_MIRRORS = ("simbad.cds.unistra.fr", "simbad.harvard.edu")

    # JPL runs no Horizons mirror. astroquery reads this endpoint from its own config,
    # so the entry only labels query_mirrors' attempts (two passes, bounded budget).
    _HORIZONS_MIRRORS = ("https://ssd.jpl.nasa.gov/api/horizons.api",)

    def _query_gaia_mirrors(self, release, query):
        """Run ``query`` on ``release``'s mirrors in turn, its ``{table}`` filled per
        mirror; see ``query_mirrors``."""
        return query_mirrors(
            lambda m: (
                gaia_client(m[0], self.query_timeout)
                .run_sync(query.format(table=m[1]))
                .to_table()
            ),
            self._GAIA_MIRRORS[release],
            f"Gaia {release}",
        )

    def _resolve_gaia_release(self, source_id):
        """The newest Gaia release whose gaia_source contains ``source_id``.

        For a bare GAIAID only: rather than assume a release, probe the archive and take
        the newest one holding the id, since the same source_id denotes different stars
        in different releases. Only a probe that answers with no match moves on to an
        older release; a failed probe returns None (warned), since an older release may
        hold a different star under that id. An id no release holds raises rather than
        being mistaken for a resolved one.

        Parameters
        ----------
        source_id : str
            Digit-only Gaia source_id, with no release prefix.

        Returns
        -------
        str or None
            The release key into ``_GAIA_MIRRORS``, or None if a probe failed.

        Raises
        ------
        HeaderError
            No queryable release contains ``source_id`` (GAIAID must be fixed).
        """
        for release in sorted(self._GAIA_MIRRORS, reverse=True):
            try:
                found = self._query_gaia_mirrors(
                    release,
                    f"SELECT source_id FROM {{table}} WHERE source_id = {source_id}",
                )
            except Exception as e:
                logger.warning(
                    "could not probe Gaia %s for source_id %s (%s: %s); release "
                    "unresolved",
                    release,
                    source_id,
                    type(e).__name__,
                    e,
                )
                return None
            if len(found):
                logger.info("resolved bare GAIAID %s to Gaia %s", source_id, release)
                return release
        raise HeaderError(
            f"{self.l0_obj.obs_id or 'unknown'}: GAIAID {source_id} is in no "
            f"Gaia release ({', '.join(sorted(self._GAIA_MIRRORS))}). L0 header "
            "must be fixed: correct GAIAID or blank it."
        )

    @staticmethod
    def _simbad_name(simbad_id):
        """A SIMBAD identifier stripped of its marker ('NAME ', 'V*', '*') and
        padding, and canonicalized (``kpf_target_name``): 'HD  21019A' ->
        'HD 21019 A', '*  61 Cyg B' -> '61 Cyg B'."""
        name = simbad_id.strip().removeprefix("NAME ").removeprefix("V*").lstrip("*")
        return kpf_target_name(" ".join(name.split()))[1]

    @classmethod
    def _simbad_common_name(cls, row):
        """The common name in a SIMBAD result row: its proper name ('NAME Vega' among
        the ``ids``) else ``main_id``, SIMBAD's own preferred identifier ('* tau Cet'
        for HD 10700), via ``_simbad_name``. For a star with neither, main_id is a
        catalog designation that ALIASES de-duplicates.
        """
        ids = str(row["ids"]).split("|")
        return cls._simbad_name(
            next((i for i in ids if i.startswith("NAME ")), str(row["main_id"]))
        )

    def _resolve_identity(self):
        """Check the header identity against the catalog matches, then write the
        PRIMARY OBJECT and ALIASES -- the same rule for every frame.

        Aliases are the header OBJECT (unless ``Unknown``), the Gaia designation and
        every SIMBAD identifier in a known catalog (``kpf_target_name``); the SIMBAD
        common name goes on ALIASES only. OBJECT is the header OBJECT if high tier,
        so a good name is never renamed (TOI-2473 stays though SIMBAD lists
        HD 287319); else the most preferred high-tier alias; else the low-tier one
        (``_HIGH_TIER``/``_LOW_TIER``). ALIASES lists OBJECT, then the other
        catalogued aliases by preference, then the common name.

        Raises
        ------
        HeaderError
            When SIMBAD's match contradicts the header: its identifiers omit the
            header OBJECT, or list a Gaia id of GAIAID's release other than GAIAID.
            The cards name different stars; the L0 header must be fixed.
        """
        primary = self.l0_obj.headers["PRIMARY"]
        obj, gaiaid = primary["OBJECT"], primary["GAIAID"]
        label = self.l0_obj.obs_id or "unknown"
        simbad_ids = self._simbad_ids
        if self._simbad:
            listed = {i.casefold() for i in simbad_ids}
            if obj != "Unknown" and obj.casefold() not in listed:
                raise HeaderError(
                    f"{label}: OBJECT {obj!r} is not an identifier of SIMBAD's "
                    f"match ({self._common_name!r}). L0 header must be fixed: "
                    "correct OBJECT/TARGNAME."
                )
            release = gaiaid.rsplit(" ", 1)[0] + " " if gaiaid else None
            same_release = {
                i.split()[-1]
                for i in simbad_ids
                if release and i.startswith(release) and kpf_target_name(i)[0] == "Gaia"
            }
            if same_release and gaiaid.split()[-1] not in same_release:
                raise HeaderError(
                    f"{label}: GAIAID {gaiaid!r} and OBJECT {obj!r} name different "
                    f"stars: SIMBAD's match ({self._common_name!r}) is "
                    f"{release}{', '.join(sorted(same_release))}. L0 header must be "
                    "fixed: make OBJECT and GAIAID match."
                )
        elif obj not in SOLAR_SYSTEM_OBJECTS:  # SIMBAD is not queried for these
            logger.warning(
                "SIMBAD gave no identifiers; OBJECT and ALIASES from the header "
                "OBJECT and Gaia designation only"
            )

        def is_low_tier(name):
            return kpf_target_name(name)[0] in self._LOW_TIER

        def order(name):  # catalog preference; an uncatalogued header OBJECT first
            catalog = kpf_target_name(name)[0]
            return (self._HIGH_TIER + self._LOW_TIER).index(catalog) if catalog else -1

        names = [obj] if obj != "Unknown" else []
        names += [self._gaia["object"]] if self._gaia else []
        names += [i for i in simbad_ids if kpf_target_name(i)[0]]
        # De-duplicated; the stable sort keeps the header's, then Gaia's, first.
        names = sorted(dict.fromkeys(names), key=order)
        high = [n for n in names if not is_low_tier(n)]
        if obj != "Unknown" and not is_low_tier(obj):
            new = obj
        elif high:
            new = high[0]
        else:  # low tier only; Unknown stays only with both queries off
            new = obj if obj != "Unknown" or not names else names[0]
        aliases = "; ".join(
            dict.fromkeys(
                n for n in (new, *names, self._common_name) if n and n != "Unknown"
            )
        )
        if new != obj:
            self.l0_obj.set_keyword("OBJECT", new)
        resolved = f" (resolved from {obj!r})" if new != obj else ""
        logger.info("OBJECT %r%s; ALIASES %r", new, resolved, aliases)
        self.l0_obj.set_keyword("ALIASES", aliases)

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

    # CATALOG_RECORD write schema (AstroQuery is the sole populator): one row per
    # resolved source (wmko/gaia/simbad) plus the merged 'kpf-drp' row, in column
    # order. A None unit marks a string column -- RA/Dec are sexagesimal strings, so
    # not subject to astropy unit conversion. catalog labels the catalog behind the
    # position (CSRC#; e.g. "Gaia DR3", "WMKO TCS/DCS"). radec_src/plx_src/rv_src
    # name the source each value block came from -- its own for a source row, the
    # winner (or "" if none) for the merged row. Values are in the EPRV C*# PRIMARY
    # format: RA/Dec sexagesimal strings (ICRS), PM [arcsec/yr] (RA incl. cos Dec),
    # parallax [mas], rv [km/s], z the redshift derived from rv, epoch/equinox
    # [Julian yr].
    # color is a color index and color_name labels it (Gaia "Gaia BP-RP", SIMBAD
    # "B-V", WMKO "G-J"), both blank when a magnitude is missing. Missing floats ->
    # NaN, strings -> "".
    _CATALOG_SCHEMA = {
        "source": None,
        "object": None,
        "catalog": None,
        "radec_src": None,
        "plx_src": None,
        "rv_src": None,
        "ra": None,
        "dec": None,
        "pmra": u.arcsec / u.yr,
        "pmdec": u.arcsec / u.yr,
        "parallax": u.mas,
        "rv": u.km / u.s,
        "z": u.dimensionless_unscaled,
        "frame": None,
        "epoch": u.yr,
        "equinox": u.yr,
        "color": u.mag,
        "color_name": None,
    }

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
                    name: row[name] for name in self._CATALOG_SCHEMA
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
        for name, unit in self._CATALOG_SCHEMA.items():
            if unit is None:
                new_table[name] = np.array(
                    ["" if r.get(name) is None else r.get(name) for r in ordered],
                    dtype=str,
                )
            else:
                new_table[name] = np.array(
                    [np.nan if r.get(name) is None else r.get(name) for r in ordered],
                    dtype=float,
                )
                new_table[name].unit = unit
        l0.set_data("CATALOG_RECORD", new_table)

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    # Expected units, checked by _verify_units so an upstream schema
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

    def query_gaia(self):
        """Query Gaia for the target's ICRS astrometry, or None (fail-soft).

        The release named in GAIAID selects the queried tables (``_GAIA_MIRRORS``), so a
        DR2 id is never looked up among DR3 source_ids; a bare id's release is found by
        ``_resolve_gaia_release`` and, once its source is found, recorded on GAIAID (and
        on an OBJECT naming the same bare id). Returns None (warned) when GAIAID is
        blank, a lookup fails on every mirror, or the source is not found. Raises
        HeaderError if a bare id matches no queryable release, or ValueError if the
        result's column units differ from ``_GAIA_UNITS``.
        """
        primary = self.l0_obj.headers["PRIMARY"]
        gaiaid = primary["GAIAID"]
        if not gaiaid:
            logger.warning("no GAIAID on L0 PRIMARY; Gaia astrometry unavailable")
            return None
        # HeaderParser's canonical 'Gaia [DR2|DR3] <source_id>'.
        *release, gaia_id = gaiaid.split()[1:]
        release = release[0] if release else None
        bare = release is None
        if bare:
            logger.warning(
                "GAIAID %r carries no data release; querying Gaia to identify it",
                gaiaid,
            )
            release = self._resolve_gaia_release(gaia_id)
            if release is None:
                logger.warning("Gaia astrometry unavailable")
                return None
        query = f"""
        SELECT ra, dec, pmra, pmdec, parallax, radial_velocity, ref_epoch,
               phot_bp_mean_mag, phot_rp_mean_mag
        FROM {{table}}
        WHERE source_id = {gaia_id}
        """
        logger.info("querying Gaia %s for source_id %s", release, gaia_id)
        try:
            results = self._query_gaia_mirrors(release, query)
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
        self._verify_units(results, self._GAIA_UNITS, f"Gaia {release}")
        designation = f"Gaia {release} {gaia_id}"
        if bare:
            for key in ("GAIAID", "OBJECT"):
                if primary[key] == gaiaid:
                    logger.info("resolved %s %r to %r", key, gaiaid, designation)
                    self.l0_obj.set_keyword(key, designation)
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
            "object": designation,
            "catalog": f"Gaia {release}",
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

    # Expected result units, verified by _verify_units so a silent upstream schema
    # change fails loudly. The votable fields asked of SIMBAD are these keys less
    # ra/dec/main_id, which every query returns unasked.
    _SIMBAD_UNITS = {
        "ra": u.deg,
        "dec": u.deg,
        "pmra": u.mas / u.yr,
        "pmdec": u.mas / u.yr,
        "plx_value": u.mas,
        "rvz_radvel": u.km / u.s,
        "B": None,
        "V": None,
        "main_id": None,
        "ids": None,
        "coo_bibcode": None,
        "otype": None,
    }

    # SIMBAD coordinate reference (coo_bibcode) -> (catalog label for CSRC#, the
    # prefix of that catalog's designation among the ids, for CID#). EDR3 and DR3
    # share source_ids, and SIMBAD lists only the DR3 designation.
    _SIMBAD_CATALOGS = {
        "2020yCat.1350....0G": ("Gaia EDR3", "Gaia DR3 "),
        "2020A&A...649A...1G": ("Gaia EDR3", "Gaia DR3 "),
        "2022yCat.1355....0G": ("Gaia DR3", "Gaia DR3 "),
        "2023A&A...674A...1G": ("Gaia DR3", "Gaia DR3 "),
        "2018yCat.1345....0G": ("Gaia DR2", "Gaia DR2 "),
        "2018A&A...616A...1G": ("Gaia DR2", "Gaia DR2 "),
        "2007A&A...474..653V": ("Hipparcos (2007)", "HIP "),
        "1997A&A...323L..49P": ("Hipparcos (1997)", "HIP "),
        "1997HIP...C......0E": ("Hipparcos (1997)", "HIP "),
        "2000A&A...355L..27H": ("Tycho-2", "TYC "),
        "2003yCat.2246....0C": ("2MASS", "2MASS J"),
        "2006AJ....131.1163S": ("2MASS", "2MASS J"),
    }

    def _simbad_catalog(self, row, obj):
        """``(catalog, designation)`` of a SIMBAD row's coordinates: the catalog its
        coo_bibcode cites and the star's canonicalized id in it (``obj`` when SIMBAD
        lists none). An unrecognized bibcode is warned and labelled 'Simbad Query
        (catalog unknown)'.
        """
        bibcode = str(row["coo_bibcode"]).strip()
        if bibcode not in self._SIMBAD_CATALOGS:
            logger.warning(
                "SIMBAD coordinates for %r cite an unrecognized catalog (coo_bibcode "
                "%r)",
                obj,
                bibcode,
            )
            return "Simbad Query (catalog unknown)", obj
        catalog, prefix = self._SIMBAD_CATALOGS[bibcode]
        ids = (i.strip() for i in str(row["ids"]).split("|"))
        designation = next((i for i in ids if i.startswith(prefix)), None)
        return catalog, kpf_target_name(designation)[1] if designation else obj

    def _fill_gaiaid(self, row):
        """Fill a blank PRIMARY GAIAID with the SIMBAD row's Gaia designation (DR3,
        else DR2), if it parses as a Gaia id (warned either way)."""
        if self.l0_obj.headers["PRIMARY"]["GAIAID"]:
            return
        ids = [i.strip() for i in str(row["ids"]).split("|")]
        simbad_id = next(
            (i for p in ("Gaia DR3 ", "Gaia DR2 ") for i in ids if i.startswith(p)),
            None,
        )
        if not simbad_id:
            return
        catalog, gaiaid, _ = kpf_target_name(simbad_id)
        if catalog == "Gaia":
            logger.warning("GAIAID blank; filled from SIMBAD as %r", gaiaid)
            self.l0_obj.set_keyword("GAIAID", gaiaid)
        else:
            logger.warning(
                "SIMBAD Gaia id %r is not a Gaia DR2/DR3 source_id; GAIAID blank",
                simbad_id,
            )

    def query_simbad(self):
        """Query SIMBAD for the OBJECT's ICRS J2000 astrometry, or None (fail-soft).

        OBJECT is sent canonical; an ``Unknown`` OBJECT (often alias cards; see
        HeaderParser._target_object) sends GAIAID instead, which HeaderParser
        guarantees is set, so SIMBAD's identifiers can name the star
        (``_resolve_identity``). A blank GAIAID is filled (``_fill_gaiaid``). Returns
        None (warned) when the name sent is ``Unknown`` or a Gaia designation without
        a release, when the lookup fails on every mirror or resolves nothing, or when
        it resolves to a planet (SIMBAD matches case-insensitively, so a component
        letter can name one). Raises ValueError if the result's column units differ
        from ``_SIMBAD_UNITS`` (astroquery 0.4.11's lowercase form, with the Johnson
        B/V magnitudes unitless).
        """
        primary = self.l0_obj.headers["PRIMARY"]
        obj = primary["OBJECT"]
        name = primary["GAIAID"] if obj == "Unknown" and primary["GAIAID"] else obj
        catalog, query_name, _ = kpf_target_name(name)
        # SIMBAD resolves a Gaia designation only with its release.
        if name == "Unknown" or (catalog == "Gaia" and len(query_name.split()) == 2):
            logger.warning(
                "%r is not a SIMBAD name (OBJECT %r); SIMBAD astrometry unavailable",
                name,
                obj,
            )
            return None
        fields = tuple(
            f for f in self._SIMBAD_UNITS if f not in ("ra", "dec", "main_id")
        )
        logger.info("querying SIMBAD for %r", query_name)
        try:
            result = query_mirrors(
                lambda m: simbad_client(m, fields, self.query_timeout).query_object(
                    query_name
                ),
                self._SIMBAD_MIRRORS,
                "SIMBAD",
            )
        except Exception as e:
            logger.warning(
                "SIMBAD query failed (%s: %s); SIMBAD astrometry unavailable",
                type(e).__name__,
                e,
            )
            return None
        if result is None or len(result) == 0:
            logger.warning(
                "SIMBAD returned no match for %r; SIMBAD astrometry unavailable",
                query_name,
            )
            return None
        self._verify_units(result, self._SIMBAD_UNITS, "SIMBAD")
        row = result[0]
        if str(row["otype"]).strip() in ("Pl", "Pl?"):
            logger.warning(
                "SIMBAD resolved %r to planet %r; SIMBAD astrometry unavailable",
                query_name,
                str(row["main_id"]),
            )
            return None
        self._common_name = self._simbad_common_name(row)
        self._simbad_ids = [
            self._simbad_name(i) for i in str(row["ids"]).split("|") if i.strip()
        ]
        self._fill_gaiaid(row)
        ra, dec = self._scalar(row["ra"]), self._scalar(row["dec"])
        pmra, pmdec = self._scalar(row["pmra"]), self._scalar(row["pmdec"])
        # To the EPRV C*# format: deg -> sexagesimal; PM mas/yr -> arcsec/yr.
        ra_str, dec_str = self._sexagesimal_radec(
            SkyCoord(ra, dec, unit=u.deg, frame="icrs")
        )
        color, color_name = self._color(
            self._scalar(row["B"]), self._scalar(row["V"]), "B-V"
        )
        catalog, designation = self._simbad_catalog(row, query_name)
        record = {
            "object": designation,
            "catalog": catalog,
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

    # Expected result units, verified by _verify_units: astrometric ICRS RA/Dec, their
    # rates (RA incl. cos Dec) and the observer range -- Horizons quantities 1, 3, 20.
    _HORIZONS_UNITS = {
        "RA": u.deg,
        "DEC": u.deg,
        "RA_rate": u.arcsec / u.hour,
        "DEC_rate": u.arcsec / u.hour,
        "delta": u.au,
    }

    def query_horizons(self):
        """Query JPL Horizons for a solar-system OBJECT's ICRS position, or None.

        Fail-soft like the catalog queries. The body is looked up by its Horizons ID
        (``SOLAR_SYSTEM_OBJECTS``) as seen from Mauna Kea (site 568) at the exposure
        midpoint, which becomes the record epoch; the RA/Dec rates and the range stand
        in for proper motion and parallax at that epoch. Returns None (warned) when
        the lookup fails or returns nothing. Raises ValueError if the result's column
        units differ from ``_HORIZONS_UNITS``.
        """
        primary = self.l0_obj.headers["PRIMARY"]
        horizons_id = SOLAR_SYSTEM_OBJECTS[primary["OBJECT"]]
        midpoint = Time(primary["JD_UTC"], format="jd", scale="utc") + (
            float(primary["EXPTIME"]) / 2 * u.s
        )
        logger.info(
            "querying JPL Horizons for %r (id %r)", primary["OBJECT"], horizons_id
        )
        try:
            result = query_mirrors(
                lambda _: horizons_client(
                    horizons_id, "568", midpoint.jd, self.query_timeout
                ).ephemerides(quantities="1,3,20", cache=False),
                self._HORIZONS_MIRRORS,
                "JPL Horizons",
            )
        except Exception as e:
            logger.warning(
                "JPL Horizons query failed (%s: %s); Horizons astrometry unavailable",
                type(e).__name__,
                e,
            )
            return None
        if len(result) == 0:
            logger.warning(
                "JPL Horizons returned nothing for id %r; Horizons astrometry "
                "unavailable",
                horizons_id,
            )
            return None
        self._verify_units(result, self._HORIZONS_UNITS, "JPL Horizons")
        row = result[0]
        ra, dec = self._scalar(row["RA"]), self._scalar(row["DEC"])
        pmra, pmdec = self._scalar(row["RA_rate"]), self._scalar(row["DEC_rate"])
        delta = self._scalar(row["delta"])
        # To the EPRV C*# format: deg -> sexagesimal; rates arcsec/hr -> arcsec/yr;
        # range -> parallax [mas].
        ra_str, dec_str = self._sexagesimal_radec(
            SkyCoord(ra, dec, unit=u.deg, frame="icrs")
        )
        per_yr = (u.arcsec / u.hour).to(u.arcsec / u.yr)
        record = {
            "object": horizons_id,
            "catalog": "JPL Horizons",
            "ra": ra_str,
            "dec": dec_str,
            "pmra": None if pmra is None else pmra * per_yr,
            "pmdec": None if pmdec is None else pmdec * per_yr,
            "parallax": (
                None
                if delta is None
                else Distance(delta * u.au).parallax.to_value(u.mas)
            ),
            "rv": None,
            "frame": "icrs",
            "epoch": midpoint.jyear,
            "equinox": 2000.0,
            "color": None,
            "color_name": None,
        }
        logger.info("successfully built record for horizons")
        self._write_catalog_record("horizons", record)
        return record

    def read_wmko_header(self):
        """Read the native WMKO/DCS astrometry from L0 PRIMARY TARG*, or None.

        The telescope-side counterpart to query_gaia/query_simbad: no query, just the
        raw TARG* pointing sanitized to the EPRV C*# format and rotated from its native
        frame to ICRS, so all three sources share one frame. KPF pointing is FK5
        (Julian TARGEQUI, default J2000) or, for some planet targets, FK4 (Besselian
        TARGEQUI, default B1950); any other or absent TARGFRAM raises rather than
        being coerced -- a wrong frame would corrupt the barycentric correction.
        Returns None (no row)
        when TARGRA is absent, or (warned) when the TARG* astrometry cannot be
        parsed. Always run: TCSOFF needs this row even when ``astrometry_priority``
        bars wmko from anchoring the position; without it the diagnostic reports
        itself not computed.
        """
        native = self.l0_obj.headers["INSTRUMENT_HEADER"]
        if native.get("TARGRA") is None:
            return None
        targfram = str(native.get("TARGFRAM") or "").strip().upper()
        if targfram not in ("FK5", "FK4"):
            raise ValueError(
                f"unexpected TARGFRAM={native.get('TARGFRAM')!r}; KPF pointing must "
                "be 'FK5' or 'FK4'. Refusing to guess the frame, which would corrupt "
                "the barycentric correction."
            )
        try:
            ra = Angle(native["TARGRA"], unit=u.hourangle)
            dec = Angle(native["TARGDEC"], unit=u.deg)
            pmra, pmdec = (
                self._scalar(native.get("TARGPMRA")),
                self._scalar(native.get("TARGPMDC")),
            )
            equinox = self._scalar(native.get("TARGEQUI"))

            # Rotate the native pointing to ICRS (no time propagation, so epoch is
            # unchanged); proper motion only when both components are present, since a
            # lone one is meaningless under rotation. TARGPMRA time-s/yr -> arcsec/yr.
            # FK5 equinoxes are Julian, FK4 Besselian.
            if targfram == "FK5":
                frame = FK5(
                    equinox="J2000"
                    if equinox is None
                    else Time(equinox, format="jyear")
                )
            else:
                frame = FK4(
                    equinox="B1950"
                    if equinox is None
                    else Time(equinox, format="byear")
                )
            components = {"ra": ra, "dec": dec, "frame": frame}
            has_pm = pmra is not None and pmdec is not None
            if has_pm:
                cosdec = np.cos(dec.radian)
                components["pm_ra_cosdec"] = pmra * 15.0 * cosdec * u.arcsec / u.yr
                components["pm_dec"] = pmdec * u.arcsec / u.yr
            icrs = SkyCoord(**components).transform_to(ICRS())

            ra_str, dec_str = self._sexagesimal_radec(icrs)
            color, color_name = self._color(
                self._scalar(native.get("GAIAMAG")),
                self._scalar(native.get("2MASSMAG")),
                "G-J",
            )
            record = {
                "object": self.l0_obj.headers["PRIMARY"].get("OBJECT"),
                "catalog": "WMKO TCS/DCS",
                "ra": ra_str,
                "dec": dec_str,
                "pmra": icrs.pm_ra_cosdec.to_value(u.arcsec / u.yr) if has_pm else None,
                "pmdec": icrs.pm_dec.to_value(u.arcsec / u.yr) if has_pm else None,
                "parallax": self._scalar(native.get("TARGPLAX")),
                "rv": self._scalar(native.get("TARGRADV")),
                "frame": "icrs",
                "epoch": self._scalar(native.get("TARGEPOC")),
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
        telescope TARGRADV on PRIMARY, never to another catalog -- except for a
        solar-system OBJECT, which needs no systemic rv and keeps it missing.
        ``color`` and ``color_name`` also ride along, but a base source without a
        color borrows one from any built row, with a mixed-catalog warning -- except
        for a solar-system OBJECT, whose color stays missing. ``radec_src``/``plx_src``
        name
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
            for source in self._SOURCES
            if getattr(self, f"_{source}") is not None
        ]

        # The astrometric block the merge takes whole from one source: proper motion
        # is meaningless without the parallax it was measured against, and both need
        # the frame/epoch that qualify the coordinates. equinox and rv are handled
        # separately.
        astrometry = ("ra", "dec", "pmra", "pmdec", "parallax", "epoch", "frame")

        # First candidate in astrometry_priority order with a complete block wins;
        # an incomplete one ahead of it is demoted with a warning.
        base_source, base_record = None, None
        gaps = {}  # source -> why it cannot be the base, for the error below
        for source in self.astrometry_priority:
            record = getattr(self, f"_{source}")
            if record is None:
                gaps[source] = "no record"
                continue
            missing = [field for field in astrometry if record[field] is None]
            if not missing:
                base_source, base_record = source, record
                break
            gaps[source] = f"missing {', '.join(missing)}"
            logger.warning(
                "%s astrometry incomplete (missing %s); skipping it as the "
                "astrometric base",
                source,
                ", ".join(missing),
            )
        if base_record is None:
            reasons = "; ".join(f"{src}: {why}" for src, why in gaps.items())
            raise ValueError(
                f"{self.l0_obj.obs_id or 'unknown'}: target identified but no "
                f"complete astrometric solution: no source in astrometry_priority "
                f"{list(self.astrometry_priority)} supplies all of "
                f"{'/'.join(astrometry)} ({reasons}). Barycentric correction needs "
                "position, proper motion and parallax, so a target without them "
                "-- a non-stellar object (galaxy, transient, nebula) or a Gaia "
                "source with only a two-parameter solution -- is unsupported. DRP "
                "limitation, not an L0 header problem."
            )

        is_ss = self.l0_obj.headers["PRIMARY"].get("OBJECT") in SOLAR_SYSTEM_OBJECTS

        # rv from the base source, else telescope TARGRADV off PRIMARY -- independent of
        # astrometry_priority, which governs only the position. A solar-system body has
        # no systemic rv (Horizons supplies none), so it stays missing.
        rv_value = base_record["rv"]
        rv_source = base_source if rv_value is not None else ""
        if rv_value is None and not is_ss:
            rv_value = self._scalar(
                self.l0_obj.headers["INSTRUMENT_HEADER"].get("TARGRADV")
            )
            rv_source = "wmko" if rv_value is not None else ""
            if rv_value is not None:
                logger.warning(
                    "no catalog supplied a radial velocity; falling back to the "
                    "telescope TARGRADV=%s km/s from L0 PRIMARY",
                    rv_value,
                )

        # A color index is independent of the astrometry, so borrowing one from another
        # catalog when the base source has none is acceptable, but flagged.
        # Never for a solar-system body: a header G-J color is not its color.
        color = base_record["color"]
        color_name = base_record["color_name"]
        if color is None and not is_ss:
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
            "catalog": base_record["catalog"],
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

    # CATALOG_RECORD canonical column -> EPRV C*# keyword base. The row is stored in
    # EPRV C*# format already, so the overlay is a direct copy (no conversion).
    _CATALOG_CARD_BASES = {
        "object": "CID",
        "catalog": "CSRC",
        "ra": "CRA",
        "dec": "CDEC",
        "pmra": "CPMR",
        "pmdec": "CPMD",
        "parallax": "CPLX",
        "rv": "CRV",
        "z": "CZ",
        "epoch": "CEPCH",
        "equinox": "CEQNX",
        "color": "CCLR",
        "color_name": "CCLRN",
    }

    def _catalog_primary_cards(self):
        """Map the merged CATALOG_RECORD 'kpf-drp' row onto the SCI-fiber C*# cards.

        Returns ``{C-keyword: value}`` for every science fiber, at its 1-based trace
        index -- a direct copy of the canonical row's already-EPRV-format cells,
        skipping any missing value (NaN / "") so the card keeps the blank the seed
        stamped rather than carrying 'nan'. Warns when the canonical astrometry was
        assembled from more than one catalog.
        """
        table = self.l0_obj.data["CATALOG_RECORD"]
        row = table[table["source"] == "kpf-drp"][0]

        provenance = {
            str(row[col]).strip()
            for col in ("radec_src", "plx_src", "rv_src")
            if str(row[col]).strip()
        }
        if len(provenance) > 1:
            logger.warning(
                "PRIMARY catalog keywords assembled from mixed sources "
                "(radec=%s, parallax=%s, rv=%s); C*# astrometry is not "
                "internally single-source",
                row["radec_src"],
                str(row["plx_src"]) or '""',
                str(row["rv_src"]) or '""',
            )

        cards = {}
        for col, base in self._CATALOG_CARD_BASES.items():
            value = row[col]
            if isinstance(value, str):
                if value.strip() == "":
                    continue
            else:
                if np.isnan(value):
                    continue
                value = float(value)
            # SCI traces only. SKY and CAL have no object on the fiber, so their
            # catalog cards keep HeaderParser's CSRC/CID "None" and stay blank
            # otherwise -- not a gap. (Most C*# types admit no "N/A".)
            for fiber in SCI_FIBERS:
                cards[f"{base}{self.fiber_positions[fiber] + 1}"] = value
        return cards

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------

    def _track_info(self):
        """Build and cache the info() summary text from instance attributes."""
        gaia = self._gaia["object"] if self._gaia else "n/a"
        simbad = self._simbad["object"] if self._simbad else "n/a"
        horizons = self._horizons["object"] if self._horizons else "n/a"
        canonical = self._canonical["radec_src"] if self._canonical else "n/a"
        lines = [
            "AstroQuery",
            f"  obs_id:  {self.l0_obj.obs_id or 'unknown'}",
            f"  Gaia:      {gaia}",
            f"  SIMBAD:    {simbad}",
            f"  Horizons:  {horizons}",
            f"  canonical position source:  {canonical}",
        ]
        self._info = "\n\n" + "\n".join(lines) + "\n\n"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(
        self,
        *,
        do_gaia_query=None,
        do_simbad_query=None,
        do_horizons_query=None,
        astrometry_priority=None,
    ):
        """
        Resolve external catalog astrometry and write it to the L0.

        Parameters
        ----------
        do_gaia_query, do_simbad_query, do_horizons_query : bool, optional
            Override the configured catalog-query toggles for this call.
        astrometry_priority : sequence of str, optional
            Override which sources may anchor the merged position, highest first.

        Returns
        -------
        l0_obj : KPF0
            The input L0, with the ``wmko``/``gaia``/``simbad`` (``horizons`` for a
            solar-system OBJECT) and merged ``kpf-drp`` rows written to
            ``CATALOG_RECORD``, the canonical astrometry overlaid onto the
            SCI-fiber ``C*#`` PRIMARY cards, OBJECT and ``ALIASES`` written by
            ``_resolve_identity``, plus an 'astro_query' receipt entry. A blank
            GAIAID that SIMBAD fills is queried on Gaia after SIMBAD.
            Unusually for a pipeline module this returns an L0, not the next level --
            AstroQuery runs before assembly.

        Raises
        ------
        HeaderError
            A bare GAIAID that no Gaia release contains, or a header OBJECT/GAIAID
            that SIMBAD's match contradicts (the L0 header must fix it).
        ValueError
            No enabled catalog query resolved the target (identity unresolved), or
            no source named in ``astrometry_priority`` supplied a complete
            astrometric block.
        """
        if do_gaia_query is not None:
            self.do_gaia_query = do_gaia_query
        if do_simbad_query is not None:
            self.do_simbad_query = do_simbad_query
        if do_horizons_query is not None:
            self.do_horizons_query = do_horizons_query
        if astrometry_priority is not None:
            self.astrometry_priority = astrometry_priority
            self._validate_priority()

        self._wmko = self.read_wmko_header()
        primary = self.l0_obj.headers["PRIMARY"]
        if primary["OBJECT"] in SOLAR_SYSTEM_OBJECTS:
            # Gaia and SIMBAD hold no solar-system bodies; Horizons replaces both.
            self._horizons = self.query_horizons() if self.do_horizons_query else None
            queries = (("JPL Horizons", self.do_horizons_query, self._horizons),)
        else:
            gaiaid_blank = self.do_gaia_query and not primary["GAIAID"]
            self._gaia = (
                self.query_gaia() if self.do_gaia_query and primary["GAIAID"] else None
            )
            self._simbad = self.query_simbad() if self.do_simbad_query else None
            # Gaia runs first so a bare GAIAID's resolution reaches OBJECT before
            # SIMBAD; a blank GAIAID waits for SIMBAD's fill (query_gaia warns if
            # none came).
            if gaiaid_blank:
                self._gaia = self.query_gaia()
            queries = (
                ("Gaia", self.do_gaia_query, self._gaia),
                ("SIMBAD", self.do_simbad_query, self._simbad),
            )

        # HeaderParser passes a Target only with a name or a GAIAID to look up
        # (else it raises); if no catalog resolves either, the target is
        # unidentified. Raise that here, not as a merge failure, and never fall
        # back to the telescope pointing (wmko) for an unidentified target.
        queried = [name for name, enabled, _ in queries if enabled]
        if queried and all(record is None for _, _, record in queries):
            gaiaid = primary["GAIAID"]
            raise ValueError(
                f"{self.l0_obj.obs_id or 'unknown'}: target identity unresolved: "
                f"{' and '.join(queried)} resolved no source for OBJECT "
                f"{primary['OBJECT']!r}, "
                f"{f'GAIAID {gaiaid!r}' if gaiaid else 'GAIAID blank'} "
                "(reasons logged above). If a catalog query failed, rerun once the "
                "service is up; if the catalogs returned no match, L0 header must "
                "be fixed: correct OBJECT/TARGNAME or set GAIAID."
            )

        self.merge_catalog_records()

        # Overlay the canonical astrometry onto the SCI-fiber C*# cards. The seed
        # has already stamped every member of each family, so this is pure
        # enrichment: a card with no catalog value stays present and blank.
        for keyword, value in self._catalog_primary_cards().items():
            self.l0_obj.set_keyword(keyword, value)
        self._resolve_identity()

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
