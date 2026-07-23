"""Diagnostics for KPF Level 0 (raw CCD) data products."""

import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from kpfpipe.quality_control.diagnostics.base import Diagnostics

logger = logging.getLogger(__name__)

# Fields the pointing offset requires: RA/Dec (position) and epoch (the propagation
# baseline). A record missing either is unusable, so its offset is emitted present-
# but-empty. Proper motion and parallax are optional -- when absent they fall back to
# zero for the offset only (see _record_skycoord), leaving CATALOG_RECORD untouched.
# RA/Dec are sexagesimal strings, epoch a float.
_OFFSET_STR_FIELDS = ("ra", "dec")
_OFFSET_REQUIRED_NUM_FIELDS = ("epoch",)

# CATALOG_RECORD presence flag (int 0/1) per source, on the extension header.
_CATALOG_FLAGS = {"gaia": "GAIACR", "simbad": "SIMBADCR", "wmko": "WMKOCR"}


class DiagL0(Diagnostics):
    """Diagnostics for KPF Level 0 raw data products.

    The pointing-offset metrics compare the telescope pointing against the target
    astrometry AstroQuery resolves into the L0 ``CATALOG_RECORD`` extension (Gaia /
    SIMBAD / DCS, one canonical ICRS schema). An unavailable source yields a
    present-but-empty offset and a WARNING -- error-raising is the checkpoint's job.
    """

    LEVEL = "L0"

    def _catalog_record(self, source):
        """The CATALOG_RECORD row for ``source``, or None with a WARNING.

        Returns None when AstroQuery has not run (no presence flag), the source's
        record is absent (flag 0), or a present record lacks a position or epoch. The
        flag is written with the row, so flag 1 guarantees exactly one matching row.
        """
        hdr = self.kpf_obj.headers["CATALOG_RECORD"]
        keyword = _CATALOG_FLAGS[source]
        if keyword not in hdr:
            logger.warning(
                "no CATALOG_RECORD flags on L0 (run AstroQuery first); "
                "%s pointing offset unavailable",
                source,
            )
            return None
        if not hdr[keyword]:
            logger.warning(
                "no %s astrometry in CATALOG_RECORD; pointing offset unavailable",
                source,
            )
            return None
        table = self.kpf_obj.data["CATALOG_RECORD"]
        row = table[table["source"] == source][0]
        missing = any(
            str(row[field]).strip() == "" for field in _OFFSET_STR_FIELDS
        ) or any(np.isnan(row[field]) for field in _OFFSET_REQUIRED_NUM_FIELDS)
        if missing:
            logger.warning(
                "incomplete %s record in CATALOG_RECORD (no position or epoch); "
                "pointing offset unavailable",
                source,
            )
            return None
        return row

    def _record_skycoord(self, rec):
        """ICRS SkyCoord from a CATALOG_RECORD record.

        The canonical schema all three sources share: RA/Dec sexagesimal strings
        (RA hour-angle, Dec deg), proper motion arcsec/yr (RA incl. cos Dec),
        parallax mas, epoch in Julian years. Proper motion and parallax fall back to
        zero (no motion, no distance) when the record omits them -- for this offset
        only; the CATALOG_RECORD values are left untouched.
        """
        pmra, pmdec = float(rec["pmra"]), float(rec["pmdec"])
        parallax = float(rec["parallax"])
        pm_missing = np.isnan(pmra) or np.isnan(pmdec)
        plx_missing = np.isnan(parallax) or parallax <= 0
        if pm_missing or plx_missing:
            logger.debug(
                "%s record missing %s; using PM=0, parallax=0 for the offset",
                rec["source"],
                " and ".join(
                    n for n, m in (("PM", pm_missing), ("parallax", plx_missing)) if m
                ),
            )
        kwargs = {
            "ra": rec["ra"],
            "dec": rec["dec"],
            "unit": (u.hourangle, u.deg),
            "pm_ra_cosdec": (0.0 if pm_missing else pmra) * u.arcsec / u.yr,
            "pm_dec": (0.0 if pm_missing else pmdec) * u.arcsec / u.yr,
            "obstime": Time(float(rec["epoch"]), format="jyear"),
            "frame": str(rec["frame"]),
        }
        if not plx_missing:
            kwargs["distance"] = (1e3 / parallax) * u.pc
        return SkyCoord(**kwargs)

    def _offset(self, source):
        """Arcsec separation of the pointing from a catalog source at obs epoch.

        Returns None (present-but-empty) when the source astrometry is unavailable
        or malformed (an unparseable value is caught and emitted empty, not raised);
        otherwise the catalog position is proper-motion propagated to the obs epoch
        before the comparison.
        """
        rec = self._catalog_record(source)
        if rec is None:
            return None
        try:
            hdr = self.kpf_obj.headers["PRIMARY"]
            pointing = SkyCoord(hdr["RA"], hdr["DEC"], unit=(u.hourangle, u.deg))
            obs_time = Time(float(hdr["MJD-OBS"]), format="mjd")
            coord = self._record_skycoord(rec).apply_space_motion(new_obstime=obs_time)
            return round(float(pointing.separation(coord).arcsec), 4)
        except Exception as exc:
            logger.warning(
                "could not compute %s pointing offset (%s: %s); emitting empty",
                source,
                type(exc).__name__,
                exc,
            )
            return None

    def gaia_ra_dec_offset(self):
        """GAIAOFF: arcsec, RA/DEC pointing vs Gaia catalog position at obs epoch."""
        return self._tag(GAIAOFF=self._offset("gaia"))

    gaia_ra_dec_offset._diag_name = "gaia_ra_dec_offset"

    def target_ra_dec_offset(self):
        """TARGOFF: arcsec, RA/DEC pointing vs DCS target position at obs epoch."""
        return self._tag(TARGOFF=self._offset("wmko"))

    target_ra_dec_offset._diag_name = "target_ra_dec_offset"

    def object_ra_dec_offset(self):
        """OBJOFF: arcsec, RA/DEC pointing vs SIMBAD(OBJECT) position at obs epoch."""
        return self._tag(OBJOFF=self._offset("simbad"))

    object_ra_dec_offset._diag_name = "object_ra_dec_offset"
