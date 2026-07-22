"""Diagnostics for KPF Level 0 (raw CCD) data products."""

import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from kpfpipe.quality_control.diagnostics.base import Diagnostics

logger = logging.getLogger(__name__)

# Record fields the pointing offset needs; a missing one (an empty RA/Dec string, a
# NaN measurement, or a non-positive parallax -- routine for faint Gaia sources)
# makes the source unusable, and the offset is emitted present-but-empty. RA/Dec are
# sexagesimal strings (EPRV C*# format), the rest floats. 'frame' is a fixed literal
# AstroQuery always sets, so it is not gated here.
_OFFSET_STR_FIELDS = ("ra", "dec")
_OFFSET_NUM_FIELDS = ("pmra", "pmdec", "parallax", "epoch")

# CATALOG_RECORD presence flag (int 0/1) per source, on the extension header.
_CATALOG_FLAGS = {"gaia": "GAIACR", "simbad": "SIMBADCR", "wmko": "WMKOCR"}


class DiagL0(Diagnostics):
    """Diagnostics for KPF Level 0 raw data products.

    The pointing-offset metrics compare the telescope pointing against the target
    astrometry resolved upstream by AstroQuery and written to the L0
    ``CATALOG_RECORD`` extension (Gaia / SIMBAD / DCS, all in one canonical ICRS
    schema). When a source's astrometry is unavailable the offset is emitted
    present-but-empty (a valueless card) and a WARNING is logged -- diagnostics
    record what they can and leave error-raising to the checkpoint layer.
    """

    LEVEL = "L0"

    def _catalog_record(self, source):
        """The CATALOG_RECORD row for ``source``, or None with a WARNING.

        Handles the three contingencies: AstroQuery not run (no presence flag on the
        CATALOG_RECORD header), the source's record absent (flag 0 -- lookup
        disabled/failed, or WMKO unavailable), or a record present but missing a
        measurement the offset needs. The flag is written with the row, so flag 1
        guarantees exactly one matching row.
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
        missing = (
            any(str(row[field]).strip() == "" for field in _OFFSET_STR_FIELDS)
            or any(np.isnan(row[field]) for field in _OFFSET_NUM_FIELDS)
            or float(row["parallax"]) <= 0
        )
        if missing:
            logger.warning(
                "incomplete %s record in CATALOG_RECORD; pointing offset unavailable",
                source,
            )
            return None
        return row

    def _record_skycoord(self, rec):
        """ICRS SkyCoord from a CATALOG_RECORD record (EPRV C*# format).

        Every source ('gaia'/'simbad'/'wmko') is sanitized to the same schema --
        ICRS, RA/Dec sexagesimal strings (RA hour-angle, Dec deg), proper motion
        arcsec/yr (RA incl. cos Dec), parallax mas, epoch in Julian years -- so one
        builder serves all three.
        """
        return SkyCoord(
            ra=rec["ra"],
            dec=rec["dec"],
            unit=(u.hourangle, u.deg),
            pm_ra_cosdec=float(rec["pmra"]) * u.arcsec / u.yr,
            pm_dec=float(rec["pmdec"]) * u.arcsec / u.yr,
            distance=(1e3 / float(rec["parallax"])) * u.pc,
            obstime=Time(float(rec["epoch"]), format="jyear"),
            frame=str(rec["frame"]),
        )

    def _offset(self, source):
        """Arcsec separation of the pointing from a catalog source at obs epoch.

        Returns None (present-but-empty keyword) when the source astrometry is
        unavailable; otherwise the catalog position is propagated to the
        observation epoch (proper motion) before the comparison, so the offset
        reflects where the source actually sits at the time of the exposure. Any
        residual malformed astrometry (e.g. an unparseable epoch) is caught and
        emitted empty rather than raised -- error-raising is the checkpoint's job.
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
