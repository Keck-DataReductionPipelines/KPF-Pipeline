"""Diagnostics for KPF Level 0 (raw CCD) data products."""

import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from kpfpipe.quality_control.diagnostics.base import Diagnostics

logger = logging.getLogger(__name__)


class DiagL0(Diagnostics):
    """Diagnostics for KPF Level 0 raw data products.

    The pointing-offset metrics compare the telescope pointing against the target
    astrometry AstroQuery resolves into the L0 ``CATALOG_RECORD`` extension, so
    AstroQuery must have run. The DCS target offset is required; the Gaia and
    SIMBAD offsets are emitted only when their catalog lookup matched.
    """

    LEVEL = "L0"

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

        The catalog position is proper-motion propagated to the observation epoch
        before the comparison.
        """
        table = self.kpf_obj.data["CATALOG_RECORD"]
        rec = table[table["source"] == source][0]
        hdr = self.kpf_obj.headers["PRIMARY"]
        pointing = SkyCoord(hdr["RA"], hdr["DEC"], unit=(u.hourangle, u.deg))
        obs_time = Time(float(hdr["MJD-OBS"]), format="mjd")
        coord = self._record_skycoord(rec).apply_space_motion(new_obstime=obs_time)
        return round(float(pointing.separation(coord).arcsec), 4)

    def gaia_ra_dec_offset(self):
        """GAIAOFF: arcsec, RA/DEC pointing vs Gaia catalog position at obs epoch.

        Gaia is an optional source; an unmatched lookup emits no keyword.
        """
        if "gaia" not in self.kpf_obj.data["CATALOG_RECORD"]["source"]:
            return {}
        return self._tag(GAIAOFF=self._offset("gaia"))

    gaia_ra_dec_offset._diag_name = "gaia_ra_dec_offset"

    def target_ra_dec_offset(self):
        """TCSOFF: arcsec, RA/DEC pointing vs DCS target position at obs epoch."""
        return self._tag(TCSOFF=self._offset("wmko"))

    target_ra_dec_offset._diag_name = "target_ra_dec_offset"

    def object_ra_dec_offset(self):
        """OBJOFF: arcsec, RA/DEC pointing vs SIMBAD(OBJECT) position at obs epoch.

        SIMBAD is an optional source; an unmatched lookup emits no keyword.
        """
        if "simbad" not in self.kpf_obj.data["CATALOG_RECORD"]["source"]:
            return {}
        return self._tag(OBJOFF=self._offset("simbad"))

    object_ra_dec_offset._diag_name = "object_ra_dec_offset"
