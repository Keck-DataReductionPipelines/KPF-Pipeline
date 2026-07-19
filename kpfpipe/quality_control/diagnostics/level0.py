"""Diagnostics for KPF Level 0 (raw CCD) data products."""

import logging

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from kpfpipe.quality_control.diagnostics.base import Diagnostics

logger = logging.getLogger(__name__)

# Record fields the pointing offset needs; a None in any of them (or a missing
# record) makes the source unusable, and the offset is emitted present-but-empty.
_OFFSET_FIELDS = ("ra", "dec", "pmra", "pmdec", "parallax", "epoch", "frame")


class DiagL0(Diagnostics):
    """Diagnostics for KPF Level 0 raw data products.

    The pointing-offset metrics compare the telescope pointing against the target
    astrometry resolved upstream by AstroQuery and attached as ``l0.catalog_query``
    (Gaia / SIMBAD / DCS, all in one canonical ICRS schema). When a source's
    astrometry is unavailable the offset is emitted present-but-empty (a valueless
    card) and a WARNING is logged -- diagnostics record what they can and leave
    error-raising to the checkpoint layer.
    """

    LEVEL = "L0"

    def _catalog_record(self, source):
        """Usable ``catalog_query`` record for ``source``, or None with a WARNING.

        Handles the three contingencies: AstroQuery not run (no ``catalog_query``),
        the source's record absent (lookup disabled/failed, or WMKO unavailable),
        or a record present but missing a field the offset needs.
        """
        catalog = getattr(self.kpf_obj, "catalog_query", None)
        if catalog is None:
            logger.warning(
                "no catalog_query on L0 (run AstroQuery first); "
                "%s pointing offset unavailable",
                source,
            )
            return None
        rec = catalog.get(source)
        if rec is None:
            logger.warning(
                "no %s astrometry in catalog_query; pointing offset unavailable",
                source,
            )
            return None
        if any(rec.get(field) is None for field in _OFFSET_FIELDS):
            logger.warning(
                "incomplete %s record in catalog_query; pointing offset unavailable",
                source,
            )
            return None
        return rec

    def _record_skycoord(self, rec):
        """ICRS SkyCoord from a ``catalog_query`` record (canonical units).

        AstroQuery normalizes every source ('gaia'/'simbad'/'wmko') to the same
        schema -- ICRS, RA/Dec deg, proper motion mas/yr (RA incl. cos Dec),
        parallax mas, epoch in Julian years -- so one builder serves all three.
        """
        return SkyCoord(
            ra=rec["ra"] * u.deg,
            dec=rec["dec"] * u.deg,
            pm_ra_cosdec=rec["pmra"] * u.mas / u.yr,
            pm_dec=rec["pmdec"] * u.mas / u.yr,
            distance=(1e3 / rec["parallax"]) * u.pc,
            obstime=Time(rec["epoch"], format="jyear"),
            frame=rec["frame"],
        )

    def _pointing(self):
        """Telescope pointing SkyCoord from L0 PRIMARY RA/DEC (sexagesimal h/deg)."""
        hdr = self.kpf_obj.headers["PRIMARY"]
        return SkyCoord(hdr["RA"], hdr["DEC"], unit=(u.hourangle, u.deg))

    def _obs_time(self):
        """Observation epoch (Time) from L0 PRIMARY MJD-OBS, for PM propagation."""
        return Time(float(self.kpf_obj.headers["PRIMARY"]["MJD-OBS"]), format="mjd")

    def _offset(self, source):
        """Arcsec separation of the pointing from a catalog source at obs epoch.

        Returns None (present-but-empty keyword) when the source astrometry is
        unavailable; otherwise the catalog position is propagated to the
        observation epoch (proper motion) before the comparison, so the offset
        reflects where the source actually sits at the time of the exposure.
        """
        rec = self._catalog_record(source)
        if rec is None:
            return None
        coord = self._record_skycoord(rec).apply_space_motion(
            new_obstime=self._obs_time()
        )
        return round(float(self._pointing().separation(coord).arcsec), 4)

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
