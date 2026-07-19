"""Diagnostics for KPF Level 0 (raw CCD) data products."""

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from kpfpipe.quality_control.diagnostics.base import Diagnostics


class DiagL0(Diagnostics):
    """Diagnostics for KPF Level 0 raw data products.

    The pointing-offset metrics compare the telescope pointing against the target
    astrometry resolved upstream by AstroQuery and attached as ``l0.catalog_query``
    (Gaia / SIMBAD / DCS, all in one canonical ICRS schema); DiagL0 assumes that
    dict is present and fully populated.
    """

    LEVEL = "L0"

    def _record_skycoord(self, source):
        """ICRS SkyCoord from a ``catalog_query`` record (canonical units).

        AstroQuery normalizes every source ('gaia'/'simbad'/'wmko') to the same
        schema -- ICRS, RA/Dec deg, proper motion mas/yr (RA incl. cos Dec),
        parallax mas, epoch in Julian years -- so one builder serves all three.
        """
        rec = self.kpf_obj.catalog_query[source]
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

        The catalog position is propagated to the observation epoch (proper
        motion) before the comparison, so the offset reflects where the source
        actually sits at the time of the exposure.
        """
        coord = self._record_skycoord(source).apply_space_motion(
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
