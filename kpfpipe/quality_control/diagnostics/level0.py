"""Diagnostics for KPF Level 0 (raw CCD) data products.

Pointing/identity checks that cross-match the telescope pointing (RA/DEC) against
three reference positions: the loaded DCS target (TARGRA/DEC), the Gaia DR3
catalog position of GAIAID, and the SIMBAD position of the OBJECT name. All three
metrics are fail-soft: a frame with no pointing/target (e.g. a calibration frame)
skips them, and a Gaia/SIMBAD lookup failure warns and skips rather than failing
the L0 checkpoint. Diagnostics that read the overscan region (read noise,
non-Gaussian RN) are owned by ImageAssembly because they need to run before gain
conversion modifies the amp data.
"""

import re
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad

from kpfpipe.quality_control.diagnostics.base import Diagnostics


class DiagL0(Diagnostics):
    LEVEL = "L0"

    # Keys each metric needs; absent -> metric is N/A for that frame (skip).
    _POINTING_KEYS = ("RA", "DEC", "MJD-OBS")
    _TARGET_KEYS = (
        "TARGRA",
        "TARGDEC",
        "TARGPMRA",
        "TARGPMDC",
        "TARGPLAX",
        "TARGFRAM",
        "TARGEPOC",
    )

    @staticmethod
    def _present(hdr, keys):
        """True if every key in `keys` is present and non-None in `hdr`."""
        return all(hdr.get(k) is not None for k in keys)

    # Astrometry helpers reproduced from BarycentricCorrection._gaia_astrometry /
    # ._wmko_astrometry (there they read INSTRUMENT_HEADER; at L0 the natives are
    # on PRIMARY). The shared copies move to utils/astro.py in a follow-up -- keep
    # the two in sync until then.

    def _gaia_source_id(self):
        """Digit-only Gaia DR3 id from L0 GAIAID, or None if absent/malformed."""
        raw = self.kpf_obj.headers["PRIMARY"].get("GAIAID")
        if raw is None:
            return None
        token = re.split(r"\s+", str(raw).strip())[-1]
        return token if token.isdigit() else None

    def _gaia_astrometry(self):
        """Gaia DR3 SkyCoord (ICRS, PM+distance, Gaia epoch) for the L0 GAIAID."""
        gaia_id = self._gaia_source_id()
        if gaia_id is None:
            raise ValueError("no usable Gaia source id in L0 PRIMARY GAIAID")
        query = f"""
        SELECT ra, dec, pmra, pmdec, parallax, ref_epoch
        FROM gaiadr3.gaia_source
        WHERE source_id = {gaia_id}
        """
        result = Gaia.launch_job(query).get_results()[0]
        return SkyCoord(
            ra=result["ra"] * u.deg,
            dec=result["dec"] * u.deg,
            pm_ra_cosdec=result["pmra"] * u.mas / u.yr,
            pm_dec=result["pmdec"] * u.mas / u.yr,
            distance=(1e3 / result["parallax"]) * u.pc,
            obstime=Time(result["ref_epoch"], format="jyear"),
            frame="icrs",
        )

    def _wmko_astrometry(self):
        """WMKO/DCS target SkyCoord from L0 PRIMARY TARG* astrometry.

        TARGPMRA is s/yr (-> mas/yr via x15 cos(dec)); TARGPLAX is mas (-> pc).
        """
        hdr = self.kpf_obj.headers["PRIMARY"]
        pos = SkyCoord(hdr["TARGRA"], hdr["TARGDEC"], unit=(u.hourangle, u.deg))
        pm_ra_cosdec = float(hdr["TARGPMRA"]) * 15.0 * np.cos(pos.dec.rad) * 1e3
        return SkyCoord(
            ra=pos.ra,
            dec=pos.dec,
            pm_ra_cosdec=pm_ra_cosdec * u.mas / u.yr,
            pm_dec=float(hdr["TARGPMDC"]) * 1e3 * u.mas / u.yr,
            distance=(1e3 / float(hdr["TARGPLAX"])) * u.pc,
            frame=str(hdr["TARGFRAM"]).lower(),
            obstime=Time(float(hdr["TARGEPOC"]), format="jyear"),
        )

    def _object_name(self):
        """SIMBAD-resolvable name from L0 PRIMARY OBJECT, or None if absent.

        KPF OBJECT for standard stars is the bare HD number (e.g. '10700'),
        which SIMBAD only resolves with an 'HD ' prefix; named targets pass
        through unchanged.
        """
        obj = self.kpf_obj.headers["PRIMARY"].get("OBJECT")
        if obj is None:
            return None
        obj = str(obj).strip()
        if not obj:
            return None
        return f"HD {obj}" if obj.isdigit() else obj

    def _simbad_astrometry(self):
        """SIMBAD SkyCoord (ICRS J2000, PM+distance) for the L0 OBJECT name.

        SIMBAD reports ICRS coordinates at epoch J2000.0; column names are the
        astroquery 0.4.11 lowercase schema (ra/dec in deg, pmra/pmdec, plx_value).
        """
        name = self._object_name()
        if name is None:
            raise ValueError("no OBJECT name in L0 PRIMARY header")
        simbad = Simbad()
        simbad.add_votable_fields("pmra", "pmdec", "plx")
        result = simbad.query_object(name)
        if result is None or len(result) == 0:
            raise ValueError(f"SIMBAD returned no match for {name!r}")
        row = result[0]
        return SkyCoord(
            ra=row["ra"] * u.deg,
            dec=row["dec"] * u.deg,
            pm_ra_cosdec=row["pmra"] * u.mas / u.yr,
            pm_dec=row["pmdec"] * u.mas / u.yr,
            distance=(1e3 / row["plx_value"]) * u.pc,
            obstime=Time(2000.0, format="jyear"),
            frame="icrs",
        )

    def _pointing(self):
        """Telescope pointing SkyCoord from L0 PRIMARY RA/DEC (sexagesimal h/deg)."""
        hdr = self.kpf_obj.headers["PRIMARY"]
        return SkyCoord(hdr["RA"], hdr["DEC"], unit=(u.hourangle, u.deg))

    def _obs_time(self):
        """Observation epoch (Time) from L0 PRIMARY MJD-OBS, for PM propagation."""
        return Time(float(self.kpf_obj.headers["PRIMARY"]["MJD-OBS"]), format="mjd")

    def gaia_ra_dec_offset(self):
        """GAIAOFF: arcsec, RA/DEC pointing vs GAIAID position at obs epoch.

        Skipped when the frame has no pointing or no usable GAIAID; a Gaia
        network/lookup failure warns and skips (fail-soft).
        """
        hdr = self.kpf_obj.headers["PRIMARY"]
        if (
            not self._present(hdr, self._POINTING_KEYS)
            or self._gaia_source_id() is None
        ):
            return {}
        try:
            gaia = self._gaia_astrometry().apply_space_motion(
                new_obstime=self._obs_time()
            )
        except Exception as e:
            warnings.warn(
                f"GAIAOFF skipped: Gaia lookup failed ({type(e).__name__}: {e})",
                stacklevel=2,
            )
            return {}
        sep = float(self._pointing().separation(gaia).arcsec)
        return self._tag(GAIAOFF=round(sep, 4))

    gaia_ra_dec_offset._diag_name = "gaia_ra_dec_offset"

    def target_ra_dec_offset(self):
        """TARGOFF: arcsec, RA/DEC pointing vs TARGRA/DEC target at obs epoch.

        Skipped when the frame has no pointing or no DCS target (e.g. a
        calibration frame).
        """
        hdr = self.kpf_obj.headers["PRIMARY"]
        if not self._present(hdr, self._POINTING_KEYS + self._TARGET_KEYS):
            return {}
        target = self._wmko_astrometry().apply_space_motion(
            new_obstime=self._obs_time()
        )
        sep = float(self._pointing().separation(target).arcsec)
        return self._tag(TARGOFF=round(sep, 4))

    target_ra_dec_offset._diag_name = "target_ra_dec_offset"

    def object_ra_dec_offset(self):
        """OBJOFF: arcsec, RA/DEC pointing vs SIMBAD(OBJECT) position at obs epoch.

        Skipped when the frame has no pointing or no OBJECT name; a SIMBAD
        network/lookup failure (or an unresolvable name) warns and skips.
        """
        hdr = self.kpf_obj.headers["PRIMARY"]
        if not self._present(hdr, self._POINTING_KEYS) or self._object_name() is None:
            return {}
        try:
            obj = self._simbad_astrometry().apply_space_motion(
                new_obstime=self._obs_time()
            )
        except Exception as e:
            warnings.warn(
                f"OBJOFF skipped: SIMBAD lookup failed ({type(e).__name__}: {e})",
                stacklevel=2,
            )
            return {}
        sep = float(self._pointing().separation(obj).arcsec)
        return self._tag(OBJOFF=round(sep, 4))

    object_ra_dec_offset._diag_name = "object_ra_dec_offset"
