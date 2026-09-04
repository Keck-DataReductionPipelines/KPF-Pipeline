"""Diagnostics for the KPF Level 0 telemetry and observing conditions."""

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    AltAz,
    SkyCoord,
    get_body,
    get_body_barycentric_posvel,
)
from astropy.time import Time

from kpfpipe.quality_control.diagnostics.base import Diagnostics
from kpfpipe.utils.astro import KECK_LOCATION


class Telemetry(Diagnostics):
    """Instrument telemetry and the observing conditions of the exposure.

    Covers the TELEMETRY table, the environment cards the native header carries
    and the solar/lunar geometry.
    """

    LEVEL = "L0"

    def _telemetry_average(self, keyword):
        """One TELEMETRY keyword's exposure-average reading."""
        table = self.kpf_obj.data["TELEMETRY"]
        return float(table[table["keyword"] == keyword]["average"][0])

    def ccd_temperature_offsets(self):
        """GTEMPOFF/RTEMPOFF: signed GREEN/RED CCD offset from setpoint [mK].

        The exposure-average kpf{green,red}.STA_CCD_T telemetry against the
        -100 C setpoint, signed so the direction of the drift is visible.
        """
        return self._tag(
            GTEMPOFF=round(
                (self._telemetry_average("kpfgreen.STA_CCD_T") + 100.0) * 1e3, 6
            ),
            RTEMPOFF=round(
                (self._telemetry_average("kpfred.STA_CCD_T") + 100.0) * 1e3, 6
            ),
        )

    ccd_temperature_offsets._diag_name = "ccd_temperature_offsets"

    def etalon_temperature_offset(self):
        """ETATOFF: signed etalon offset from setpoint [mK], worst chamber.

        The inner bottom lid (ETAV1C3T) and the outer chamber (ETAV1C4T), each
        against its own setpoint keyword, falling back to the design value when
        the setpoint is not recorded. One keyword covers both, so the chamber
        furthest from its setpoint is the one reported.
        """
        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        offsets = []
        for temp_key, set_key, design in (
            ("ETAV1C3T", "ETAV1C3S", 23.6),
            ("ETAV1C4T", "ETAV1C4S", 23.9),
        ):
            setpoint = float(hdr[set_key]) if set_key in hdr else design
            offsets.append((float(hdr[temp_key]) - setpoint) * 1e3)
        return self._tag(ETATOFF=round(max(offsets, key=abs), 6))

    etalon_temperature_offset._diag_name = "etalon_temperature_offset"

    def site_conditions(self):
        """INHUM, DEWPOINT, OUTPRES, M1TMP, M2TEMP: conditions at mid-exposure.

        Read from the native cards, all of them the keyheader ExposureMiddle
        snapshot. RELH and PRES are the in-dome Vaisala humidity and pressure,
        the latter in hPa where OUTPRES is in kPa. The DCS reports the dewpoint
        only as DIFFPTDW, its offset below the primary mirror temperature, and
        to a tenth of a degree.
        """
        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        return self._tag(
            INHUM=round(float(hdr["RELH"]), 6),
            DEWPOINT=round(float(hdr["PRIMTEMP"]) - float(hdr["DIFFPTDW"]), 1),
            OUTPRES=round(float(hdr["PRES"]) / 10.0, 6),
            M1TMP=round(float(hdr["PRIMTEMP"]), 6),
            M2TEMP=round(float(hdr["SECMTEMP"]), 6),
        )

    site_conditions._diag_name = "site_conditions"

    def solar_lunar_geometry(self):
        """SUNEL, MOONEL, MOONANG, MOONILLU: Sun and Moon geometry [deg, %].

        Evaluated at mid-exposure from the WMKO site; the altitudes are negative
        with the body below the horizon. MOONILLU is the illuminated fraction of
        the lunar disc, from the Sun-Moon elongation.
        """
        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        obs_time = Time(str(hdr["DATE-MID"]), scale="utc")
        horizon = AltAz(obstime=obs_time, location=KECK_LOCATION)
        sun = get_body("sun", obs_time, KECK_LOCATION)
        moon = get_body("moon", obs_time, KECK_LOCATION)
        pointing = SkyCoord(hdr["RA"], hdr["DEC"], unit=(u.hourangle, u.deg))
        elongation = float(sun.separation(moon).rad)
        # The target is a direction at infinity, so the Moon's topocentric angles
        # are compared as angles: transforming the Moon out of its observer-centred
        # frame would move it by the lunar parallax, up to a degree.
        moon_direction = SkyCoord(moon.ra, moon.dec)
        # EPRV-defined, so these route straight to PRIMARY rather than to the
        # QUALITY_CONTROL extension the other diagnostics land in.
        return self._tag(
            SUNEL=round(float(sun.transform_to(horizon).alt.deg), 5),
            MOONEL=round(float(moon.transform_to(horizon).alt.deg), 5),
            MOONANG=round(float(pointing.separation(moon_direction).deg), 2),
            MOONILLU=round(float(50 * (1 - np.cos(elongation))), 2),
        )

    solar_lunar_geometry._diag_name = "solar_lunar_geometry"

    @staticmethod
    def _recession(origin, target):
        """Rate the target recedes from the origin [km/s].

        Each argument is a barycentric ``(position, velocity)`` pair.
        """
        (origin_pos, origin_vel), (target_pos, target_vel) = origin, target
        line = target_pos - origin_pos
        velocity = (target_vel - origin_vel).dot(line / line.norm())
        return float(velocity.to_value(u.km / u.s))

    def moon_radial_velocity(self):
        """MOONRV: RV of sunlight reflected off the Moon [km/s].

        The two legs of the reflected path at mid-exposure: the rate the Moon
        recedes from the Sun, plus the rate the observer recedes from the Moon.
        """
        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        obs_time = Time(str(hdr["DATE-MID"]), scale="utc")
        sun = get_body_barycentric_posvel("sun", obs_time)
        moon = get_body_barycentric_posvel("moon", obs_time)
        earth_pos, earth_vel = get_body_barycentric_posvel("earth", obs_time)
        site_pos, site_vel = KECK_LOCATION.get_gcrs_posvel(obs_time)
        observer = (earth_pos + site_pos, earth_vel + site_vel)
        return self._tag(
            MOONRV=round(
                self._recession(sun, moon) + self._recession(moon, observer), 6
            )
        )

    moon_radial_velocity._diag_name = "moon_radial_velocity"
