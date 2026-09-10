"""Diagnostics for the KPF Level 0 telemetry and observing conditions."""

from datetime import datetime, timedelta

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
from kpfpipe.utils.network import cfht_archive, retry_request


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
        return {
            "GTEMPOFF": round(
                (self._telemetry_average("kpfgreen.STA_CCD_T") + 100.0) * 1e3, 6
            ),
            "RTEMPOFF": round(
                (self._telemetry_average("kpfred.STA_CCD_T") + 100.0) * 1e3, 6
            ),
        }

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
        return {"ETATOFF": round(max(offsets, key=abs), 6)}

    etalon_temperature_offset._diag_name = "etalon_temperature_offset"

    def site_conditions(self):
        """INHUM, OUTPRES, OUTTMP, OUTHUM, ENVWINDS, ENVWINDD: conditions at
        mid-exposure.

        RELH and PRES are the in-dome Vaisala humidity and pressure, the latter in
        hPa where OUTPRES is in kPa. Nothing WMKO records carries the weather
        outside the dome, so the rest come from the CFHT tower a few hundred metres
        away -- a proxy, close enough for conditions monitoring and outside the
        science chain.
        """
        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        return {
            "INHUM": round(float(hdr["RELH"]), 6),
            "OUTPRES": round(float(hdr["PRES"]) / 10.0, 6),
            **self._cfht_weather(Time(str(hdr["DATE-MID"]), scale="utc").to_datetime()),
        }

    site_conditions._diag_name = "site_conditions"

    @staticmethod
    def _cfht_weather(mid):
        """OUTTMP, OUTHUM, ENVWINDS, ENVWINDD from the CFHT tower at ``mid`` UTC.

        The archive stamps one row per minute in HST and reports wind in knots;
        its fields are documented at
        http://mkwc.ifa.hawaii.edu/archive/wx/cfht/format.txt.

        A year is one 21 MB file, sorted but unindexed, so the minute is sought
        rather than downloaded, each window's own timestamps placing the next seek.
        Nothing infers a row size or rate: the station's outages leave gaps that
        make either drift over a year. The first seek reads the tail because the
        current year's file stops at today, so the bracket must be dated rather
        than assumed to span the year. A row either cut left partial is dropped.

        The station also drops a minute here and there, so the nearest row within
        half an hour stands in. Conditions move about 0.3 C and 2% humidity over
        that span, and a longer gap is a station outage worth reporting as one.
        """
        stamp = (mid - timedelta(hours=10)).replace(second=0, microsecond=0)
        window = 1 << 16  # About a day of rows, so a hit carries hours either side.
        low, low_time = 0, datetime(stamp.year, 1, 1)
        high = high_time = None
        start = -window
        rows = {}

        for _ in range(6):
            text, size = retry_request(
                lambda start=start: cfht_archive(stamp.year, start, window),
                "CFHT weather",
            )
            lines = text.splitlines()[0 if start == 0 else 1 :]
            if not text.endswith("\n"):
                lines = lines[:-1]
            rows = {}
            for line in lines:
                fields = line.split()
                if len(fields) >= 9:
                    rows[datetime(*(int(f) for f in fields[:5]))] = fields
            if not rows:
                break
            first = min(rows)
            if first <= stamp <= max(rows):
                break
            if high is None:
                high, high_time = size - window, first
            elif stamp < first:
                high, high_time = start, first
            else:
                low, low_time = start, first
            if stamp > high_time or high <= low:
                break
            reach = (stamp - low_time) / (high_time - low_time)
            guess = low + int((high - low) * reach) - window // 2
            start = min(max(guess, 0), max(size - window, 0))

        for offset in sorted(range(-30, 31), key=abs):
            fields = rows.get(stamp + timedelta(minutes=offset))
            if fields is not None:
                wind, direction, temperature, humidity = (float(f) for f in fields[5:9])
                return {
                    "OUTTMP": temperature,
                    "OUTHUM": humidity,
                    "ENVWINDS": round(wind * 0.514444, 6),
                    "ENVWINDD": direction,
                }

        raise ValueError(f"the CFHT weather archive has no reading near {stamp} HST")

    def mirror_temperatures(self):
        """M1TMP, M2TMP: primary and secondary mirror temperatures [deg C]."""
        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        return {
            "M1TMP": round(float(hdr["PRIMTEMP"]), 6),
            "M2TMP": round(float(hdr["SECMTEMP"]), 6),
        }

    mirror_temperatures._diag_name = "mirror_temperatures"

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
        return {
            "SUNEL": round(float(sun.transform_to(horizon).alt.deg), 5),
            "MOONEL": round(float(moon.transform_to(horizon).alt.deg), 5),
            "MOONANG": round(float(pointing.separation(moon_direction).deg), 2),
            "MOONILLU": round(float(50 * (1 - np.cos(elongation))), 2),
        }

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
        return {
            "MOONRV": round(
                self._recession(sun, moon) + self._recession(moon, observer), 6
            )
        }

    moon_radial_velocity._diag_name = "moon_radial_velocity"
