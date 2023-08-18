# This file contains assorted utility functions for computing astronomical 
# quantities associated with KPF data.

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun, get_moon

def get_sun_alt(UTdatetime):
    """
    Returns the altitude of the Sun at Maunakea on a specific UT datetime.

    Args:
        UTdatetime - an astropy Time object with the UT datetime

    Returns:
        alt - Altitude of the Sun in degrees at the UT datetime.  
              Negative values correspond to the Sun below the horizon.
    """
    
    sun = get_sun(UTdatetime)
    maunakea = EarthLocation(lat='19d49m42.6s', lon='-155d28m48.9s', height=4205) 
    altaz_sun = sun.transform_to(AltAz(obstime=UTdatetime, location=maunakea))
    alt = altaz_sun.alt.deg

    return alt
    
def get_moon_sep(UTdatetime, RA, dec):
    """
    Returns the separation in degrees between the Moon and an object with 
    coordinates RA/dec at a specific UT datetime.

    Args:
        UTdatetime - an astropy Time object with the UT datetime
        RA - right ascension of the object (string format, e.g. "10:24:36.5")
        dec - declination of the object (string format, e.g. "+45:10:45.1")

    Returns:
        sep - separation (degrees)
    """

    target = SkyCoord(RA, dec, unit=(u.hourangle, u.deg))
    moon = get_moon(UTdatetime)
    sep = target.separation(moon)

    return sep.deg # in degrees
