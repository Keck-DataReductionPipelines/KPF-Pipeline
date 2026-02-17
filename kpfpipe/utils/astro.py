from astropy.constants import c
from astropy.time import Time
from datetime import timezone, timedelta


def compute_doppler_shift(v):
    # TODO: sanitize input to ensure unit consistency
    beta = v / c
    z = ((1 - beta) / (1 + beta))**0.5
    return z


def utc_to_date(utc_str, tz='utc'):
    t = Time(utc_str, format="isot", scale="utc")

    if tz == 'hawaii':
        dt = t.to_datetime(timezone=timezone(timedelta(hours=-10)))
    elif tz == 'utc':
        dt = t.to_datetime()
    else:
        raise ValueError("tz must be 'utc' or 'hawaii'")

    return dt.strftime("%Y%m%d")