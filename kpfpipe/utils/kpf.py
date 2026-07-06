"""Obs-id, datecode, and timestamp parsing plus UTC/HST and EPRV conversions."""

import os
import re
from datetime import datetime, timedelta

_OBS_ID_PATTERN = re.compile(r"(KP\.\d{8}\.\d{5}\.\d{2})")
_DATECODE_PATTERN = re.compile(r"\d{8}")
_KPF_TIMESTAMP_PATTERN = re.compile(r"(\d{8}\.\d{5}\.\d{2})")

# Seconds per day
_SECONDS_PER_DAY = 86400

# Hawaii Standard Time is UTC-10 (KPF timestamps are UTC)
_HST_UTC_OFFSET_SECONDS = 36000

# J2000.0 epoch (2000-01-01 12:00 UTC)
_J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)


def _validate_kpf_timestamp(timestamp):
    """
    Validate that `timestamp` is a well-formed KPF timestamp string of the
    form 'YYYYMMDD.SSSSS.FF' with a real calendar date and SSSSS in
    [0, 86399]. Raises ValueError otherwise.
    """
    if not isinstance(timestamp, str):
        raise ValueError(
            f"KPF timestamp must be a string; got {type(timestamp).__name__}"
        )
    if not _KPF_TIMESTAMP_PATTERN.fullmatch(timestamp):
        raise ValueError(
            f"Invalid KPF timestamp format {timestamp!r}; expected 'YYYYMMDD.SSSSS.FF'"
        )
    date_str, seconds_str, _ = timestamp.split(".")
    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError as e:
        raise ValueError(
            f"Invalid date in KPF timestamp {timestamp!r}: {date_str!r}"
        ) from e
    seconds = int(seconds_str)
    if seconds >= _SECONDS_PER_DAY:
        raise ValueError(
            f"Invalid seconds-past-midnight in KPF timestamp {timestamp!r}: "
            f"{seconds} not in [0, {_SECONDS_PER_DAY - 1}]"
        )


def is_obs_id(s):
    """
    Return True if `s` is a valid KPF observation ID, e.g.
    'KP.20240113.23249.10'. Both the format and the embedded date/seconds
    are checked.
    """
    if not isinstance(s, str) or not _OBS_ID_PATTERN.fullmatch(s):
        return False
    try:
        _validate_kpf_timestamp(s[3:])
    except ValueError:
        return False
    return True


def is_datecode(s):
    """
    Return True if `s` is a valid 8-digit datecode that parses as a real
    calendar date, e.g. '20240405'.
    """
    if not isinstance(s, str) or not _DATECODE_PATTERN.fullmatch(s):
        return False
    try:
        datetime.strptime(s, "%Y%m%d")
    except ValueError:
        return False
    return True


def is_timestamp(s):
    """
    Return True if `s` is a valid KPF timestamp, e.g. '20240113.23249.10'.
    Both the format and the embedded date/seconds are checked.
    """
    try:
        _validate_kpf_timestamp(s)
    except ValueError:
        return False
    return True


def get_obs_id(fn):
    """
    Extract the obs_id from a filename or path.

    Parameters
    ----------
    fn : str
        Filename or path, e.g.
        '/data/L1/20240113/KP.20240113.23249.10_L1.fits'.

    Returns
    -------
    str
        The obs_id, e.g. 'KP.20240113.23249.10'.

    Raises
    ------
    ValueError
        If no valid obs_id is found in `fn`.
    """
    if not isinstance(fn, str):
        raise ValueError(f"input must be a string; got {type(fn).__name__}")
    match = _OBS_ID_PATTERN.search(os.path.basename(fn))
    if not match:
        raise ValueError(f"No obs_id found in: {fn}")
    obs_id = match.group(1)
    _validate_kpf_timestamp(obs_id[3:])
    return obs_id


def get_datecode(s):
    """
    Extract the datecode from an obs_id or filename.

    Parameters
    ----------
    s : str
        An obs_id or filename, e.g. 'KP.20230708.04519.63' or
        'KP.20230708.04519.63_L1.fits'.

    Returns
    -------
    str
        The datecode, e.g. '20230708'.

    Raises
    ------
    ValueError
        If no valid obs_id is found in `s`.
    """
    if not isinstance(s, str):
        raise ValueError(f"input must be a string; got {type(s).__name__}")
    match = _OBS_ID_PATTERN.search(s)
    if not match:
        raise ValueError(f"Cannot extract datecode from: {s}")
    obs_id = match.group(1)
    _validate_kpf_timestamp(obs_id[3:])
    return obs_id.split(".")[1]


def get_timestamp(s):
    """
    Extract the KPF timestamp from an obs_id, filename, or path.

    Parameters
    ----------
    s : str
        An obs_id, filename, or path, e.g. 'KP.20240113.23249.10' or
        '/data/L0/20240113/KP.20240113.23249.10.fits'.

    Returns
    -------
    str
        The KPF timestamp, e.g. '20240113.23249.10'.

    Raises
    ------
    ValueError
        If no valid KPF timestamp is found in `s`.
    """
    if not isinstance(s, str):
        raise ValueError(f"input must be a string; got {type(s).__name__}")
    match = _KPF_TIMESTAMP_PATTERN.search(os.path.basename(s))
    if not match:
        raise ValueError(f"No KPF timestamp found in: {s}")
    timestamp = match.group(1)
    _validate_kpf_timestamp(timestamp)
    return timestamp


def get_seconds_since_j2000(s):
    """
    Compute seconds since the J2000.0 epoch (2000-01-01 12:00 UTC) for the
    KPF timestamp embedded in `s`. Suitable as a monotonic scalar for sorting
    and gap detection.

    Parameters
    ----------
    s : str
        A KPF timestamp ('YYYYMMDD.SSSSS.FF'), an obs_id
        ('KP.YYYYMMDD.SSSSS.FF'), or any filename or path containing one.

    Returns
    -------
    int
        Seconds since J2000.0 (naive UTC).

    Raises
    ------
    ValueError
        If no valid KPF timestamp is found in `s`.

    Notes
    -----
    Arithmetic is naive UTC; leap seconds are ignored. Fine for frame
    ordering and cluster-gap detection but does not give TT/TAI precision
    and should not be used for astronomical timing.
    """
    dt = kpf_timestamp_to_datetime(get_timestamp(s))
    return int((dt - _J2000_EPOCH).total_seconds())


def utc_to_hst(timestamp):
    """
    Convert a KPF UTC timestamp to HST (Hawaii Standard Time, UTC-10).

    Parameters
    ----------
    timestamp : str
        KPF timestamp string of the form 'YYYYMMDD.SSSSS.FF'.

    Returns
    -------
    str
        HST timestamp in the same KPF format.
    """
    _validate_kpf_timestamp(timestamp)
    date_str, seconds_str, frame_str = timestamp.split(".")
    hst_seconds = int(seconds_str) - _HST_UTC_OFFSET_SECONDS
    date = datetime.strptime(date_str, "%Y%m%d")
    if hst_seconds < 0:
        hst_seconds += _SECONDS_PER_DAY
        date -= timedelta(days=1)
    return f"{date.strftime('%Y%m%d')}.{hst_seconds:05d}.{frame_str}"


def hst_to_utc(timestamp):
    """
    Convert a KPF HST timestamp to UTC.

    Parameters
    ----------
    timestamp : str
        KPF timestamp string of the form 'YYYYMMDD.SSSSS.FF'.

    Returns
    -------
    str
        UTC timestamp in the same KPF format.
    """
    _validate_kpf_timestamp(timestamp)
    date_str, seconds_str, frame_str = timestamp.split(".")
    utc_seconds = int(seconds_str) + _HST_UTC_OFFSET_SECONDS
    date = datetime.strptime(date_str, "%Y%m%d")
    if utc_seconds >= _SECONDS_PER_DAY:
        utc_seconds -= _SECONDS_PER_DAY
        date += timedelta(days=1)
    return f"{date.strftime('%Y%m%d')}.{utc_seconds:05d}.{frame_str}"


def kpf_timestamp_to_datetime(timestamp):
    """
    Parse a KPF UTC timestamp string into a `datetime`, e.g.
    '20240405.40113.57' -> ``datetime(2024, 4, 5, 11, 8, 33)``.

    Parameters
    ----------
    timestamp : str
        KPF timestamp string of the form 'YYYYMMDD.SSSSS.FF' (the sub-second
        frame field is ignored).

    Returns
    -------
    datetime
        Naive UTC datetime at the timestamp's seconds-past-midnight.
    """
    _validate_kpf_timestamp(timestamp)
    date_str, seconds_str, _ = timestamp.split(".")
    return datetime.strptime(date_str, "%Y%m%d") + timedelta(seconds=int(seconds_str))


def kpf_timestamp_to_eprv_timestamp(timestamp):
    """
    Convert a KPF timestamp to EPRV standard format, e.g. '20240405.40113.57'
    -> '20240405T110833'.

    EPRV timestamps have 1-second resolution; the sub-second frame field
    is dropped.

    Parameters
    ----------
    timestamp : str
        KPF timestamp string of the form 'YYYYMMDD.SSSSS.FF'.

    Returns
    -------
    str
        EPRV timestamp of the form 'YYYYMMDDTHHMMSS'.
    """
    _validate_kpf_timestamp(timestamp)
    date_str, seconds_str, _ = timestamp.split(".")
    total_seconds = int(seconds_str)
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{date_str}T{hh:02d}{mm:02d}{ss:02d}"
