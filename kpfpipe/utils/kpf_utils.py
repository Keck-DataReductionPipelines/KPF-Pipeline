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


def is_timestamp(s):
    """
    Return True if `s` is a valid KPF timestamp, e.g. '20240113.23249.10'.

    Checks all three: the 'YYYYMMDD.SSSSS.FF' format, that YYYYMMDD is a real
    calendar date, and that SSSSS (seconds past midnight) is in [0, 86399]; the
    sub-second frame field FF is format-checked only. The module's single source
    of timestamp validity -- every other predicate and raising converter
    validates through it.
    """
    if not isinstance(s, str) or not _KPF_TIMESTAMP_PATTERN.fullmatch(s):
        return False
    date_str, seconds_str, _ = s.split(".")
    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return False
    return int(seconds_str) < _SECONDS_PER_DAY


def is_obs_id(s):
    """
    Return True if `s` is a valid KPF observation ID, e.g.
    'KP.20240113.23249.10'. Both the format and the embedded date/seconds
    are checked.
    """
    if not isinstance(s, str) or not _OBS_ID_PATTERN.fullmatch(s):
        return False
    return is_timestamp(s[3:])


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


def get_obs_id(fn):
    """Extract the obs_id (e.g. 'KP.20240113.23249.10') from a filename or path;
    raises ``ValueError`` if none is found."""
    if not isinstance(fn, str):
        raise ValueError(f"input must be a string; got {type(fn).__name__}")
    match = _OBS_ID_PATTERN.search(os.path.basename(fn))
    if not match:
        raise ValueError(f"No obs_id found in: {fn}")
    obs_id = match.group(1)
    if not is_timestamp(obs_id[3:]):
        raise ValueError(f"Invalid KPF timestamp in obs_id: {obs_id!r}")
    return obs_id


def get_datecode(s):
    """Extract the datecode (e.g. '20230708') from an obs_id or filename; raises
    ``ValueError`` if no valid obs_id is found."""
    if not isinstance(s, str):
        raise ValueError(f"input must be a string; got {type(s).__name__}")
    match = _OBS_ID_PATTERN.search(s)
    if not match:
        raise ValueError(f"Cannot extract datecode from: {s}")
    obs_id = match.group(1)
    if not is_timestamp(obs_id[3:]):
        raise ValueError(f"Invalid KPF timestamp in {s!r}")
    return obs_id.split(".")[1]


def get_timestamp(s):
    """Extract the KPF timestamp (e.g. '20240113.23249.10') from an obs_id,
    filename, or path; raises ``ValueError`` if none is found."""
    if not isinstance(s, str):
        raise ValueError(f"input must be a string; got {type(s).__name__}")
    match = _KPF_TIMESTAMP_PATTERN.search(os.path.basename(s))
    if not match:
        raise ValueError(f"No KPF timestamp found in: {s}")
    timestamp = match.group(1)
    if not is_timestamp(timestamp):
        raise ValueError(f"Invalid KPF timestamp: {timestamp!r}")
    return timestamp


def utc_to_hst(timestamp):
    """Convert a KPF UTC timestamp ('YYYYMMDD.SSSSS.FF') to HST (Hawaii Standard
    Time, UTC-10), returned in the same KPF format."""
    if not is_timestamp(timestamp):
        raise ValueError(f"Invalid KPF timestamp: {timestamp!r}")
    date_str, seconds_str, frame_str = timestamp.split(".")
    hst_seconds = int(seconds_str) - _HST_UTC_OFFSET_SECONDS
    date = datetime.strptime(date_str, "%Y%m%d")
    if hst_seconds < 0:
        hst_seconds += _SECONDS_PER_DAY
        date -= timedelta(days=1)
    return f"{date.strftime('%Y%m%d')}.{hst_seconds:05d}.{frame_str}"


def hst_to_utc(timestamp):
    """Convert a KPF HST timestamp ('YYYYMMDD.SSSSS.FF') to UTC, in the same KPF
    format. The inverse of `utc_to_hst`, kept as its symmetric counterpart: the
    pipeline only calls `utc_to_hst` (the data tree is UTC-keyed), so this backs
    round-trip tests and any future HST-keyed input."""
    if not is_timestamp(timestamp):
        raise ValueError(f"Invalid KPF timestamp: {timestamp!r}")
    date_str, seconds_str, frame_str = timestamp.split(".")
    utc_seconds = int(seconds_str) + _HST_UTC_OFFSET_SECONDS
    date = datetime.strptime(date_str, "%Y%m%d")
    if utc_seconds >= _SECONDS_PER_DAY:
        utc_seconds -= _SECONDS_PER_DAY
        date += timedelta(days=1)
    return f"{date.strftime('%Y%m%d')}.{utc_seconds:05d}.{frame_str}"


def kpf_timestamp_to_datetime(timestamp):
    """Parse a KPF UTC timestamp ('YYYYMMDD.SSSSS.FF', sub-second field ignored)
    into a naive UTC `datetime`, e.g. '20240405.40113.57' ->
    ``datetime(2024, 4, 5, 11, 8, 33)``."""
    if not is_timestamp(timestamp):
        raise ValueError(f"Invalid KPF timestamp: {timestamp!r}")
    date_str, seconds_str, _ = timestamp.split(".")
    return datetime.strptime(date_str, "%Y%m%d") + timedelta(seconds=int(seconds_str))


def kpf_timestamp_to_eprv_timestamp(timestamp):
    """Convert a KPF timestamp to EPRV format ('YYYYMMDDTHHMMSS', 1-second
    resolution, sub-second field dropped), e.g. '20240405.40113.57' ->
    '20240405T110833'."""
    if not is_timestamp(timestamp):
        raise ValueError(f"Invalid KPF timestamp: {timestamp!r}")
    date_str, seconds_str, _ = timestamp.split(".")
    total_seconds = int(seconds_str)
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{date_str}T{hh:02d}{mm:02d}{ss:02d}"
