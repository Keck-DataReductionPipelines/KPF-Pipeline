import os
import re
from datetime import datetime, timedelta


_OBS_ID_PATTERN    = re.compile(r'(KP\.\d{8}\.\d{5}\.\d{2})')
_DATECODE_PATTERN  = re.compile(r'\d{8}')
_KPF_TIMESTAMP_PATTERN  = re.compile(r'(\d{8}\.\d{5}\.\d{2})')
_EPRV_TIMESTAMP_PATTERN = re.compile(r'\d{8}T\d{6}')

# Hawaii Standard Time is UTC-10 (KPF timestamps are UTC)
_HST_UTC_OFFSET_SECONDS = 36000


def is_obs_id(s):
    """
    Returns True if s is a valid KPF observation ID, e.g. 'KP.20240113.23249.10'
    """
    return bool(_OBS_ID_PATTERN.fullmatch(s))


def is_datecode(s):
    """
    Returns True if s is a valid 8-digit datecode, e.g. '20240405'
    """
    return bool(_DATECODE_PATTERN.fullmatch(s))


def is_timestamp(s):
    """
    Returns True if s is a valid KPF timestamp, e.g. '20240113.23249.10'
    """
    return bool(_KPF_TIMESTAMP_PATTERN.fullmatch(s))


def get_obs_id(filename):
    """
    Extract obs_id from a filename or path.

    Args:
        filename: e.g. '/data/L1/20240113/KP.20240113.23249.10_L1.fits'

    Returns:
        obs_id: e.g. 'KP.20240113.23249.10'

    Raises:
        ValueError: if no obs_id pattern found in filename
    """
    match = _OBS_ID_PATTERN.search(os.path.basename(filename))
    if match:
        return match.group(1)
    raise ValueError(f"No obs_id found in: {filename}")


def get_datecode(input_str):
    """
    Extract datecode from an obs_id or filename.

    Args:
        input_str: e.g. 'KP.20230708.04519.63' or 'KP.20230708.04519.63_L1.fits'

    Returns:
        datecode: e.g. '20230708'

    Raises:
        ValueError: if no obs_id pattern found in input_str
    """
    match = _OBS_ID_PATTERN.search(input_str)
    if match:
        return match.group(1).split('.')[1]
    raise ValueError(f"Cannot extract datecode from: {input_str}")


def get_timestamp(input_str):
    """
    Extract KPF timestamp from an obs_id, filename, or path.

    Args:
        input_str: e.g. 'KP.20240113.23249.10' or '/data/L0/20240113/KP.20240113.23249.10.fits'

    Returns:
        timestamp: e.g. '20240113.23249.10'

    Raises:
        ValueError: if no KPF timestamp pattern found in input_str
    """
    match = _KPF_TIMESTAMP_PATTERN.search(os.path.basename(input_str))
    if match:
        return match.group(1)
    raise ValueError(f"No KPF timestamp found in: {input_str}")


def utc_to_hst(timestamp):
    """
    Convert a KPF UTC timestamp to HST (Hawaii Standard Time, UTC-10).

    Args:
        timestamp: KPF timestamp string of the form 'YYYYMMDD.SSSSS.FF'

    Returns:
        str: HST timestamp in the same KPF format
    """
    date_str, seconds_str, frame_str = timestamp.split('.')
    hst_seconds = int(seconds_str) - _HST_UTC_OFFSET_SECONDS
    date = datetime.strptime(date_str, '%Y%m%d')
    if hst_seconds < 0:
        hst_seconds += 86400
        date -= timedelta(days=1)
    return f'{date.strftime("%Y%m%d")}.{hst_seconds:05d}.{frame_str}'


def hst_to_utc(timestamp):
    """
    Convert a KPF HST timestamp to UTC.

    Args:
        timestamp: KPF timestamp string of the form 'YYYYMMDD.SSSSS.FF'

    Returns:
        str: UTC timestamp in the same KPF format
    """
    date_str, seconds_str, frame_str = timestamp.split('.')
    utc_seconds = int(seconds_str) + _HST_UTC_OFFSET_SECONDS
    date = datetime.strptime(date_str, '%Y%m%d')
    if utc_seconds >= 86400:
        utc_seconds -= 86400
        date += timedelta(days=1)
    return f'{date.strftime("%Y%m%d")}.{utc_seconds:05d}.{frame_str}'


def kpf_timestamp_to_eprv_timestamp(timestamp):
    """
    Convert a KPF timestamp to EPRV standard format.

    EPRV timestamps have 1-second resolution; the sub-second frame field
    is dropped.

    Args:
        timestamp: KPF timestamp string of the form 'YYYYMMDD.SSSSS.FF'

    Returns:
        str: EPRV timestamp of the form 'YYYYMMDDTHHMMSS'

    Example:
        kpf_timestamp_to_eprv_timestamp('20240405.40113.57') -> '20240405T110833'
    """
    date_str, seconds_str, _ = timestamp.split('.')
    total_seconds = int(seconds_str)
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f'{date_str}T{hh:02d}{mm:02d}{ss:02d}'


def eprv_timestamp_to_kpf_timestamp(timestamp):
    """
    Convert an EPRV standard timestamp to KPF format.

    The frame field is set to '00' since EPRV timestamps have 1-second
    resolution and carry no sub-second information.

    Args:
        timestamp: EPRV timestamp string of the form 'YYYYMMDDTHHMMSS'

    Returns:
        str: KPF timestamp of the form 'YYYYMMDD.SSSSS.00'

    Example:
        eprv_timestamp_to_kpf_timestamp('20240405T110833') -> '20240405.40113.00'
    """
    date_str = timestamp[:8]
    time_str = timestamp[9:]
    hh = int(time_str[0:2])
    mm = int(time_str[2:4])
    ss = int(time_str[4:6])
    total_seconds = hh * 3600 + mm * 60 + ss
    return f'{date_str}.{total_seconds:05d}.00'
