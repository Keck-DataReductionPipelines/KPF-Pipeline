import os
import re


_OBS_ID_PATTERN = re.compile(r'(KP\.\d{8}\.\d{5}\.\d{2})')
_DATECODE_PATTERN = re.compile(r'\d{8}')
_TIMESTAMP_PATTERN = re.compile(r'(\d{8}\.\d{5}\.\d{2})')


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
    return bool(_TIMESTAMP_PATTERN.fullmatch(s))


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
    Extract timestamp from an obs_id, filename, or path.

    Args:
        input_str: e.g. 'KP.20240113.23249.10' or '/data/L0/20240113/KP.20240113.23249.10.fits'

    Returns:
        timestamp: e.g. '20240113.23249.10'

    Raises:
        ValueError: if no timestamp pattern found in input_str
    """
    match = _TIMESTAMP_PATTERN.search(os.path.basename(input_str))
    if match:
        return match.group(1)
    raise ValueError(f"No timestamp found in: {input_str}")
