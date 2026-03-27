import os
import re


_OBS_ID_PATTERN = re.compile(r'(KP\.\d{8}\.\d{5}\.\d{2})')
_DATECODE_PATTERN = re.compile(r'^\d{8}$')


def is_obs_id(s):
    """
    Returns True if s is a valid KPF observation ID, e.g. 'KP.20240113.23249.10'
    """
    return bool(_OBS_ID_PATTERN.fullmatch(s))


def is_datecode(s):
    """
    Returns True if s is a valid 8-digit datecode, e.g. '20240405'
    """
    return bool(_DATECODE_PATTERN.match(s))


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


def build_filepath(input_str, data_root, level, *, master=None):
    """
    Build an absolute filepath for a KPF data product.

    Args:
        input_str: obs_id (e.g. 'KP.20240405.49597.71') for science products;
                   obs_id or datecode (e.g. '20240405') for master products.
        data_root: root data directory from config (e.g. '/data/kpf/').
        level:     data level string, one of 'L0', 'L1', 'L2', 'L4'.
        master:    master calibration type, one of 'bias', 'dark', 'flat',
                   'thar-wls'. If provided, builds a master calibration path.
                   If omitted, builds a science data path.

    Returns:
        Absolute filepath as a string.

    Raises:
        ValueError: if level is unrecognized, if input_str is not a valid
                    obs_id for science products, or if master type is
                    unrecognized.
    """
    if level not in ('L0', 'L1', 'L2', 'L4'):
        raise ValueError(f"'level' must be 'L0', 'L1', 'L2', or 'L4'; got '{level}'")

    if master is not None:
        if master not in ('bias', 'dark', 'flat', 'thar-wls'):
            raise ValueError(f"'master' must be 'bias', 'dark', 'flat', or 'thar-wls'; got '{master}'")

        datecode = input_str if is_datecode(input_str) else get_datecode(input_str)
        filename = f'kpf_{datecode}_{master}_{level}.fits'
        return os.path.join(data_root, 'masters', datecode, filename)

    if not is_obs_id(input_str):
        raise ValueError("input_str must be a valid obs_id for science data products")

    datecode = get_datecode(input_str)
    filename = f'{input_str}.fits' if level == 'L0' else f'{input_str}_{level}.fits'
    return os.path.join(data_root, level, datecode, filename)
