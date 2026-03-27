import numpy as np
import re


def get_datecode(input_str):
    """
    Extract the datecode from an obs_id or filename

    Args:
        input_str, e.g. 'KP.20230708.04519.63' or 'KP.20230708.04519.63_2D.fits'

    Returns:
        datecode, e.g. '20230708'
    """
    # TODO: modify to properly handle masters files, use regex
    if is_obs_id(input_str):
        obs_id = input_str
    else:
        obs_id = get_obs_id(input_str)
    
    datecode = obs_id.split('.')[1]

    return datecode


def get_obs_id(filename):
    """
    Extracts an obs_id from a filename
    
    Args:
        filename, e.g. '/data/L1/20240113/KP.20240113.23249.10_L1.fits').

    Returns:
        obs_id, e.g. 'KP.20240113.23249.10'
    """
    # TODO: modify to properly handle masters files, use regex
    obs_id = filename.split('/')[-1]
    for substring in ['.fits', '_2D', '_L1', '_L2']:
        obs_id = obs_id.replace(substring, '')
    return obs_id


def is_obs_id(obs_id):
    """
    Returns True if the input is a properly formatted ObsID, e.g. 'KP.20240113.23249.10'
    """
    pattern = r'^KP\.\d{8}\.\d{5}\.\d{2}$'
    is_obs_id_bool = bool(re.match(pattern, obs_id))
    return is_obs_id_bool


def fetch_filepath(input_str, *, level=None, master=None, abspath=True):
    # TODO: fix logic to allow L1, L2 masters to be pulled
    if (level is None) == (master is None):
        raise ValueError("Exactly one of 'level' or 'master' must be provided.")
    
    datecode = get_datecode(input_str)

    if level is not None:
        if not is_obs_id(input_str):
            raise ValueError("input string must be a valid obs_id when 'level' is provided")
        
        if level == 'L0':
            filepath = f'{input_str}.fits'
        elif np.isin(level, ['L1', 'L2', 'FFI']):
            filepath = f'{input_str}_{level}.fits'
        else:
            raise ValueError(f"'level' must be one of 'L0', 'L1', 'L2', or 'FFI'; got {level}")

        if abspath:
            filepath = f'/data/{level}/{datecode}/{filepath}/'

        return filepath

    if master is not None:
        if np.isin(master, ['bias', 'dark', 'flat', 'thar-wls']):
            filename = f'kpf_{datecode}_{master}.fits'
        else:
            raise ValueError(f"'master' must be one of 'bias', 'dark', 'flat', or 'thar-wls'; got {master}")

        if abspath:
            filepath = f'/data/masters/{datecode}/{filepath}'

        return filepath
