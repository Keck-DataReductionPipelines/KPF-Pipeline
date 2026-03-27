import glob
import os
import warnings

import pandas as pd
from astropy.io import fits

from kpfpipe.utils.kpf import is_datecode, get_datecode, is_obs_id


_METADATA_KEYS = ['FILENAME', 'TARGNAME', 'IMTYPE', 'OBJECT', 'EXPTIME', 'ELAPSED']

_OBJECT_MAP = {
    'bias': 'autocal-bias',
    'dark': 'autocal-dark',
    'flat': 'autocal-flat-all',
}


def build_mini_database(data_dir):
    """
    Build a metadata table for all FITS files in a directory and write
    it to disk as KP.{datecode}_{level}.csv in that directory.

    Reads the PRIMARY header of each FITS file and extracts a standard
    set of keys used for frame selection (e.g. filtering by OBJECT type
    to identify bias, dark, or flat frames).

    Assumes data_dir follows the convention .../{level}/{datecode}/.

    Args:
        data_dir: path to directory containing L0 FITS files.

    Returns:
        pandas DataFrame with columns:
            FILENAME  -- absolute path to the FITS file
            TARGNAME  -- target name
            IMTYPE    -- image type
            OBJECT    -- object identifier (e.g. 'autocal-bias')
            EXPTIME   -- requested exposure time (s)
            ELAPSED   -- actual elapsed time (s)

        Rows where a header key is missing are included with NaN for
        that column and a warning is issued.
    """
    data_dir = os.path.normpath(data_dir)
    datecode = os.path.basename(data_dir)
    level = os.path.basename(os.path.dirname(data_dir))

    file_list = sorted(glob.glob(os.path.join(data_dir, '*.fits')))

    metadata = {k: [] for k in _METADATA_KEYS}

    for fn in file_list:
        try:
            header = fits.getheader(fn, ext=0)
        except Exception as e:
            warnings.warn(f"Could not read header from {fn}: {e}")
            continue

        metadata['FILENAME'].append(fn)

        for k in _METADATA_KEYS[1:]:
            metadata[k].append(header.get(k, None))

    df = pd.DataFrame(metadata)
    csv_path = os.path.join(data_dir, f'KP.{datecode}_{level}.csv')
    df.to_csv(csv_path, index=False)
    return df


def build_l0_file_list(data_dir, imtype):
    """
    Build a sorted list of L0 FITS files of a given calibration type.

    Loads the mini database CSV from data_dir if it exists; otherwise
    calls build_mini_database to scan headers and write it. Filters
    by the OBJECT header keyword.

    Args:
        data_dir: path to directory containing L0 FITS files.
        imtype:   calibration frame type. One of 'bias', 'dark', 'flat'.

    Returns:
        Sorted list of absolute file paths matching the requested type.

    Raises:
        ValueError: if imtype is not a recognized calibration type.
    """
    if imtype not in _OBJECT_MAP:
        raise ValueError(
            f"imtype must be one of {list(_OBJECT_MAP.keys())}; got '{imtype}'"
        )

    data_dir = os.path.normpath(data_dir)
    datecode = os.path.basename(data_dir)
    level = os.path.basename(os.path.dirname(data_dir))
    csv_path = os.path.join(data_dir, f'KP.{datecode}_{level}.csv')

    if os.path.isfile(csv_path):
        metadata = pd.read_csv(csv_path)
    else:
        metadata = build_mini_database(data_dir)

    mask = metadata['OBJECT'] == _OBJECT_MAP[imtype]
    return sorted(metadata.loc[mask, 'FILENAME'].tolist())


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
    if master is not None:
        # Masters: {data_root}/masters/{datecode}/kpf_{datecode}_{master}_{level}.fits
        # Level is in the filename only — no level subdirectory.
        if master not in ('bias', 'dark', 'flat', 'thar-wls'):
            raise ValueError(f"'master' must be 'bias', 'dark', 'flat', or 'thar-wls'; got '{master}'")
        if level not in ('L1', 'L2', 'L4'):
            raise ValueError(f"'level' for master products must be 'L1', 'L2', or 'L4'; got '{level}'")

        datecode = input_str if is_datecode(input_str) else get_datecode(input_str)
        filename = f'kpf_{datecode}_{master}_{level}.fits'
        return os.path.join(data_root, 'masters', datecode, filename)

    # Science: {data_root}/{level}/{datecode}/{obs_id}[_{level}].fits
    if level not in ('L0', 'L1', 'L2', 'L4'):
        raise ValueError(f"'level' must be 'L0', 'L1', 'L2', or 'L4'; got '{level}'")
    if not is_obs_id(input_str):
        raise ValueError("input_str must be a valid obs_id for science data products")

    datecode = get_datecode(input_str)
    filename = f'{input_str}.fits' if level == 'L0' else f'{input_str}_{level}.fits'
    return os.path.join(data_root, level, datecode, filename)
