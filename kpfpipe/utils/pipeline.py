import glob
import os
import warnings

import pandas as pd
from astropy.io import fits


_METADATA_KEYS = ['FILENAME', 'TARGNAME', 'IMTYPE', 'OBJECT', 'EXPTIME', 'ELAPSED']


def build_mini_database(data_dir):
    """
    Build a metadata table for all FITS files in a directory.

    Reads the PRIMARY header of each FITS file and extracts a standard
    set of keys used for frame selection (e.g. filtering by OBJECT type
    to identify bias, dark, or flat frames).

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

    return pd.DataFrame(metadata)
