import glob
import os
import warnings

import pandas as pd
from astropy.io import fits
from datetime import datetime, timedelta

from kpfpipe.utils.kpf import is_datecode, get_datecode, is_obs_id, get_timestamp


_METADATA_KEYS = ['FILENAME', 'TARGNAME', 'IMTYPE', 'OBJECT', 'EXPTIME', 'ELAPSED']

_OBJECT_MAP = {
    'bias': 'autocal-bias',
    'dark': 'autocal-dark',
    'flat': 'autocal-flat-all',
}

_HST_UTC_OFFSET_SECONDS = 36000
_CALIBRATION_CLUSTER_GAP_SECONDS = 7200

_TIMEOFDAY_BOUNDARIES = [
    ( 0,  4, 'midnight'),
    ( 4, 10, 'morn'),
    (10, 15, 'midday'),
    (15, 19, 'eve'),
    (19, 23, 'night'),
    (23, 24, 'midnight'),
]

def _utc_to_hst(utc_timestamp):
    """
    Convert a KPF UTC timestamp to an HST timestamp.

    Args:
        utc_timestamp: str of the form 'YYYYMMDD.SSSSS.FF'

    Returns:
        str: HST timestamp in the same format
    """
    date_str, seconds_str, frame_str = utc_timestamp.split('.')
    hst_seconds = int(seconds_str) - _HST_UTC_OFFSET_SECONDS
    hst_date = datetime.strptime(date_str, '%Y%m%d')
    if hst_seconds < 0:
        hst_seconds += 86400
        hst_date -= timedelta(days=1)
    return f'{hst_date.strftime("%Y%m%d")}.{hst_seconds:05d}.{frame_str}'


def _detect_and_assign_timeofday(hst_timestamp):
    """
    Assign a local Hawaii time of day to an HST timestamp.

    Args:
        hst_timestamp: str of the form 'YYYYMMDD.SSSSS.FF' in HST

    Returns:
        str: one of 'midnight', 'morn', 'midday', 'eve', 'night'
    """
    hst_hour = int(hst_timestamp.split('.')[1]) / 3600
    for start, end, label in _TIMEOFDAY_BOUNDARIES:
        if start <= hst_hour < end:
            return label
    raise RuntimeError(f"No time-of-day label for HST hour {hst_hour:.2f}")


def _check_cluster_timeofday_consistency(df):
    """
    Verify that all files within a calibration cluster share the same TIMEOFDAY label.

    Calibrations are taken in rapid succession, then a multi-hour gap separates
    the next cluster. Files within a cluster should all map to the same time-of-day.
    A mismatch indicates a boundary-straddling cluster or a timestamp anomaly.

    Groups rows by OBJECT, sorts each group by UTC seconds, detects cluster
    boundaries using _CALIBRATION_CLUSTER_GAP_SECONDS, then checks that all
    TIMEOFDAY values within each cluster are identical.

    Args:
        df: DataFrame with FILENAME, OBJECT, and TIMEOFDAY columns.

    Raises:
        ValueError: if any cluster contains more than one TIMEOFDAY label.
    """
    for obj_val, group in df.groupby('OBJECT', dropna=False):
        # Build a sortable total-seconds value from the UTC timestamp.
        def utc_total_seconds(filename):
            date_str, seconds_str, _ = get_timestamp(filename).split('.')
            date_ordinal = datetime.strptime(date_str, '%Y%m%d').toordinal()
            return date_ordinal * 86400 + int(seconds_str)

        group = group.copy()
        group['_UTC_TOTAL'] = group['FILENAME'].apply(utc_total_seconds)
        group = group.sort_values('_UTC_TOTAL')

        gaps = group['_UTC_TOTAL'].diff()
        cluster_id = (gaps > _CALIBRATION_CLUSTER_GAP_SECONDS).cumsum()

        for cid, cluster_rows in group.groupby(cluster_id):
            labels = cluster_rows['TIMEOFDAY'].unique()
            if len(labels) > 1:
                raise ValueError(
                    f"OBJECT='{obj_val}' cluster {cid} spans multiple time-of-day "
                    f"labels: {sorted(labels)}. Check timestamps near "
                    f"{cluster_rows['FILENAME'].iloc[0]}"
                )


def build_mini_database(data_dir):
    """
    Build a metadata table for all FITS files in a directory and write
    it to disk as KP.{datecode}_{level}.csv in that directory.

    Reads the PRIMARY header of each FITS file and extracts a standard
    set of keys used for frame selection (e.g. filtering by OBJECT type
    to identify bias, dark, or flat frames). Also assigns a TIMEOFDAY
    label to each row based on its HST timestamp.

    Assumes data_dir follows the convention .../{level}/{datecode}/.

    Args:
        data_dir: path to directory containing L0 FITS files.

    Returns:
        pandas DataFrame with columns:
            FILENAME   -- absolute path to the FITS file
            TARGNAME   -- target name
            IMTYPE     -- image type
            OBJECT     -- object identifier (e.g. 'autocal-bias')
            EXPTIME    -- requested exposure time (s)
            ELAPSED    -- actual elapsed time (s)
            TIMEOFDAY  -- time-of-day label inferred from each file's HST timestamp

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

    timeofday = []
    for fn in df['FILENAME']:
        timestamp = get_timestamp(fn)
        hst_timestamp = _utc_to_hst(timestamp)
        timeofday.append(_detect_and_assign_timeofday(hst_timestamp))
    df['TIMEOFDAY'] = timeofday
    _check_cluster_timeofday_consistency(df)

    csv_path = os.path.join(data_dir, f'KP.{datecode}_{level}.csv')
    df.to_csv(csv_path, index=False)
    return df


def build_l0_file_list(data_dir, imtype, time_of_day=None):
    """
    Build a sorted list of L0 FITS files of a given calibration type.

    Loads the mini database CSV from data_dir if it exists and already
    contains a TIMEOFDAY column; otherwise calls build_mini_database to
    scan headers and write it. Filters by the OBJECT header keyword and,
    optionally, by time-of-day label.

    Args:
        data_dir:    path to directory containing L0 FITS files.
        imtype:      calibration frame type. One of 'bias', 'dark', 'flat'.
        time_of_day: optional label to filter by, e.g. 'morn', 'eve'.
                     If None, returns all frames of the requested type.

    Returns:
        Sorted list of absolute file paths matching the requested type
        (and time-of-day, if specified).

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
        if 'TIMEOFDAY' not in metadata.columns:
            metadata = build_mini_database(data_dir)
    else:
        metadata = build_mini_database(data_dir)

    mask = metadata['OBJECT'] == _OBJECT_MAP[imtype]
    if time_of_day is not None:
        mask &= metadata['TIMEOFDAY'] == time_of_day

    return sorted(metadata.loc[mask, 'FILENAME'].tolist())


def build_filepath(input_str, data_root, level, *, master=None):
    """
    Build an absolute filepath for a KPF data product.

    Args:
        input_str: obs_id (e.g. 'KP.20240405.49597.71') for both science
                   and master products. For masters, this should be the
                   obs_id of the first frame in the stack.
                   A bare datecode (e.g. '20240405') is also accepted for
                   master products (deprecated; prefer obs_id).
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
        # Masters: {data_root}/masters/{datecode}/{obs_id}_master_{master}_{level}.fits
        # Level is in the filename only — no level subdirectory.
        if master not in ('bias', 'dark', 'flat', 'thar-wls'):
            raise ValueError(f"'master' must be 'bias', 'dark', 'flat', or 'thar-wls'; got '{master}'")
        if level not in ('L1', 'L2', 'L4'):
            raise ValueError(f"'level' for master products must be 'L1', 'L2', or 'L4'; got '{level}'")

        if is_obs_id(input_str):
            datecode = get_datecode(input_str)
            filename = f'{input_str}_master_{master}_{level}.fits'
        elif is_datecode(input_str):
            datecode = input_str
            filename = f'kpf_{datecode}_{master}_{level}.fits'
        else:
            raise ValueError(
                f"input_str must be a valid obs_id or datecode for master products; got '{input_str}'"
            )
        return os.path.join(data_root, 'masters', datecode, filename)

    # Science: {data_root}/{level}/{datecode}/{obs_id}[_{level}].fits
    if level not in ('L0', 'L1', 'L2', 'L4'):
        raise ValueError(f"'level' must be 'L0', 'L1', 'L2', or 'L4'; got '{level}'")
    if not is_obs_id(input_str):
        raise ValueError("input_str must be a valid obs_id for science data products")

    datecode = get_datecode(input_str)
    filename = f'{input_str}.fits' if level == 'L0' else f'{input_str}_{level}.fits'
    return os.path.join(data_root, level, datecode, filename)
