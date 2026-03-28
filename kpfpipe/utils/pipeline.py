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


def _utc_to_hst(utc_timestamp):
    """
    Convert a KPF UTC timestamp to HST.

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


def _detect_calibration_stack_clusters(df):
    """
    Assign CAL_START and CAL_END columns to calibration frames in df.

    Calibration frames are grouped by OBJECT, sorted by UTC seconds, and
    split into clusters wherever the gap between consecutive frames exceeds
    _CALIBRATION_CLUSTER_GAP_SECONDS. Every frame in a cluster receives
    the UTC timestamp of the first frame as CAL_START and the UTC timestamp
    of the last frame as CAL_END. Science frames (IMTYPE == 'Object') receive
    an empty string for both columns.

    Args:
        df: DataFrame with FILENAME, IMTYPE, and OBJECT columns.

    Returns:
        df with CAL_START and CAL_END columns added (same row order as input).
    """
    def _utc_total_seconds(filename):
        date_str, seconds_str, _ = get_timestamp(filename).split('.')
        date_ordinal = datetime.strptime(date_str, '%Y%m%d').toordinal()
        return date_ordinal * 86400 + int(seconds_str)

    cal_start = pd.Series('', index=df.index)
    cal_end   = pd.Series('', index=df.index)

    cal_objects = set(_OBJECT_MAP.values())
    cal_df = df[df['OBJECT'].isin(cal_objects)].copy()
    cal_df['_UTC_TOTAL'] = cal_df['FILENAME'].apply(_utc_total_seconds)

    for _, group in cal_df.groupby('OBJECT', dropna=False):
        group = group.sort_values('_UTC_TOTAL')
        gaps = group['_UTC_TOTAL'].diff()
        cluster_id = (gaps > _CALIBRATION_CLUSTER_GAP_SECONDS).cumsum()

        for _, cluster_rows in group.groupby(cluster_id):
            start_time = get_timestamp(cluster_rows.iloc[0]['FILENAME'])
            end_time   = get_timestamp(cluster_rows.iloc[-1]['FILENAME'])
            cal_start.loc[cluster_rows.index] = start_time
            cal_end.loc[cluster_rows.index]   = end_time

    df = df.copy()
    df['CAL_START'] = cal_start
    df['CAL_END']   = cal_end
    return df


def build_mini_database(data_dir):
    """
    Build a metadata table for all FITS files in a directory and write
    it to disk as KP.{datecode}_{level}.csv in that directory.

    Reads the PRIMARY header of each FITS file and extracts a standard
    set of keys used for frame selection (e.g. filtering by OBJECT type
    to identify bias, dark, or flat frames). Also detects calibration stack
    clusters and records the start and end timestamps of each cluster.

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
            CAL_START -- UTC timestamp of the first frame in the calibration
                         stack cluster; empty string for science frames
            CAL_END   -- UTC timestamp of the last frame in the calibration
                         stack cluster; empty string for science frames

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
    df = _detect_calibration_stack_clusters(df)

    csv_path = os.path.join(data_dir, f'KP.{datecode}_{level}.csv')
    df.to_csv(csv_path, index=False)
    return df


def build_l0_file_list(data_dir, imtype, utc_time, min_file_count=5):
    """
    Build a sorted list of L0 FITS files for the calibration cluster most
    recently preceding a given UTC time.

    Loads the mini database CSV from data_dir if it exists; otherwise calls
    build_mini_database to scan headers and write it. Selects the calibration
    cluster of the requested type whose CAL_START is the most recent timestamp
    strictly before utc_time.

    If the selected cluster contains fewer than min_file_count files, all
    calibration frames of the requested type from the previous 24 hours are
    returned instead, with a warning. Raises if no cluster precedes utc_time
    within 24 hours, or if the expanded 24-hour window still yields fewer than
    min_file_count files.

    Args:
        data_dir:        path to directory containing L0 FITS files.
        imtype:          calibration frame type. One of 'bias', 'dark', 'flat'.
        utc_time:        UTC timestamp string ('YYYYMMDD.SSSSS.FF') of the
                         observation for which calibrations are being selected.
        min_file_count:  minimum number of files required in the returned list.
                         Default is 5.

    Returns:
        Sorted list of absolute file paths belonging to the selected cluster,
        or all frames within 24 hours if the cluster is below min_file_count.

    Raises:
        ValueError: if imtype is not a recognized calibration type, if no
                    calibration cluster precedes utc_time within 24 hours, or
                    if the 24-hour window contains fewer than min_file_count
                    files.
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
        if 'CAL_START' not in metadata.columns:
            metadata = build_mini_database(data_dir)
    else:
        metadata = build_mini_database(data_dir)

    def _to_seconds(ts):
        date_str, seconds_str, _ = ts.split('.')
        date_ordinal = datetime.strptime(date_str, '%Y%m%d').toordinal()
        return date_ordinal * 86400 + int(seconds_str)

    ref_seconds = _to_seconds(utc_time)

    cal_mask = metadata['OBJECT'] == _OBJECT_MAP[imtype]
    clusters = (
        metadata[cal_mask & (metadata['CAL_START'] != '')]
        [['CAL_START']]
        .drop_duplicates()
        .copy()
    )
    clusters['_SECONDS'] = clusters['CAL_START'].apply(_to_seconds)

    before = clusters[clusters['_SECONDS'] < ref_seconds]
    if before.empty:
        raise ValueError(
            f"No '{imtype}' calibration cluster found before {utc_time}"
        )

    best = before.loc[before['_SECONDS'].idxmax()]
    gap = ref_seconds - best['_SECONDS']
    if gap > 86400:
        raise ValueError(
            f"Nearest '{imtype}' calibration cluster (CAL_START={best['CAL_START']}) "
            f"is {gap / 3600:.1f} hours before {utc_time}; exceeds 24-hour limit"
        )

    cluster_mask = cal_mask & (metadata['CAL_START'] == best['CAL_START'])
    files = sorted(metadata.loc[cluster_mask, 'FILENAME'].tolist())

    if len(files) >= min_file_count:
        return files

    # Cluster is below min_file_count — expand to all frames within 24 hours.
    cal_files = metadata.loc[cal_mask, 'FILENAME'].tolist()
    cal_seconds = {f: _to_seconds(get_timestamp(f)) for f in cal_files}
    expanded_files = sorted(
        f for f in cal_files
        if ref_seconds - 86400 <= cal_seconds[f] < ref_seconds
    )

    if len(expanded_files) < min_file_count:
        raise ValueError(
            f"Only {len(expanded_files)} '{imtype}' frame(s) found in the 24 hours "
            f"before {utc_time}; need at least {min_file_count}"
        )

    warnings.warn(
        f"'{imtype}' cluster at CAL_START={best['CAL_START']} has only "
        f"{len(files)} file(s); using all {len(expanded_files)} frames "
        f"from the previous 24 hours instead."
    )
    return expanded_files


def get_calibration_stack_clusters(mini_db, imtype):
    """
    Return one sorted file list per calibration cluster of the requested type.

    Args:
        mini_db: DataFrame returned by build_mini_database.
        imtype:  calibration frame type. One of 'bias', 'dark', 'flat'.

    Returns:
        List of sorted file lists, one per cluster, ordered by CAL_START.

    Raises:
        ValueError: if imtype is not a recognized calibration type.
    """
    if imtype not in _OBJECT_MAP:
        raise ValueError(
            f"imtype must be one of {list(_OBJECT_MAP.keys())}; got '{imtype}'"
        )
    mask = (mini_db['OBJECT'] == _OBJECT_MAP[imtype]) & (mini_db['CAL_START'] != '')
    return [
        sorted(cluster['FILENAME'].tolist())
        for _, cluster in mini_db[mask].groupby('CAL_START')
    ]


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
