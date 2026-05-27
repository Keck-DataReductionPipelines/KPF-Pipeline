import glob
import os
import warnings

import pandas as pd
from astropy.io import fits

from kpfpipe.utils.kpf import (
    get_datecode,
    get_seconds_since_j2000,
    get_timestamp,
    is_obs_id,
    kpf_timestamp_to_eprv_timestamp,
)


_MINI_DB_KEYS = ['FILENAME', 'TARGNAME', 'IMTYPE', 'OBJECT', 'EXPTIME', 'ELAPSED']

_OBJECT_MAP = {
    'bias':     ['autocal-bias'],
    'dark':     ['autocal-dark'],
    'flat':     ['autocal-flat-all'],
    'thar': [
        'autocal-thar-all-morn',
        'autocal-thar-all-midday',
        'autocal-thar-all-eve',
        'autocal-thar-all-night',
        'autocal-thar-all-midnight',
    ],
}

def _detect_calibration_stack_clusters(df, cluster_gap_seconds=7200):
    """
    Assign CAL_START and CAL_END columns to calibration frames in df.

    Calibration frames are grouped by OBJECT, sorted by UTC seconds, and
    split into clusters wherever the gap between consecutive frames exceeds
    cluster_gap_seconds. Every frame in a cluster receives the UTC timestamp
    of the first frame as CAL_START and the UTC timestamp of the last frame
    as CAL_END. Science frames (IMTYPE == 'Object') receive an empty string
    for both columns.

    Args:
        df:                  DataFrame with FILENAME, IMTYPE, and OBJECT columns.
        cluster_gap_seconds: gap between consecutive frames that splits a
                             calibration sequence into separate clusters.
                             Default 7200s (2 hours) reliably distinguishes
                             morning vs. evening KPF calibration clusters,
                             which are separated by science observations.

    Returns:
        df with CAL_START and CAL_END columns added (same row order as input).
    """
    cal_start = pd.Series('', index=df.index)
    cal_end   = pd.Series('', index=df.index)

    cal_objects = {obj for objs in _OBJECT_MAP.values() for obj in objs}
    cal_df = df[df['OBJECT'].isin(cal_objects)].copy()
    cal_df['_UTC_TOTAL'] = [get_seconds_since_j2000(f) for f in cal_df['FILENAME']]

    for _, group in cal_df.groupby('OBJECT', dropna=False):
        group = group.sort_values('_UTC_TOTAL')
        gaps = group['_UTC_TOTAL'].diff()
        cluster_id = (gaps > cluster_gap_seconds).cumsum()

        for _, cluster_rows in group.groupby(cluster_id):
            start_time = get_timestamp(cluster_rows.iloc[0]['FILENAME'])
            end_time   = get_timestamp(cluster_rows.iloc[-1]['FILENAME'])
            cal_start.loc[cluster_rows.index] = start_time
            cal_end.loc[cluster_rows.index]   = end_time

    df = df.copy()
    df['CAL_START'] = cal_start
    df['CAL_END']   = cal_end
    return df


def build_mini_database(data_dir, write=True):
    """
    Build the mini database for all FITS files in a directory and write
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

    if not file_list:
        raise ValueError(f"No FITS files found in {data_dir}")

    mini_db = {k: [] for k in _MINI_DB_KEYS}

    for fn in file_list:
        try:
            header = fits.getheader(fn, ext=0)
        except Exception as e:
            warnings.warn(f"Could not read header from {fn}: {e}")
            continue

        mini_db['FILENAME'].append(fn)

        for k in _MINI_DB_KEYS[1:]:
            mini_db[k].append(header.get(k, None))

    df = pd.DataFrame(mini_db)
    df = _detect_calibration_stack_clusters(df)

    if write:
        csv_path = os.path.join(data_dir, f'KP.{datecode}_{level}.csv')
        df.to_csv(csv_path, index=False)
    return df


def build_l0_file_lists(imtype, min_file_count=5, *, data_dir=None, mini_db=None):
    """
    Return sorted file lists for all calibration clusters of the requested type.

    Exactly one of data_dir or mini_db must be provided. When data_dir is given,
    loads the mini database CSV if it exists, otherwise calls build_mini_database
    to scan headers and write it. When mini_db is given, uses it directly to
    avoid redundant I/O. Groups calibration frames into clusters. Clusters with
    at least min_file_count files are returned as individual lists. If any
    cluster falls below min_file_count, all clusters of that type are merged
    into a single list with a warning.

    Args:
        imtype:          calibration frame type. One of 'bias', 'dark', 'flat', 'thar'.
        min_file_count:  minimum number of files required per returned list.
                         Default is 5.
        data_dir:        path to directory containing L0 FITS files.
        mini_db:         DataFrame returned by build_mini_database.

    Returns:
        List of sorted file lists, one per cluster or one merged list if any
        cluster fell below min_file_count.

    Raises:
        ValueError: if imtype is not a recognized calibration type, if exactly
                    one of data_dir or mini_db is not provided, if no
                    calibration frames of the requested type are found, or if
                    the merged total still contains fewer than min_file_count
                    files.
    """
    if imtype not in _OBJECT_MAP:
        raise ValueError(
            f"imtype must be one of {list(_OBJECT_MAP.keys())}; got '{imtype}'"
        )

    if (data_dir is None) == (mini_db is None):
        raise ValueError("Exactly one of data_dir or mini_db must be provided")

    if mini_db is None:
        data_dir = os.path.normpath(data_dir)
        datecode = os.path.basename(data_dir)
        level = os.path.basename(os.path.dirname(data_dir))
        csv_path = os.path.join(data_dir, f'KP.{datecode}_{level}.csv')

        if os.path.isfile(csv_path):
            mini_db = pd.read_csv(csv_path)
            on_disk = set(glob.glob(os.path.join(data_dir, '*.fits')))
            cached = set(mini_db['FILENAME']) if 'FILENAME' in mini_db.columns else set()
            if 'CAL_START' not in mini_db.columns:
                warnings.warn(
                    f"Mini database at {csv_path} is missing CAL_START column; rebuilding.",
                    UserWarning,
                )
                mini_db = build_mini_database(data_dir)
            elif on_disk != cached:
                added = on_disk - cached
                removed = cached - on_disk
                warnings.warn(
                    f"Mini database at {csv_path} is stale "
                    f"(+{len(added)} added, -{len(removed)} removed on disk); rebuilding.",
                    UserWarning,
                )
                mini_db = build_mini_database(data_dir)
        else:
            mini_db = build_mini_database(data_dir)

    mask = mini_db['OBJECT'].isin(_OBJECT_MAP[imtype]) & (mini_db['CAL_START'] != '')
    cal_df = mini_db[mask]

    if cal_df.empty:
        raise ValueError(
            f"No '{imtype}' calibration frames found in {data_dir or 'the provided mini_db'}"
        )

    clusters = [
        sorted(group['FILENAME'].tolist())
        for _, group in cal_df.groupby('CAL_START')
    ]

    if all(len(c) >= min_file_count for c in clusters):
        return clusters

    merged = sorted(f for c in clusters for f in c)
    if len(merged) < min_file_count:
        raise ValueError(
            f"Only {len(merged)} '{imtype}' frame(s) found in {data_dir or 'the provided mini_db'}; "
            f"need at least {min_file_count}"
        )
    warnings.warn(
        f"'{imtype}' clusters below min_file_count={min_file_count}; "
        f"merged into one list of {len(merged)} files.",
        UserWarning,
    )
    return [merged]


def build_qlp_dir(obs_id, level, *, data_root):
    """
    Build the QLP output directory for a given observation and level.

    Args:
        obs_id:    observation ID (e.g. 'KP.20240405.49597.71').
        level:     data level string, one of 'L0', 'L1', 'L2', 'L4'.
        data_root: root data directory (e.g. '/data/kpf-next/').

    Returns:
        Absolute path: {data_root}/QLP/{datecode}/{obs_id}/{level}/

    Raises:
        ValueError: if obs_id is not a valid observation ID.
    """
    if not is_obs_id(obs_id):
        raise ValueError(f"obs_id must be a valid observation ID (e.g. 'KP.20240405.49597.71'); got '{obs_id}'")
    return os.path.join(data_root, 'QLP', get_datecode(obs_id), obs_id, level)


def build_filepath(obs_id, level, *, data_root=None, master=None):
    """
    Build a filepath for a KPF data product.

    Args:
        obs_id:    observation ID (e.g. 'KP.20240405.49597.71'). For master
                   products this should be the obs_id of the first frame in
                   the stack.
        level:     data level string, one of 'L0', 'L1', 'L2', 'L4'.
        data_root: root data directory (e.g. '/data/kpf/'). When provided,
                   returns an absolute path. When omitted, returns the bare
                   filename only.
        master:    master calibration type, one of 'bias', 'dark', 'flat',
                   'thar'. If provided, builds a master calibration path.
                   If omitted, builds a science data path.

    Returns:
        Absolute filepath as a string if data_root is given, else bare filename.

    Raises:
        ValueError: if level is unrecognized, if obs_id is not a valid
                    observation ID, or if master type is unrecognized.
    """
    if not is_obs_id(obs_id):
        raise ValueError(f"obs_id must be a valid observation ID (e.g. 'KP.20240405.49597.71'); got '{obs_id}'")

    datecode = get_datecode(obs_id)

    if master is not None:
        # Masters: {data_root}/masters/{datecode}/{obs_id}_master_{master}_{level}.fits
        # Level is in the filename only — no level subdirectory.
        if master not in ('bias', 'dark', 'flat', 'thar'):
            raise ValueError(f"'master' must be 'bias', 'dark', 'flat', or 'thar'; got '{master}'")
        if level not in ('L1', 'L2', 'L4'):
            raise ValueError(f"'level' for master products must be 'L1', 'L2', or 'L4'; got '{level}'")
        filename = f'{obs_id}_master_{master}_{level}.fits'
        if data_root is None:
            return filename
        return os.path.join(data_root, 'masters', datecode, filename)

    # Science paths by level:
    #   L0:       {obs_id}.fits                                       (KPF-native)
    #   L1/L2/L4: kpf_SL{N}_{YYYYMMDD}T{HHmmss}.fits                (EPRV standard)
    if level not in ('L0', 'L1', 'L2', 'L4'):
        raise ValueError(f"'level' must be 'L0', 'L1', 'L2', or 'L4'; got '{level}'")

    if level == 'L0':
        filename = f'{obs_id}.fits'
    else:
        eprv_ts = kpf_timestamp_to_eprv_timestamp(get_timestamp(obs_id))
        filename = f'kpf_SL{level[1]}_{eprv_ts}.fits'

    if data_root is None:
        return filename
    return os.path.join(data_root, level, datecode, filename)
