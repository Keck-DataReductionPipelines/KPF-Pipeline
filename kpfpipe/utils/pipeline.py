"""L0 file-list discovery, calibration clustering, and the mini-database."""

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

_MINI_DB_KEYS = ["FILENAME", "TARGNAME", "IMTYPE", "OBJECT", "EXPTIME", "ELAPSED"]

_OBJECT_MAP = {
    "bias": ["autocal-bias"],
    "dark": ["autocal-dark"],
    "flat": ["autocal-flat-all"],
    "thar": [
        "autocal-thar-all-morn",
        "autocal-thar-all-midday",
        "autocal-thar-all-eve",
        "autocal-thar-all-night",
        "autocal-thar-all-midnight",
    ],
}


def build_mini_database(data_dir, write=True):
    """
    Build the mini database for all FITS files in a directory and write
    it to disk as KP.{datecode}_{level}.csv in that directory.

    Reads the PRIMARY header of each FITS file and extracts a standard
    set of keys used for frame selection (e.g. filtering by OBJECT to
    identify bias, dark, flat, or thar frames). Assumes `data_dir` follows
    the convention .../{level}/{datecode}/.

    Parameters
    ----------
    data_dir : str
        Path to directory containing L0 FITS files.
    write : bool, default True
        Whether to write the mini database CSV to `data_dir`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns FILENAME (absolute path to the FITS file),
        TARGNAME (target name), IMTYPE (image type), OBJECT (object
        identifier, e.g. 'autocal-bias'), EXPTIME (requested exposure time
        [s]), and ELAPSED (actual elapsed time [s]). Rows where a header key
        is missing are included with NaN for that column and a warning is
        issued.
    """
    data_dir = os.path.normpath(data_dir)
    datecode = os.path.basename(data_dir)
    level = os.path.basename(os.path.dirname(data_dir))

    file_list = sorted(glob.glob(os.path.join(data_dir, "*.fits")))

    if not file_list:
        raise ValueError(f"No FITS files found in {data_dir}")

    mini_db = {k: [] for k in _MINI_DB_KEYS}

    for fn in file_list:
        try:
            header = fits.getheader(fn, ext=0)
        except Exception as e:
            warnings.warn(f"Could not read header from {fn}: {e}", stacklevel=2)
            continue

        mini_db["FILENAME"].append(fn)

        for k in _MINI_DB_KEYS[1:]:
            mini_db[k].append(header.get(k, None))

    df = pd.DataFrame(mini_db)

    if write:
        csv_path = os.path.join(data_dir, f"KP.{datecode}_{level}.csv")
        df.to_csv(csv_path, index=False)
    return df


def build_l0_file_lists(
    imtype, *, data_dir=None, mini_db=None, min_file_count=5, cluster_gap_seconds=7200
):
    """
    Return sorted file lists for all calibration clusters of the requested
    type.

    Exactly one of `data_dir` or `mini_db` must be provided. When `data_dir`
    is given, loads the mini database CSV if it exists, otherwise calls
    `build_mini_database` to scan headers and write it. When `mini_db` is
    given, uses it directly to avoid redundant I/O. Filters by OBJECT, then
    groups frames into clusters by detecting gaps larger than
    `cluster_gap_seconds` between consecutive timestamps. Every returned
    cluster must have at least `min_file_count` files; otherwise raises.

    Parameters
    ----------
    imtype : str
        Calibration frame type. One of 'bias', 'dark', 'flat', 'thar'.
    data_dir : str, optional
        Path to directory containing L0 FITS files.
    mini_db : pandas.DataFrame, optional
        DataFrame returned by `build_mini_database`.
    min_file_count : int, default 5
        Minimum number of files required per cluster.
    cluster_gap_seconds : int, default 7200
        Gap [s] between consecutive frames that splits a calibration sequence
        into separate clusters. The default of 7200 (2 hours) reliably
        distinguishes morning vs. evening KPF calibration clusters, which are
        separated by science obs.

    Returns
    -------
    list of list of str
        Sorted file lists, one per cluster.

    Raises
    ------
    ValueError
        If `imtype` is not a recognized calibration type, if exactly one of
        `data_dir` or `mini_db` is not provided, if no calibration frames of
        the requested type are found, or if any cluster contains fewer than
        `min_file_count` files.
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
        csv_path = os.path.join(data_dir, f"KP.{datecode}_{level}.csv")

        if os.path.isfile(csv_path):
            mini_db = pd.read_csv(csv_path)
            # Normalize both sides via realpath so symlinks and relative
            # paths don't trigger gratuitous rebuilds.
            on_disk = {
                os.path.realpath(p) for p in glob.glob(os.path.join(data_dir, "*.fits"))
            }
            cached = (
                {os.path.realpath(p) for p in mini_db["FILENAME"]}
                if "FILENAME" in mini_db.columns
                else set()
            )
            if on_disk != cached:
                added = on_disk - cached
                removed = cached - on_disk
                warnings.warn(
                    f"Mini database at {csv_path} is stale "
                    f"(+{len(added)} added, -{len(removed)} removed on disk); "
                    "rebuilding.",
                    UserWarning,
                    stacklevel=2,
                )
                mini_db = build_mini_database(data_dir)
        else:
            mini_db = build_mini_database(data_dir)

    cal_df = mini_db[mini_db["OBJECT"].isin(_OBJECT_MAP[imtype])]

    if cal_df.empty:
        raise ValueError(
            f"No '{imtype}' calibration frames found in "
            f"{data_dir or 'the provided mini_db'}"
        )

    # Cluster per-OBJECT (morning vs. evening thar etc. have different OBJECT
    # suffixes), splitting wherever consecutive frames are more than
    # cluster_gap_seconds apart. Final list sorted chronologically.
    clusters = []
    for _, group in cal_df.groupby("OBJECT", dropna=False):
        timed = sorted((get_seconds_since_j2000(fn), fn) for fn in group["FILENAME"])
        if not timed:
            continue
        cluster = [timed[0][1]]
        for (prev_t, _), (t, fn) in zip(timed, timed[1:], strict=False):
            if t - prev_t > cluster_gap_seconds:
                clusters.append(cluster)
                cluster = [fn]
            else:
                cluster.append(fn)
        clusters.append(cluster)
    clusters.sort(key=lambda c: get_seconds_since_j2000(c[0]))

    short = [c for c in clusters if len(c) < min_file_count]
    if short:
        raise ValueError(
            f"'{imtype}' has {len(short)} cluster(s) below "
            f"min_file_count={min_file_count}; sizes: {[len(c) for c in short]}"
        )
    return clusters


def build_qlp_dir(obs_id, level, *, data_root):
    """
    Build the QLP output directory for a given observation and level.

    Parameters
    ----------
    obs_id : str
        Observation ID (e.g. 'KP.20240405.49597.71').
    level : str
        Data level string, one of 'L0', 'L1', 'L2', 'L4'.
    data_root : str
        Root data directory (e.g. '/data/kpf-next/').

    Returns
    -------
    str
        Absolute path {data_root}/QLP/{datecode}/{obs_id}/{level}/.

    Raises
    ------
    ValueError
        If `obs_id` is not a valid observation ID, or if `data_root` is not a
        non-empty string.
    """
    if not isinstance(data_root, str) or not data_root:
        raise ValueError(f"data_root must be a non-empty string; got {data_root!r}")
    if not is_obs_id(obs_id):
        raise ValueError(
            "obs_id must be a valid observation ID "
            f"(e.g. 'KP.20240405.49597.71'); got '{obs_id}'"
        )
    return os.path.join(data_root, "QLP", get_datecode(obs_id), obs_id, level)


def build_filepath(obs_id, level, *, data_root=None, master=None):
    """
    Build a filepath for a KPF data product.

    Parameters
    ----------
    obs_id : str
        Observation ID (e.g. 'KP.20240405.49597.71'). For master products
        this should be the obs_id of the first frame in the stack.
    level : str
        Data level string, one of 'L0', 'L1', 'L2', 'L4'.
    data_root : str or None, optional
        Root data directory (e.g. '/data/kpf/'). When None (the default),
        returns the bare filename. Otherwise must be a non-empty string and a
        full path is returned.
    master : str or None, optional
        Master calibration type, one of 'bias', 'dark', 'flat', 'thar'. If
        provided, builds a master calibration path. If omitted, builds a
        science data path.

    Returns
    -------
    str
        Filepath (full path if `data_root` is set, bare filename if
        `data_root` is None).

    Raises
    ------
    ValueError
        If `level` is unrecognized, if `obs_id` is not a valid observation
        ID, if `master` type is unrecognized, or if `data_root` is not None
        and not a non-empty string.
    """
    if data_root is not None and (not isinstance(data_root, str) or not data_root):
        raise ValueError(
            f"data_root must be None or a non-empty string; got {data_root!r}"
        )
    if not is_obs_id(obs_id):
        raise ValueError(
            "obs_id must be a valid observation ID "
            f"(e.g. 'KP.20240405.49597.71'); got '{obs_id}'"
        )

    datecode = get_datecode(obs_id)

    if master is not None:
        # Masters: {data_root}/masters/{datecode}/{obs_id}_master_{master}_{level}.fits
        # Level is in the filename only — no level subdirectory.
        if master not in ("bias", "dark", "flat", "thar"):
            raise ValueError(
                f"'master' must be 'bias', 'dark', 'flat', or 'thar'; got '{master}'"
            )
        if level not in ("L1", "L2", "L4"):
            raise ValueError(
                "'level' for master products must be 'L1', 'L2', or 'L4'; "
                f"got '{level}'"
            )
        filename = f"{obs_id}_master_{master}_{level}.fits"
        if data_root is None:
            return filename
        return os.path.join(data_root, "masters", datecode, filename)

    # Science paths by level:
    #   L0:       {obs_id}.fits                                       (KPF-native)
    #   L1/L2/L4: kpf_SL{N}_{YYYYMMDD}T{HHmmss}.fits                (EPRV standard)
    if level not in ("L0", "L1", "L2", "L4"):
        raise ValueError(f"'level' must be 'L0', 'L1', 'L2', or 'L4'; got '{level}'")

    if level == "L0":
        filename = f"{obs_id}.fits"
    else:
        eprv_ts = kpf_timestamp_to_eprv_timestamp(get_timestamp(obs_id))
        filename = f"kpf_SL{level[1]}_{eprv_ts}.fits"

    if data_root is None:
        return filename
    return os.path.join(data_root, level, datecode, filename)
