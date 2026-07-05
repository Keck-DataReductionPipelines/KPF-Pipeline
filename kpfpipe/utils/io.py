"""L0 file-list discovery, calibration clustering, and the mini-database."""

import glob
import logging
import os
import warnings

import pandas as pd
from astropy.io import fits

from kpfpipe.utils.kpf import (
    get_datecode,
    get_obs_id,
    get_seconds_since_j2000,
    get_timestamp,
    is_obs_id,
    kpf_timestamp_to_eprv_timestamp,
    utc_to_hst,
)

logger = logging.getLogger(__name__)

_MINI_DB_KEYS = ["FILENAME", "TARGNAME", "IMTYPE", "OBJECT", "EXPTIME", "ELAPSED"]
# Derived, not read from a header key: UTC/HST from the filename's KPF timestamp,
# ISJUNK from the observer junk list (see load_junk_obs_ids).
_MINI_DB_DERIVED_KEYS = ["UTC", "HST", "ISJUNK"]

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


def load_junk_obs_ids(data_input):
    """Set of observer-flagged junk obs_ids for a data tree.

    "Junk" is a manual flag observers set at exposure time (e.g. the wrong
    telescope settings): such a frame can pass every automated QC yet be
    scientifically useless, so it must be excluded from masters and science. The
    list is WMKO's ``{data_input}/reference/Junk_Observations_for_KPF.csv`` -- a
    title line, then an ``observation_id`` header, then one obs_id per row (read
    the same way as v2.12's ``kpf_processing_progress.py``). An absent file
    yields the empty set, so exclusion becomes a no-op -- matching WMKO's rule
    that a missing list means nothing is junk.
    """
    junk_csv = os.path.join(data_input, "reference", "Junk_Observations_for_KPF.csv")
    if not os.path.isfile(junk_csv):
        return set()
    df = pd.read_csv(junk_csv, header=1)
    return set(df.iloc[:, 0].astype(str).str.strip())


def build_mini_database(data_dir, write=False):
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
    write : bool, default False
        Whether to write the mini database CSV to `data_dir`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns FILENAME (absolute path to the FITS file),
        TARGNAME (target name), IMTYPE (image type), OBJECT (object
        identifier, e.g. 'autocal-bias'), EXPTIME (requested exposure time
        [s]), ELAPSED (actual elapsed time [s]), UTC (observation time in
        universal time, taken from the KPF timestamp in the filename), and
        HST (that same time converted to Hawaii Standard Time, UTC-10), and
        ISJUNK (bool: the frame is on the observer junk list, see
        load_junk_obs_ids). Both UTC and HST are KPF-format timestamp strings
        ('YYYYMMDD.SSSSS.FF'). Rows where a header key is missing are included
        with NaN for that column and a warning is issued.

        Junk rows are flagged (ISJUNK), not dropped, so callers keep them
        visible; build_l0_file_lists(exclude_junk=True) does the actual
        exclusion for master construction.
    """
    data_dir = os.path.normpath(data_dir)
    datecode = os.path.basename(data_dir)
    level = os.path.basename(os.path.dirname(data_dir))

    file_list = sorted(glob.glob(os.path.join(data_dir, "*.fits")))

    if not file_list:
        raise ValueError(f"No FITS files found in {data_dir}")

    # data_dir is {root}/{level}/{datecode}; the junk list lives under {root}.
    junk = load_junk_obs_ids(os.path.dirname(os.path.dirname(data_dir)))

    logger.info("scanning %d FITS headers in %s", len(file_list), data_dir)
    mini_db = {k: [] for k in _MINI_DB_KEYS + _MINI_DB_DERIVED_KEYS}

    for fn in file_list:
        logger.debug("reading header of %s", fn)
        try:
            header = fits.getheader(fn, ext=0)
        except Exception as e:
            warnings.warn(f"Could not read header from {fn}: {e}", stacklevel=2)
            continue

        mini_db["FILENAME"].append(fn)

        for k in _MINI_DB_KEYS[1:]:
            mini_db[k].append(header.get(k, None))

        # UTC is the KPF timestamp; HST is that converted to Hawaii time.
        utc = get_timestamp(fn)
        mini_db["UTC"].append(utc)
        mini_db["HST"].append(utc_to_hst(utc))
        mini_db["ISJUNK"].append(get_obs_id(fn) in junk)

    df = pd.DataFrame(mini_db)

    if write:
        csv_path = os.path.join(data_dir, f"KP.{datecode}_{level}.csv")
        df.to_csv(csv_path, index=False)
        logger.info("wrote mini database to %s", csv_path)
    return df


def build_l0_file_lists(
    cal_type,
    *,
    data_dir=None,
    mini_db=None,
    min_file_count=5,
    cluster_gap_seconds=7200,
    merge_small_clusters=False,
    enforce_hst_midnight_boundary=True,
    exclude_junk=True,
):
    """
    Return sorted file lists for all calibration clusters of the requested
    type.

    Exactly one of `data_dir` or `mini_db` must be provided. When `data_dir`
    is given, loads the mini database CSV if it exists, otherwise calls
    `build_mini_database` to scan headers and write it. When `mini_db` is
    given, uses it directly to avoid redundant I/O. Drops observer-flagged junk
    frames (unless `exclude_junk=False`), filters by OBJECT, then
    groups frames into clusters by detecting gaps larger than
    `cluster_gap_seconds` between consecutive timestamps. By default a cluster
    never spans two HST (Hawaii) calendar days: frames on either side of HST
    midnight are always split, even though the UTC-keyed data directory places
    them together. Set `enforce_hst_midnight_boundary=False` to lift that split
    (used for darks, whose sparse 1200 s sequences routinely straddle the HST
    midnight of a single observing night). Every returned cluster has at least
    `min_file_count` files: undersized clusters are dropped, or (with
    `merge_small_clusters`) folded into a neighbor. Raises only when no cluster
    meets the threshold.

    Parameters
    ----------
    cal_type : str
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
    merge_small_clusters : bool, default False
        How to handle clusters smaller than `min_file_count`. If False, drop
        them. If True, iteratively merge each into its nearest-in-time neighbor
        until every cluster meets the threshold; when the HST-midnight boundary
        is enforced the neighbor must be on the same HST day (a cluster with no
        such neighbor is dropped), otherwise any chronological neighbor is
        eligible.
    enforce_hst_midnight_boundary : bool, default True
        If True, clusters never span HST midnight: frames on opposite sides of
        it are always split, and merging is restricted to same-HST-day
        neighbors. If False, HST midnight is ignored entirely (only
        `cluster_gap_seconds` splits clusters, and any neighbor may be merged) --
        set this for darks, whose sparse sequences legitimately span the HST
        midnight of one observing night.
    exclude_junk : bool, default True
        Drop observer-flagged junk frames (the mini database's ISJUNK column)
        before clustering, so they never enter a master stack. Rarely disabled.
        Requires the ISJUNK column: a mini database built before ISJUNK existed
        (a stale cached CSV) raises KeyError -- rebuild it.

    Returns
    -------
    list of list of str
        Sorted file lists, one per cluster.

    Raises
    ------
    ValueError
        If `cal_type` is not a recognized calibration type, if exactly one of
        `data_dir` or `mini_db` is not provided, if no calibration frames of
        the requested type are found, or if no cluster meets `min_file_count`.
    """
    if cal_type not in _OBJECT_MAP:
        raise ValueError(
            f"cal_type must be one of {list(_OBJECT_MAP.keys())}; got '{cal_type}'"
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

    if exclude_junk:
        mini_db = mini_db[~mini_db["ISJUNK"].astype(bool)]

    cal_df = mini_db[mini_db["OBJECT"].isin(_OBJECT_MAP[cal_type])]

    if cal_df.empty:
        raise ValueError(
            f"No '{cal_type}' calibration frames found in "
            f"{data_dir or 'the provided mini_db'}"
        )

    # HST calendar day per file, from the mini_db HST timestamp ('YYYYMMDD...').
    hst_day = {
        fn: str(hst).split(".")[0]
        for fn, hst in zip(cal_df["FILENAME"], cal_df["HST"], strict=True)
    }

    # Cluster per-OBJECT (morning vs. evening thar etc. have different OBJECT
    # suffixes), splitting wherever consecutive frames are more than
    # cluster_gap_seconds apart or fall on different HST days (the UTC-keyed
    # directory can straddle HST midnight). Final list sorted chronologically.
    clusters = []
    for _, group in cal_df.groupby("OBJECT", dropna=False):
        timed = sorted((get_seconds_since_j2000(fn), fn) for fn in group["FILENAME"])
        if not timed:
            continue
        cluster = [timed[0][1]]
        for (prev_t, prev_fn), (t, fn) in zip(timed, timed[1:], strict=False):
            crosses_midnight = (
                enforce_hst_midnight_boundary and hst_day[fn] != hst_day[prev_fn]
            )
            if t - prev_t > cluster_gap_seconds or crosses_midnight:
                clusters.append(cluster)
                cluster = [fn]
            else:
                cluster.append(fn)
        clusters.append(cluster)
    clusters.sort(key=lambda c: get_seconds_since_j2000(c[0]))

    if merge_small_clusters:
        # Fold each undersized cluster (smallest first) into its nearer
        # chronological neighbor. When the HST-midnight boundary is enforced the
        # neighbor must share the HST day (so a master never spans HST midnight),
        # and a cluster with no same-day neighbor is dropped; otherwise any
        # adjacent cluster is eligible.
        while len(clusters) > 1 and any(len(c) < min_file_count for c in clusters):
            i = min(
                (k for k, c in enumerate(clusters) if len(c) < min_file_count),
                key=lambda k: len(clusters[k]),
            )
            day_i = hst_day[clusters[i][0]]
            # A neighbor is eligible when the midnight boundary is off, or when
            # it shares cluster i's HST day.
            prev_ok = i > 0 and (
                not enforce_hst_midnight_boundary
                or hst_day[clusters[i - 1][0]] == day_i
            )
            next_ok = i < len(clusters) - 1 and (
                not enforce_hst_midnight_boundary
                or hst_day[clusters[i + 1][0]] == day_i
            )
            prev_gap = (
                get_seconds_since_j2000(clusters[i][0])
                - get_seconds_since_j2000(clusters[i - 1][-1])
                if prev_ok
                else float("inf")
            )
            next_gap = (
                get_seconds_since_j2000(clusters[i + 1][0])
                - get_seconds_since_j2000(clusters[i][-1])
                if next_ok
                else float("inf")
            )
            if prev_gap == float("inf") and next_gap == float("inf"):
                clusters.pop(i)
                continue
            j = i - 1 if prev_gap <= next_gap else i + 1
            merged = sorted(clusters[i] + clusters[j], key=get_seconds_since_j2000)
            for idx in sorted((i, j), reverse=True):
                clusters.pop(idx)
            clusters.append(merged)
        clusters.sort(key=lambda c: get_seconds_since_j2000(c[0]))
    else:
        # Drop undersized clusters; one usable cluster is enough.
        clusters = [c for c in clusters if len(c) >= min_file_count]

    if not any(len(c) >= min_file_count for c in clusters):
        raise ValueError(
            f"'{cal_type}' has no cluster with at least "
            f"min_file_count={min_file_count} files"
        )
    # Which frames feed each master is a decision point (DRP-RUN-08).
    logger.info(
        "'%s' frames form %d cluster(s); sizes: %s",
        cal_type,
        len(clusters),
        [len(c) for c in clusters],
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

    This is the pipeline's authoritative path builder: it constructs the output
    path (directory layout + filename) from an obs_id string, before/without a
    populated data object, and is what every recipe uses to decide where to
    write. The parallel `<model>.generate_standard_filename()` builds only the
    basename from a populated object's headers and is the `to_fits(fn=None)`
    fallback (rvdata-owned for the EPRV-standard levels L2/L4, KPF-overridden for
    the non-standard levels L0/L1). The two encode the same naming rule and must
    agree per level; `TestFilenameConsistency` enforces that contract.

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
    #   L0:    {obs_id}.fits                            (KPF-native)
    #   L1:    kpf_L1_{YYYYMMDD}T{HHmmss}.fits          (no EPRV "S": no L1 standard)
    #   L2/L4: kpf_SL{N}_{YYYYMMDD}T{HHmmss}.fits       (EPRV standard)
    if level not in ("L0", "L1", "L2", "L4"):
        raise ValueError(f"'level' must be 'L0', 'L1', 'L2', or 'L4'; got '{level}'")

    if level == "L0":
        filename = f"{obs_id}.fits"
    else:
        eprv_ts = kpf_timestamp_to_eprv_timestamp(get_timestamp(obs_id))
        # L1 has no EPRV standard, so it keeps the KPF "kpf_L1" prefix (no "S");
        # L2/L4 use the EPRV "kpf_SL{N}" prefix.
        prefix = "kpf_L1" if level == "L1" else f"kpf_SL{level[1]}"
        filename = f"{prefix}_{eprv_ts}.fits"

    if data_root is None:
        return filename
    return os.path.join(data_root, level, datecode, filename)


def glob_masters(data_root, cal_type, level, datecode):
    """
    Glob pattern matching every ``cal_type``/``level`` master written under
    ``data_root`` for ``datecode`` (the KOAID prefix wildcarded):
    ``{data_root}/masters/{datecode}/*_master_{cal_type}_{level}.fits``.

    The reader counterpart to a `build_filepath` master path — it builds the same
    masters directory and filename independently, with the KOAID wildcarded.
    `test_glob_masters_matches_build_filepath` guards that the two stay in step.
    """
    return os.path.join(
        data_root, "masters", datecode, f"*_master_{cal_type}_{level}.fits"
    )
