"""KPF data-file discovery: the FileHandler class plus product-path builders."""

import glob
import logging
import math
import os
import warnings
from datetime import datetime

import pandas as pd
from astropy.io import fits

from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.kpf_utils import (
    get_datecode,
    get_obs_id,
    get_timestamp,
    is_obs_id,
    kpf_timestamp_to_datetime,
    kpf_timestamp_to_eprv_timestamp,
    utc_to_hst,
)

logger = logging.getLogger(__name__)

# J2000.0 epoch (2000-01-01 12:00 UTC); reference for the monotonic sort/gap
# scalar FileHandler._seconds_since_j2000 computes when clustering frames.
_J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)

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

# The calibration/master frame types, in canonical order. Derived from
# _OBJECT_MAP so it stays the single source: both L0 frame selection
# (build_calibration_stacks) and master-product filenames (kpf_filename) validate
# against the same set and cannot drift.
_CAL_TYPES = tuple(_OBJECT_MAP)


def load_junk_obs_ids(data_input):
    """Set of observer-flagged junk obs_ids, to be stored with L0 data at
    ``{data_input}/vNext/reference/junk_obs.csv`` (a title line, an
    ``observation_id`` header, then one obs_id per row). An absent file yields
    the empty set, so exclusion becomes a no-op.
    """
    junk_csv = os.path.join(data_input, "vNext", "reference", "junk_obs.csv")
    if not os.path.isfile(junk_csv):
        return set()
    df = pd.read_csv(junk_csv, header=1)
    return set(df.iloc[:, 0].astype(str).str.strip())


class FileHandler:
    """Discover KPF data files across the L0-input and masters-output trees.

    Resolves the pipeline's data-tree roots once and exposes file discovery as
    methods keyed by ``datecode``/``cal_type``, so recipes and scripts never
    assemble data paths by hand.

    Not thread-safe: ``build_mini_database`` stores the scanned night on
    ``self._mini_db`` and returns it, so concurrent calls on one instance race.
    Use one instance per thread (the on-disk cache is shared safely, keyed by
    datecode).

    Parameters
    ----------
    config : None | dict | ConfigHandler
        Source of the ``[DATA_DIRS]`` roots ``KPF_DATA_INPUT`` (the L0 input
        tree) and ``KPF_MASTERS_OUTPUT`` (the masters output tree). ``None``
        leaves both unset -- fine for an instance that only calls methods needing
        the other root; a method whose root is unset raises ``ValueError``.
    """

    def __init__(self, config=None):
        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        self._data_input = params.get("KPF_DATA_INPUT")
        self._masters_output = params.get("KPF_MASTERS_OUTPUT")
        self._mini_db = None  # the loaded night, set by build_mini_database

    def _mini_db_cache_path(self, datecode):
        """On-disk mini-database cache path for one night:
        ``{KPF_DATA_INPUT}/vNext/mini_db/{datecode}_L0.csv``."""
        return os.path.join(self._data_input, "vNext", "mini_db", f"{datecode}_L0.csv")

    def _validate_mini_db_cache(self, cache_path, file_list, data_dir):
        """True if the on-disk mini-db cache is trustworthy for the current L0
        directory, else False (the caller then rescans). Validation only -- the
        cache is read (loaded) separately by the caller.

        Two guardrails protect against a stale cache:

        * **Count** -- the cache's row count must equal the number of FITS files
          on disk, so a frame added or removed since the cache was written
          invalidates it. (An unreadable frame is dropped from the scan but still
          counts on disk, so a persistently-corrupt frame defeats the cache -- an
          acceptable, rare edge that errs toward rescanning.)
        * **Freshness** -- the cache must be at least as new as every input's
          timestamp: for each file the later of its modification time
          (``st_mtime``, when its contents were written) and its change time
          (``st_ctime``, when it was last moved/linked into the directory -- so a
          frame copied in with an old mtime is still caught); plus the directory's
          own ``st_mtime`` (its entry list changes when a file is added or
          removed). A same-count swap thus still invalidates the cache.
        """
        if not file_list or not os.path.isfile(cache_path):
            return False

        n_cached = len(pd.read_csv(cache_path))
        if n_cached != len(file_list):
            logger.info(
                "mini database cache %s is stale (%d cached rows vs %d files on "
                "disk); rescanning",
                cache_path,
                n_cached,
                len(file_list),
            )
            return False

        cache_mtime = os.stat(cache_path).st_mtime
        newest_input = os.stat(data_dir).st_mtime
        for fn in file_list:
            st = os.stat(fn)
            newest_input = max(newest_input, st.st_mtime, st.st_ctime)
        if cache_mtime < newest_input:
            logger.info(
                "mini database cache %s is out of date (older than an L0 input); "
                "rescanning",
                cache_path,
            )
            return False

        return True

    def build_mini_database(self, datecode, cache=False):
        """
        Scan the PRIMARY header of every L0 FITS file for one observing night,
        cache the resulting DataFrame on the instance, and return it.

        Reads ``{KPF_DATA_INPUT}/L0/{datecode}/*.fits`` and stores the result as
        ``self._mini_db``, so a subsequent ``build_calibration_stacks`` needs
        only the calibration type; callers rarely need the return value directly.

        With ``cache=True`` the on-disk CSV cache at
        ``{KPF_DATA_INPUT}/vNext/mini_db/{datecode}_L0.csv`` is used for both
        read and write: a *current* cache is loaded instead of re-scanning the L0
        directory, otherwise the directory is scanned and the result written to
        the cache (directory created as needed) for next time. A cache is current
        only when it passes the count and freshness guardrails in
        ``_validate_mini_db_cache``; a frame added, removed, replaced, or
        rewritten since the cache was built forces a rescan. With ``cache=False``
        (default) the directory is always scanned and nothing is read from or
        written to disk.

        Parameters
        ----------
        datecode : str
            Observing-night datecode 'YYYYMMDD'.
        cache : bool, default False
            When True, load the on-disk cache CSV if it is current, else scan and
            (re)write it. When False, always scan fresh and never touch the cache.

        Returns
        -------
        pandas.DataFrame
            One row per readable frame. Columns: the header keys FILENAME
            (absolute path), TARGNAME, IMTYPE, OBJECT, EXPTIME, ELAPSED; plus
            derived UTC and HST (KPF-format timestamps from the filename, HST =
            UTC-10) and ISJUNK (frame is on the observer junk list -- flagged,
            not dropped). A missing header key gives None for that column; an
            unreadable frame is skipped; both warn.

        Raises
        ------
        ValueError
            If KPF_DATA_INPUT is unset, or if the L0 directory holds no FITS files.
        """
        if self._data_input is None:
            raise ValueError("FileHandler has no KPF_DATA_INPUT configured")

        data_dir = os.path.join(self._data_input, "L0", datecode)
        file_list = sorted(glob.glob(os.path.join(data_dir, "*.fits")))

        cache_path = self._mini_db_cache_path(datecode)
        if cache and self._validate_mini_db_cache(cache_path, file_list, data_dir):
            logger.info("loading mini database cache from %s", cache_path)
            self._mini_db = pd.read_csv(cache_path)
            return self._mini_db

        if not file_list:
            raise ValueError(f"No FITS files found in {data_dir}")

        junk = load_junk_obs_ids(self._data_input)

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

        self._mini_db = pd.DataFrame(mini_db)

        if cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            self._mini_db.to_csv(cache_path, index=False)
            logger.info("wrote mini database cache to %s", cache_path)

        return self._mini_db

    def _seconds_since_j2000(self, s):
        """Seconds since J2000.0 (2000-01-01 12:00 UTC) for the KPF timestamp in
        `s` (a timestamp, obs_id, filename, or path) -- the monotonic scalar this
        handler sorts and gap-detects on when clustering frames; raises
        ``ValueError`` if `s` holds no valid KPF timestamp.

        Naive UTC arithmetic (leap seconds ignored): fine for frame ordering and
        cluster-gap detection, not for astronomical (TT/TAI) timing.
        """
        dt = kpf_timestamp_to_datetime(get_timestamp(s))
        return int((dt - _J2000_EPOCH).total_seconds())

    def _merge_undersized_clusters(
        self, clusters, hst_day, min_file_count, enforce_hst_midnight_boundary
    ):
        """
        Fold each undersized cluster into its nearer chronological neighbor.

        Repeatedly takes the smallest cluster below `min_file_count` and merges
        it into whichever adjacent cluster is nearer in time, until every
        remaining cluster meets the threshold (or none can grow). When the
        HST-midnight boundary is enforced the neighbor must share the cluster's
        HST day (so a master never spans HST midnight), and a cluster with no
        same-day neighbor is dropped; otherwise any adjacent cluster is eligible.

        `clusters` is a chronologically-sorted list of filename lists and
        `hst_day` maps each filename to its HST calendar day ('YYYYMMDD'); the
        input is not modified and the merged clusters are returned
        chronologically sorted.
        """
        clusters = list(clusters)

        def gap_to(i, j):
            # Chronological gap [s] from cluster i to neighbor j (i - 1 or i + 1);
            # inf when j is out of range or -- boundary enforced -- on a different
            # HST day, marking it ineligible to merge into.
            if not 0 <= j < len(clusters) or (
                enforce_hst_midnight_boundary
                and hst_day[clusters[j][0]] != hst_day[clusters[i][0]]
            ):
                return math.inf
            earlier, later = sorted((i, j))
            return self._seconds_since_j2000(
                clusters[later][0]
            ) - self._seconds_since_j2000(clusters[earlier][-1])

        while len(clusters) > 1 and any(len(c) < min_file_count for c in clusters):
            i = min(
                (k for k, c in enumerate(clusters) if len(c) < min_file_count),
                key=lambda k: len(clusters[k]),
            )
            prev_gap = gap_to(i, i - 1)
            next_gap = gap_to(i, i + 1)
            if prev_gap == next_gap == math.inf:
                clusters.pop(i)
                continue
            j = i - 1 if prev_gap <= next_gap else i + 1
            merged = sorted(clusters[i] + clusters[j], key=self._seconds_since_j2000)
            for idx in sorted((i, j), reverse=True):
                clusters.pop(idx)
            clusters.append(merged)
        clusters.sort(key=lambda c: self._seconds_since_j2000(c[0]))
        return clusters

    def build_calibration_stacks(
        self,
        cal_type,
        *,
        mini_db=None,
        min_file_count=5,
        cluster_gap_seconds=7200,
        merge_small_clusters=False,
        enforce_hst_midnight_boundary=True,
        exclude_junk=True,
    ):
        """
        Return sorted file lists for all calibration clusters of the requested
        type, grouped from the loaded mini database.

        Drops observer-flagged junk frames (unless `exclude_junk=False`), filters
        by OBJECT, then groups frames into clusters by detecting gaps larger than
        `cluster_gap_seconds` between consecutive timestamps. By default a cluster
        never spans two HST (Hawaii) calendar days: frames on either side of HST
        midnight are always split, even though the UTC-keyed data directory places
        them together. Set `enforce_hst_midnight_boundary=False` to lift that split
        (used for darks, whose sparse sequences routinely straddle HST midnight).
        Every returned cluster has at least `min_file_count` files: undersized
        clusters are dropped, or (with `merge_small_clusters`) folded into a
        neighbor. Raises only when no cluster meets the threshold. Clusters the
        mini database carried on the instance, so the recipe never handles the
        DataFrame itself.

        Parameters
        ----------
        cal_type : str
            Calibration frame type. One of 'bias', 'dark', 'flat', 'thar'.
        mini_db : pandas.DataFrame, optional
            DataFrame to cluster; defaults to ``self._mini_db``. Pass one only to
            cluster a database the handler did not build.
        min_file_count : int, default 5
            Minimum number of files required per cluster.
        cluster_gap_seconds : int, default 7200
            Gap [s] between consecutive frames that splits one cluster from the
            next. The 2-hour default separates KPF morning vs. evening clusters.
        merge_small_clusters : bool, default False
            When False, drop clusters below `min_file_count`; when True, merge
            each into its nearest-in-time (and, if the boundary is enforced,
            same-HST-day) neighbor.
        enforce_hst_midnight_boundary : bool, default True
            Whether clusters may span HST midnight (see summary). Set False for
            darks.
        exclude_junk : bool, default True
            Drop junk frames (the ISJUNK column) before clustering. Rarely
            disabled.

        Returns
        -------
        list of list of str
            Sorted file lists, one per cluster.

        Raises
        ------
        ValueError
            If `cal_type` is not a recognized calibration type, if no mini
            database is available (none loaded and none passed), if no calibration
            frames of the requested type are found, or if no cluster meets
            `min_file_count`.
        """
        if cal_type not in _OBJECT_MAP:
            raise ValueError(
                f"cal_type must be one of {list(_OBJECT_MAP.keys())}; got '{cal_type}'"
            )

        if mini_db is None:
            mini_db = self._mini_db
        if mini_db is None:
            raise ValueError(
                "no mini database available; call build_mini_database(datecode) "
                "first (or pass mini_db=)"
            )

        if exclude_junk:
            mini_db = mini_db[~mini_db["ISJUNK"].astype(bool)]

        cal_df = mini_db[mini_db["OBJECT"].isin(_OBJECT_MAP[cal_type])]

        if cal_df.empty:
            raise ValueError(f"No '{cal_type}' calibration frames found")

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
            timed = sorted(
                (self._seconds_since_j2000(fn), fn) for fn in group["FILENAME"]
            )
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
        clusters.sort(key=lambda c: self._seconds_since_j2000(c[0]))

        if merge_small_clusters:
            clusters = self._merge_undersized_clusters(
                clusters, hst_day, min_file_count, enforce_hst_midnight_boundary
            )
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

    def find_masters(self, cal_type, level, datecode):
        """
        Sorted list of the master files matching ``cal_type``/``level`` written
        under ``KPF_MASTERS_OUTPUT`` for ``datecode`` -- everything matching
        ``{root}/masters/{datecode}/*_master_{cal_type}_{level}.fits`` (the KOAID
        prefix wildcarded).

        The reader counterpart to a `kpf_filepath` master path;
        `TestFindMasters.test_finds_kpf_filepath_output` guards that the two stay
        in step.

        Raises
        ------
        ValueError
            If KPF_MASTERS_OUTPUT is unset.
        """
        if self._masters_output is None:
            raise ValueError("FileHandler has no KPF_MASTERS_OUTPUT configured")
        pattern = os.path.join(
            self._masters_output,
            "masters",
            datecode,
            f"*_master_{cal_type}_{level}.fits",
        )
        return sorted(glob.glob(pattern))


def kpf_filename(obs_id, level, *, master=None):
    """
    Base filename for a KPF data product (no directory). The single source of
    the naming rule; ``<model>.generate_standard_filename()`` and `kpf_filepath`
    both delegate here.

    Science basenames by level:
      L0:    {obs_id}.fits                       (KPF-native)
      L1:    kpf_L1_{YYYYMMDD}T{HHmmss}.fits      (no EPRV "S": no L1 standard)
      L2/L4: kpf_SL{N}_{YYYYMMDD}T{HHmmss}.fits   (EPRV standard)
    Master basename: {obs_id}_master_{master}_{level}.fits.

    Parameters
    ----------
    obs_id : str
        Observation ID (e.g. 'KP.20240405.49597.71'). For master products this
        is the obs_id of the first frame in the stack.
    level : str
        Data level 'L0'/'L1'/'L2'/'L4' (masters: 'L1'/'L2'/'L4').
    master : str or None, optional
        Master calibration type ('bias'/'dark'/'flat'/'thar') for a master
        product, or None (the default) for a science product.

    Returns
    -------
    str
        The base filename.

    Raises
    ------
    ValueError
        If `obs_id` is not a valid observation ID, `level` is unrecognized, or
        `master` is an unrecognized type or paired with an invalid level.
    """
    if not is_obs_id(obs_id):
        raise ValueError(
            "obs_id must be a valid observation ID "
            f"(e.g. 'KP.20240405.49597.71'); got '{obs_id}'"
        )

    if master is not None:
        if master not in _CAL_TYPES:
            raise ValueError(
                f"'master' must be one of {list(_CAL_TYPES)}; got '{master}'"
            )
        if level not in ("L1", "L2", "L4"):
            raise ValueError(
                "'level' for master products must be 'L1', 'L2', or 'L4'; "
                f"got '{level}'"
            )
        return f"{obs_id}_master_{master}_{level}.fits"

    if level not in ("L0", "L1", "L2", "L4"):
        raise ValueError(f"'level' must be 'L0', 'L1', 'L2', or 'L4'; got '{level}'")

    if level == "L0":
        return f"{obs_id}.fits"
    eprv_ts = kpf_timestamp_to_eprv_timestamp(get_timestamp(obs_id))
    # L1 has no EPRV standard, so it keeps the KPF "kpf_L1" prefix (no "S");
    # L2/L4 use the EPRV "kpf_SL{N}" prefix.
    prefix = "kpf_L1" if level == "L1" else f"kpf_SL{level[1]}"
    return f"{prefix}_{eprv_ts}.fits"


_DIRECTORY_KINDS = ("science", "masters", "QLP")


def kpf_directory(obs_id, *, level=None, data_root, kind):
    """
    Output directory for a KPF product tree.

    The single authority for the on-disk output layout; `kpf_filepath` and the
    recipes go through it rather than re-deriving ``os.path.join(data_root, ...)``
    by hand. The datecode is parsed from ``obs_id``. Pure path construction -- it
    does not touch the filesystem.

    ===========  =========================================================
    ``kind``     directory
    ===========  =========================================================
    ``science``  ``{data_root}/{level}/{datecode}`` -- L0/L1/L2/L4 products
    ``masters``  ``{data_root}/masters/{datecode}`` (``level`` unused)
    ``QLP``      ``{data_root}/QLP/{datecode}/{obs_id}/{level}`` -- quicklook
    ===========  =========================================================

    Parameters
    ----------
    obs_id : str
        Observation ID (e.g. 'KP.20240405.49597.71'); the datecode source and,
        for ``QLP``, a path component.
    level : str or None, optional
        Data level 'L0'/'L1'/'L2'/'L4'. Required for ``science`` and ``QLP``;
        unused for ``masters``.
    data_root : str
        Root data directory (e.g. '/data/kpf/'); a non-empty string.
    kind : str
        Which output tree: 'science', 'masters', or 'QLP'.

    Returns
    -------
    str
        The directory path (no trailing separator).

    Raises
    ------
    ValueError
        If `data_root` is not a non-empty string, `kind` is unrecognized,
        `obs_id` is invalid, or `level` is missing/invalid for a kind that needs
        it.
    """
    if not isinstance(data_root, str) or not data_root:
        raise ValueError(f"data_root must be a non-empty string; got {data_root!r}")
    if kind not in _DIRECTORY_KINDS:
        raise ValueError(f"kind must be one of {list(_DIRECTORY_KINDS)}; got {kind!r}")
    if not is_obs_id(obs_id):
        raise ValueError(
            "obs_id must be a valid observation ID "
            f"(e.g. 'KP.20240405.49597.71'); got '{obs_id}'"
        )

    datecode = get_datecode(obs_id)

    if kind == "masters":
        return os.path.join(data_root, "masters", datecode)

    # science and QLP both carry a data level.
    if level not in ("L0", "L1", "L2", "L4"):
        raise ValueError(
            f"'level' must be 'L0', 'L1', 'L2', or 'L4' for kind={kind!r}; "
            f"got {level!r}"
        )
    if kind == "science":
        return os.path.join(data_root, level, datecode)
    # kind == "QLP": the obs_id names a path component.
    return os.path.join(data_root, "QLP", datecode, obs_id, level)


def kpf_filepath(obs_id, level, *, data_root=None, master=None):
    """
    Build a filepath for a KPF data product: the product's directory
    (`kpf_directory`) joined with its basename (`kpf_filename`).

    The pipeline's authoritative path builder: constructs the output path from
    an obs_id string (before/without a populated data object), as every recipe
    does when deciding where to write. Its basename twin,
    ``<model>.generate_standard_filename()``, builds the same name from a
    populated object; `TestFilenameConsistency` enforces they agree per level.

    Parameters
    ----------
    obs_id : str
        Observation ID (e.g. 'KP.20240405.49597.71'). For master products
        this should be the obs_id of the first frame in the stack.
    level : str
        Data level string, one of 'L0', 'L1', 'L2', 'L4'.
    data_root : str or None, optional
        Root data directory (e.g. '/data/kpf/'). When None (the default),
        returns the bare filename (`kpf_filename`). Otherwise must be a
        non-empty string and a full path is returned.
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

    filename = kpf_filename(obs_id, level, master=master)
    if data_root is None:
        return filename

    kind = "masters" if master is not None else "science"
    return os.path.join(
        kpf_directory(obs_id, level=level, data_root=data_root, kind=kind),
        filename,
    )
