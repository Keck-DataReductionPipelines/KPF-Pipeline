"""KPF data-file discovery: the FileHandler class plus product-path builders."""

import glob
import logging
import os
import tempfile
import warnings
from datetime import datetime

import pandas as pd
from astropy.io import fits

from kpfpipe.utils.kpf_utils import (
    get_datecode,
    get_obs_id,
    get_timestamp,
    is_datecode,
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

# How build_calibration_stacks groups a cal_type's frames into stacks:
# 'time_of_day' (per observing session, gap-split), 'hst_day' (per HST calendar
# day), or 'obs_night' (the whole loaded night, one stack spanning HST midnight).
_GROUPBY_MODES = ("time_of_day", "hst_day", "obs_night")


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


def datecode_dirs_in_range(root, start, end):
    """Sorted datecode subdirs of `root` within the inclusive [start, end] range.

    Used by the masters orchestrator to expand a ``--date_range`` against the L0
    tree. Non-datecode names and plain files are skipped; datecodes sort
    lexicographically, which coincides with chronological order for ``YYYYMMDD``.
    """
    return [
        d
        for d in sorted(os.listdir(root))
        if is_datecode(d) and start <= d <= end and os.path.isdir(os.path.join(root, d))
    ]


def read_token_file(path):
    """Read whitespace-stripped, non-blank lines from a text file.

    Backs the ``--dates`` (masters) and ``--obs_ids`` (science) flags, which
    each accept a reference file listing one unit -- a datecode or an obs_id --
    per line instead of inline values. Whitespace is stripped from each line and
    blank lines are dropped; validating that each token is a real datecode/obs_id
    is left to the caller.
    """
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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
    data_dirs : dict
        The already-extracted ``[DATA_DIRS]`` mapping, holding the roots
        ``KPF_DATA_INPUT`` (the L0 input tree) and ``KPF_MASTERS_OUTPUT`` (the
        masters output tree). Required: this is a util, not a pipeline module, so
        it has no config defaults to fall back on. Callers with a
        ``ConfigHandler`` pass ``config.get_params(["DATA_DIRS"])`` (this class
        deliberately does not import ``ConfigHandler``, keeping construction
        light). Either root may be absent -- fine for an instance that only calls
        methods needing the other root; a method whose root is unset raises
        ``ValueError``. Pass ``{}`` for an instance that touches neither root.
    """

    def __init__(self, data_dirs):
        self._data_input = data_dirs.get("KPF_DATA_INPUT")
        self._masters_output = data_dirs.get("KPF_MASTERS_OUTPUT")
        self._mini_db = None  # the loaded night, set by build_mini_database

    def _mini_db_cache_path(self, datecode):
        """On-disk mini-database cache path for one night:
        ``{KPF_DATA_INPUT}/vNext/mini_db/{datecode}_L0.csv``."""
        return os.path.join(self._data_input, "vNext", "mini_db", f"{datecode}_L0.csv")

    def _read_mini_db_cache(self, datecode):
        """The cached mini database for `datecode` as a DataFrame if the on-disk
        cache is trustworthy for the current L0 directory, else None (the caller
        then rescans). Reads the cache CSV at most once: the loaded frame is both
        validated and returned. The cache path, L0 directory, and its FITS file
        list are all derived from `datecode` plus the instance's ``KPF_DATA_INPUT``
        root.

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
        cache_path = self._mini_db_cache_path(datecode)
        data_dir = os.path.join(self._data_input, "L0", datecode)
        file_list = sorted(glob.glob(os.path.join(data_dir, "*.fits")))
        if not file_list or not os.path.isfile(cache_path):
            return None

        cached = pd.read_csv(cache_path)
        if len(cached) != len(file_list):
            logger.warning(
                "mini database cache %s is stale (%d cached rows vs %d files on "
                "disk); rescanning",
                cache_path,
                len(cached),
                len(file_list),
            )
            return None

        cache_mtime = os.stat(cache_path).st_mtime
        newest_input = os.stat(data_dir).st_mtime
        for fn in file_list:
            st = os.stat(fn)
            newest_input = max(newest_input, st.st_mtime, st.st_ctime)
        if cache_mtime < newest_input:
            logger.warning(
                "mini database cache %s is out of date (older than an L0 input); "
                "rescanning",
                cache_path,
            )
            return None

        return cached

    def _write_mini_db_cache(self, datecode):
        """Atomically write the carried mini database (``self._mini_db``) to the
        on-disk cache CSV for `datecode`, creating the cache directory as needed.

        pandas' ``to_csv`` truncates the target in place, so a concurrent reader
        could observe an empty, partial, or half-written CSV. Fill a same-dir temp
        file and ``os.replace`` it into position instead (atomic on POSIX), so
        readers always see the old or new CSV whole; clean up the temp file if the
        write fails.
        """
        cache_path = self._mini_db_cache_path(datecode)
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
        os.close(fd)
        try:
            self._mini_db.to_csv(tmp, index=False)
            os.replace(tmp, cache_path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        logger.info("wrote mini database cache to %s", cache_path)

    def build_mini_database(self, datecode, cache=False):
        """
        Scan the PRIMARY header of every L0 FITS file for one observing night,
        cache the resulting DataFrame on the instance, and return it.

        Reads ``{KPF_DATA_INPUT}/L0/{datecode}/*.fits`` and stores the result as
        ``self._mini_db``, so a subsequent ``build_calibration_stacks`` needs
        only the calibration type; callers rarely need the return value directly.

        The on-disk CSV cache lives at
        ``{KPF_DATA_INPUT}/vNext/mini_db/{datecode}_L0.csv``. `cache` selects which
        side of it to use, read and write being independent:

        * read (``"r"``) -- load a *current* cache instead of re-scanning the L0
          directory. A cache is current only when it passes the count and
          freshness guardrails in ``_read_mini_db_cache``; a frame added, removed,
          replaced, or rewritten since the cache was built forces a rescan.
        * write (``"w"``) -- after scanning, (re)write the cache (directory created
          as needed) for next time. A cache hit on read short-circuits the scan, so
          nothing is rewritten.

        Parameters
        ----------
        datecode : str
            Observing-night datecode 'YYYYMMDD'.
        cache : {False, "r", "w", "rw", "wr"}, default False
            Which side(s) of the on-disk cache to use. ``False`` (default) always
            scans fresh and never touches the cache. ``"r"`` reads a current cache
            (else scans); ``"w"`` writes the scan result; ``"rw"``/``"wr"`` do both
            (the read-then-write behavior a plain cache once had).

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
            If `cache` is not one of the allowed modes, if KPF_DATA_INPUT is
            unset, or if the L0 directory holds no FITS files.
        """
        if cache not in (False, "r", "w", "rw", "wr"):
            raise ValueError(
                f"cache must be False or one of 'r'/'w'/'rw'/'wr', got {cache!r}"
            )
        read_cache = cache and "r" in cache
        write_cache = cache and "w" in cache

        if self._data_input is None:
            raise ValueError("FileHandler has no KPF_DATA_INPUT configured")

        if read_cache:
            cached = self._read_mini_db_cache(datecode)
            if cached is not None:
                logger.info(
                    "loaded mini database cache from %s",
                    self._mini_db_cache_path(datecode),
                )
                self._mini_db = cached
                return self._mini_db

        data_dir = os.path.join(self._data_input, "L0", datecode)
        file_list = sorted(glob.glob(os.path.join(data_dir, "*.fits")))
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

        if write_cache:
            self._write_mini_db_cache(datecode)

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

    def _select_frames(self, cal_type, *, exclude_junk=True):
        """
        The junk-excluded, OBJECT-filtered frames of `cal_type` from the carried
        mini database (``self._mini_db``).

        The single frame-selection entry point the grouping paths share. Raises
        ``ValueError`` if no mini database is loaded (call ``build_mini_database``
        first) or if the type has no frames; a mini database lacking the ISJUNK
        column raises ``KeyError`` under ``exclude_junk=True`` (a fail-loud guard
        against a database built before junk tracking existed).
        """
        mini_db = self._mini_db
        if mini_db is None:
            raise ValueError(
                "no mini database available; call build_mini_database(datecode) first"
            )
        if exclude_junk:
            mini_db = mini_db[~mini_db["ISJUNK"].astype(bool)]
        cal_df = mini_db[mini_db["OBJECT"].isin(_OBJECT_MAP[cal_type])]
        if cal_df.empty:
            raise ValueError(f"No '{cal_type}' calibration frames found")
        return cal_df

    def _identify_clusters(self, cal_type, *, cluster_gap_seconds, exclude_junk=True):
        """
        Group `cal_type` frames into observing-session clusters by time gaps.

        Reads the carried mini database (``self._mini_db``) via `_select_frames`
        and infers the morn/midday/eve/night/midnight sessions purely from the
        frame timestamps: the OBJECT morn/eve/night suffix is only a label and
        does not partition the result, so a mislabeled frame clusters with its
        true session instead of splitting off. Consecutive time-sorted frames
        start a new cluster wherever they are more than `cluster_gap_seconds`
        apart; for the allowlisted cal types the morn and eve sessions sit hours
        apart, so the gap alone separates them.

        Parameters
        ----------
        cal_type : str
            Calibration frame type (a key of `_OBJECT_MAP`).
        cluster_gap_seconds : int
            Gap [s] between consecutive frames that starts a new cluster.
        exclude_junk : bool, default True
            Drop observer-flagged junk frames before clustering.

        Returns
        -------
        clusters : list of list of str
            Chronologically-sorted filename lists, one per detected session.
        """
        cal_df = self._select_frames(cal_type, exclude_junk=exclude_junk)
        timed = sorted((self._seconds_since_j2000(fn), fn) for fn in cal_df["FILENAME"])
        clusters = []
        cluster = [timed[0][1]]
        prev_t = timed[0][0]
        for t, fn in timed[1:]:
            if t - prev_t > cluster_gap_seconds:
                clusters.append(cluster)
                cluster = [fn]
            else:
                cluster.append(fn)
            prev_t = t
        clusters.append(cluster)
        clusters.sort(key=lambda c: self._seconds_since_j2000(c[0]))
        return clusters

    def build_calibration_stacks(
        self,
        cal_type,
        *,
        min_file_count=5,
        cluster_gap_seconds=7200,
        groupby="time_of_day",
        exclude_junk=True,
    ):
        """
        Return sorted file lists for all calibration stacks of the requested
        type, grouped from the mini database carried on the instance
        (``self._mini_db``, set by `build_mini_database`).

        Drops observer-flagged junk frames (unless `exclude_junk=False`), filters
        by OBJECT, then groups the surviving frames into stacks according to
        `groupby`:

        * ``'time_of_day'`` (default) -- one stack per observing session, split
          wherever consecutive frames are more than `cluster_gap_seconds` apart
          (the morn/eve/night sessions). The split is purely temporal, so the
          morn/eve/night OBJECT suffix does not partition it and a mislabeled
          frame stacks with its true session. Used for bias and thar.
        * ``'hst_day'`` -- one stack per HST (Hawaii) calendar day.
        * ``'obs_night'`` -- one stack for the whole loaded night (the UTC-keyed
          datecode), spanning HST midnight. Used for darks, whose sparse sequences
          routinely straddle HST midnight and belong in a single nightly stack.

        Every returned stack has at least `min_file_count` files; undersized
        stacks are dropped, and it raises when none meets the threshold. Reads the
        mini database off the instance, so the recipe never handles the DataFrame
        itself.

        Parameters
        ----------
        cal_type : str
            Calibration frame type. One of 'bias', 'dark', 'flat', 'thar'.
        min_file_count : int, default 5
            Minimum number of files required per stack.
        cluster_gap_seconds : int, default 7200
            Gap [s] between consecutive frames that splits one session from the
            next. The 2-hour default separates KPF morning vs. evening sessions.
            Applies only to ``groupby='time_of_day'``; ignored otherwise.
        groupby : {'time_of_day', 'hst_day', 'obs_night'}, default 'time_of_day'
            How to group frames into stacks (see summary).
        exclude_junk : bool, default True
            Drop junk frames (the ISJUNK column) before grouping. Rarely disabled.

        Returns
        -------
        list of list of str
            Sorted file lists, one per stack.

        Raises
        ------
        ValueError
            If `groupby` or `cal_type` is not recognized, if no mini database is
            loaded on the instance, if no calibration frames of the requested type
            are found, or if no stack meets `min_file_count`.
        """
        if groupby not in _GROUPBY_MODES:
            raise ValueError(
                f"groupby must be one of {list(_GROUPBY_MODES)}; got '{groupby}'"
            )
        if cal_type not in _OBJECT_MAP:
            raise ValueError(
                f"cal_type must be one of {list(_OBJECT_MAP.keys())}; got '{cal_type}'"
            )

        if groupby == "time_of_day":
            clusters = self._identify_clusters(
                cal_type,
                cluster_gap_seconds=cluster_gap_seconds,
                exclude_junk=exclude_junk,
            )
        else:
            cal_df = self._select_frames(cal_type, exclude_junk=exclude_junk)
            if groupby == "hst_day":
                # One stack per HST calendar day (the 'YYYYMMDD' HST-timestamp prefix).
                by_day = {}
                for fn, hst in zip(cal_df["FILENAME"], cal_df["HST"], strict=True):
                    by_day.setdefault(str(hst).split(".")[0], []).append(fn)
                clusters = [
                    sorted(v, key=self._seconds_since_j2000)
                    for _, v in sorted(by_day.items())
                ]
            else:  # obs_night -- the whole loaded night, one stack spanning midnight
                clusters = [sorted(cal_df["FILENAME"], key=self._seconds_since_j2000)]

        clusters = [c for c in clusters if len(c) >= min_file_count]

        if not clusters:
            raise ValueError(
                f"'{cal_type}' groupby={groupby} produced no cluster with at least "
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
