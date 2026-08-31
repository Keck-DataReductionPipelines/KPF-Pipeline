"""
KPF Calibration Association module.

Given a KPF observation frame, finds the most appropriate master calibration
file for each calibration type (bias, dark, flat, wls) by searching
the masters directory and selecting the nearest-in-time match.
"""

import logging
from datetime import datetime, timedelta

from kpfpipe import DEFAULTS
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import FileHandler
from kpfpipe.utils.kpf import get_timestamp, kpf_timestamp_to_datetime

logger = logging.getLogger(__name__)

_DEFAULTS = {
    **DEFAULTS,
    "masters_search_window_days": [-1, 0],
}

# Level suffix of the master FITS file for each supported calibration type,
# used to build the *_master_<cal_type>_<level>.fits glob.
_LEVEL_BY_CAL_TYPE = {
    "bias": "L1",
    "dark": "L1",
    "flat": "L1",
    "thar": "L2",
}

# PRIMARY header prefix written for each supported calibration type.
_HEADER_PREFIX = {
    "bias": "BIAS",
    "dark": "DARK",
    "flat": "FLAT",
    "thar": "WLS",
}


class CalibrationAssociation:
    """
    Associate a KPF observation frame with master calibration files.

    For each requested calibration type, scans the masters directory and
    selects the master whose observation timestamp is nearest to the
    frame's observation time. A configurable search window limits how far
    back (or forward) in time the search extends.

    Parameters
    ----------
    l1_obj : KPF1
        KPF observation frame. The observation timestamp is read from its
        PRIMARY header (DATE-OBS).
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: KPF_MASTERS_OUTPUT (root
        directory holding the master calibration files) and
        masters_search_window_days ([days_before, days_after] search window
        relative to the frame's observation date; default [-1, 0]).
    """

    def __init__(self, l1_obj, config=None):
        self.l1_obj = l1_obj

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(
                ["DATA_DIRS", "TRACES", "MODULE_CALIBRATION_ASSOCIATION"]
            )
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in _DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self._masters_output = params.get("KPF_MASTERS_OUTPUT")
        self._file_handler = FileHandler(params)  # masters discovery
        self._calibrations = None  # per-cal {filepath} for _set_headers
        self._info = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_master_files(self, cal_type, date_obs, masters_search_window_days=None):
        """
        Return sorted (filepath, kpf_timestamp) tuples for all masters of
        ``cal_type`` within the [days_before, days_after] window around
        ``date_obs``. Raises ValueError for an unsupported ``cal_type``.
        """
        if cal_type not in _LEVEL_BY_CAL_TYPE:
            raise ValueError(
                f"unsupported cal_type {cal_type!r}; "
                f"expected one of {sorted(_LEVEL_BY_CAL_TYPE)}"
            )
        if masters_search_window_days is None:
            masters_search_window_days = self.masters_search_window_days

        obs_date = datetime.fromisoformat(date_obs).date()
        days_before, days_after = masters_search_window_days
        level = _LEVEL_BY_CAL_TYPE[cal_type]

        master_files = []
        for delta in range(days_before, days_after + 1):
            search_date = obs_date + timedelta(days=delta)
            datecode = search_date.strftime("%Y%m%d")
            for filepath in self._file_handler.find_masters(cal_type, level, datecode):
                try:
                    ts = get_timestamp(filepath)
                except ValueError as e:
                    logger.warning(
                        "dropping master with unparseable timestamp: %r (%s)",
                        filepath,
                        e,
                    )
                    continue
                master_files.append((filepath, ts))

        return sorted(master_files, key=lambda x: x[1])

    def _select_nearest(self, date_obs, master_files):
        """
        Return the filepath of the candidate nearest in time to ``date_obs``,
        or None if ``master_files`` is empty (callers treat None as a failure).
        """
        if not master_files:
            return None

        obs_dt = datetime.fromisoformat(date_obs)
        return min(
            master_files,
            key=lambda x: abs(
                (kpf_timestamp_to_datetime(x[1]) - obs_dt).total_seconds()
            ),
        )[0]

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------

    def _track_info(self):
        """Build and cache the info() summary text from instance attributes."""
        lines = [
            "CalibrationAssociation",
            f"  obs_id:         {self.l1_obj.obs_id}",
            f"  masters output: {self._masters_output}",
            f"  search window:  {self.masters_search_window_days} days [before, after]",
            f"\n  {'cal_type':<12s} {'master file'}",
            "  " + "-" * 60,
        ]
        for cal_type, cal in self._calibrations.items():
            lines.append(f"  {cal_type:<12s} {cal['filepath']}")
            lines.append("")
        self._info = "\n\n" + "\n".join(lines) + "\n\n"

    def _set_headers(self, l1_obj):
        """Write the master-path and signed-age keywords for each association.

        Reads self._calibrations (from perform()); each cal type contributes
        {PREFIX}FILE (full master path, routed to RECEIPT) and {PREFIX}AGE (the
        master's timestamp minus PRIMARY DATE-OBS, in days, routed to
        QUALITY_CONTROL).
        """
        obs_dt = datetime.fromisoformat(l1_obj.headers["PRIMARY"]["DATE-OBS"])
        for cal_type, cal in self._calibrations.items():
            prefix = _HEADER_PREFIX[cal_type]
            l1_obj.set_keyword(f"{prefix}FILE", cal["filepath"])
            age = kpf_timestamp_to_datetime(get_timestamp(cal["filepath"])) - obs_dt
            l1_obj.set_keyword(f"{prefix}AGE", age.total_seconds() / 86400.0)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, cal_types, *, masters_search_window_days=None):
        """
        Run calibration association for the given calibration types.

        Parameters
        ----------
        cal_types : list of str
            Calibration types to associate (e.g. ['bias', 'dark', 'flat', 'thar']).
        masters_search_window_days : [int, int], optional
            Search window as [days_before, days_after]. Defaults to
            self.masters_search_window_days.

        Returns
        -------
        KPF1
            The input frame with calibration headers populated and a
            receipt entry added.

        Raises
        ------
        ValueError
            If any requested ``cal_type`` is unsupported.
        FileNotFoundError
            If no master file is found for any requested calibration type.
        """
        if masters_search_window_days is None:
            masters_search_window_days = self.masters_search_window_days

        unknown = [c for c in cal_types if c not in _HEADER_PREFIX]
        if unknown:
            raise ValueError(
                f"unsupported cal_type(s) {unknown}; "
                f"expected subset of {sorted(_HEADER_PREFIX)}"
            )

        date_obs = self.l1_obj.headers["PRIMARY"]["DATE-OBS"]

        self._calibrations = {}
        for cal_type in cal_types:
            master_files = self._find_master_files(
                cal_type, date_obs, masters_search_window_days
            )
            filepath = self._select_nearest(date_obs, master_files)
            if filepath is None:
                raise FileNotFoundError(
                    f"No '{cal_type}' master found for {date_obs} "
                    f"within window {masters_search_window_days} days"
                )

            self._calibrations[cal_type] = {"filepath": filepath}
            logger.debug(
                "%s: selected %s from %d candidate(s)",
                cal_type,
                filepath,
                len(master_files),
            )

        self._set_headers(self.l1_obj)
        self._track_info()
        self.l1_obj.receipt_add_entry("calibration_association", "", "PASS")

        logger.info("%s", self._info)
        return self.l1_obj

    def info(self):
        """Print a summary of the module configuration and association results."""
        if self._info is None:
            print(f"{type(self).__name__}: perform() has not been called")
        else:
            print(self._info)
