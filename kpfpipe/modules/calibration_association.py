"""
KPF Calibration Association module.

Given a KPF observation frame, finds the most appropriate master calibration
file for each calibration type (bias, dark, flat, thar) by searching
the masters directory and selecting the nearest-in-time match.
"""

import glob
import os
import warnings
from datetime import datetime, timedelta

from kpfpipe import DEFAULTS
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.kpf import get_datecode, get_timestamp, kpf_timestamp_to_datetime

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
        KPF observation frame. For now this is always an L1 frame. The
        observation timestamp is read from its PRIMARY header (DATE-OBS).
    config : None | dict | ConfigHandler
        Module configuration. Recognised keys:
            KPF_MASTERS_OUTPUT : str
                Root directory under which master calibration files live
                (searched as {root}/masters/{datecode}/). This is where the
                masters recipe writes its products.
            masters_search_window_days : [int, int]
                Search window as [days_before, days_after] relative to the
                science frame's observation date. Negative values are in the
                past, positive in the future. Default: [-1, 0] (search up to
                1 day before and same day only).
    """

    def __init__(self, l1_obj, config=None):
        self.l1_obj = l1_obj

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(
                ["DATA_DIRS", "KPFPIPE", "MODULE_CALIBRATION_ASSOCIATION"]
            )
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in _DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self._masters_root = params.get("KPF_MASTERS_OUTPUT")
        self._results = None  # populated by perform()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_master_files(
        self, cal_type, date_obs, masters_search_window_days=None, verbose=True
    ):
        """
        Return a list of (filepath, timestamp) tuples for all available
        masters of the given calibration type within the search window.

        Parameters
        ----------
        cal_type : str
            One of 'bias', 'dark', 'flat', 'thar'.
        date_obs : str
            ISO-format observation datetime from the frame's PRIMARY header
            (e.g. '2024-04-05T11:08:33').
        masters_search_window_days : [int, int], optional
            Search window as [days_before, days_after]. Defaults to
            self.masters_search_window_days.
        verbose : bool, optional
            If True (default), emit a UserWarning when a candidate master is
            dropped because its filename has no parseable KPF timestamp
            (a silent drop could invisibly shift which master is selected).

        Returns
        -------
        list of (str, str)
            Sorted list of (filepath, kpf_timestamp) tuples.

        Raises
        ------
        ValueError
            If `cal_type` is not a key of `_LEVEL_BY_CAL_TYPE`.
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
            pattern = os.path.join(
                self._masters_root,
                "masters",
                datecode,
                f"*_master_{cal_type}_{level}.fits",
            )
            for filepath in sorted(glob.glob(pattern)):
                try:
                    ts = get_timestamp(filepath)
                except ValueError as e:
                    if verbose:
                        warnings.warn(
                            f"dropping master with unparseable timestamp: "
                            f"{filepath!r} ({e})",
                            stacklevel=2,
                        )
                    continue
                master_files.append((filepath, ts))

        return sorted(master_files, key=lambda x: x[1])

    def _select_nearest(self, date_obs, master_files):
        """
        Select the candidate whose timestamp is nearest to date_obs.

        Parameters
        ----------
        date_obs : str
            ISO-format observation datetime from the frame's PRIMARY header
            (e.g. '2024-04-05T11:08:33').
        master_files : list of (str, str)
            (filepath, kpf_timestamp) pairs from _find_master_files.

        Returns
        -------
        str or None
            Filepath of the selected master, or None if master_files is empty.
            Callers should treat None as a failure.
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
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, cal_types, *, masters_search_window_days=None, verbose=True):
        """
        Run calibration association for the given calibration types.

        Parameters
        ----------
        cal_types : list of str
            Calibration types to associate (e.g. ['bias', 'dark', 'flat', 'thar']).
        masters_search_window_days : [int, int], optional
            Search window as [days_before, days_after]. Defaults to
            self.masters_search_window_days.
        verbose : bool, optional
            If True (default), warn when a candidate master is dropped for an
            unparseable timestamp (passed through to _find_master_files).

        Returns
        -------
        KPF1
            The input frame with calibration headers populated and a
            receipt entry added.

        Raises
        ------
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
        obs_date = datetime.fromisoformat(date_obs).date()
        primary = self.l1_obj.headers["PRIMARY"]

        for cal_type in cal_types:
            master_files = self._find_master_files(
                cal_type, date_obs, masters_search_window_days, verbose=verbose
            )
            filepath = self._select_nearest(date_obs, master_files)
            if filepath is None:
                raise FileNotFoundError(
                    f"No '{cal_type}' master found for {date_obs} "
                    f"within window {masters_search_window_days} days"
                )

            if cal_type == "thar":
                # Match legacy WLS header convention exactly: full path in
                # WLSFILE (no WLSDIR), AGEWLS in fractional days using the
                # master and obs timestamps (sign convention: master - obs,
                # so AGEWLS is negative when the master predates the obs).
                obs_dt = datetime.fromisoformat(date_obs)
                master_dt = kpf_timestamp_to_datetime(get_timestamp(filepath))
                primary["WLSFILE"] = filepath
                primary["AGEWLS"] = (master_dt - obs_dt).total_seconds() / 86400.0
            else:
                prefix = _HEADER_PREFIX[cal_type]
                master_date = datetime.strptime(get_datecode(filepath), "%Y%m%d").date()
                primary[f"{prefix}FILE"] = os.path.basename(filepath)
                primary[f"{prefix}DIR"] = os.path.dirname(filepath)
                primary[f"AGE{prefix}"] = (obs_date - master_date).days

        self._results = {
            cal_type: primary[f"{_HEADER_PREFIX[cal_type]}FILE"]
            for cal_type in cal_types
        }
        self.l1_obj.receipt_add_entry("calibration_association", "PASS")

        return self.l1_obj

    def info(self):
        """Print a summary of the module configuration and association results."""
        print("CalibrationAssociation")
        print(f"  obs_id:        {self.l1_obj.obs_id}")
        print(f"  masters root:  {self._masters_root}")
        print(
            f"  search window: {self.masters_search_window_days} days [before, after]"
        )

        if self._results is None:
            print("  perform() has not been called")
            return

        print(f"\n  {'cal_type':<12s} {'master file'}")
        print("  " + "-" * 60)
        h = self.l1_obj.headers["PRIMARY"]
        for cal_type, filename in self._results.items():
            prefix = _HEADER_PREFIX[cal_type]
            age = h.get(f"AGE{prefix}", "n/a")
            print(f"  {cal_type:<12s} {filename}")
            print(f"  {'':12s} age = {age}d")
            print()
