"""
KPF Calibration Association module.

Given a KPF observation frame, finds the most appropriate master calibration
file for each calibration type (bias, dark, flat, thar-wls) by searching
the masters directory and selecting the nearest-in-time match.
"""
import glob
import os
from datetime import datetime, timedelta

from kpfpipe import DEFAULTS
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.kpf import get_datecode, get_timestamp

DEFAULTS.update({
    'masters_search_window_days': [-1, 0],
})


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
            params = config.get_params(["DATA_DIRS", "KPFPIPE", "MODULE_CALIBRATION_ASSOCIATION"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self._data_root = params.get('KPF_DATA_INPUT')
        self._results = None  # populated by perform()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_master_files(self, cal_type, date_obs, search_window=None):
        """
        Return a list of (filepath, timestamp) tuples for all available
        masters of the given calibration type within the search window.

        Parameters
        ----------
        cal_type : str
            One of 'bias', 'dark', 'flat', 'thar-wls'.
        date_obs : str
            ISO-format observation datetime from the frame's PRIMARY header
            (e.g. '2024-04-05T11:08:33').
        search_window : [int, int], optional
            Search window as [days_before, days_after]. Defaults to
            self.masters_search_window_days.

        Returns
        -------
        list of (str, str)
            Sorted list of (filepath, kpf_timestamp) tuples.
        """
        obs_date = datetime.fromisoformat(date_obs).date()
        days_before, days_after = search_window if search_window is not None else self.masters_search_window_days

        master_files = []
        for delta in range(days_before, days_after + 1):
            search_date = obs_date + timedelta(days=delta)
            datecode = search_date.strftime('%Y%m%d')
            pattern = os.path.join(
                self._data_root, 'masters', datecode,
                f'*_master_{cal_type}_L1.fits'
            )
            for filepath in sorted(glob.glob(pattern)):
                try:
                    ts = get_timestamp(filepath)
                    master_files.append((filepath, ts))
                except ValueError:
                    pass

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

        def _candidate_dt(ts):
            date_str, seconds_str, _ = ts.split('.')
            return datetime.strptime(date_str, '%Y%m%d') + timedelta(seconds=int(seconds_str))

        return min(master_files, key=lambda x: abs((_candidate_dt(x[1]) - obs_dt).total_seconds()))[0]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, cal_types, search_window=None):
        """
        Run calibration association for the given calibration types.

        Parameters
        ----------
        cal_types : list of str
            Calibration types to associate (e.g. ['bias', 'dark', 'flat', 'thar-wls']).
        search_window : [int, int], optional
            Search window as [days_before, days_after]. Defaults to
            self.masters_search_window_days.

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
        date_obs = self.l1_obj.headers['PRIMARY']['DATE-OBS']
        obs_date = datetime.fromisoformat(date_obs).date()
        active_window = search_window if search_window is not None else self.masters_search_window_days

        _header_prefix = {'bias': 'BIAS', 'dark': 'DARK', 'flat': 'FLAT'}

        for cal_type in cal_types:
            master_files = self._find_master_files(cal_type, date_obs, search_window)
            filepath = self._select_nearest(date_obs, master_files)
            if filepath is None:
                raise FileNotFoundError(
                    f"No '{cal_type}' master found for {date_obs} "
                    f"within window {active_window} days"
                )

            prefix = _header_prefix.get(cal_type)
            if prefix is not None:
                master_date = datetime.strptime(get_datecode(filepath), '%Y%m%d').date()
                self.l1_obj.headers['PRIMARY'][f'{prefix}FILE'] = os.path.basename(filepath)
                self.l1_obj.headers['PRIMARY'][f'{prefix}DIR'] = os.path.dirname(filepath)
                self.l1_obj.headers['PRIMARY'][f'AGE{prefix}'] = (obs_date - master_date).days

        self._results = {
            cal_type: self.l1_obj.headers['PRIMARY'].get(f'{_header_prefix[cal_type]}FILE')
                      if cal_type in _header_prefix else None
            for cal_type in cal_types
        }
        self.l1_obj.receipt_add_entry('calibration_association', 'PASS')

        return self.l1_obj

    def info(self):
        """Print a summary of the module configuration and association results."""
        print("CalibrationAssociation")
        print(f"  obs_id:        {self.l1_obj.obs_id}")
        print(f"  data root:     {self._data_root}")
        print(f"  search window: {self.masters_search_window_days} days [before, after]")

        if self._results is None:
            print("  perform() has not been called")
            return

        print(f"\n  {'cal_type':<12s} {'master file'}")
        print("  " + "-" * 60)
        h = self.l1_obj.headers['PRIMARY']
        _prefix = {'bias': 'BIAS', 'dark': 'DARK', 'flat': 'FLAT'}
        for cal_type, filename in self._results.items():
            prefix = _prefix.get(cal_type)
            if prefix is not None:
                age = h.get(f'AGE{prefix}', 'n/a')
                print(f"  {cal_type:<12s} {filename}")
                print(f"  {'':12s} age = {age}d")
                print()
            else:
                print(f"  {cal_type:<12s} (no header written)")
                print()
