"""
KPF Wavelength Calibration module.

Copies the per-fiber wavelength solution from a precomputed master WLS L2
product onto a science L2. The master is located by CalibrationAssociation,
which writes the full WLSFILE path to the L1 PRIMARY header. KPF1.to_kpf2()
preserves the full L1 PRIMARY in the L2 INSTRUMENT_HEADER extension, where
this module reads it from.
"""
import os

import numpy as np

from kpfpipe import DEFAULTS
from kpfpipe.data_models.masters.level2 import KPFMasterL2
from kpfpipe.utils.config import ConfigHandler

_DEFAULTS = dict(DEFAULTS)


class WavelengthCalibration:
    """
    Apply a precomputed wavelength solution to an extracted KPF L2 frame.

    Reads `WLSFILE` (full path, legacy convention) from the L2
    INSTRUMENT_HEADER extension (populated by CalibrationAssociation on
    the L1 PRIMARY and carried through by to_kpf2), loads the corresponding
    KPFMasterL2, and copies each per-fiber {CHIP}_{FIBER}_WAVE array onto
    the science L2.

    Parameters
    ----------
    l2_obj : KPF2
        Extracted L2 frame. The INSTRUMENT_HEADER extension must contain
        a WLSFILE keyword.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: chips, fibers.
    """

    def __init__(self, l2_obj, config=None):
        self.l2_obj = l2_obj

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS", "KPFPIPE", "MODULE_WAVELENGTH_CALIBRATION"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in _DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self._wls_path = None  # set by load_wls()
        self._results = None   # populated by perform()

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def load_wls(self, wls_path=None):
        """
        Load the master wavelength solution from disk.

        If `wls_path` is provided it is used directly, bypassing the header
        lookup. Otherwise the path is read from `WLSFILE` in the L2
        INSTRUMENT_HEADER extension (which preserves the L1 PRIMARY, where
        CalibrationAssociation wrote WLSFILE as a full path per the legacy
        WLS convention).

        Parameters
        ----------
        wls_path : str, optional
            Direct path to a master WLS L2 FITS file.

        Returns
        -------
        KPFMasterL2
            The loaded master WLS data model.

        Raises
        ------
        KeyError
            If neither `wls_path` is given nor WLSFILE is present in the L2
            INSTRUMENT_HEADER.
        FileNotFoundError
            If the resolved path does not exist.
        """
        if wls_path is None:
            inst_header = self.l2_obj.headers.get('INSTRUMENT_HEADER', {})
            if 'WLSFILE' not in inst_header:
                raise KeyError(
                    "WLSFILE missing from L2 INSTRUMENT_HEADER; "
                    "run CalibrationAssociation with 'thar' on the L1 first"
                )
            wls_path = inst_header['WLSFILE']

        if not os.path.isfile(wls_path):
            raise FileNotFoundError(f"Master WLS file not found: {wls_path}")

        self._wls_path = wls_path
        return KPFMasterL2.from_fits(wls_path)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, chips=None, fibers=None, wls_path=None):
        """
        Copy the master wavelength solution onto the science L2.

        For every (chip, fiber) in the configured chips × fibers, copies
        `{CHIP}_{FIBER}_WAVE` from the master L2 onto the science L2. The
        chip-prefix keys resolve through the KPF2 alias system into slices
        of the underlying concatenated TRACE{n}_WAVE arrays.

        Parameters
        ----------
        chips : list of str, optional
            Chip identifiers, e.g. ['GREEN', 'RED']. Defaults to self.chips.
        fibers : list of str, optional
            Fiber identifiers, e.g. ['SKY', 'SCI1', 'SCI2', 'SCI3', 'CAL'].
            Defaults to self.fibers.
        wls_path : str, optional
            Direct path to the master WLS L2 file. If omitted, the path is
            read from WLSFILE on the L2 INSTRUMENT_HEADER extension.

        Returns
        -------
        l2_obj : KPF2
            The input L2 with per-fiber _WAVE extensions populated and a
            'wavelength_calibration' receipt entry. WLSFILE on
            INSTRUMENT_HEADER is left untouched.
        """
        if chips is None:
            chips = self.chips
        if fibers is None:
            fibers = self.fibers

        master = self.load_wls(wls_path=wls_path)

        for chip in chips:
            for fiber in fibers:
                key = f'{chip}_{fiber}_WAVE'
                src = master.data[key]
                if src is None or np.size(src) == 0:
                    raise KeyError(
                        f"WLS master has no data for {key}; "
                        f"cannot apply wavelength solution"
                    )
                self.l2_obj.set_data(key, np.asarray(src, dtype=np.float64))

        self.l2_obj.receipt_add_entry('wavelength_calibration', 'PASS')

        self._results = {
            'wls_path': self._wls_path,
            'chips': list(chips),
            'fibers': list(fibers),
        }

        return self.l2_obj

    def info(self):
        """Print a summary of the module configuration and association results."""
        print("WavelengthCalibration")
        obs_id = self.l2_obj.headers.get('PRIMARY', {}).get('ORIGID', 'unknown')
        if isinstance(obs_id, tuple):
            obs_id = obs_id[0]
        print(f"  obs_id:  {obs_id}")
        print(f"  chips:   {self.chips}")
        print(f"  fibers:  {self.fibers}")

        if self._results is None:
            print("  perform() has not been called")
            return

        inst = self.l2_obj.headers.get('INSTRUMENT_HEADER', {})
        agewls = inst.get('AGEWLS')
        print(f"  wls_path: {self._results['wls_path']}")
        if agewls is not None:
            print(f"  AGEWLS:   {agewls:+.4f} d  (master - obs)")
