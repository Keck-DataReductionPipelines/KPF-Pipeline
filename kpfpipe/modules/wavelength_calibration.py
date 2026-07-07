"""
KPF Wavelength Calibration module.

Copies the per-fiber wavelength solution from a precomputed master WLS L2
product onto a science L2. The master is located by CalibrationAssociation,
which writes the full WLSFILE path to the RECEIPT header (its registry home);
to_kpf2 forwards the L1 RECEIPT header to the L2, where this module reads it.
"""

import logging
import os

import numpy as np

from kpfpipe import DEFAULTS
from kpfpipe.data_models.masters.level2 import KPFMasterL2
from kpfpipe.utils.config import ConfigHandler

logger = logging.getLogger(__name__)

_DEFAULTS = {**DEFAULTS}


class WavelengthCalibration:
    """
    Apply a precomputed wavelength solution to an extracted KPF L2 frame.

    Reads `WLSFILE` (full path, legacy convention) from the L2 RECEIPT
    header (written by CalibrationAssociation on the L1 RECEIPT and carried
    through by to_kpf2), loads the corresponding KPFMasterL2, and copies each
    per-fiber {CHIP}_{FIBER}_WAVE array onto the science L2.

    Parameters
    ----------
    l2_obj : KPF2
        Extracted L2 frame. The RECEIPT header must contain a WLSFILE keyword.
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
            params = config.get_params(
                ["DATA_DIRS", "TRACES", "MODULE_WAVELENGTH_CALIBRATION"]
            )
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in _DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self._wls_path = None  # set by load_wls()
        self._info = None

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def load_wls(self, wls_path=None):
        """
        Load the master wavelength solution from disk.

        If `wls_path` is provided it is used directly, bypassing the header
        lookup. Otherwise the path is read from `WLSFILE` in the L2 RECEIPT
        header (where CalibrationAssociation wrote it as a full path per the
        legacy WLS convention).

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
            RECEIPT header.
        FileNotFoundError
            If the resolved path does not exist.
        """
        if wls_path is None:
            receipt = self.l2_obj.headers.get("RECEIPT", {})
            if "WLSFILE" not in receipt:
                raise KeyError(
                    "WLSFILE missing from L2 RECEIPT; "
                    "run CalibrationAssociation with 'thar' on the L1 first"
                )
            wls_path = receipt.get("WLSFILE")

        if not os.path.isfile(wls_path):
            raise FileNotFoundError(f"Master WLS file not found: {wls_path}")

        self._wls_path = wls_path
        return KPFMasterL2.from_fits(wls_path)

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------

    def _track_info(self):
        """Build and cache the info() summary text from instance attributes."""
        obs_id = self.l2_obj.obs_id or "unknown"
        lines = [
            "WavelengthCalibration",
            f"  obs_id:  {obs_id}",
            f"  chips:   {self.chips}",
            f"  fibers:  {self.fibers}",
        ]
        # WLSAGE is written to QUALITY_CONTROL by DiagL1 (CalibrationAssociation
        # writes WLSFILE to RECEIPT; DiagL1 derives the age from it).
        agewls = self.l2_obj.headers.get("QUALITY_CONTROL", {}).get("WLSAGE")
        lines.append(f"  wls_path: {self._wls_path}")
        if agewls is not None:
            lines.append(f"  WLSAGE:   {agewls:+.4f} d  (master - obs)")
        self._info = "\n".join(lines)

    def _set_headers(self, l2_obj):
        """Write all PRIMARY-header keywords for wavelength calibration.

        Reserved: this module writes no PRIMARY metadata yet. Present so every
        module consolidates header writes in one place, called just before the
        receipt entry.
        """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, chips=None, fibers=None, *, wls_path=None):
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
            read from WLSFILE on the L2 RECEIPT header.

        Returns
        -------
        l2_obj : KPF2
            The input L2 with per-fiber _WAVE extensions populated and a
            'wavelength_calibration' receipt entry. WLSFILE on RECEIPT is
            left untouched.
        """
        if chips is None:
            chips = self.chips
        if fibers is None:
            fibers = self.fibers

        master = self.load_wls(wls_path=wls_path)

        for chip in chips:
            for fiber in fibers:
                key = f"{chip}_{fiber}_WAVE"
                src = master.data[key]
                if src is None or np.size(src) == 0:
                    raise KeyError(
                        f"WLS master has no data for {key}; "
                        f"cannot apply wavelength solution"
                    )
                self.l2_obj.set_data(key, np.asarray(src, dtype=np.float64))

        self._set_headers(self.l2_obj)
        self._track_info()
        self.l2_obj.receipt_add_entry("wavelength_calibration", "", "PASS")

        logger.info("summary:\n%s", self._info)
        return self.l2_obj

    def info(self):
        """Print a summary of the module configuration and association results."""
        if self._info is None:
            print(f"{type(self).__name__}: perform() has not been called")
        else:
            print(self._info)
