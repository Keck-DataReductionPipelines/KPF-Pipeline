"""
KPF Masters Level 1 data model.

Stacked 2D calibration frame product (bias, dark, flat).

Inherits from KPFMasterModel and KPF1. All KPF1 methods (from_fits,
to_fits, _read, info, to_kpf2) are inherited unchanged. Extension names
differ from science L1 to reflect masters-specific normalization:

    GREEN_IMG  -- stacked mean image
    GREEN_SNR  -- signal-to-noise ratio
    GREEN_MASK -- boolean bad pixel mask (1=good, 0=bad)
    RED_IMG, RED_SNR, RED_MASK -- same for red chip

Filename convention (WMKO DRP-RUN-05): masters are written as
{KOAID-of-first-input}_master_{type}_L1.fits (e.g.
KP.20240405.49597.71_master_bias_L1.fits), built by
KPFMasterModel.generate_standard_filename().
"""

import importlib.resources

import numpy as np
import pandas as pd

from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.masters.base import KPFMasterModel

_config_path = importlib.resources.files("kpfpipe.data_models.config")
_MASTERS_L1_EXTENSIONS = pd.read_csv(_config_path / "ML1-extensions.csv")


class KPFMasterL1(KPFMasterModel, KPF1):
    """
    KPF Masters Level 1 stacked calibration frame.

    Thin wrapper around KPF1 with masters-specific extension names.
    Inherits all KPF1 methods (from_fits, to_fits, _read, info).

    Extensions:
        GREEN_IMG   -- stacked mean image, green chip
        GREEN_SNR   -- signal-to-noise ratio, green chip
        GREEN_MASK  -- bad pixel mask, green chip (1=good, 0=bad)
        RED_IMG, RED_SNR, RED_MASK -- same for red chip

    Construct empty with `KPFMasterL1()`, or load a stacked product from
    disk with `KPFMasterL1.from_fits(path)`.
    """

    _DATALVL = "ML1"
    _known_extensions = set(_MASTERS_L1_EXTENSIONS["Name"])

    def __init__(self):
        KPFMasterModel.__init__(self)
        self.level = 1

        for _, row in _MASTERS_L1_EXTENSIONS.iterrows():
            if row["Required"] and row["Name"] not in self.extensions:
                self.create_extension(row["Name"], row["DataType"])

        # Masters carry their own minimal PRIMARY (no EPRV science skeleton); stamp
        # DATALVL so it is present in-memory, not only at to_fits (DRP-RUN data
        # level). Routed via set_keyword like the science models.
        self.set_keyword("DATALVL", self._DATALVL)

    def _create_hdul(self):
        """Cast MASK extensions to uint8 before building HDUs, then restore."""
        originals = {}
        for ext in list(self.data.keys()):
            if ext.endswith("_MASK") and self.data[ext] is not None:
                originals[ext] = self.data[ext]
                self.data[ext] = self.data[ext].astype(np.uint8)
        try:
            return super()._create_hdul()
        finally:
            for ext, arr in originals.items():
                self.data[ext] = arr

    def _read(self, hdul):
        """Read extensions from FITS; cast MASK extensions back to bool."""
        super()._read(hdul)
        for ext in list(self.data.keys()):
            if ext.endswith("_MASK") and self.data[ext] is not None:
                self.data[ext] = self.data[ext].astype(bool)
