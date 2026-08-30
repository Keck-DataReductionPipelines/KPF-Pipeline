"""
KPF Masters Level 1 data model.

Stacked FFI calibration (bias, dark, flat). Extends KPF1 with
masters-specific extension names (GREEN_IMG, GREEN_SNR, GREEN_MASK and the
red-chip equivalents) that reflect the calibration normalization.
"""

import numpy as np

from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.masters.base import KPFMasterModel


class KPFMasterL1(KPFMasterModel, KPF1):
    """
    KPF Masters Level 1 stacked FFI calibration.

    Thin wrapper around KPF1 with masters-specific extension names.
    Inherits KPF1's from_fits, to_fits, and info; overrides ``_read`` and
    ``_create_hdul`` to cast the MASK extensions (bool <-> uint8) across I/O.

    Extensions:
        GREEN_IMG   -- stacked mean image, green chip
        GREEN_SNR   -- signal-to-noise ratio, green chip
        GREEN_MASK  -- bad pixel mask, green chip (1=good, 0=bad)
        RED_IMG, RED_SNR, RED_MASK -- same for red chip

    Construct empty with ``KPFMasterL1()``, or load a stacked product from
    disk with ``KPFMasterL1.from_fits(path)``.
    """

    def __init__(self):
        # KPF1.__init__ builds the extensions and seeds PRIMARY off the ML1
        # data model: KPFMasterModel redirects both to the masters' tables.
        super().__init__()
        self.set_keyword("DATALVL", "ML1")

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
