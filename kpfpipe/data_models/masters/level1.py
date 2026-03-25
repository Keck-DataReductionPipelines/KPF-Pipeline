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
"""

import importlib.resources

import pandas as pd

from kpfpipe.data_models.base import KPFDataModel
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.masters.base import KPFMasterModel

_config_path = importlib.resources.files("kpfpipe.data_models.config")
_MASTERS_L1_EXTENSIONS = pd.read_csv(_config_path / "Masters-L1-extensions.csv")
_KNOWN_MASTERS_L1_EXTENSIONS = set(_MASTERS_L1_EXTENSIONS["Name"].tolist())


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

    Usage:
        m1 = KPFMasterL1()
        m1 = KPFMasterL1.from_fits("/path/to/master_bias.fits")
    """

    _DATALVL = "ML1"
    _FILENAME_PREFIX = "kpf_ML1"
    _known_extensions = _KNOWN_MASTERS_L1_EXTENSIONS

    def __init__(self):
        KPFMasterModel.__init__(self)
        self.level = 1

        for _, row in _MASTERS_L1_EXTENSIONS.iterrows():
            if row["Required"] and row["Name"] not in self.extensions:
                self.create_extension(row["Name"], row["DataType"])
