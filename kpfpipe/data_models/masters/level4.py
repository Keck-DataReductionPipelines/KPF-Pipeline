"""
KPF Masters Level 4 data model.

Masters RV/CCF calibration products. Not yet implemented.
"""

from kpfpipe.data_models.level4 import KPF4
from kpfpipe.data_models.masters.base import KPFMasterModel


class KPFMasterL4(KPFMasterModel, KPF4):
    """
    KPF Masters Level 4 RV/CCF calibration products.

    Not yet implemented.
    """

    _DATALVL = "ML4"
    _FILENAME_PREFIX = "kpf_ML4"

    def __init__(self):
        raise NotImplementedError("KPFMasterL4 is not yet implemented")
