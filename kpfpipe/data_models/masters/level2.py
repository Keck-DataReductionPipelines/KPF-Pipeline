"""
KPF Masters Level 2 data model.

Extracted masters spectra (e.g., flat lamp spectra). Not yet implemented.
"""

from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.masters.base import KPFMasterModel


class KPFMasterL2(KPFMasterModel, KPF2):
    """
    KPF Masters Level 2 extracted masters spectra.

    Not yet implemented.
    """

    _DATALVL = "ML2"
    _FILENAME_PREFIX = "kpf_ML2"

    def __init__(self):
        raise NotImplementedError("KPFMasterL2 is not yet implemented")
