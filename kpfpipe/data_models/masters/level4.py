"""
KPF Masters Level 4 data model.

Masters RV/CCF calibration products. Not yet implemented.

Filename convention (WMKO DRP-RUN-05): masters are written as
{KOAID-of-first-input}_master_{type}_L4.fits, built by
KPFMasterModel.generate_standard_filename().
"""

from kpfpipe.data_models.level4 import KPF4
from kpfpipe.data_models.masters.base import KPFMasterModel


class KPFMasterL4(KPFMasterModel, KPF4):
    """
    KPF Masters Level 4 RV/CCF calibration products.

    Not yet implemented.
    """

    _DATALVL = "ML4"

    def __init__(self):
        raise NotImplementedError("KPFMasterL4 is not yet implemented")
