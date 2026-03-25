"""
KPF masters base data model.

Base class for all masters calibration data models. Inherits from
KPFDataModel and initializes the data model infrastructure without
creating any science-level extensions.

Level-specific masters classes use double inheritance, e.g.:

    class KPFMasterL1(KPFMasterModel, KPF1):
        ...

This gives the level-specific class access to science model methods
(from_fits, to_fits, _read, info, etc.) while the extension setup is
controlled entirely by KPFMasterModel and its subclasses.

Masters products differ from science products in extension naming to
avoid confusion: units and normalization conventions differ by
calibration type (bias, dark, flat) and are not the same as raw
science counts (e.g., GREEN_CCD).
"""

from kpfpipe.data_models.base import KPFDataModel


class KPFMasterModel(KPFDataModel):
    """
    Base class for KPF masters calibration data models.

    Inherits from KPFDataModel and initializes only the base data model
    infrastructure. Science-level extension setup is intentionally skipped
    so that level-specific subclasses can install masters extensions instead.
    Normalization conventions differ by calibration type (bias, dark, flat).
    """

    def __init__(self):
        KPFDataModel.__init__(self)
