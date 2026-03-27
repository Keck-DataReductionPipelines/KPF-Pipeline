"""
KPF-specific base data model.

Thin layer on top of RVDataModel that adds KPF-specific attributes
and override points for filename conventions. L0 and L1 subclass this.
L2 and L4 data products use KPF2 and KPF4 (which extend RV2/RV4
with KPF-friendly extension aliases).
"""

from rvdata.core.models.base import RVDataModel

from kpfpipe.utils.kpf_parse import _OBS_ID_PATTERN, _DATECODE_PATTERN


class KPFDataModel(RVDataModel):
    """Base class for KPF pre-extraction data models (L0, L1)."""

    OBS_ID_PATTERN = _OBS_ID_PATTERN
    DATECODE_PATTERN = _DATECODE_PATTERN

    def __init__(self):
        super().__init__()
        self.obs_id = None

    def check_filename_convention(self, filename):
        """Override: KPF L0/L1 files do not use the EPRV SL# pattern."""
        return True
