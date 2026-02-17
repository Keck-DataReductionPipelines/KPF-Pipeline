"""
KPF-specific base data model.

Thin layer on top of RVDataModel that adds KPF-specific attributes
and override points for filename conventions. L0 and L1 subclass this.
L2+ data products use RVData classes directly (RV2, RV4).
"""

import os
import re

from rvdata.core.models.base import RVDataModel


class KPFDataModel(RVDataModel):
    """Base class for KPF pre-extraction data models (L0, L1)."""

    OBS_ID_PATTERN = re.compile(r"^KP\.\d{8}\.\d{5}\.\d{2}$")

    def __init__(self):
        super().__init__()
        self.obs_id = None

    def check_filename_convention(self, filename):
        """Override: KPF L0/L1 files do not use the EPRV SL# pattern."""
        return True
