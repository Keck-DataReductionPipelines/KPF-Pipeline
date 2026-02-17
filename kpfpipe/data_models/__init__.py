"""
KPF data models.

KPF0 -- Raw CCD data (Level 0)
KPF1 -- Assembled 2D frames (Level 1)

For extracted spectra (L2) and RVs (L4), use RVData directly:
    from rvdata.core.models.level2 import RV2
    from rvdata.core.models.level4 import RV4
"""

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
