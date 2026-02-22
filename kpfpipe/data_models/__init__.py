"""
KPF data models.

KPF0 -- Raw CCD data (Level 0)
KPF1 -- Assembled 2D frames (Level 1)
KPF2 -- Extracted spectra (Level 2, extends RV2 with KPF aliases)
KPF4 -- RVs and CCFs (Level 4, extends RV4 with KPF aliases)
"""

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.level4 import KPF4
