"""
KPF masters data models.

* KPFMasterL1 -- FFI calibrations (bias, dark, flat)
* KPFMasterL2 -- Extracted spectra (wls, flat)
* KPFMasterL4 -- RV/CCF products (stub; NotYetImplemented)
"""

from kpfpipe.data_models.masters.level1 import KPFMasterL1
from kpfpipe.data_models.masters.level2 import KPFMasterL2
from kpfpipe.data_models.masters.level4 import KPFMasterL4
