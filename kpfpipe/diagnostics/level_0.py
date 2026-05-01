"""Diagnostics for KPF Level 0 (raw CCD) data products.

Currently a placeholder. Most L0 metrics computable from the raw product
alone are header pass-throughs from the telescope and don't need
re-computation here. Diagnostics that read the overscan region (read
noise, non-Gaussian RN) are owned by ImageAssembly because they need
to run before gain conversion modifies the amp data.
"""

from kpfpipe.diagnostics.base import Diagnostics


class DiagL0(Diagnostics):
    LEVEL = "L0"
