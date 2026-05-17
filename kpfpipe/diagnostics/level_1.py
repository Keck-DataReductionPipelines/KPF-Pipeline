"""Diagnostics for KPF Level 1 (assembled 2D) data products.

Currently a placeholder. The L1 metrics consumed by QCL1 (read noise,
master ages, BIASUB flag) are all written by the modules that produce
them — ImageAssembly, CalibrationAssociation, ImageProcessing — because
they depend on intermediate processing state. Diagnostics that can be
computed from the finished L1 product alone (e.g. flux percentiles,
order-trace alignment metrics) will live here.
"""

from kpfpipe.diagnostics.base import Diagnostics


class DiagL1(Diagnostics):
    LEVEL = "L1"
