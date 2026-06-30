"""Checkpoints for KPF Level 4 (RVs and CCFs) data products."""

from kpfpipe.quality_control.checkpoints.base import Checkpoint
from kpfpipe.quality_control.diagnostics import DiagL4
from kpfpipe.quality_control.qc_flags import QCL4


class CheckpointL4(Checkpoint):
    """Checkpoints for KPF Level 4 RV/CCF products."""

    LEVEL = "L4"
    RAISE_FLAGS = ("DATAPRL4",)  # missing science CCF/RV is fatal; other flags warn.
    DIAGNOSTICS = DiagL4
    QC = QCL4
