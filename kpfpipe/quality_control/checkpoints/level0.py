"""Checkpoints for KPF Level 0 (raw CCD) data products."""

from kpfpipe.quality_control.checkpoints.base import Checkpoint
from kpfpipe.quality_control.diagnostics import DiagL0
from kpfpipe.quality_control.qc_flags import QCL0


class CheckpointL0(Checkpoint):
    """Checkpoints for KPF Level 0 raw data products."""

    LEVEL = "L0"
    RAISE_FLAGS = ("DATAPRL0",)  # missing raw data is fatal; other flags warn.
    DIAGNOSTICS = DiagL0
    QC = QCL0
