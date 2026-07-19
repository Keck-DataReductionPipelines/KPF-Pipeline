"""Checkpoints for KPF Level 1 (assembled FFI) data products."""

from kpfpipe.quality_control.checkpoints.base import Checkpoint
from kpfpipe.quality_control.diagnostics import DiagL1
from kpfpipe.quality_control.qc_flags import QCL1


class CheckpointL1(Checkpoint):
    """Checkpoints for KPF Level 1 assembled FFI products."""

    LEVEL = "L1"
    # Missing assembled CCDs or a required PRIMARY keyword is fatal; others warn.
    RAISE_FLAGS = ("DATAPRL1", "KWRDPRL1")
    DIAGNOSTICS = DiagL1
    QC = QCL1
