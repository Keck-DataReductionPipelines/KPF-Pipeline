"""Checkpoints for KPF Level 0 (raw CCD) data products."""

from kpfpipe.quality_control.checkpoints.base import Checkpoint
from kpfpipe.quality_control.diagnostics import (
    DiagL0,
    ExposureMeter,
    Guider,
    Telemetry,
)
from kpfpipe.quality_control.qc_flags import QCL0


class CheckpointL0(Checkpoint):
    """Checkpoints for KPF Level 0 raw data products."""

    LEVEL = "L0"
    # Missing raw data or a missing required PRIMARY keyword is fatal; others warn.
    RAISE_FLAGS = ("DATAPRL0",)
    # Guider before Telemetry: SEEING is the guider's GDRSEEV, carried to PRIMARY.
    DIAGNOSTICS = (DiagL0, Guider, ExposureMeter, Telemetry)
    QC = QCL0
