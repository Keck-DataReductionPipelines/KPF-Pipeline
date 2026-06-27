"""Checkpoints for KPF Level 1 (assembled FFI) data products."""

from kpfpipe.quality_control.checkpoints.base import Checkpoint


class CheckpointL1(Checkpoint):
    """Checkpoints for KPF Level 1 assembled FFI products."""

    LEVEL = "L1"
    RAISE_FLAGS = ("DATAPRL1",)  # missing assembled CCDs is fatal; other flags warn.
