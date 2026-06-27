"""Checkpoints for KPF Level 2 (extracted spectra) data products."""

from kpfpipe.quality_control.checkpoints.base import Checkpoint


class CheckpointL2(Checkpoint):
    """Checkpoints for KPF Level 2 extracted spectra products."""

    LEVEL = "L2"
    RAISE_FLAGS = ("DATAPRL2",)  # missing extracted flux is fatal; other flags warn.
