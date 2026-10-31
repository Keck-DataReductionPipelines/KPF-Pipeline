"""Checkpoints for KPF Level 2 (extracted spectra) data products."""

from kpfpipe.quality_control.checkpoints.base import Checkpoint
from kpfpipe.quality_control.diagnostics import DiagL2
from kpfpipe.quality_control.qc_flags import QCL2


class CheckpointL2(Checkpoint):
    """Checkpoints for KPF Level 2 extracted spectra products."""

    LEVEL = "L2"
    # Missing extracted flux or a required PRIMARY keyword is fatal; others warn.
    RAISE_FLAGS = {
        "none": (),
        "default": ("DATAPRL2",),
        # DATAPRL2 requires WAVE and, for science frames, barycentric arrays. The
        # engineering recipe makes no catalog query (so no barycentric correction)
        # and may lack a viable WLS master, so will very often fail DATAPRL2.
        "engineering": (),
    }
    DIAGNOSTICS = (DiagL2,)
    QC = QCL2
