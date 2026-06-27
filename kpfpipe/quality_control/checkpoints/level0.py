"""Checkpoints for KPF Level 0 (raw CCD) data products."""

from kpfpipe.quality_control.checkpoints.base import Checkpoint


class CheckpointL0(Checkpoint):
    """Checkpoints for KPF Level 0 raw data products.

    The raw WMKO L0 PRIMARY is not registry-governed, so ``unregistered_keywords``
    skips it and validates only the KPF-custom extensions (QUALITY_CONTROL,
    RECEIPT) present on the product.
    """

    LEVEL = "L0"
    RAISE_FLAGS = ("DATAPRL0",)  # missing raw data is fatal; other flags warn.
