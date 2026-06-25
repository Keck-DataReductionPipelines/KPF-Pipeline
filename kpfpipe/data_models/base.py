"""
KPF-specific base data model.

Thin layer on top of RVDataModel that adds KPF-specific attributes
and override points for filename conventions. L0 and L1 subclass this.
L2 and L4 data products use KPF2 and KPF4 (which extend RV2/RV4
with KPF-friendly extension aliases).
"""

from astropy.table import Table
from rvdata.core.models.base import RVDataModel

from kpfpipe.utils.kpf import _DATECODE_PATTERN, _OBS_ID_PATTERN

# Receipt names that are data-model conversions / serialization rather than
# pipeline modules — excluded from DRPSTATU so it names the last real stage.
# ``from_fits`` is here too: reading a product back must not clobber the status
# the writer stamped.
_INTERNAL_RECEIPTS = frozenset({"to_l1", "to_kpf2", "to_kpf4", "to_fits", "from_fits"})


def update_drpstatus(data_model, module):
    """Stamp DRPSTATU = '<Module Name> module complete' for a completed module.

    Called from the ``receipt_add_entry`` override on every KPF data model.
    """
    if module in _INTERNAL_RECEIPTS:
        return
    primary = data_model.headers.get("PRIMARY")
    if primary is None:
        return
    label = module.replace("_", " ").title()
    primary["DRPSTATU"] = (
        f"{label} module complete",
        "DRP reduction status (DRP-RUN-20)",
    )


class KPFDataModel(RVDataModel):
    """Base class for KPF pre-extraction data models (L0, L1)."""

    OBS_ID_PATTERN = _OBS_ID_PATTERN
    DATECODE_PATTERN = _DATECODE_PATTERN

    def __init__(self):
        super().__init__()
        self.obs_id = None

    def receipt_add_entry(self, module, status):
        """Record a processing step, and update DRPSTATU for pipeline modules."""
        super().receipt_add_entry(module, status)
        if status == "PASS":
            update_drpstatus(self, module)

    def _create_hdul(self):
        """Sync self.receipt into the RECEIPT extension before writing; rvdata
        serializes self.data["RECEIPT"], not self.receipt. L0/L1 omit RECEIPT
        from their default extensions, so create it if absent."""
        if self.receipt is not None and not self.receipt.empty:
            if "RECEIPT" not in self.extensions:
                self.create_extension("RECEIPT", "BinTableHDU")
            self.data["RECEIPT"] = Table.from_pandas(self.receipt)
        return super()._create_hdul()

    def check_filename_convention(self, filename):
        """Override: KPF L0/L1 files do not use the EPRV SL# pattern."""
        return True
