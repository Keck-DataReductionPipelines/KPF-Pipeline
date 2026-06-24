"""
KPF-specific base data model.

Thin layer on top of RVDataModel that adds KPF-specific attributes
and override points for filename conventions. L0 and L1 subclass this.
L2 and L4 data products use KPF2 and KPF4 (which extend RV2/RV4
with KPF-friendly extension aliases).
"""

from importlib.metadata import PackageNotFoundError, version

from astropy.table import Table
from rvdata.core.models.base import RVDataModel

from kpfpipe.utils.kpf import _DATECODE_PATTERN, _OBS_ID_PATTERN

_KPF_DRP_PACKAGE = "kpf-drp"
_PROVENANCE_KEYS = ("PROGID", "KOAID")


def _header_value(value):
    """Return the raw FITS value, dropping any in-memory comment tuple."""
    if isinstance(value, tuple):
        return value[0]
    return value


def kpf_drp_version():
    """Return the installed KPF DRP package version used for FITS metadata."""
    try:
        return version(_KPF_DRP_PACKAGE)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"Cannot determine {_KPF_DRP_PACKAGE} package version for DRPVERNO"
        ) from exc


def apply_provenance_metadata(data_model):
    """
    Apply WMKO provenance/status keywords to a data model before FITS write.

    Processing history is kept in the existing RECEIPT extension. This helper
    only handles the PRIMARY header keywords required at write finalization.
    """
    primary = data_model.headers.get("PRIMARY")
    if primary is None:
        return

    for key in _PROVENANCE_KEYS:
        if key in primary:
            continue
        instrument_header = data_model.headers.get("INSTRUMENT_HEADER", {})
        if key in instrument_header:
            primary[key] = (
                _header_value(instrument_header[key]),
                f"Propagated original {key}",
            )

    primary["DRPVERNO"] = (kpf_drp_version(), "KPF DRP package version")
    primary["DRPSTATU"] = ("COMPLETE", "DRP reduction status")


def sync_receipt_extension(data_model):
    """Copy the in-memory processing receipt into the FITS RECEIPT extension."""
    if data_model.receipt is None or data_model.receipt.empty:
        return
    if "RECEIPT" not in data_model.extensions:
        data_model.create_extension("RECEIPT", "BinTableHDU")
    data_model.data["RECEIPT"] = Table.from_pandas(data_model.receipt)


class KPFDataModel(RVDataModel):
    """Base class for KPF pre-extraction data models (L0, L1)."""

    OBS_ID_PATTERN = _OBS_ID_PATTERN
    DATECODE_PATTERN = _DATECODE_PATTERN

    def __init__(self):
        super().__init__()
        self.obs_id = None

    def _create_hdul(self):
        """
        Sync receipt/provenance metadata into FITS content before writing.

        rvdata writes data["RECEIPT"]; KPF L0/L1 processing steps are tracked in
        self.receipt, so copy that table into the extension at finalization.
        """
        apply_provenance_metadata(self)
        sync_receipt_extension(self)
        return super()._create_hdul()

    def check_filename_convention(self, filename):
        """Override: KPF L0/L1 files do not use the EPRV SL# pattern."""
        return True
