"""
KPF Level 2 (extracted spectra) data model.

Extracted, wavelength-calibrated spectra. Built from ``L2-extensions.csv`` and
carrying KPF-friendly extension aliases, so data can be accessed by either EPRV
name (``TRACE3_FLUX``) or KPF name (``SCI2_FLUX``), including per-chip views
(``GREEN_SCI2_FLUX``, ``RED_SCI2_FLUX``).
"""

import pandas as pd

from kpfpipe.data_models.aliased_dict import (
    NORDER_GREEN,
    ChipPrefixDict,
)
from kpfpipe.data_models.base import TRACE_MAP, KPFDataModel
from kpfpipe.data_models.config import PATH as _config_path
from kpfpipe.data_models.level4 import KPF4

# Re-exported: NORDER_GREEN is the chip split point, defined once beside the
# chip-prefix dict that applies it.
__all__ = ["KPF2", "NORDER_GREEN"]

_ORDER_TABLE_COLUMNS = pd.read_csv(_config_path / "L2-ORDER_TABLE-columns.csv")

# Extension name suffixes for each trace (e.g., TRACE3_FLUX, TRACE3_WAVE)
_TRACE_SUFFIXES = ["FLUX", "WAVE", "VAR", "BLAZE"]

# Per-order ancillary extensions (not traces) that also support chip-prefix
# access, e.g. GREEN_BARYCORR_Z -> BARYCORR_Z[:NORDER_GREEN]. These are 1-D
# (norder,) arrays aligned with the concatenated trace orders.
_ANCILLARY_PER_ORDER = ["BJD_TDB", "BARYCORR_KMS", "BARYCORR_Z"]

# e.g. "GREEN_CAL_FLUX", "RED_SCI1_FLUX", "GREEN_BARYCORR_Z", ...
_L2_CHIP_PREFIX_KEYS = {}  # chip-prefixed key -> (base_key, chip)
for _, _row in TRACE_MAP.iterrows():
    _fiber = str(_row["Fiber"]).strip()
    for _suffix in _TRACE_SUFFIXES:
        _fiber_alias = f"{_fiber}_{_suffix}"
        for _chip in ("GREEN", "RED"):
            _L2_CHIP_PREFIX_KEYS[f"{_chip}_{_fiber}_{_suffix}"] = (_fiber_alias, _chip)
for _ext in _ANCILLARY_PER_ORDER:
    for _chip in ("GREEN", "RED"):
        _L2_CHIP_PREFIX_KEYS[f"{_chip}_{_ext}"] = (_ext, _chip)


class _KPF2DataDict(ChipPrefixDict):
    """L2 data dict: chip-prefix views over the trace and per-order ancillaries.

    ``d["GREEN_SCI2_FLUX"]`` returns ``d["SCI2_FLUX"][:NORDER_GREEN]``, a numpy
    view into the first 35 orders of TRACE3_FLUX.
    """

    _PREFIX_KEYS = _L2_CHIP_PREFIX_KEYS


class KPF2(KPFDataModel):
    """
    KPF Level 2 extracted spectra data model.

    Built from ``L2-extensions.csv``, with KPF-friendly extension aliases and
    per-chip access; EPRV-standard names remain canonical and aliases are
    transparent synonyms. Each trace holds the green and red orders concatenated
    (green first); a GREEN_/RED_ prefix returns a numpy view of that chip's
    orders. For example, ``data["SCI2_FLUX"]`` is ``data["TRACE3_FLUX"]`` and
    ``data["GREEN_SCI2_FLUX"]`` returns its green orders.
    """

    _ALIAS_TEMPLATES = tuple((f"TRACE#_{s}", s) for s in _TRACE_SUFFIXES)

    _DATA_DICT = _KPF2DataDict

    def __init__(self):
        super().__init__()
        self.level = 2
        self._build()

    def _fill_typed_empty_tables(self):
        """Add L2's own empty table to the base skeletons.

        ORDER_TABLE takes its columns from ``config/L2-ORDER_TABLE-columns.csv``,
        as the ``RV#`` tables take theirs at L4. Every L2 manifest declares it --
        the science one and both ML2 masters -- so it needs no membership gate.
        """
        super()._fill_typed_empty_tables()
        self.set_data(
            "ORDER_TABLE",
            pd.DataFrame(columns=_ORDER_TABLE_COLUMNS["Name"].tolist()),
        )

    def to_kpf4(self):
        """
        Create a KPF4 scaffold from this KPF2, carrying over headers and receipt.

        RV and CCF data extensions are created but empty -- the caller (RV
        computation) fills those in.
        """
        kpf4 = KPF4()

        # Mirrors to_kpf2: PRIMARY overlays onto kpf4's EPRV seed (native wins), the
        # rest are verbatim copies. The receipt *table* propagates via the copy
        # below, as does CATALOG_RECORD's table -- its rows carry forward too.
        self._forward_headers(
            kpf4,
            (
                "PRIMARY",
                "INSTRUMENT_HEADER",
                "QUALITY_CONTROL",
                "CATALOG_RECORD",
                "RECEIPT",
            ),
        )
        if self.data.get("CATALOG_RECORD") is not None:
            kpf4.set_data("CATALOG_RECORD", self.data["CATALOG_RECORD"])

        if self.receipt is not None and not self.receipt.empty:
            kpf4.receipt = self.receipt.copy()
        kpf4.obs_id = self.obs_id

        kpf4.set_keyword("DATALVL", "L4")
        kpf4.receipt_add_entry("to_kpf4", "", "PASS")
        return kpf4
