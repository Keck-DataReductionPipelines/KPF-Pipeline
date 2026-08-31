"""
KPF Level 4 (RVs and CCFs) data model.

Radial velocities and cross-correlation functions. Built from
``L4-extensions.csv`` and carrying KPF-friendly extension aliases for the CCF
and RV data, following the same pattern as KPF2 -- data can be accessed by
either EPRV name
(``CCF3``, ``RV3``) or KPF name (``SCI2_CCF``, ``SCI2_RV``), with per-chip
views for the CCF cubes (``GREEN_SCI2_CCF``, ``RED_SCI2_CCF``).
"""

import pandas as pd

from kpfpipe.data_models.aliased_dict import (
    NORDER_GREEN,
    ChipPrefixDict,
)
from kpfpipe.data_models.base import TRACE_MAP, KPFDataModel
from kpfpipe.data_models.config import PATH as _config_path

# Re-exported: NORDER_GREEN is the chip split point, defined once beside the
# chip-prefix dict that applies it.
__all__ = ["KPF4", "NORDER_GREEN"]

_RV_COLUMNS = pd.read_csv(_config_path / "L4-RV-columns.csv")

# Build a set of valid chip-prefix keys for fast membership testing.
# e.g., {"GREEN_SCI2_CCF": ("SCI2_CCF", "GREEN"),
#        "GREEN_SCI2_RV": ("SCI2_RV", "GREEN")}.
# Each maps a chip-prefixed key -> (fiber_alias, chip).
_L4_CHIP_PREFIX_KEYS = {}
for _, _row in TRACE_MAP.iterrows():
    _fiber = str(_row["Fiber"]).strip()
    for _suffix in ("CCF", "CCF_VAR", "RV"):
        _fiber_alias = f"{_fiber}_{_suffix}"
        for _chip in ("GREEN", "RED"):
            _L4_CHIP_PREFIX_KEYS[f"{_chip}_{_fiber_alias}"] = (_fiber_alias, _chip)


class _KPF4DataDict(ChipPrefixDict):
    """L4 data dict: chip-prefix views over the CCF cubes and RV tables.

    ``d["GREEN_SCI2_CCF"]`` returns ``d["SCI2_CCF"][:NORDER_GREEN]``, a numpy
    view into the first 35 orders of CCF3 (order axis is axis 0, like the trace
    flux arrays). ``d["GREEN_SCI2_RV"]`` returns the green rows of the SCI2_RV
    table; RV chip-prefix access is read-only, since each RV table is written
    whole (one BinTable per orderlet).
    """

    _PREFIX_KEYS = _L4_CHIP_PREFIX_KEYS
    _READONLY_BASES = ("_RV",)


class KPF4(KPFDataModel):
    """
    KPF Level 4 RV and CCF data model.

    Built from ``L4-extensions.csv``, with KPF-friendly extension aliases and
    per-chip access for the CCF cubes; EPRV-standard names (``CCF1...N``,
    ``RV1...N``) remain canonical and aliases are transparent synonyms. Each CCF
    holds the green and red orders concatenated (green first); a GREEN_/RED_
    prefix returns a numpy view of that chip's orders, while RV tables hold one
    row per order. For example, ``data["SCI2_CCF"]`` is ``data["CCF3"]`` and
    ``data["SCI2_RV"]`` is ``data["RV3"]``.
    """

    _ALIAS_TEMPLATES = (("CCF#", "CCF"), ("CCF#_VAR", "CCF_VAR"), ("RV#", "RV"))

    _DATA_DICT = _KPF4DataDict

    def __init__(self):
        super().__init__()
        self.level = 4
        self._build()

    def _fill_typed_empty_tables(self):
        """Add L4's own empty tables to the base skeletons.

        All five ``RV#`` tables get the same skeleton from
        ``config/L4-RV-columns.csv``, so a dark fiber ships the same shape as an
        illuminated one.
        """
        super()._fill_typed_empty_tables()
        rv_columns = _RV_COLUMNS["Name"].tolist()
        for trace_num in range(1, 6):
            ext = f"RV{trace_num}"
            if ext in self.extensions:
                self.set_data(ext, pd.DataFrame(columns=rv_columns))
