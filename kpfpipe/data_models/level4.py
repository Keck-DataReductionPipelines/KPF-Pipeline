"""
KPF Level 4 (RVs and CCFs) data model.

Inherits from RV4 (EPRV standard) and adds KPF-friendly extension aliases
for CCF and RV data, following the same AliasedOrderedDict pattern as KPF2.

The EPRV standard defines CCF1...N (ImageHDU) and RV1...N (BinTableHDU), one
per orderlet. KPF numbers both to match the L2 traces (CCF{n}/RV{n} hold the
CCF/RV for the orderlet on TRACE{n}): n=1 CAL, 2 SCI1, 3 SCI2, 4 SCI3, 5 SKY.
The KPF name → extension mapping is therefore derived from the shared trace map
(trace-map.csv), exactly like the L2 TRACE*_FLUX aliases — not from aliases.csv.

Each CCF stores green+red orders concatenated (green first), like TRACE*_FLUX,
so per-chip access mirrors KPF2. RV tables hold one row per order (no chip
prefix — slice by ORDER_INDEX instead):

    kpf4.data["SCI2_CCF"]        is kpf4.data["CCF3"]        # True (alias)
    kpf4.data["GREEN_SCI2_CCF"]  # CCF3[:NORDER_GREEN]  (green orders, a view)
    kpf4.data["RED_SCI2_CCF"]    # CCF3[NORDER_GREEN:]  (red orders, a view)
    kpf4.data["SCI2_RV"]         is kpf4.data["RV3"]         # True (alias)
"""

import importlib.resources

import numpy as np
import pandas as pd
from astropy.table import Table
from rvdata.core.models.level4 import RV4

from kpfpipe import DETECTOR
from kpfpipe.data_models.aliased_dict import AliasedOrderedDict

NORDER_GREEN = DETECTOR['norder']['GREEN']
NORDER_RED   = DETECTOR['norder']['RED']
NORDER       = NORDER_GREEN + NORDER_RED

_config_path = importlib.resources.files("kpfpipe.data_models.config")
# CCF and RV extensions reuse the shared trace map (CCF{n}/RV{n} <-> TRACE{n});
# the trace-to-fiber mapping is identical for L2 and L4. aliases.csv holds only
# the non-trace 1:1 aliases (shared with L2), matching that convention.
_TRACE_MAP = pd.read_csv(_config_path / "trace-map.csv")
_ALIASES = pd.read_csv(_config_path / "aliases.csv")

# Build a set of valid chip-prefix keys for fast membership testing.
# e.g., {"GREEN_SCI2_CCF": ("SCI2_CCF", "GREEN"), "GREEN_SCI2_RV": ("SCI2_RV", "GREEN")}.
# Each maps a chip-prefixed key -> (fiber_alias, chip). CCF cubes are sliced on
# their order axis (axis 0) and support chip-prefix read and write; RV tables are
# row-sliced (green = rows 0:NORDER_GREEN, red the rest) and support read only —
# each is written whole (one BinTable per orderlet), so a chip-prefix write raises.
_CHIP_PREFIX_KEYS = {}  # chip-prefixed key → (fiber_alias, chip)
for _, _row in _TRACE_MAP.iterrows():
    _fiber = str(_row["Fiber"]).strip()
    for _suffix in ("CCF", "RV"):
        _fiber_alias = f"{_fiber}_{_suffix}"
        for _chip in ("GREEN", "RED"):
            _CHIP_PREFIX_KEYS[f"{_chip}_{_fiber_alias}"] = (_fiber_alias, _chip)


class _KPF4DataDict(AliasedOrderedDict):
    """Data dict supporting GREEN_/RED_ chip-prefix access for CCF cubes and RV tables.

    Accessing d["GREEN_SCI2_CCF"] returns d["SCI2_CCF"][:NORDER_GREEN], a numpy
    view into the first 35 orders of CCF3 (order axis is axis 0, like the trace
    flux arrays). d["GREEN_SCI2_RV"] returns the green rows of the SCI2_RV table
    (rows 0:NORDER_GREEN); RV chip-prefix access is read-only, since each RV
    table is written whole.
    """

    def _chip_split(self, key):
        """If key is a chip-prefix pattern, return (fiber_alias, chip), else None."""
        return _CHIP_PREFIX_KEYS.get(key)

    def __setitem__(self, key, value):
        split = self._chip_split(key)
        if split is not None:
            fiber_alias, chip = split
            # RV tables are written whole (one BinTable per orderlet); a
            # chip-prefixed RV key is read-only.
            if fiber_alias.endswith('_RV'):
                raise KeyError(
                    f"chip-prefixed RV key {key!r} is read-only; write the full "
                    f"table via {fiber_alias!r} (rows are green-then-red)")
            resolved = self._resolve(fiber_alias)
            # Allocate the full concatenated cube on first write (or if empty).
            # value.shape[1:] keeps this correct for the (norder_chip, nvel) CCF.
            existing = super().__getitem__(resolved) if super().__contains__(resolved) else None
            if existing is None or np.size(existing) == 0:
                full = np.zeros((NORDER_GREEN + NORDER_RED, *value.shape[1:]), dtype=value.dtype)
                super().__setitem__(resolved, full)
            arr = super().__getitem__(resolved)
            if chip == 'GREEN':
                arr[:NORDER_GREEN] = value
            else:
                arr[NORDER_GREEN:] = value
        else:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        split = self._chip_split(key)
        if split is not None:
            fiber_alias, chip = split
            data = super().__getitem__(self._resolve(fiber_alias))
            return data[:NORDER_GREEN] if chip == "GREEN" else data[NORDER_GREEN:]
        return super().__getitem__(self._resolve(key))

    def __contains__(self, key):
        split = self._chip_split(key)
        if split is not None:
            fiber_alias, _ = split
            return super().__contains__(self._resolve(fiber_alias))
        return super().__contains__(self._resolve(key))

    def get(self, key, default=None):
        split = self._chip_split(key)
        if split is not None:
            fiber_alias, chip = split
            resolved = self._resolve(fiber_alias)
            if not super().__contains__(resolved):
                return default
            data = super().__getitem__(resolved)
            return data[:NORDER_GREEN] if chip == "GREEN" else data[NORDER_GREEN:]
        return super().get(self._resolve(key), default)

    @classmethod
    def from_ordered_dict(cls, od):
        """Create a _KPF4DataDict from an existing OrderedDict."""
        from collections import OrderedDict
        aliased = cls()
        for key, value in od.items():
            OrderedDict.__setitem__(aliased, key, value)
        return aliased


class KPF4(RV4):
    """
    KPF Level 4 RV and CCF data model.

    Extends RV4 with KPF-friendly extension aliases and per-chip access for the
    CCF cubes. EPRV-standard extension names (CCF1...N, RV1...N) remain
    canonical; aliases are transparent synonyms.

    Each CCF holds green+red orders concatenated (35 green + 32 red = 67 orders
    total). Per-chip access via the GREEN_/RED_ prefix returns numpy views into
    the concatenated array. RV tables hold one row per order.

    Alias examples:
        kpf4.data["SCI2_CCF"]    is kpf4.data["CCF3"]   # True
        kpf4.data["SCI2_RV"]     is kpf4.data["RV3"]    # True

    Per-chip access:
        kpf4.data["GREEN_SCI2_CCF"]   # CCF3[:35]  (green orders, a numpy view)
        kpf4.data["RED_SCI2_CCF"]     # CCF3[35:]  (red orders, a numpy view)
        kpf4.data["GREEN_SCI2_RV"]    # RV3[:35]   (green rows; read-only)
        kpf4.data["RED_SCI2_RV"]      # RV3[35:]   (red rows; read-only)
    """

    def __init__(self):
        super().__init__()

        # RV4 creates only the required CCF1 / RV1; KPF stores one CCF and one
        # RV table per orderlet (CCF{n}/RV{n} <-> TRACE{n}).
        for trace_num in range(1, 6):
            for prefix, hdu_type in (("CCF", "ImageHDU"), ("RV", "BinTableHDU")):
                ext = f"{prefix}{trace_num}"
                if ext not in self.extensions:
                    self.create_extension(ext, hdu_type)

        # Replace plain OrderedDicts with alias-aware versions
        self.extensions = AliasedOrderedDict.from_ordered_dict(self.extensions)
        self.headers = AliasedOrderedDict.from_ordered_dict(self.headers)
        self.data = _KPF4DataDict.from_ordered_dict(self.data)

        self._register_aliases()

    def _register_aliases(self):
        """Register KPF-friendly aliases from config CSVs."""
        # Simple 1:1 non-trace extension aliases (shared with L2; none apply to
        # L4 today, but kept for consistency with KPF2).
        for _, row in _ALIASES.iterrows():
            alias = str(row["KPF"]).strip()
            canonical = str(row["EPRV"]).strip()
            if canonical in self.extensions:
                for d in (self.extensions, self.headers, self.data):
                    d.register_alias(alias, canonical)

        # Per-orderlet CCF/RV aliases derived from the trace map
        # (CCF{n}/RV{n} ↔ TRACE{n}): e.g. SCI2_CCF → CCF3, SCI2_RV → RV3.
        for _, row in _TRACE_MAP.iterrows():
            trace_num = int(row["Trace"])
            fiber = str(row["Fiber"]).strip()
            for prefix in ("CCF", "RV"):
                canonical = f"{prefix}{trace_num}"
                alias = f"{fiber}_{prefix}"
                if canonical in self.extensions:
                    for d in (self.extensions, self.headers, self.data):
                        d.register_alias(alias, canonical)

    def set_data(self, ext_name, data):
        """Override to resolve aliases before the base class .keys() check.
        Chip-prefix keys (e.g. 'GREEN_SCI2_CCF') are routed directly through
        _KPF4DataDict.__setitem__, which writes into the appropriate slice of
        the concatenated CCF cube.
        """
        if hasattr(self.data, '_chip_split') and self.data._chip_split(ext_name) is not None:
            self.data[ext_name] = data
            return
        if hasattr(self.extensions, '_resolve'):
            ext_name = self.extensions._resolve(ext_name)
        # astropy reads BinTableHDUs back as numpy record arrays; convert to Table.
        if (ext_name in self.extensions
                and self.extensions[ext_name] == "BinTableHDU"
                and isinstance(data, np.ndarray)
                and data.dtype.names is not None):
            data = Table(data)
        super().set_data(ext_name, data)
        # Sync self.receipt when the RECEIPT extension is loaded from FITS.
        if ext_name == "RECEIPT" and isinstance(data, Table):
            self.receipt = data.to_pandas()

    def set_header(self, ext_name, header):
        """Override to resolve aliases before the base class .keys() check."""
        if hasattr(self.extensions, '_resolve'):
            ext_name = self.extensions._resolve(ext_name)
        super().set_header(ext_name, header)

    def _create_hdul(self):
        """Override to sync self.receipt into self.data["RECEIPT"] before writing.

        rvdata's to_fits writes self.data["RECEIPT"] (the default empty table),
        not self.receipt (the processing history DataFrame). This override syncs
        them so the full receipt is written to the FITS file.
        """
        if self.receipt is not None and not self.receipt.empty:
            self.data["RECEIPT"] = Table.from_pandas(self.receipt)
        return super()._create_hdul()

    def info(self):
        """Print summary of KPF4 data model contents."""
        if self.filename:
            print(f"KPF L4: {self.filename}")
        else:
            print("Empty KPF4 data product")

        print(f"\n{'Extension':<25s} {'Aliases':<25s} {'Type':<15s} {'Shape/Size':<20s}")
        print("=" * 85)
        for name, ext_type in self.extensions.items():
            if name == "PRIMARY":
                n_cards = len(self.headers.get(name, {}))
                print(f"{'PRIMARY':<25s} {'':<25s} {'header':<15s} {n_cards} cards")
                continue
            aliases = self.extensions.aliases_for(name)
            alias_str = ", ".join(sorted(aliases)) if aliases else ""
            ext = self.data.get(name)
            if isinstance(ext, np.ndarray) and ext.size > 0:
                print(f"{name:<25s} {alias_str:<25s} {'array':<15s} {str(ext.shape):<20s}")
            elif hasattr(ext, "__len__") and len(ext) > 0:
                print(f"{name:<25s} {alias_str:<25s} {'table':<15s} {len(ext)} rows")
            else:
                print(f"{name:<25s} {alias_str:<25s} {ext_type:<15s} {'(empty)':<20s}")
