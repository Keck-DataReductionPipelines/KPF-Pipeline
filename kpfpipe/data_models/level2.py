"""
KPF Level 2 (extracted spectra) data model.

Inherits from RV2 (EPRV standard) and adds KPF-friendly extension aliases
so pipeline code can use either EPRV names (TRACE3_FLUX) or KPF names
(SCI2_FLUX). Per-chip access (GREEN_SCI2_FLUX) is provided via green()
and red() convenience methods that slice traces by order.

The alias mechanism (AliasedOrderedDict) is generic and could be
upstreamed into the rvdata standard.
"""

import importlib.resources

import numpy as np
import pandas as pd
from rvdata.core.models.level2 import RV2

from kpfpipe.data_models.aliased_dict import AliasedOrderedDict

_config_path = importlib.resources.files("kpfpipe.data_models.config")
_TRACE_MAP = pd.read_csv(_config_path / "L2-trace-map.csv")
_L2_ALIASES = pd.read_csv(_config_path / "L2-aliases.csv")

# Data types that each trace carries
_TRACE_DTYPES = ["FLUX", "WAVE", "VAR", "BLAZE"]


class KPF2(RV2):
    """
    KPF Level 2 extracted spectra data model.

    Extends RV2 with KPF-friendly extension aliases and per-chip
    access methods. EPRV-standard extension names remain canonical;
    aliases are transparent synonyms.

    Alias examples:
        kpf2.data["SCI2_FLUX"]   is kpf2.data["TRACE3_FLUX"]   # True
        kpf2.data["CAL_WAVE"]    is kpf2.data["TRACE1_WAVE"]    # True
        kpf2.data["CA_HK"]       is kpf2.data["ANCILLARY_SPECTRUM"]  # True

    Per-chip access:
        kpf2.green("SCI2", "FLUX")  # green orders of TRACE3_FLUX
        kpf2.red("SCI2", "FLUX")    # red orders of TRACE3_FLUX
    """

    def __init__(self):
        super().__init__()

        # RV2 creates only TRACE1 by default; KPF uses 5 traces
        for trace_num in range(2, 6):
            for dtype in _TRACE_DTYPES:
                ext = f"TRACE{trace_num}_{dtype}"
                if ext not in self.extensions:
                    self.create_extension(ext, "ImageHDU")

        # Pass-through extensions not in RV2 base
        for ext, ext_type in [("ANCILLARY_SPECTRUM", "BinTableHDU"),
                               ("EXPMETER", "BinTableHDU"),
                               ("TELEMETRY", "BinTableHDU")]:
            if ext not in self.extensions:
                self.create_extension(ext, ext_type)

        # Replace plain OrderedDicts with alias-aware versions
        self.extensions = AliasedOrderedDict.from_ordered_dict(self.extensions)
        self.headers = AliasedOrderedDict.from_ordered_dict(self.headers)
        self.data = AliasedOrderedDict.from_ordered_dict(self.data)

        self._register_aliases()

    def _register_aliases(self):
        """Register KPF-friendly aliases from config CSVs."""
        # Simple 1:1 extension aliases (e.g., CA_HK → ANCILLARY_SPECTRUM)
        for _, row in _L2_ALIASES.iterrows():
            alias = str(row["Alias"]).strip()
            canonical = str(row["Canonical"]).strip()
            if canonical in self.extensions:
                self.extensions.register_alias(alias, canonical)
                self.headers.register_alias(alias, canonical)
                self.data.register_alias(alias, canonical)

        # Fiber-based trace aliases (e.g., SCI2_FLUX → TRACE3_FLUX)
        for _, row in _TRACE_MAP.iterrows():
            trace_num = int(row["Trace"])
            fiber = str(row["Fiber"]).strip()
            for dtype in _TRACE_DTYPES:
                canonical = f"TRACE{trace_num}_{dtype}"
                alias = f"{fiber}_{dtype}"
                if canonical in self.extensions:
                    self.extensions.register_alias(alias, canonical)
                    self.headers.register_alias(alias, canonical)
                    self.data.register_alias(alias, canonical)

    def set_data(self, ext_name, data):
        """Override to resolve aliases before the base class .keys() check."""
        if hasattr(self.extensions, '_resolve'):
            ext_name = self.extensions._resolve(ext_name)
        super().set_data(ext_name, data)

    def set_header(self, ext_name, header):
        """Override to resolve aliases before the base class .keys() check."""
        if hasattr(self.extensions, '_resolve'):
            ext_name = self.extensions._resolve(ext_name)
        super().set_header(ext_name, header)

    @property
    def n_green_orders(self):
        """Number of green orders, derived from ORDER_TABLE wavelength data.

        Green orders have wavelengths below ~5900 A. Returns None if
        ORDER_TABLE is not populated.
        """
        order_table = self.data.get("ORDER_TABLE")
        if order_table is None or len(order_table) == 0:
            return None
        if hasattr(order_table, "to_pandas"):
            df = order_table.to_pandas()
        else:
            df = order_table
        if "WAVE_END" not in df.columns or df["WAVE_END"].isna().all():
            return None
        # Green CCD covers roughly < 5900 A
        return int((df["WAVE_END"] < 5900).sum())

    def _chip_slice(self, fiber, dtype, chip):
        """Return the green or red slice of a trace array.

        Args:
            fiber: Fiber name (e.g., "SCI2", "CAL", "SKY")
            dtype: Data type (e.g., "FLUX", "WAVE", "VAR", "BLAZE")
            chip: "GREEN" or "RED"

        Returns:
            numpy array slice (view into the trace data)
        """
        alias = f"{fiber}_{dtype}"
        trace_data = self.data[alias]
        n_green = self.n_green_orders
        if n_green is None:
            raise ValueError(
                "Cannot determine green/red split: ORDER_TABLE not populated"
            )
        if chip == "GREEN":
            return trace_data[:n_green]
        elif chip == "RED":
            return trace_data[n_green:]
        else:
            raise ValueError(f"chip must be 'GREEN' or 'RED', got '{chip}'")

    def green(self, fiber, dtype):
        """Return the green-chip portion of a trace.

        Args:
            fiber: Fiber name (e.g., "SCI2", "CAL", "SKY")
            dtype: Data type (e.g., "FLUX", "WAVE", "VAR", "BLAZE")

        Returns:
            numpy array view of the green orders
        """
        return self._chip_slice(fiber, dtype, "GREEN")

    def red(self, fiber, dtype):
        """Return the red-chip portion of a trace.

        Args:
            fiber: Fiber name (e.g., "SCI2", "CAL", "SKY")
            dtype: Data type (e.g., "FLUX", "WAVE", "VAR", "BLAZE")

        Returns:
            numpy array view of the red orders
        """
        return self._chip_slice(fiber, dtype, "RED")

    def info(self):
        """Print summary of KPF2 data model contents."""
        if self.filename:
            print(f"KPF L2: {self.filename}")
        else:
            print("Empty KPF2 data product")

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
