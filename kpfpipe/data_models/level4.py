"""
KPF Level 4 (RVs and CCFs) data model.

Inherits from RV4 (EPRV standard) and adds KPF-friendly extension aliases
for CCF and RV data. Follows the same AliasedOrderedDict pattern as KPF2.
"""

import numpy as np
from rvdata.core.models.level4 import RV4

from kpfpipe.data_models.aliased_dict import AliasedOrderedDict

# KPF-specific aliases for L4 extensions
_L4_ALIASES = {
    "GREEN_CCF": "CCF1",
    "RED_CCF": "CCF1",
}


class KPF4(RV4):
    """
    KPF Level 4 RV and CCF data model.

    Extends RV4 with KPF-friendly extension aliases for CCF data
    and RV table columns.

    Alias examples:
        kpf4.data["RV"]  # RV1 table via alias
    """

    def __init__(self):
        super().__init__()

        # Replace plain OrderedDicts with alias-aware versions
        self.extensions = AliasedOrderedDict.from_ordered_dict(self.extensions)
        self.headers = AliasedOrderedDict.from_ordered_dict(self.headers)
        self.data = AliasedOrderedDict.from_ordered_dict(self.data)

        self._register_aliases()

    def _register_aliases(self):
        """Register KPF-friendly aliases."""
        # RV table alias
        if "RV1" in self.extensions:
            self.extensions.register_alias("RV", "RV1")
            self.headers.register_alias("RV", "RV1")
            self.data.register_alias("RV", "RV1")

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
