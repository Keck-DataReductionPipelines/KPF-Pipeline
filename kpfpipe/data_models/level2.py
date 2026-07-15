"""
KPF Level 2 (extracted spectra) data model.

Extracted, wavelength-calibrated spectra. Extends the EPRV RV2 model with
KPF-friendly extension aliases, so data can be accessed by either EPRV name
(``TRACE3_FLUX``) or KPF name (``SCI2_FLUX``), including per-chip views
(``GREEN_SCI2_FLUX``, ``RED_SCI2_FLUX``).
"""

import importlib.resources
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
from rvdata.core.models.definitions import LEVEL2_EXTENSIONS
from rvdata.core.models.level2 import RV2

from kpfpipe import DETECTOR
from kpfpipe.data_models.aliased_dict import AliasedOrderedDict
from kpfpipe.data_models.base import KPFDataModel, keyword_registry
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.utils.io import kpf_filename

# Make rvdata's RV2._read aware of KPF's QUALITY_CONTROL extension so an L2
# written with it reads back (KPF2.__init__ creates the empty extension).
keyword_registry.register_rvdata_extension(
    LEVEL2_EXTENSIONS,
    "QUALITY_CONTROL",
    "BinTableHDU",
    "Quality-control booleans and diagnostic metrics",
)

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]

logger = logging.getLogger(__name__)

_config_path = importlib.resources.files("kpfpipe.data_models.config")
_TRACE_MAP = pd.read_csv(_config_path / "trace-map.csv")
_ALIASES = pd.read_csv(_config_path / "aliases.csv")

# Extension name suffixes for each trace (e.g., TRACE3_FLUX, TRACE3_WAVE)
_TRACE_SUFFIXES = ["FLUX", "WAVE", "VAR", "BLAZE"]

# Per-order ancillary extensions (not traces) that also support chip-prefix
# access, e.g. GREEN_BARYCORR_Z -> BARYCORR_Z[:NORDER_GREEN]. These are 1-D
# (norder,) arrays aligned with the concatenated trace orders.
_ANCILLARY_PER_ORDER = ["BJD_TDB", "BARYCORR_KMS", "BARYCORR_Z"]

# Build a set of valid chip-prefix keys for fast membership testing.
# e.g., {"GREEN_CAL_FLUX", "RED_CAL_FLUX", "GREEN_SCI1_FLUX", "GREEN_BARYCORR_Z", ...}
_CHIP_PREFIX_KEYS = {}  # chip-prefixed key → (base_key, chip)
for _, _row in _TRACE_MAP.iterrows():
    _fiber = str(_row["Fiber"]).strip()
    for _suffix in _TRACE_SUFFIXES:
        _fiber_alias = f"{_fiber}_{_suffix}"
        for _chip in ("GREEN", "RED"):
            _CHIP_PREFIX_KEYS[f"{_chip}_{_fiber}_{_suffix}"] = (_fiber_alias, _chip)
for _ext in _ANCILLARY_PER_ORDER:
    for _chip in ("GREEN", "RED"):
        _CHIP_PREFIX_KEYS[f"{_chip}_{_ext}"] = (_ext, _chip)


class _KPF2DataDict(AliasedOrderedDict):
    """
    Data dict that supports GREEN_/RED_ chip-prefix access.

    Accessing ``d["GREEN_SCI2_FLUX"]`` returns ``d["SCI2_FLUX"][:NORDER_GREEN]``,
    a numpy view into the first 35 orders of TRACE3_FLUX.
    """

    def _chip_split(self, key):
        """
        If key is a chip-prefix pattern, return (fiber_alias, chip).

        Returns None if key is not a chip-prefix pattern.
        """
        return _CHIP_PREFIX_KEYS.get(key)

    def __setitem__(self, key, value):
        split = self._chip_split(key)
        if split is not None:
            fiber_alias, chip = split
            resolved = self._resolve(fiber_alias)
            # Allocate the full concatenated array on first write (or if empty).
            # value.shape[1:] keeps this correct for 2-D traces (norder, ncol)
            # and 1-D per-order ancillary arrays (norder,).
            existing = (
                super().__getitem__(resolved)
                if super().__contains__(resolved)
                else None
            )
            if existing is None or np.size(existing) == 0:
                full = np.zeros(
                    (NORDER_GREEN + NORDER_RED, *value.shape[1:]), dtype=value.dtype
                )
                super().__setitem__(resolved, full)
            arr = super().__getitem__(resolved)
            if chip == "GREEN":
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
            if chip == "GREEN":
                return data[:NORDER_GREEN]
            else:
                return data[NORDER_GREEN:]
        return super().__getitem__(self._resolve(key))

    def __contains__(self, key):
        if self._chip_split(key) is not None:
            fiber_alias, _ = self._chip_split(key)
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
            if chip == "GREEN":
                return data[:NORDER_GREEN]
            else:
                return data[NORDER_GREEN:]
        return super().get(self._resolve(key), default)

    @classmethod
    def from_ordered_dict(cls, od):
        """Create a _KPF2DataDict from an existing OrderedDict."""
        aliased = cls()
        for key, value in od.items():
            OrderedDict.__setitem__(aliased, key, value)
        return aliased


class KPF2(KPFDataModel, RV2):
    """
    KPF Level 2 extracted spectra data model.

    Extends RV2 with KPF-friendly extension aliases and per-chip access;
    EPRV-standard names remain canonical and aliases are transparent
    synonyms. Each trace holds the green and red orders concatenated (green
    first); a GREEN_/RED_ prefix returns a numpy view of that chip's orders.
    For example, ``data["SCI2_FLUX"]`` is ``data["TRACE3_FLUX"]`` and
    ``data["GREEN_SCI2_FLUX"]`` returns its green orders.
    """

    def __init__(self):
        super().__init__()

        # RV2 creates only TRACE1 by default; KPF uses 5 traces
        for trace_num in range(2, 6):
            for suffix in _TRACE_SUFFIXES:
                ext = f"TRACE{trace_num}_{suffix}"
                if ext not in self.extensions:
                    self.create_extension(ext, "ImageHDU")

        # Pass-through extensions not in the RV2 base. NB: the EPRV standard defines
        # ANCILLARY_SPECTRUM (Ca H&K) as an ImageHDU, but we keep it a BinTableHDU
        # placeholder for now -- Ca H&K extraction is WIP and existing L2/master
        # products encode it as BinTableHDU, so flipping the type breaks reading
        # them back (a deliberate EPRV deviation). QUALITY_CONTROL is
        # created here (L2 has no extensions CSV); RECEIPT and barycentric/RV#
        # already exist via RV2.
        for ext, ext_type in [
            ("ANCILLARY_SPECTRUM", "BinTableHDU"),
            ("EXPMETER", "BinTableHDU"),
            ("TELEMETRY", "BinTableHDU"),
            ("QUALITY_CONTROL", "BinTableHDU"),
        ]:
            if ext not in self.extensions:
                self.create_extension(ext, ext_type)

        # Replace plain OrderedDicts with alias-aware versions
        self.extensions = AliasedOrderedDict.from_ordered_dict(self.extensions)
        self.headers = AliasedOrderedDict.from_ordered_dict(self.headers)
        self.data = _KPF2DataDict.from_ordered_dict(self.data)

        self._register_aliases()

    def _register_aliases(self):
        """Register KPF-friendly aliases from config CSVs."""
        # Simple 1:1 extension aliases (e.g., CA_HK → ANCILLARY_SPECTRUM)
        for _, row in _ALIASES.iterrows():
            alias = str(row["KPF"]).strip()
            canonical = str(row["EPRV"]).strip()
            if canonical in self.extensions:
                self.extensions.register_alias(alias, canonical)
                self.headers.register_alias(alias, canonical)
                self.data.register_alias(alias, canonical)

        # Fiber-based trace aliases (e.g., SCI2_FLUX → TRACE3_FLUX)
        for _, row in _TRACE_MAP.iterrows():
            trace_num = int(row["Trace"])
            fiber = str(row["Fiber"]).strip()
            for suffix in _TRACE_SUFFIXES:
                canonical = f"TRACE{trace_num}_{suffix}"
                alias = f"{fiber}_{suffix}"
                if canonical in self.extensions:
                    self.extensions.register_alias(alias, canonical)
                    self.headers.register_alias(alias, canonical)
                    self.data.register_alias(alias, canonical)

    def check_filename_convention(self, filename):
        """KPF L2 is EPRV-standard (SL2 name); delegate to rvdata's check."""
        return RV2.check_filename_convention(self, filename)

    def generate_standard_filename(self):
        """KPF L2 standard filename (EPRV-standard SL2 name).

        Raises
        ------
        ValueError
            If ``obs_id`` is unset or invalid.
        """
        return kpf_filename(self.obs_id, "L2")

    def to_fits(self, fn=None):
        """KPF keeps a single-filepath ``to_fits``; rvdata >=0.4.0 renamed the
        parameter to ``out_filename``. Delegate so all our call sites can keep
        passing one path (``to_fits(fn)``)."""
        out_path = super().to_fits(out_filename=fn)
        logger.info("wrote %s to %s", type(self).__name__, out_path)
        return out_path

    def to_kpf4(self):
        """
        Create a KPF4 scaffold from this KPF2, carrying over headers and receipt.

        Returns a KPF4 with PRIMARY header keywords forwarded from L2,
        and the receipt chain preserved. RV and CCF data extensions are
        created but empty -- the caller (RV computation) fills those in.
        """
        kpf4 = KPF4()

        # Forward PRIMARY, INSTRUMENT_HEADER, QUALITY_CONTROL, and RECEIPT
        # card-by-card, mirroring to_kpf2: PRIMARY overlays onto kpf4's EPRV seed
        # (native wins), the rest are verbatim copies. The receipt *table*
        # propagates separately via the copy below.
        self._forward_headers(
            kpf4, ("PRIMARY", "INSTRUMENT_HEADER", "QUALITY_CONTROL", "RECEIPT")
        )

        if self.receipt is not None and not self.receipt.empty:
            kpf4.receipt = self.receipt.copy()
        kpf4.obs_id = self.obs_id

        kpf4.set_keyword("DATALVL", "L4")
        kpf4.receipt_add_entry("to_kpf4", "", "PASS")
        return kpf4

    def info(self):
        """Print summary of KPF2 data model contents."""
        if self.filename:
            print(f"KPF L2: {self.filename}")
        else:
            print("Empty KPF2 data product")

        print(
            f"\n{'Extension':<25s} {'Aliases':<25s} {'Type':<15s} {'Shape/Size':<20s}"
        )
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
                print(
                    f"{name:<25s} {alias_str:<25s} {'array':<15s} {str(ext.shape):<20s}"
                )
            elif hasattr(ext, "__len__") and len(ext) > 0:
                print(f"{name:<25s} {alias_str:<25s} {'table':<15s} {len(ext)} rows")
            else:
                print(f"{name:<25s} {alias_str:<25s} {ext_type:<15s} {'(empty)':<20s}")
