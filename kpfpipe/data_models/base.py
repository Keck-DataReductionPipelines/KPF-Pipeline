"""
KPF-specific base data model.

Thin layer on top of RVDataModel that adds KPF-specific attributes and the
behaviour shared by every KPF data model: fits.Header storage, the DRPSTATU
receipt stamp, alias-aware set_data/set_header, and lossless PRIMARY
serialization. All four KPF models inherit it — L0/L1 directly, and L2/L4 via
multiple inheritance alongside rvdata's RV2/RV4 (KPFDataModel listed first so
its overrides win while RV2/RV4 remain reachable through ``super()``).
"""

import importlib.resources

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from rvdata.core.models.base import RVDataModel

from kpfpipe.utils.kpf import _DATECODE_PATTERN, _OBS_ID_PATTERN

# --- WMKO ↔ EPRV ↔ KPF PRIMARY header reference data -------------------------
# Single source of truth for the WMKO→EPRV keyword mapping (rvdata's
# header_map.csv) and the sets that define what may legitimately appear on a KPF
# EPRV PRIMARY header. Loaded once at import. ``validate_eprv_primary`` (below)
# consumes the derived keyword sets; ``KPF0.wmko_to_eprv`` (level0.py) imports
# ``HEADER_MAP`` from here for the L0→L1 conversion.
#
# rvdata (installed package): the wmko↔EPRV map and the EPRV PRIMARY keyword
# definitions. KPF consumes these rather than re-encoding the standard.
_kpf_cfg = importlib.resources.files("rvdata.instruments.kpf.config")
_rv_cfg = importlib.resources.files("rvdata.core.models.config")
# This repo: the KPF-pipeline keyword registry (validator allowlist).
_kpf_data_cfg = importlib.resources.files("kpfpipe.data_models.config")

HEADER_MAP = pd.read_csv(_kpf_cfg / "header_map.csv")


def _load_keyword_csv(handle):
    """Read an rvdata PRIMARY-keyword CSV, stripping any UTF-8 BOM on column 1."""
    df = pd.read_csv(handle)
    df.columns = [str(c).lstrip("﻿").strip() for c in df.columns]
    return df


_L2_KW = _load_keyword_csv(_rv_cfg / "L2-PRIMARY-keywords.csv")
_L4_KW = _load_keyword_csv(_rv_cfg / "L4-PRIMARY-keywords.csv")
# KPF-pipeline keyword registry, split by the level that first writes the
# keyword (config/L{0,1,2,4}-headers.csv). All levels are combined into one
# allowlist, since a keyword written at an earlier level persists on PRIMARY.
_KPFPIPE_KW = pd.concat(
    [
        pd.read_csv(_kpf_data_cfg / f"L{level}-headers.csv")
        for level in ("0", "1", "2", "4")
    ],
    ignore_index=True,
)


# The EPRV keyword CSVs encode per-target/per-telescope families as a
# "BASE1 ... BASE#" template (e.g. "CDEC1 ... CDEC#"). rvdata seeds these
# expanded at index 1 (CDEC1); a producer may add CDEC2.. up to NUMTRACE/NUMTEL.
def _family_base(first_token):
    """'CDEC1' -> 'CDEC' (strip the trailing index)."""
    return first_token.rstrip("0123456789")


def _keyword_literals(df):
    """All EPRV keyword names, expanding 'BASE1 ... BASE#' families to BASE1-9."""
    out = set()
    for kw in df["Keyword"]:
        name = str(kw).strip()
        if not name:
            continue
        if "..." in name:
            base = _family_base(name.split("...")[0].strip())
            out.update(f"{base}{i}" for i in range(1, 10))
        else:
            out.add(name)
    return out


def _required_literals(df):
    """Required EPRV keyword names; families require only index 1 (rvdata's seed)."""
    flags = df["Required"].astype(str).str.strip().str.lower() == "true"
    out = set()
    for kw in df.loc[flags, "Keyword"]:
        name = str(kw).strip()
        if not name:
            continue
        if "..." in name:
            out.add(f"{_family_base(name.split('...')[0].strip())}1")
        else:
            out.add(name)
    return out


# Every EPRV-standard PRIMARY keyword (L4 = L2 plus the L4-only RV summary keys).
EPRV_L2_KEYS = _keyword_literals(_L2_KW)
EPRV_L4_KEYS = EPRV_L2_KEYS | _keyword_literals(_L4_KW)
REQUIRED_L2_KEYS = _required_literals(_L2_KW)
REQUIRED_L4_KEYS = REQUIRED_L2_KEYS | _required_literals(_L4_KW)

# header_map STANDARD names are the EPRV-side targets the conversion produces.
HEADERMAP_STANDARD_KEYS = {
    str(r["STANDARD"]).strip()
    for _, r in HEADER_MAP.iterrows()
    if pd.notna(r["STANDARD"])
}
# header_map INSTRUMENT names that differ from their STANDARD target are the raw
# WMKO names that must NOT remain on a converted PRIMARY (they belong in
# INSTRUMENT_HEADER). Used for a precise "wmko leak" diagnostic.
WMKO_PRIMARY_KEYS = {
    str(r["INSTRUMENT"]).strip()
    for _, r in HEADER_MAP.iterrows()
    if pd.notna(r["INSTRUMENT"])
    and str(r["INSTRUMENT"]).strip()
    and str(r["INSTRUMENT"]).strip() != str(r["STANDARD"]).strip()
}

# KPF-pipeline keywords legitimately written to PRIMARY. Each registry entry
# is an explicit 8-character-max FITS keyword (no wildcards); families are
# enumerated per member (e.g. RNGREEN1-4, CCD1BJD/CCD2BJD).
KPFPIPE_PRIMARY_KEYS = {str(k).strip() for k in _KPFPIPE_KW["Keyword"]}

# FITS structural / bookkeeping cards plus KPF-internal PRIMARY keys that are
# neither EPRV nor in the registry but are always permitted.
_STRUCTURAL_KEYS = {
    "SIMPLE",
    "BITPIX",
    "EXTEND",
    "XTENSION",
    "PCOUNT",
    "GCOUNT",
    "BSCALE",
    "BZERO",
    "COMMENT",
    "HISTORY",
    "CONTINUE",
    "CHECKSUM",
    "DATASUM",
    "",
    "DATALVL",
    "ORIGID",
    "FILENAME",
    "DATE",
}

# Receipt names that are data-model conversions / serialization rather than
# pipeline modules — excluded from DRPSTATU so it names the last real stage.
# ``from_fits`` is here too: reading a product back must not clobber the status
# the writer stamped.
_INTERNAL_RECEIPTS = frozenset({"to_l1", "to_kpf2", "to_kpf4", "to_fits", "from_fits"})


class KPFDataModel(RVDataModel):
    """Shared base for every KPF data model (L0, L1, and — multiply-inherited
    with RV2/RV4 — L2, L4)."""

    OBS_ID_PATTERN = _OBS_ID_PATTERN
    DATECODE_PATTERN = _DATECODE_PATTERN

    def __init__(self):
        super().__init__()
        self.obs_id = None

    @staticmethod
    def as_fits_header(src):
        """Return ``src`` as an ``astropy.io.fits.Header``, preserving comments.

        KPF stores every extension header as a ``fits.Header`` so reads and writes
        go through astropy natively, with no value-vs-``(value, comment)``
        ambiguity. This is the single bridge from the two legacy in-memory forms:

        - a ``fits.Header`` (e.g. from ``from_fits``) is returned as a copy, so a
          caller can rebuild an HDU without aliasing the stored header;
        - a plain mapping (RVData seeds PRIMARY defaults as an ``OrderedDict`` whose
          entries are ``(value, comment)`` tuples) is rebuilt card by card —
          ``head[kw] = (value, comment)`` sets the value and comment together.
        """
        if isinstance(src, fits.Header):
            return src.copy()
        head = fits.Header()
        for keyword, content in src.items():
            head[keyword] = content
        return head

    def create_extension(self, ext_name, ext_type, header=None, data=None):
        """Create an extension, storing its header as a ``fits.Header``.

        rvdata initializes a new header as a plain ``OrderedDict``; KPF keeps every
        header as a ``fits.Header`` so all reads/writes are native astropy.
        """
        super().create_extension(ext_name, ext_type, header=header, data=data)
        self.headers[ext_name] = self.as_fits_header(self.headers[ext_name])

    def set_data(self, ext_name, data):
        """Set extension data, resolving KPF aliases first.

        For aliased models (KPF2/KPF4) this resolves chip-prefix keys (e.g.
        'GREEN_SCI2_FLUX', routed through the data dict's ``__setitem__``) and
        extension aliases before the base class ``.keys()`` check. The
        ``hasattr`` guards make it a no-op passthrough for non-aliased L0/L1.
        """
        if (
            hasattr(self.data, "_chip_split")
            and self.data._chip_split(ext_name) is not None
        ):
            self.data[ext_name] = data
            return
        if hasattr(self.extensions, "_resolve"):
            ext_name = self.extensions._resolve(ext_name)
        # astropy reads BinTableHDUs back as numpy record arrays; convert to Table.
        if (
            ext_name in self.extensions
            and self.extensions[ext_name] == "BinTableHDU"
            and isinstance(data, np.ndarray)
            and data.dtype.names is not None
        ):
            data = Table(data)
        super().set_data(ext_name, data)
        # Sync self.receipt when the RECEIPT extension is loaded from FITS.
        if ext_name == "RECEIPT" and isinstance(data, Table):
            self.receipt = data.to_pandas()

    def set_header(self, ext_name, header):
        """Set an extension header, resolving KPF aliases before the base class
        ``.keys()`` check (a no-op for non-aliased L0/L1)."""
        if hasattr(self.extensions, "_resolve"):
            ext_name = self.extensions._resolve(ext_name)
        super().set_header(ext_name, header)

    def receipt_add_entry(self, module, status):
        """Record a processing step, and update DRPSTATU for pipeline modules."""
        super().receipt_add_entry(module, status)
        if status == "PASS":
            self._update_drpstatus(module)

    def _update_drpstatus(self, module):
        """Stamp DRPSTATU = '<Module Name> module complete' for a completed module.

        Called from ``receipt_add_entry``; conversions/serialization receipts
        (``_INTERNAL_RECEIPTS``) are skipped so DRPSTATU names the last real stage.
        """
        if module in _INTERNAL_RECEIPTS:
            return
        primary = self.headers.get("PRIMARY")
        if primary is None:
            return
        label = module.replace("_", " ").title()
        primary["DRPSTATU"] = (
            f"{label} module complete",
            "DRP reduction status (DRP-RUN-20)",
        )

    def _create_hdul(self):
        """Sync self.receipt into the RECEIPT extension before writing; rvdata
        serializes self.data["RECEIPT"], not self.receipt. L0/L1 omit RECEIPT
        from their default extensions, so create it if absent."""
        if self.receipt is not None and not self.receipt.empty:
            if "RECEIPT" not in self.extensions:
                self.create_extension("RECEIPT", "BinTableHDU")
            self.data["RECEIPT"] = Table.from_pandas(self.receipt)
        return self._restore_primary_comments(super()._create_hdul())

    def _restore_primary_comments(self, hdu_list):
        """Rebuild the PRIMARY HDU so its keyword comments survive serialization.

        RVData's ``_create_hdul`` builds PRIMARY by iterating
        ``headers["PRIMARY"].items()``, which on a ``fits.Header`` yields only
        ``(key, value)`` and silently drops the comments. Replace that one HDU with
        a fresh ``PrimaryHDU`` built from the stored header (a ``fits.Header`` that
        holds the comments). The ``PrimaryHDU`` constructor re-adds the structural
        cards.
        """
        primary = self.headers.get("PRIMARY")
        if primary is None:
            return hdu_list
        for i, hdu in enumerate(hdu_list):
            if isinstance(hdu, fits.PrimaryHDU):
                hdu_list[i] = fits.PrimaryHDU(header=self.as_fits_header(primary))
                break
        return hdu_list

    @staticmethod
    def validate_eprv_primary(header, level):
        """Validate a converted EPRV PRIMARY header, raising on any inconsistency.

        The shared fail-loud guard (no silent fallback) for the L1->L2 and
        L2->L4 boundaries; called by ``KPF1.to_kpf2`` / ``KPF2.to_kpf4``. Three
        rules:

        1. **WMKO leak** — a raw WMKO keyword name (header_map INSTRUMENT name
           differing from its EPRV target) is present on PRIMARY; it should have
           been converted or left in INSTRUMENT_HEADER.
        2. **Unregistered keyword** — a card that is neither an EPRV-standard
           keyword, a header_map STANDARD target, a registered KPF-pipeline
           keyword (``config/L{0,1,2,4}-headers.csv``), nor a structural card.
        3. **Missing required** — a Required EPRV PRIMARY keyword is absent.

        Parameters
        ----------
        header : Mapping
            The PRIMARY header to validate.
        level : str
            ``"L2"`` or ``"L4"`` (selects the EPRV keyword set).

        Raises
        ------
        ValueError
            On the first inconsistency found.
        """
        level = str(level).upper()
        eprv_keys = EPRV_L4_KEYS if level == "L4" else EPRV_L2_KEYS
        required_keys = REQUIRED_L4_KEYS if level == "L4" else REQUIRED_L2_KEYS

        for raw_key in list(header):
            key = str(raw_key).strip()
            if key in _STRUCTURAL_KEYS or key.startswith("NAXIS"):
                continue
            if (
                key in eprv_keys
                or key in HEADERMAP_STANDARD_KEYS
                or key in KPFPIPE_PRIMARY_KEYS
            ):
                continue
            if key in WMKO_PRIMARY_KEYS:
                raise ValueError(
                    f"native WMKO keyword {key!r} found on {level} PRIMARY; it must "
                    "be converted to its EPRV name or kept in INSTRUMENT_HEADER"
                )
            raise ValueError(
                f"unregistered keyword {key!r} on {level} PRIMARY; add it to "
                "config/L{0,1,2,4}-headers.csv or fix the writer"
            )

        missing = sorted(k for k in required_keys if k not in header)
        if missing:
            raise ValueError(
                f"missing required EPRV PRIMARY keyword(s) on {level}: {missing}"
            )

    def generate_standard_filename(self):
        """Abstract: every concrete KPF model builds its own standard filename.

        KPFDataModel is never instantiated directly — only inherited — so reaching
        this means a subclass failed to define the method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define generate_standard_filename"
        )

    def check_filename_convention(self, filename):
        """Abstract: every concrete KPF model declares its own filename convention.

        KPFDataModel is never instantiated directly — only inherited — so reaching
        this means a subclass failed to define the method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define check_filename_convention"
        )
