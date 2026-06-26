"""
Header/extension keyword registry for the KPF data models.

Single home for the header keyword reference data, organized by the three
use-cases it serves. ``KPFDataModel`` (data_models/base.py) is the only module
that imports this one; it re-exposes what sibling ``data_models`` files need
(``HEADER_MAP``, ``register_rvdata_extension``, ``REGISTERED_KEYWORDS``) and
surfaces the routing/validation lookups as class attributes, so consumers handed
a ``kpf_obj`` (the qc_booleans validator, tests) read them off the model rather
than importing from here.

Source of truth: ``KEYWORD_REGISTRY``, one DataFrame unioning the KPF
``L{0,1,2,4}-headers.csv`` registries with the EPRV keyword definitions (rvdata's
``LEVEL2/4_PRIMARY_KEYWORDS`` plus the per-extension keyword CSVs). Columns:
``Keyword, Description, Extension, DataType, Populated by, Required, Level``. The
fast lookups below are precomputed views of this table.

The three use-cases:
  (1) Mapping  — ``HEADER_MAP`` (WMKO->EPRV), consumed by ``KPF0.wmko_to_eprv``,
      filtered through ``REGISTERED_KEYWORDS``.
  (2) Validation — ``EXT_ALLOWED`` / ``EXT_REQUIRED`` (per-extension, from the
      table) plus ``STRUCTURAL_KEYS`` (FITS bookkeeping cards).
  (3) Routing — ``KEYWORD_ROUTING`` (keyword -> (extension, comment)), consumed
      by ``KPFDataModel.set_keyword``.
"""

import importlib.resources
import warnings

import pandas as pd
from rvdata.core.models.definitions import (
    LEVEL2_PRIMARY_KEYWORDS,
    LEVEL4_PRIMARY_KEYWORDS,
)

_kpf_cfg = importlib.resources.files("rvdata.instruments.kpf.config")
_rv_cfg = importlib.resources.files("rvdata.core.models.config")
_kpf_data_cfg = importlib.resources.files("kpfpipe.data_models.config")

# Unified-table column order.
_COLUMNS = [
    "Keyword",
    "Description",
    "Extension",
    "DataType",
    "Populated by",
    "Required",
    "Level",
]


def _load_keyword_csv(handle):
    """Read an rvdata keyword CSV, stripping any UTF-8 BOM on column 1."""
    df = pd.read_csv(handle)
    df.columns = [str(c).lstrip("﻿").strip() for c in df.columns]
    return df


def _family_base(first_token):
    """'CDEC1' -> 'CDEC' (strip the trailing index)."""
    return first_token.rstrip("0123456789")


def _eprv_rows(df, extension, level):
    """Expand an EPRV keyword CSV into unified-registry row dicts.

    EPRV CSVs encode per-target/per-telescope families as a "BASE1 ... BASE#"
    template; expand each to literal rows BASE1-9 (only index 1 inherits the
    Required flag, mirroring rvdata's seed).
    """
    rows = []
    req = df["Required"].astype(str).str.strip().str.lower() == "true"
    for i, kw in enumerate(df["Keyword"]):
        name = str(kw).strip()
        if not name:
            continue
        descr = str(df["Description"].iloc[i]) if "Description" in df else ""
        dtype = str(df["DataType"].iloc[i]) if "DataType" in df else ""
        required = bool(req.iloc[i])
        if "..." in name:
            base = _family_base(name.split("...")[0].strip())
            for j in range(1, 10):
                rows.append(
                    [
                        f"{base}{j}",
                        descr,
                        extension,
                        dtype,
                        "EPRV",
                        required and j == 1,
                        level,
                    ]
                )
        else:
            rows.append([name, descr, extension, dtype, "EPRV", required, level])
    return rows


def _kpf_rows():
    """KPF-pipeline keyword rows from config/L{0,1,2,4}-headers.csv (Required=False)."""
    rows = []
    for level in (0, 1, 2, 4):
        df = pd.read_csv(_kpf_data_cfg / f"L{level}-headers.csv")
        for _, r in df.iterrows():
            descr = "" if pd.isna(r["Description"]) else str(r["Description"]).strip()
            rows.append(
                [
                    str(r["Keyword"]).strip(),
                    descr,
                    str(r["Extension"]).strip(),
                    str(r.get("DataType", "")).strip(),
                    str(r.get("Populated by", "")).strip(),
                    False,
                    level,
                ]
            )
    return rows


# Per-extension EPRV keyword CSVs (rvdata does not load these as constants).
# RV#/CCF# share one template CSV; BARYCORR_*/BJD_TDB are per-extension.
_EPRV_EXT_CSV = {
    "BJD_TDB": (_rv_cfg / "L2-BJD_TDB-keywords.csv", 2),
    "BARYCORR_KMS": (_rv_cfg / "L2-BARYCORR_KMS-keywords.csv", 2),
    "BARYCORR_Z": (_rv_cfg / "L2-BARYCORR_Z-keywords.csv", 2),
    **{f"RV{i}": (_rv_cfg / "L4-RV1-keywords.csv", 4) for i in range(1, 6)},
    **{f"CCF{i}": (_rv_cfg / "L4-CCF1-keywords.csv", 4) for i in range(1, 6)},
}


def _build_registry():
    """Build the unified KEYWORD_REGISTRY rows (EPRV first, then KPF)."""
    # EPRV PRIMARY: L2 (Level 2) plus the L4-only extras (Level 4); never both.
    l2 = _eprv_rows(LEVEL2_PRIMARY_KEYWORDS, "PRIMARY", 2)
    l2_keys = {r[0] for r in l2}
    l4 = [
        r
        for r in _eprv_rows(LEVEL4_PRIMARY_KEYWORDS, "PRIMARY", 4)
        if r[0] not in l2_keys
    ]
    # EPRV per-extension keywords (governed extensions that carry standard cards).
    ext = []
    for name, (path, level) in _EPRV_EXT_CSV.items():
        ext += _eprv_rows(_load_keyword_csv(path), name, level)
    eprv = l2 + l4 + ext
    return eprv, _kpf_rows()


_EPRV_ROWS, _KPF_ROWS = _build_registry()
KEYWORD_REGISTRY = pd.DataFrame(_EPRV_ROWS + _KPF_ROWS, columns=_COLUMNS)
# Every registered keyword (the allowlist; also the filter for wmko_to_eprv).
REGISTERED_KEYWORDS = set(KEYWORD_REGISTRY["Keyword"])


# === (3) Routing: keyword -> home extension + comment =========================
# Write targets only: EPRV PRIMARY keywords (-> PRIMARY) and every KPF keyword
# (its explicit Extension); KPF wins a name collision. EPRV per-extension cards
# (RVMETHOD on RV#, CTYPE*, ...) are validation-only, never set_keyword targets.
def _build_routing():
    # Row: [Keyword, Description, Extension, DataType, Populated by, Required, Level]
    routing = {}
    for r in _EPRV_ROWS:
        if r[2] == "PRIMARY":
            routing.setdefault(r[0], ("PRIMARY", r[1]))
    for r in _KPF_ROWS:
        routing[r[0]] = (r[2], r[1])
    return routing


KEYWORD_ROUTING = _build_routing()


# === (2) Validation: per-extension allowed / required =========================
# allowed: every keyword registered for an extension (no level gate). required:
# keyword -> minimal Level it is Required at (PRIMARY warnings filter Level<=N).
def _build_validation():
    # Row: [Keyword, Description, Extension, DataType, Populated by, Required, Level]
    allowed = {}
    required = {}
    for r in _EPRV_ROWS + _KPF_ROWS:
        kw, extn, req, lvl = r[0], r[2], r[5], r[6]
        allowed.setdefault(extn, set()).add(kw)
        if req:
            d = required.setdefault(extn, {})
            d[kw] = min(d.get(kw, lvl), lvl)
    return allowed, required


EXT_ALLOWED, EXT_REQUIRED = _build_validation()

# FITS structural / bookkeeping cards plus KPF-internal PRIMARY keys that are
# neither EPRV nor in the registry but are always permitted.
STRUCTURAL_KEYS = {
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


# === (1) Mapping: WMKO-native -> EPRV-standard ================================
HEADER_MAP = pd.read_csv(_kpf_cfg / "header_map.csv")

# header_map STANDARD targets that are not registered keywords. wmko_to_eprv
# filters its output to REGISTERED_KEYWORDS, so these would be silently dropped;
# warn once at import to surface the rvdata header_map / registry inconsistency.
_UNREGISTERED_HEADERMAP_TARGETS = sorted(
    {
        str(r["STANDARD"]).strip()
        for _, r in HEADER_MAP.iterrows()
        if pd.notna(r["STANDARD"])
    }
    - REGISTERED_KEYWORDS
    - {""}
)
if _UNREGISTERED_HEADERMAP_TARGETS:
    warnings.warn(
        "header_map.csv maps to STANDARD targets absent from the keyword registry; "
        f"wmko_to_eprv will not emit them: {_UNREGISTERED_HEADERMAP_TARGETS}",
        UserWarning,
        stacklevel=2,
    )


def register_rvdata_extension(level_extensions, name, datatype, description):
    """Register a KPF-custom extension into an rvdata ``LEVELn_EXTENSIONS`` table.

    rvdata's ``RVn._read`` resolves each HDU's DataType by Name from its
    ``LEVELn_EXTENSIONS`` DataFrame; a KPF-only extension (e.g. QUALITY_CONTROL)
    is absent there, so reading an Ln product that contains it raises ``KeyError``.
    This appends the row in-memory (idempotent). ``Required`` is False so rvdata
    neither auto-creates it nor lists it in EXT_DESCRIPT — the KPF model
    ``__init__`` creates the (empty) extension explicitly.
    """
    if name in set(level_extensions["Name"]):
        return
    row = {col: "" for col in level_extensions.columns}
    row.update(
        HDU=int(level_extensions["HDU"].max()) + 1,
        Name=name,
        DataType=datatype,
        Required=False,
        Multiplicity=False,
        Description=description,
    )
    level_extensions.loc[len(level_extensions)] = row
