"""
WMKO ↔ EPRV ↔ KPF PRIMARY header reference data.

This module is the single source of truth for the WMKO→EPRV keyword mapping
(rvdata's ``header_map.csv``) and the sets that define what may legitimately
appear on a KPF EPRV PRIMARY header. It holds only module-level reference data —
loaded once at import — consumed by the methods that actually convert and
validate:

- ``KPF0.wmko_to_eprv`` / ``KPF0.build_instrument_header`` (``data_models/level0.py``)
  convert the raw WMKO L0 PRIMARY to an EPRV-standard L1 PRIMARY and preserve the
  verbatim raw PRIMARY in ``INSTRUMENT_HEADER``.
- ``KPFDataModel.validate_eprv_primary`` (``data_models/base.py``), called by
  ``KPF1.to_kpf2`` / ``KPF2.to_kpf4``, fails loudly if a raw WMKO keyword leaked
  onto PRIMARY, an unregistered keyword appears, or a required EPRV keyword is
  missing.

KPF stores every extension header as an ``astropy.io.fits.Header``, so reads use
``header.get(key)`` / ``header[key]`` and writes use ``header[key] = (value,
comment)`` natively. KPF-pipeline keywords written directly to PRIMARY (read
noise, QC booleans, diagnostics, RV/barycentric cards, …) are catalogued in the
packaged registry (``config/L{0,1,2,4}-headers.csv``) and allowed by the validator.
"""

import importlib.resources

import pandas as pd

from kpfpipe import DETECTOR

# --- Authoritative configs -------------------------------------------------
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


# --- Derived keyword sets --------------------------------------------------
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
KPFPIPE_PRIMARY_KEYS = {str(k).strip() for k in _KPFPIPE_KW["keyword"]}

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

# Detector truth: KPF has 35 green + 32 red echelle orders (rvdata's header_map
# default of 65 is wrong — see notes/header_audit.md A2).
_NUMORDER = int(DETECTOR["norder"]["GREEN"]) + int(DETECTOR["norder"]["RED"])

# Initial DRPSTATU value on the L1 EPRV PRIMARY, before any pipeline module runs.
# Each module overwrites it via the receipt_add_entry override (see base.py).
_DRPSTATU_DEFAULT = "File ingested into KPF-DRP"
