"""
Header/extension keyword registry for the KPF data models.

Single home for the header keyword reference data, organized by the three
use-cases it serves. ``KeywordRegistry`` builds every lookup once in
``__init__`` from one source-of-truth table; the module exposes a single
instance, ``keyword_registry``. ``KPFDataModel`` (data_models/base.py) is the
only module that imports this one; it surfaces the instance as a class attribute
so consumers handed a ``kpf_obj`` (the qc_booleans validator, level0's
WMKO->EPRV mapping, tests) reach the registry through ``kpf.keyword_registry``
rather than importing from here.

Source of truth: ``self.table``, one DataFrame unioning the KPF
``L{0,1,2,4}-headers.csv`` registries with the EPRV keyword definitions (rvdata's
``LEVEL2/4_PRIMARY_KEYWORDS`` plus the per-extension keyword CSVs). Columns:
``Keyword, Description, Extension, DataType, Populated by, Required, Level``. The
``routing``/``allowed``/``required`` lookups are derived from this table.

The three use-cases:
  (1) Mapping  — ``header_map`` (WMKO->EPRV), consumed by ``KPF0.wmko_to_eprv``,
      filtered through ``registered``.
  (2) Validation — ``allowed`` / ``required`` (per-extension, from the table)
      plus ``structural`` (FITS bookkeeping cards).
  (3) Routing — ``routing`` (keyword -> (extension, comment)), consumed by
      ``KPFDataModel.set_keyword``.
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


class KeywordRegistry:
    """Owns the KPF keyword reference data and the lookups derived from it.

    Built once at import (the module exposes the singleton ``keyword_registry``).
    All attributes are read-only reference data; see the module docstring for the
    three use-cases (mapping / validation / routing).
    """

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

    # Per-extension EPRV keyword CSVs (rvdata does not load these as constants).
    # RV#/CCF# share one template CSV; BARYCORR_*/BJD_TDB are per-extension.
    _EPRV_EXT_CSV = {
        "BJD_TDB": (_rv_cfg / "L2-BJD_TDB-keywords.csv", 2),
        "BARYCORR_KMS": (_rv_cfg / "L2-BARYCORR_KMS-keywords.csv", 2),
        "BARYCORR_Z": (_rv_cfg / "L2-BARYCORR_Z-keywords.csv", 2),
        **{f"RV{i}": (_rv_cfg / "L4-RV1-keywords.csv", 4) for i in range(1, 6)},
        **{f"CCF{i}": (_rv_cfg / "L4-CCF1-keywords.csv", 4) for i in range(1, 6)},
    }

    # FITS structural / bookkeeping cards plus KPF-internal PRIMARY keys that are
    # neither EPRV nor in the registry but are always permitted.
    _STRUCTURAL = {
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

    def __init__(self):
        # Source table (EPRV rows first, then KPF rows; KPF wins a collision).
        eprv_rows, kpf_rows = self._build_rows()
        self.table = pd.DataFrame(eprv_rows + kpf_rows, columns=self._COLUMNS)
        # Every registered keyword (the allowlist; also the wmko_to_eprv filter).
        self.registered = set(self.table["Keyword"])
        # Derived lookups (all read off self.table — no parallel row lists).
        self.routing = self._build_routing()
        self.allowed, self.required = self._build_validation()
        self.structural = set(self._STRUCTURAL)
        # (1) Mapping: the rvdata WMKO-native -> EPRV-standard header_map.
        self.header_map = pd.read_csv(_kpf_cfg / "header_map.csv")
        self._warn_unregistered_targets()

    # --- Source table construction -------------------------------------------

    @staticmethod
    def _load_keyword_csv(handle):
        """Read an rvdata keyword CSV, stripping any UTF-8 BOM on column 1."""
        df = pd.read_csv(handle)
        df.columns = [str(c).lstrip("﻿").strip() for c in df.columns]
        return df

    @staticmethod
    def _family_base(first_token):
        """'CDEC1' -> 'CDEC' (strip the trailing index)."""
        return first_token.rstrip("0123456789")

    @classmethod
    def _eprv_rows(cls, df, extension, level):
        """Expand an EPRV keyword CSV into unified-registry rows.

        EPRV CSVs encode per-target/per-telescope families as a "BASE1 ... BASE#"
        template; expand each to literal rows BASE1-9 (only index 1 inherits the
        Required flag, mirroring rvdata's seed). EPRV rows carry ``"EPRV"`` in the
        ``Populated by`` column — the discriminator the derived lookups use to
        tell EPRV rows from KPF rows.
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
                base = cls._family_base(name.split("...")[0].strip())
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

    @staticmethod
    def _kpf_rows():
        """KPF-pipeline keyword rows from config/L{0,1,2,4}-headers.csv.

        ``Required`` is always False; ``Populated by`` is whatever the CSV says
        (never the literal ``"EPRV"``).
        """
        rows = []
        for level in (0, 1, 2, 4):
            df = pd.read_csv(_kpf_data_cfg / f"L{level}-headers.csv")
            for _, r in df.iterrows():
                descr = (
                    "" if pd.isna(r["Description"]) else str(r["Description"]).strip()
                )
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

    @classmethod
    def _build_rows(cls):
        """Build the EPRV and KPF row lists that union into ``self.table``."""
        # EPRV PRIMARY: L2 (Level 2) plus the L4-only extras (Level 4); never both.
        l2 = cls._eprv_rows(LEVEL2_PRIMARY_KEYWORDS, "PRIMARY", 2)
        l2_keys = {r[0] for r in l2}
        l4 = [
            r
            for r in cls._eprv_rows(LEVEL4_PRIMARY_KEYWORDS, "PRIMARY", 4)
            if r[0] not in l2_keys
        ]
        # EPRV per-extension keywords (governed extensions with standard cards).
        ext = []
        for name, (path, level) in cls._EPRV_EXT_CSV.items():
            ext += cls._eprv_rows(cls._load_keyword_csv(path), name, level)
        return l2 + l4 + ext, cls._kpf_rows()

    # --- Derived lookups (all read self.table) -------------------------------

    def _build_routing(self):
        """keyword -> (home extension, comment), derived from ``self.table``.

        Write targets only: EPRV PRIMARY keywords (-> PRIMARY) and every KPF
        keyword (its explicit Extension); KPF wins a name collision. EPRV
        per-extension cards (RVMETHOD on RV#, CTYPE*, ...) are validation-only,
        never set_keyword targets, so they are excluded.

        Column order (``_COLUMNS``): 0 Keyword, 1 Description, 2 Extension,
        3 DataType, 4 Populated by, 5 Required, 6 Level.
        """
        routing = {}
        # EPRV PRIMARY rows first (setdefault), then KPF rows override (KPF wins).
        for row in self.table.itertuples(index=False):
            if row[4] == "EPRV" and row[2] == "PRIMARY":
                routing.setdefault(row[0], ("PRIMARY", row[1]))
        for row in self.table.itertuples(index=False):
            if row[4] != "EPRV":
                routing[row[0]] = (row[2], row[1])
        return routing

    def _build_validation(self):
        """Per-extension allowed / required lookups, derived from ``self.table``.

        allowed: every keyword registered for an extension (no level gate).
        required: keyword -> minimal Level it is Required at (PRIMARY warnings
        filter Level<=N). Column order as in ``_build_routing``.
        """
        allowed = {}
        required = {}
        for row in self.table.itertuples(index=False):
            kw, extn, req, lvl = row[0], row[2], row[5], row[6]
            allowed.setdefault(extn, set()).add(kw)
            if req:
                d = required.setdefault(extn, {})
                d[kw] = min(d.get(kw, lvl), lvl)
        return allowed, required

    def _warn_unregistered_targets(self):
        """Warn once if header_map maps to STANDARD targets we don't register.

        wmko_to_eprv filters its output to ``registered``, so these would be
        silently dropped; surfacing the rvdata header_map / registry inconsistency
        (e.g. PARANG, PARANG2) keeps the drop intentional, not invisible.
        """
        unregistered = sorted(
            {
                str(r["STANDARD"]).strip()
                for _, r in self.header_map.iterrows()
                if pd.notna(r["STANDARD"])
            }
            - self.registered
            - {""}
        )
        if unregistered:
            warnings.warn(
                "header_map.csv maps to STANDARD targets absent from the keyword "
                f"registry; wmko_to_eprv will not emit them: {unregistered}",
                UserWarning,
                stacklevel=2,
            )

    # --- rvdata extension registration ---------------------------------------

    @staticmethod
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


# Module singleton — the one registry instance every consumer reaches through.
keyword_registry = KeywordRegistry()
