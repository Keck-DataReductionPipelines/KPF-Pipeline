"""
Header/extension keyword registry for the KPF data models.

Single home for the header keyword reference data, organized by the three
use-cases it serves. ``KeywordRegistry`` builds every lookup once in
``__init__`` from one source-of-truth table; the module exposes a single
instance, ``keyword_registry``. ``KPFDataModel`` (data_models/base.py) is the
only module that imports this one; it surfaces the instance as a class attribute
so consumers handed a ``kpf_obj`` (the checkpoints validator, level0's
WMKO->EPRV mapping, tests) reach the registry through ``kpf.keyword_registry``
rather than importing from here.

Source of truth: ``self.table``, one DataFrame unioning the KPF
``L{0,1,2,4}-headers.csv`` registries with the EPRV keyword definitions (rvdata's
``LEVEL2/4_PRIMARY_KEYWORDS`` plus the per-extension keyword CSVs). Columns:
``Keyword, Description, Extension, DataType, PopulatedBy, Required, Level, Default,
Units`` (``Default``/``Units`` carry the EPRV CSV values for EPRV rows, ``""`` for
KPF rows). The ``routing``/``allowed``/``required``/``eprv_primary_*`` lookups are
derived from this table.

The three use-cases:
  (1) Mapping  — ``header_map`` (WMKO->EPRV), consumed by ``KPF0._map_header``;
      ``eprv_primary_datatypes`` types the values it emits. ``header_map`` is
      **sanitized on load** so it holds only genuine static native->EPRV mappings:
      rows whose target is unregistered (PARANG/PARANG2) or handled elsewhere
      (``_HEADER_MAP_NON_NATIVE``) are dropped, so ``_map_header`` needs no
      in-loop filter or per-keyword correction.
  (2) Validation — ``allowed`` / ``required`` (per-extension, from the table)
      plus ``structural`` (FITS bookkeeping cards).
  (3) Routing — ``routing`` (keyword -> (extension, comment)), consumed by
      ``KPFDataModel.set_keyword``.

It also exposes ``eprv_primary_seed`` (the typed EPRV Required PRIMARY skeleton
``KPF1.__init__`` stamps, mirroring rvdata's ``RV2.__init__``). The header_map
corrections live in one place each: ``NUMORDER``/``DRPTAG`` via ``_DEFAULT_OVERRIDES``
(table sanitization), ``DATALVL`` via ``KPF1.__init__`` (the model level), and the
``JD_UTC`` epoch transform in ``_map_header`` (a per-frame value, not a static default).
"""

import importlib.resources
import warnings
from types import MappingProxyType

import pandas as pd
from rvdata.core.models.definitions import (
    LEVEL2_PRIMARY_KEYWORDS,
    LEVEL4_PRIMARY_KEYWORDS,
)
from rvdata.core.tools.headers import parse_value_to_datatype

from kpfpipe import DETECTOR, __version__

_kpf_cfg = importlib.resources.files("rvdata.instruments.kpf.config")
_rv_cfg = importlib.resources.files("rvdata.core.models.config")
_kpf_data_cfg = importlib.resources.files("kpfpipe.data_models.config")

# Number of echelle orders (green + red); the value header_map.csv gets wrong (65).
_NUMORDER = int(DETECTOR["norder"]["GREEN"]) + int(DETECTOR["norder"]["RED"])


class KeywordRegistry:
    """Owns the KPF keyword reference data and the lookups derived from it.

    Built once at import (the module exposes the singleton ``keyword_registry``).
    All attributes are read-only reference data; see the module docstring for the
    three use-cases (mapping / validation / routing).
    """

    # Sentinel written into the "PopulatedBy" column for EPRV-sourced rows; the
    # derived lookups use it to tell EPRV rows from KPF rows, so it is the single
    # source of that string (and _kpf_rows asserts no KPF row reuses it).
    _EPRV_TAG = "EPRV"

    # Unified-table column order. All names are valid Python identifiers so the
    # derived lookups can read rows by attribute via itertuples (row.PopulatedBy).
    _COLUMNS = [
        "Keyword",
        "Description",
        "Extension",
        "DataType",
        "PopulatedBy",
        "Required",
        "Level",
        "Default",
        "Units",
    ]

    # header_map.csv STANDARD targets that are NOT genuine static native->EPRV
    # mappings, so they are dropped when header_map is sanitized on load (and the
    # raw header_map default/native they carried is wrong or vestigial). Each one's
    # real value home: NUMORDER/DRPTAG -> _DEFAULT_OVERRIDES (table), DATALVL ->
    # KPF1.__init__ (the model's data level), JD_UTC -> the _map_header epoch
    # transform (a per-frame value, not a static default).
    _HEADER_MAP_NON_NATIVE = frozenset({"NUMORDER", "DATALVL", "DRPTAG", "JD_UTC"})

    # Table Default corrections applied during the __init__ sanitize phase, for
    # keywords header_map.csv gets wrong (NUMORDER's default is 65; KPF has 67) or
    # whose default is runtime (DRPTAG is the DRP version). Stored as strings like
    # every other Default (the column is string-typed); parse_value_to_datatype
    # types them. After sanitization the table is clean, so _eprv_primary_lookup
    # seeds them without special-casing.
    _DEFAULT_OVERRIDES = {"NUMORDER": str(_NUMORDER), "DRPTAG": __version__}

    # Per-extension EPRV keyword CSVs (rvdata does not load these as constants).
    # RV#/CCF# share one template CSV; BARYCORR_*/BJD_TDB are per-extension.
    _EPRV_EXT_CSV = {
        "BJD_TDB": (_rv_cfg / "L2-BJD_TDB-keywords.csv", 2),
        "BARYCORR_KMS": (_rv_cfg / "L2-BARYCORR_KMS-keywords.csv", 2),
        "BARYCORR_Z": (_rv_cfg / "L2-BARYCORR_Z-keywords.csv", 2),
        **{f"RV{i}": (_rv_cfg / "L4-RV1-keywords.csv", 4) for i in range(1, 6)},
        **{f"CCF{i}": (_rv_cfg / "L4-CCF1-keywords.csv", 4) for i in range(1, 6)},
    }

    # Registry "PopulatedBy" values that mark a QUALITY_CONTROL row as a 0/1 QC
    # flag. The generic "QC" tags the cross-level aggregate ISGOOD; "QCL{n}" tags
    # a level-N check. ``qc_flag_keywords`` unions them (drives the ISGOOD
    # aggregate); ``qc_flag_keywords_by_level`` keys the level-N checks by their
    # LEVEL tag (drives the per-level checkpoint, which flags only its own checks).
    _QC_POPULATORS = frozenset({"QC", "QCL0", "QCL1", "QCL2"})

    # FITS structural / bookkeeping cards that are always permitted on any
    # extension and are never registered keywords. This is the single source of
    # truth for "structural" (see is_structural); the checkpoint validator reads
    # it rather than keeping its own list. Two parts: exact-match cards below, and
    # the WCS / binary-table column-descriptor families in _STRUCTURAL_PREFIXES
    # (matched by prefix because they are enumerated, e.g. NAXIS1/TTYPE3). Only
    # genuine FITS cards belong here -- EPRV keywords (DATALVL) live in the
    # registry table, KPF keywords (ORIGID) are registered to their home extension.
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
        "FILENAME",
        "DATE",
        "EXTNAME",
        "TFIELDS",
    }

    # Structural card families astropy adds when serializing an extension header
    # (binary-table column descriptors, image WCS); matched by prefix.
    _STRUCTURAL_PREFIXES = (
        "NAXIS",
        "TTYPE",
        "TFORM",
        "TUNIT",
        "TDIM",
        "TDISP",
        "TNULL",
        "TSCAL",
        "TZERO",
        "CTYPE",
        "CUNIT",
        "CRPIX",
        "CRVAL",
        "CDELT",
        "CROTA",
    )

    def __init__(self):
        # --- Read the raw reference inputs ---
        # The unified keyword table (EPRV rows first, then KPF rows; KPF wins a
        # collision) and the rvdata WMKO-native -> EPRV-standard header_map.
        eprv_rows, kpf_rows = self._build_rows()
        table = pd.DataFrame(eprv_rows + kpf_rows, columns=self._COLUMNS)
        header_map = pd.read_csv(_kpf_cfg / "header_map.csv")

        # --- Sanitize the inputs once, here, before any lookup is derived ---
        # Correct the Defaults header_map.csv gets wrong (NUMORDER 65 -> 67) or that
        # are runtime (DRPTAG -> version); these feed eprv_primary_seed, so the build
        # step reads a clean table with no per-keyword special-casing.
        for keyword, value in self._DEFAULT_OVERRIDES.items():
            table.loc[table["Keyword"] == keyword, "Default"] = value
        self.table = table
        # The keyword allowlist -- a distributed lookup, and the gate header_map
        # sanitization filters against (so it is derived before that runs).
        self.registered = frozenset(self.table["Keyword"])
        # Drop header_map rows that aren't genuine static native -> EPRV mappings.
        self.header_map = self._sanitize_header_map(header_map)

        # --- Build / distribute the derived lookups from the sanitized table ---
        # All are read-only reference data shared process-wide via the singleton;
        # freeze them (frozenset / MappingProxyType) so a stray consumer mutation
        # can't corrupt the registry for everyone (notably under parallel tests).
        # self.table is only ever read.
        self.routing = MappingProxyType(self._routing_lookup())
        allowed, required = self._validation_lookup()
        self.allowed = MappingProxyType(
            {ext: frozenset(kws) for ext, kws in allowed.items()}
        )
        self.required = MappingProxyType(
            {ext: MappingProxyType(d) for ext, d in required.items()}
        )
        self.structural = frozenset(self._STRUCTURAL)
        # QC-flag keyword sets: the full set (ISGOOD aggregate) and the per-level
        # split (current-level checkpoint scope). See _qc_flag_sets_lookup.
        qc_all, qc_by_level = self._qc_flag_sets_lookup()
        self.qc_flag_keywords = frozenset(qc_all)
        self.qc_flag_keywords_by_level = MappingProxyType(
            {lvl: frozenset(kws) for lvl, kws in qc_by_level.items()}
        )
        # EPRV PRIMARY skeleton: the typed Required seed (KPF1.__init__ stamps it,
        # mirroring RV2.__init__) and the datatypes _map_header types its emitted
        # values with. See _eprv_primary_lookup.
        seed, datatypes = self._eprv_primary_lookup()
        self.eprv_primary_seed = MappingProxyType(seed)
        self.eprv_primary_datatypes = MappingProxyType(datatypes)

    def is_structural(self, key):
        """True for a FITS structural / bookkeeping card (never a registered keyword).

        The single structural test, consumed by the checkpoint header validator: a
        card is structural if it is an exact-match bookkeeping card (``structural``)
        or belongs to a WCS / binary-table column-descriptor family
        (``_STRUCTURAL_PREFIXES``, e.g. ``NAXIS2``/``TTYPE3``).
        """
        k = str(key).strip()
        return k in self.structural or k.startswith(self._STRUCTURAL_PREFIXES)

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
        ``PopulatedBy`` column — the discriminator the derived lookups use to
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
            # Default/Units feed the typed PRIMARY seed (eprv_primary_seed); kept
            # raw (NaN-as-is) here, the seed builder normalizes them.
            default = df["Default"].iloc[i] if "Default" in df else ""
            units = df["Units"].iloc[i] if "Units" in df else ""
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
                            cls._EPRV_TAG,
                            required and j == 1,
                            level,
                            default,
                            units,
                        ]
                    )
            else:
                rows.append(
                    [
                        name,
                        descr,
                        extension,
                        dtype,
                        cls._EPRV_TAG,
                        required,
                        level,
                        default,
                        units,
                    ]
                )
        return rows

    @classmethod
    def _kpf_rows(cls):
        """KPF-pipeline keyword rows from config/L{0,1,2,4}-headers.csv.

        ``Required`` is always False; ``PopulatedBy`` is whatever the CSV says.
        It must NEVER be the EPRV sentinel: the derived routing/validation lookups
        treat a ``PopulatedBy == _EPRV_TAG`` row as EPRV-sourced, so a KPF row
        reusing that string would be silently misclassified (dropped from routing,
        given the wrong Required/Level). Guard against it loudly here.
        """
        rows = []
        for level in (0, 1, 2, 4):
            df = pd.read_csv(_kpf_data_cfg / f"L{level}-headers.csv")
            for _, r in df.iterrows():
                descr = (
                    "" if pd.isna(r["Description"]) else str(r["Description"]).strip()
                )
                populated_by = str(r.get("PopulatedBy", "")).strip()
                if populated_by == cls._EPRV_TAG:
                    raise ValueError(
                        f"L{level}-headers.csv: keyword "
                        f"{str(r['Keyword']).strip()!r} has 'PopulatedBy' == "
                        f"{cls._EPRV_TAG!r}, which is reserved as the EPRV-row "
                        "discriminator; use a real populating site instead"
                    )
                rows.append(
                    [
                        str(r["Keyword"]).strip(),
                        descr,
                        str(r["Extension"]).strip(),
                        str(r.get("DataType", "")).strip(),
                        populated_by,
                        False,
                        level,
                        "",  # Default: EPRV-only attribute, blank for KPF rows
                        "",  # Units: EPRV-only attribute, blank for KPF rows
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

    def _routing_lookup(self):
        """keyword -> (home extension, comment), derived from ``self.table``.

        Write targets only: EPRV PRIMARY keywords (-> PRIMARY) and every KPF
        keyword (its explicit Extension); KPF wins a name collision. EPRV
        per-extension cards (RVMETHOD on RV#, CTYPE*, ...) are validation-only,
        never set_keyword targets, so they are excluded.
        """
        routing = {}
        # EPRV PRIMARY rows first (setdefault), then KPF rows override (KPF wins).
        for row in self.table.itertuples(index=False):
            if row.PopulatedBy == self._EPRV_TAG and row.Extension == "PRIMARY":
                routing.setdefault(row.Keyword, ("PRIMARY", row.Description))
        for row in self.table.itertuples(index=False):
            if row.PopulatedBy != self._EPRV_TAG:
                routing[row.Keyword] = (row.Extension, row.Description)
        return routing

    def _validation_lookup(self):
        """Per-extension allowed / required lookups, derived from ``self.table``.

        allowed: every keyword registered for an extension (no level gate).
        required: keyword -> minimal Level it is Required at (PRIMARY warnings
        filter Level<=N).
        """
        allowed = {}
        required = {}
        for row in self.table.itertuples(index=False):
            extn = row.Extension
            allowed.setdefault(extn, set()).add(row.Keyword)
            if row.Required:
                d = required.setdefault(extn, {})
                d[row.Keyword] = min(d.get(row.Keyword, row.Level), row.Level)
        return allowed, required

    def _eprv_primary_lookup(self):
        """EPRV PRIMARY lookups, derived from ``self.table``.

        Returns ``(seed, datatypes)``:

        - ``seed`` — ``{keyword: (typed_default, comment)}`` for the EPRV Required
          PRIMARY keywords at Level <= 2 (the skeleton ``KPF1.__init__`` stamps).
          Built exactly like ``RV2.__init__``: format the unit/description comment,
          then type the default via ``parse_value_to_datatype`` so consumers assign
          it straight into a header. Insertion order follows the standard's.
        - ``datatypes`` — ``{keyword: DataType}`` for *all* EPRV PRIMARY keywords
          (required + optional), so ``_map_header`` can type the native/default
          values it overlays. Scoped to EPRV PRIMARY (rvdata-vocab datatypes), so
          it never feeds KPF's ``int``/``str`` to ``parse_value_to_datatype``.
        """
        seed = {}
        datatypes = {}
        for row in self.table.itertuples(index=False):
            if row.PopulatedBy != self._EPRV_TAG or row.Extension != "PRIMARY":
                continue
            datatypes[row.Keyword] = row.DataType
            if not (row.Required and row.Level <= 2):
                continue
            units = None if pd.isna(row.Units) else str(row.Units).strip()
            unitstr = "" if not units or units.lower() == "n/a" else f"[{units}] "
            comment = f"{unitstr}{row.Description}"
            # The table is already sanitized (_DEFAULT_OVERRIDES applied in __init__),
            # so the Default is read straight, no per-keyword correction here.
            default = None if pd.isna(row.Default) else row.Default
            seed[row.Keyword] = parse_value_to_datatype(
                row.Keyword, row.DataType, (default, comment)
            )
        return seed, datatypes

    def _qc_flag_sets_lookup(self):
        """QC-flag keyword sets, derived from ``self.table``.

        Returns
        -------
        tuple
            ``(all_flags, by_level)``. ``all_flags`` is every QUALITY_CONTROL row
            tagged by a QC populator (used for the cross-level ISGOOD aggregate).
            ``by_level`` maps a LEVEL tag ("L0"/"L1"/"L2") to that level's own
            ``QCL{n}`` flags (used by the per-level checkpoint). The generic "QC"
            aggregate keyword (ISGOOD) is in ``all_flags`` only — in no per-level
            set, so the checkpoint never warns/raises on the aggregate itself.
        """
        all_flags = set()
        by_level = {}
        for row in self.table.itertuples(index=False):
            if row.Extension != "QUALITY_CONTROL" or row.PopulatedBy not in (
                self._QC_POPULATORS
            ):
                continue
            all_flags.add(row.Keyword)
            if row.PopulatedBy != "QC":  # "QCL0"/"QCL1"/"QCL2" -> "L0"/"L1"/"L2"
                by_level.setdefault(row.PopulatedBy[2:], set()).add(row.Keyword)
        return all_flags, by_level

    # --- Input sanitization (run once in __init__ before the lookups) --------

    def _sanitize_header_map(self, raw):
        """Return header_map with only genuine static native->EPRV mapping rows.

        _map_header then applies the map with no in-loop filter or per-keyword
        correction. Two row classes are dropped:
          - STANDARD targets absent from the registry (PARANG/PARANG2) -- warned,
            so the rvdata header_map / registry inconsistency stays visible;
          - ``_HEADER_MAP_NON_NATIVE`` targets, whose value comes from the seed
            (NUMORDER/DRPTAG), the model level (DATALVL), or the _map_header epoch
            transform (JD_UTC), not a static map row.
        """
        standard = raw["STANDARD"].astype(str).str.strip()
        unregistered = sorted(
            set(standard[raw["STANDARD"].notna()]) - self.registered - {""}
        )
        if unregistered:
            warnings.warn(
                "header_map.csv maps to STANDARD targets absent from the keyword "
                f"registry; they are dropped (not emitted): {unregistered}",
                UserWarning,
                stacklevel=2,
            )
        keep = standard.isin(self.registered) & ~standard.isin(
            self._HEADER_MAP_NON_NATIVE
        )
        return raw[keep].reset_index(drop=True)

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
