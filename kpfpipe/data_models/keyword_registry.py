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
``L{0,1,2,4}-headers.csv`` registries (plus ``Masters-headers.csv`` for the
out-of-EPRV-scope masters keywords, which still route through ``set_keyword``)
with the EPRV keyword definitions (rvdata's ``LEVEL2/4_PRIMARY_KEYWORDS`` plus the
per-extension keyword CSVs). Columns:
``Keyword, Description, Extension, DataType, PopulatedBy, Required, Level, Default,
Units`` (``Default``/``Units`` carry the EPRV CSV values for EPRV rows, ``""`` for
KPF rows). The ``routing``/``allowed``/``required``/``eprv_primary_*`` lookups are
derived from this table.

The three use-cases:
  (1) Mapping  — ``header_map`` (WMKO->EPRV), consumed by ``KPF0._map_header``;
      ``eprv_primary_datatypes`` types the values it emits. ``header_map`` is
      **sanitized on load** so it holds only genuine static native->EPRV mappings:
      rows whose key is unregistered (PARANG/PARANG2) or handled elsewhere
      (``_DEFAULT_OVERRIDES``) are dropped, so ``_map_header`` needs no
      in-loop filter or per-keyword correction.
  (2) Validation — ``allowed`` / ``required`` (per-extension, from the table)
      plus ``structural`` (FITS bookkeeping cards).
  (3) Routing — ``routing`` (keyword -> (extension, comment)), consumed by
      ``KPFDataModel.set_keyword``.

It also exposes ``eprv_primary_seed`` (the typed EPRV Required PRIMARY skeleton
``KPF1.__init__`` stamps, mirroring rvdata's ``RV2.__init__``). The header_map
corrections live in one place each: ``NUMORDER``/``DRPTAG``/``EPRVTAG``/``VOCLASS``
via ``_DEFAULT_OVERRIDES`` (table sanitization), ``DATALVL`` via ``KPF1.__init__``
(the model level), and the ``JD_UTC`` epoch transform in ``_map_header`` (a
per-frame value, not a static default).
"""

import importlib.metadata
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

_rvdata_inst_cfg = importlib.resources.files("rvdata.instruments.kpf.config")
_rvdata_core_cfg = importlib.resources.files("rvdata.core.models.config")
_kpf_pipe_cfg = importlib.resources.files("kpfpipe.data_models.config")

# Number of echelle orders (green + red); the value header_map.csv gets wrong (65).
_NUMORDER = int(DETECTOR["norder"]["GREEN"]) + int(DETECTOR["norder"]["RED"])

# EPRV-standard compliance is pinned to the installed rv-data-standard release
# (environment.yml pins it exactly): EPRVTAG is its version ("v0.4.0"), VOCLASS
# the release month ("EPRVSTANDARD2026.06"). The release date is not in package
# metadata (PyPI-only), so map it from the exact pin here; bump both together.
_RVDATA_VERSION = importlib.metadata.version("rv-data-standard")
_RVDATA_RELEASE_MONTHS = {"0.4.0": "2026.06"}


class KeywordRegistry:
    """Owns the KPF keyword reference data and the lookups derived from it.

    Built once at import (the module exposes the singleton ``keyword_registry``).
    All attributes are read-only reference data; see the module docstring for the
    three use-cases (mapping / validation / routing).
    """

    # "PopulatedBy" value marking an EPRV-sourced row -- the discriminator the
    # derived lookups use to tell EPRV rows from KPF rows.
    _EPRV_TAG = "EPRV"

    # Unified-table columns; valid identifiers for itertuples attribute access.
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

    # header_map keys whose entry is not a genuine native->EPRV mapping. Two
    # consumers: non-None values correct the table Default (NUMORDER 65->67; DRPTAG,
    # EPRVTAG, VOCLASS are runtime tags), and every key is dropped from header_map.
    # None = no static default; value set elsewhere (DATALVL by KPF1.__init__,
    # JD_UTC by the _map_header epoch transform).
    _DEFAULT_OVERRIDES = {
        "NUMORDER": str(_NUMORDER),
        "DRPTAG": __version__,
        "EPRVTAG": f"v{_RVDATA_VERSION}",
        "VOCLASS": f"EPRVSTANDARD{_RVDATA_RELEASE_MONTHS[_RVDATA_VERSION]}",
        "DATALVL": None,
        "JD_UTC": None,
    }

    # rvdata's header_map numbers per-trace keywords CAL-first (trace 1=CAL .. 5=SKY),
    # the stale translator convention; KPF is SKY-first (1=SKY .. 5=CAL, see
    # EPRV_DATA_STANDARD.md and trace-map.csv). These are the fiber-indexed families
    # whose STANDARD index _load_header_map realigns 1<->5. NOT here (also end in 1/5
    # but not fiber-indexed): EXSNR/EXSNRW (wavelength band 452/852nm), DQLVL, T*.
    _FIBER_INDEXED_BASES = (
        "TRACE",
        "CSRC",
        "CID",
        "CRA",
        "CDEC",
        "CEQNX",
        "CEPCH",
        "CPLX",
        "CPMR",
        "CPMD",
        "CRV",
        "CZ",
        "CCLR",
        "CLSRC",
    )

    # Per-extension EPRV keyword CSVs (rvdata exposes no constant). RV#/CCF# share a
    # template; BARYCORR_*/BJD_TDB are per-extension.
    _EPRV_EXT_CSV = {
        "BJD_TDB": (_rvdata_core_cfg / "L2-BJD_TDB-keywords.csv", 2),
        "BARYCORR_KMS": (_rvdata_core_cfg / "L2-BARYCORR_KMS-keywords.csv", 2),
        "BARYCORR_Z": (_rvdata_core_cfg / "L2-BARYCORR_Z-keywords.csv", 2),
        **{
            f"RV{i}": (_rvdata_core_cfg / "L4-RV1-keywords.csv", 4) for i in range(1, 6)
        },
        **{
            f"CCF{i}": (_rvdata_core_cfg / "L4-CCF1-keywords.csv", 4)
            for i in range(1, 6)
        },
    }

    # "PopulatedBy" values marking a QUALITY_CONTROL row as a 0/1 QC flag: "QC" tags
    # the cross-level ISGOOD aggregate, "QCL{n}" a level-N check.
    _QC_POPULATORS = frozenset({"QC", "QCL0", "QCL1", "QCL2", "QCL4"})

    # FITS structural cards permitted on any extension (never registered keywords) --
    # the validator's allowlist. Exact matches here; enumerated families in
    # _STRUCTURAL_PREFIXES.
    _STRUCTURAL = {
        "SIMPLE",
        "BITPIX",
        "EXTEND",
        "XTENSION",
        "PCOUNT",
        "GCOUNT",
        "BSCALE",
        "BZERO",
        "BUNIT",
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

    # Structural card families (NAXIS*, bintable column descriptors, WCS); by prefix.
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
        # Order matters: _load_header_map filters against the registry
        # _build_registry produces.
        self._build_registry()
        self._load_header_map()

    def _build_registry(self):
        """Build ``self.table`` and every lookup derived from it.

        Assembles the unified table (EPRV rows first, then KPF rows; KPF wins a
        collision), corrects the Defaults header_map.csv gets wrong (NUMORDER
        65 -> 67) or that are runtime (DRPTAG -> version) so downstream lookups
        read a clean table, then derives the read-only lookups. All are shared
        process-wide via the singleton, so they are frozen (frozenset /
        MappingProxyType) against a stray consumer mutation; ``self.table`` is
        only ever read.
        """
        eprv_rows, kpf_rows = self._build_rows()
        table = pd.DataFrame(eprv_rows + kpf_rows, columns=self._COLUMNS)
        for keyword, value in self._DEFAULT_OVERRIDES.items():
            if value is not None:
                table.loc[table["Keyword"] == keyword, "Default"] = value
        self.table = table
        self.registered = frozenset(self.table["Keyword"])

        self.routing = MappingProxyType(self._routing_lookup())
        allowed, required = self._validation_lookup()
        self.allowed = MappingProxyType(
            {ext: frozenset(kws) for ext, kws in allowed.items()}
        )
        self.required = MappingProxyType(
            {ext: MappingProxyType(d) for ext, d in required.items()}
        )
        self.structural = frozenset(self._STRUCTURAL)
        qc_all, qc_by_level = self._qc_flag_sets_lookup()
        self.qc_flag_keywords = frozenset(qc_all)
        self.qc_flag_keywords_by_level = MappingProxyType(
            {lvl: frozenset(kws) for lvl, kws in qc_by_level.items()}
        )
        seed, datatypes = self._eprv_primary_lookup()
        self.eprv_primary_seed = MappingProxyType(seed)
        self.eprv_primary_datatypes = MappingProxyType(datatypes)

    def _load_header_map(self):
        """Read and sanitize rvdata's WMKO-native -> EPRV-standard ``header_map``.

        First realigns the fiber-indexed STANDARD keys from rvdata's CAL-first to
        KPF's SKY-first numbering (see ``_FIBER_INDEXED_BASES``), then keeps only
        genuine static native->EPRV mapping rows so KPF0._map_header applies the map
        with no in-loop filter or per-keyword correction. Two row classes are dropped:
          - STANDARD keys absent from the registry (PARANG/PARANG2) -- warned, so
            the rvdata header_map / registry inconsistency stays visible;
          - ``_DEFAULT_OVERRIDES`` keys, whose value comes from the corrected table
            default, the model level, or the _map_header epoch transform, not a
            static map row.
        Runs after ``_build_registry`` -- it filters against ``self.registered``.
        """
        raw = pd.read_csv(_rvdata_inst_cfg / "header_map.csv")
        # Swap trace index 1<->5 for the fiber-indexed families (SKY<->CAL). A dict
        # replace maps each value once, so the two-way exchange is atomic.
        swap = {}
        for base in self._FIBER_INDEXED_BASES:
            swap[f"{base}1"], swap[f"{base}5"] = f"{base}5", f"{base}1"
        raw["STANDARD"] = raw["STANDARD"].replace(swap)
        eprv_keys = raw["STANDARD"].astype(str).str.strip()
        unregistered = sorted(
            set(eprv_keys[raw["STANDARD"].notna()]) - self.registered - {""}
        )
        if unregistered:
            warnings.warn(
                "header_map.csv maps to STANDARD keys absent from the keyword "
                f"registry; they are dropped (not emitted): {unregistered}",
                UserWarning,
                stacklevel=2,
            )
        keep = eprv_keys.isin(self.registered) & ~eprv_keys.isin(
            self._DEFAULT_OVERRIDES.keys()
        )
        self.header_map = raw[keep].reset_index(drop=True)

    def is_structural(self, key):
        """True for a FITS structural / bookkeeping card (never a registered keyword).

        The single structural test, consumed by the checkpoint header validator: a
        card is structural if it is an exact-match bookkeeping card (``structural``)
        or belongs to an enumerated card family (``_STRUCTURAL_PREFIXES``, e.g.
        ``NAXIS2``/``TTYPE3``).
        """
        k = str(key).strip()
        return k in self.structural or k.startswith(self._STRUCTURAL_PREFIXES)

    # --- Source table construction -------------------------------------------

    @classmethod
    def _eprv_rows(cls, df, extension, level):
        """Expand an EPRV keyword CSV into unified-registry rows.

        EPRV CSVs encode per-key/per-telescope families as a "BASE1 ... BASE#"
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
            descr = str(df["Description"].iloc[i])
            dtype = str(df["DataType"].iloc[i])
            # Kept raw (NaN preserved); _eprv_primary_lookup normalizes them.
            default = df["Default"].iloc[i]
            units = df["Units"].iloc[i]
            required = bool(req.iloc[i])
            if "..." in name:
                # "CDEC1 ... CDEC#" -> "CDEC" (strip the trailing index).
                base = name.split("...")[0].strip().rstrip("0123456789")
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
    def _parse_kpf_keyword_config(cls, df, source, level_of):
        """Expand a KPF header CSV into unified-registry rows.

        Shared by ``_kpf_rows`` and ``_masters_rows``. ``Required`` is always
        False; ``PopulatedBy`` is whatever the CSV says, but it must NEVER be the
        EPRV sentinel: the derived routing/validation lookups treat a
        ``PopulatedBy == _EPRV_TAG`` row as EPRV-sourced, so a KPF row reusing that
        string would be silently misclassified (dropped from routing, given the
        wrong Required/Level). Guard against it loudly here. ``source`` names the
        CSV for that error; ``level_of(row)`` yields each row's Level.
        """
        rows = []
        for _, r in df.iterrows():
            descr = "" if pd.isna(r["Description"]) else str(r["Description"]).strip()
            populated_by = str(r.get("PopulatedBy", "")).strip()
            if populated_by == cls._EPRV_TAG:
                raise ValueError(
                    f"{source}: keyword {str(r['Keyword']).strip()!r} has "
                    f"'PopulatedBy' == {cls._EPRV_TAG!r}, which is reserved as the "
                    "EPRV-row discriminator; use a real populating site instead"
                )
            rows.append(
                [
                    str(r["Keyword"]).strip(),
                    descr,
                    str(r["Extension"]).strip(),
                    str(r.get("DataType", "")).strip(),
                    populated_by,
                    False,
                    level_of(r),
                    "",  # Default (EPRV-only)
                    "",  # Units (EPRV-only)
                ]
            )
        return rows

    @classmethod
    def _kpf_rows(cls):
        """KPF-pipeline keyword rows from config/L{0,1,2,4}-headers.csv.

        Level is the per-file level (one CSV per data level).
        """
        rows = []
        for level in (0, 1, 2, 4):
            df = pd.read_csv(_kpf_pipe_cfg / f"L{level}-headers.csv")
            rows += cls._parse_kpf_keyword_config(
                df, f"L{level}-headers.csv", lambda r, lvl=level: lvl
            )
        return rows

    @classmethod
    def _masters_rows(cls):
        """KPF masters keyword rows from config/Masters-headers.csv.

        Masters (bias/dark/flat/WLS calibration products) are out of EPRV scope
        but route their PRIMARY keywords (MASTYPE + the WLS metadata) through
        ``set_keyword`` like the science models, so they are registered here. Level
        comes from the CSV's own ``Level`` column (masters span L1/L2, not one
        per-file level).
        """
        df = pd.read_csv(_kpf_pipe_cfg / "Masters-headers.csv")
        return cls._parse_kpf_keyword_config(
            df, "Masters-headers.csv", lambda r: int(r["Level"])
        )

    @classmethod
    def _build_rows(cls):
        """Build the EPRV and KPF row lists that union into ``self.table``."""
        # EPRV defines no L1, so KPF holds L1 PRIMARY to the EPRV L2 PRIMARY spec:
        # these are Required from Level 1 (what makes KWRDPRL1 meaningful). L4-only
        # extras stay Level 4.
        l2 = cls._eprv_rows(LEVEL2_PRIMARY_KEYWORDS, "PRIMARY", 1)
        l2_keys = {r[0] for r in l2}
        l4 = [
            r
            for r in cls._eprv_rows(LEVEL4_PRIMARY_KEYWORDS, "PRIMARY", 4)
            if r[0] not in l2_keys
        ]
        # EPRV per-extension keywords (governed extensions with standard cards).
        ext = []
        for name, (path, level) in cls._EPRV_EXT_CSV.items():
            ext += cls._eprv_rows(pd.read_csv(path), name, level)
        return l2 + l4 + ext, cls._kpf_rows() + cls._masters_rows()

    # --- Derived lookups (all read self.table) -------------------------------

    def _routing_lookup(self):
        """keyword -> (home extension, comment), derived from ``self.table``.

        Write keys only: EPRV PRIMARY keywords (-> PRIMARY) and every KPF
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
          PRIMARY keywords at Level <= 1 (the skeleton ``KPF1.__init__`` stamps;
          these are tagged Level 1 in the table -- see ``_build_rows``).
          Built exactly like ``RV2.__init__``: format the unit/description comment,
          then type the default via ``parse_value_to_datatype`` so consumers assign
          it straight into a header. Insertion order follows the EPRV standard's.
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
            if not (row.Required and row.Level <= 1):
                continue
            units = None if pd.isna(row.Units) else str(row.Units).strip()
            unitstr = "" if not units or units.lower() == "n/a" else f"[{units}] "
            comment = f"{unitstr}{row.Description}"
            # Default already corrected in _build_registry (_DEFAULT_OVERRIDES).
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
            if (
                row.Extension != "QUALITY_CONTROL"
                or row.PopulatedBy not in self._QC_POPULATORS
            ):
                continue
            all_flags.add(row.Keyword)
            if row.PopulatedBy != "QC":  # "QCL0"/"QCL1"/"QCL2" -> "L0"/"L1"/"L2"
                by_level.setdefault(row.PopulatedBy[2:], set()).add(row.Keyword)
        return all_flags, by_level

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
