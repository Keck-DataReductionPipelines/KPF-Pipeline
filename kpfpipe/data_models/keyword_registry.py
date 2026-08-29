"""
Header/extension keyword registry for the KPF data models.

Single home for the header keyword reference data, organized by the three
use-cases it serves. ``KeywordRegistry`` builds every lookup once in
``__init__`` from one source-of-truth table; the module exposes a single
instance, ``keyword_registry``. ``KPFDataModel`` (data_models/base.py) is the
only module that imports this one; it surfaces the instance as a class attribute
so consumers handed a ``kpf_obj`` (the checkpoints validator, the
WMKO->EPRV standardization, tests) reach the registry through
``kpf.keyword_registry`` rather than importing from here.

Source of truth: ``self.table``, one DataFrame unioning every
``config/{prefix}-{EXTENSION}-keywords.csv``. The filename carries what the rows
do not: ``prefix`` (one of ``L0 L1 L2 L4 ML1 ML2-flat ML2-wls``) gives the
``Level``, and ``EXTENSION`` names the extension the rows are registered on --
either literally, or as a family stem whose ``1..DETECTOR["numtrace"]``
expansion the level's extension manifest contains (``L2-TRACE_FLUX`` ->
``TRACE1_FLUX`` .. ``TRACE5_FLUX``). A ``#`` in a *keyword* is the same
template marker and expands the same way, with no family exceptions; ``#`` is
therefore reserved, and the CSVs carry no comment rows.

Table columns: ``Keyword, Description, Extension, DataType, PopulatedBy, Level,
Units``. A blank ``PopulatedBy`` means the keyword is registered but nothing
writes an informative value to it yet (see ``notes/keywords/audit_20260827.md``),
which keeps that audit machine-readable. ``ExampleValue`` is documentation only
and is not read here.

The three use-cases:
  (1) Mapping  -- ``header_map`` (``config/EPRV-header-map.csv``: WMKO-native ->
      EPRV-standard), consumed by ``StandardizeDataFormat``; ``datatype_for``
      types the values it emits, and ``primary_seed`` stamps the skeleton it
      fills in. The map is PRIMARY-only by definition and is the definition of
      the EPRV PRIMARY keyword set: every ``EPRV_KEY`` must be registered on
      PRIMARY here, or the load raises.
  (2) Validation -- ``allowed`` (per-extension, from the table) plus
      ``structural`` (FITS bookkeeping cards).
  (3) Routing -- ``routing`` (keyword -> home extension name) for the default
      write, plus ``comment_for`` ((keyword, extension) -> FITS comment) for
      ``set_keyword``'s targeted (``ext=``) write of per-extension cards. Both
      consumed by ``KPFDataModel.set_keyword``.
"""

import importlib.resources
import logging
from types import MappingProxyType

import pandas as pd

from kpfpipe import DETECTOR, OBSERVATORY

logger = logging.getLogger(__name__)

_kpf_pipe_cfg = importlib.resources.files("kpfpipe.data_models.config")

# SCI_TRACES holds the science-fiber indices, the fibers the catalog C*# overlay
# targets -- the single definition, imported by astro_query rather than
# re-derived there. The trace count itself is DETECTOR["numtrace"]; the numbered
# per-orderlet extensions CCF#/RV#/CCF_VAR# run 1..that.
_TRACE_MAP = pd.read_csv(_kpf_pipe_cfg / "trace-map.csv")
SCI_TRACES = tuple(
    _TRACE_MAP.loc[_TRACE_MAP["Fiber"].isin({"SCI1", "SCI2", "SCI3"}), "Trace"]
)


class KeywordRegistry:
    """Owns the KPF keyword reference data and the lookups derived from it.

    Built once at import (the module exposes the singleton ``keyword_registry``).
    All attributes are read-only reference data; see the module docstring for the
    three use-cases (mapping / validation / routing).
    """

    # Unified-table columns; valid identifiers for itertuples attribute access.
    _COLUMNS = [
        "Keyword",
        "Description",
        "Extension",
        "DataType",
        "PopulatedBy",
        "Level",
        "Units",
    ]

    # The keyword-CSV filename prefixes, and the data level each implies. Also
    # the manifest vocabulary: every model resolves its tables to one of these.
    _PROFILE_LEVELS = {
        "L0": 0,
        "L1": 1,
        "L2": 2,
        "L4": 4,
        "ML1": 1,
        "ML2-flat": 2,
        "ML2-wls": 2,
    }

    # PRIMARY keywords whose seed default is the observatory config rather than a
    # DEFAULT cell: EPRV_KEY -> kpfpipe.OBSERVATORY key.
    _OBSERVATORY_DEFAULTS = {
        "GEOSYS": "geosys",
        "OBSLON": "longitude",
        "OBSLAT": "latitude",
        "OBSALT": "altitude",
    }

    # The CSVs' DataType vocabulary, matched case-insensitively. A blank DataType
    # passes the value through unchanged.
    _TYPE_PARSERS = {
        "str": str,
        "string": str,
        "int": int,
        "uint": int,
        "float": float,
        "float32": float,
        "double": float,
        "bool": lambda v: v[0].lower() == "t" if isinstance(v, str) else bool(v),
        "boolean": lambda v: v[0].lower() == "t" if isinstance(v, str) else bool(v),
    }

    # "PopulatedBy" values marking a QUALITY_CONTROL row as a 0/1 QC flag: "QCL{n}"
    # tags a level-N check.
    _QC_POPULATORS = frozenset({"QCL0", "QCL1", "QCL2", "QCL4"})

    # FITS structural cards: written by the I/O layer (astropy) from the HDU's
    # structure, never authored by the pipeline -- so always permitted on any
    # extension and never registered keywords. Exact matches here; enumerated
    # families in _STRUCTURAL_PREFIXES. The lone exception is BUNIT: it carries
    # content (physical units) but is stamped directly on the masters images
    # rather than registered (see masters/base.py).
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
        "EXTNAME",
        "TFIELDS",
    }

    # Structural card families (NAXIS*, bintable column descriptors); by prefix. No
    # WCS family here: KPF authors no WCS, and CTYPE is registered content (rvdata
    # uses it to name each extension's axes), so a stray WCS card should be flagged.
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
    )

    def __init__(self):
        # Order matters: _load_header_map filters against the registry
        # _build_registry produces.
        self._build_registry()
        self._load_header_map()

    def _build_registry(self):
        """Build ``self.table`` and every read-only lookup derived from it.

        All lookups are frozen (frozenset / MappingProxyType) against stray
        mutation, since the singleton shares them process-wide.
        """
        rows, profile_primary = self._load_keyword_rows()
        self.table = pd.DataFrame(rows, columns=self._COLUMNS)
        self.registered = frozenset(self.table["Keyword"])
        self.structural = frozenset(self._STRUCTURAL)
        # PRIMARY keywords contributed by each profile's own CSVs -- the seed set
        # for the masters profiles, which are outside EPRV scope.
        self._profile_primary = MappingProxyType(
            {p: tuple(kws) for p, kws in profile_primary.items()}
        )

        self.routing = MappingProxyType(self._routing_lookup())
        # (keyword, extension) -> FITS comment / DataType, for set_keyword's
        # targeted (ext=) write path (per-extension cards like VELSTART on CCF#,
        # which have no single routed home) and for typing the mapped fill.
        self.comments = MappingProxyType(
            {
                (row.Keyword, row.Extension): self._compose_comment(
                    row.Description, row.Units
                )
                for row in self.table.itertuples(index=False)
            }
        )
        self.datatypes = MappingProxyType(
            {
                (row.Keyword, row.Extension): row.DataType
                for row in self.table.itertuples(index=False)
            }
        )
        self.allowed = MappingProxyType(
            {ext: frozenset(kws) for ext, kws in self._allowed_lookup().items()}
        )
        # keyword -> the lowest Level it is registered on PRIMARY at; the level
        # gate primary_seed applies to the header map.
        self._primary_levels = MappingProxyType(self._primary_level_lookup())
        qc_all, qc_by_level = self._qc_flag_sets_lookup()
        self.qc_flag_keywords = frozenset(qc_all)
        self.qc_flag_keywords_by_level = MappingProxyType(
            {lvl: frozenset(kws) for lvl, kws in qc_by_level.items()}
        )

    # --- Source table construction -------------------------------------------

    @classmethod
    def _manifest_names(cls):
        """``profile -> set of extension names`` from the extension manifests.

        Read here (rather than imported from ``base.py``, which imports this
        module) purely to resolve keyword-CSV filenames; the data models get the
        manifests themselves from ``base._MANIFESTS``.
        """
        return {
            profile: set(
                pd.read_csv(_kpf_pipe_cfg / f"{profile}-extensions.csv")["Name"]
            )
            for profile in cls._PROFILE_LEVELS
        }

    @classmethod
    def _expand_template(cls, name):
        """Expand a ``#`` template to its ``1..DETECTOR["numtrace"]`` members.

        ``#`` is the one template marker, with no family exceptions: it means
        exactly ``1..numtrace`` wherever it appears, in a keyword or in a family
        stem. A name without ``#`` expands to itself.
        """
        if "#" not in name:
            return [name]
        return [name.replace("#", str(i)) for i in range(1, DETECTOR["numtrace"] + 1)]

    @classmethod
    def _resolve_extensions(cls, extension, names, source):
        """Resolve a keyword CSV's extension field to concrete extension names.

        The field either *is* a literal extension name in that profile's
        manifest, or is a family stem: insert an index at every position and
        accept the single candidate whose full expansion the manifest contains
        (``CCF_VAR`` -> ``CCF_VAR1..5``, not ``CCF1_VAR``; ``TRACE_FLUX`` ->
        ``TRACE1_FLUX..``, not ``TRACE_FLUX1``). Zero or two qualifying positions
        raise -- a typo must not resolve.
        """
        if extension in names:
            return [extension]
        candidates = []
        for pos in range(len(extension) + 1):
            family = cls._expand_template(f"{extension[:pos]}#{extension[pos:]}")
            if all(member in names for member in family):
                candidates.append(family)
        if len(candidates) != 1:
            raise ValueError(
                f"{source}: extension {extension!r} is neither a name in that "
                f"level's extension manifest nor an unambiguous family stem "
                f"({len(candidates)} qualifying index positions)"
            )
        return candidates[0]

    @classmethod
    def _load_keyword_rows(cls):
        """Read every ``{prefix}-{EXTENSION}-keywords.csv`` into registry rows.

        Returns ``(rows, profile_primary)``; ``profile_primary`` maps each
        profile to the PRIMARY keywords its own CSVs contribute.
        """
        manifest_names = cls._manifest_names()
        rows = []
        profile_primary = {profile: [] for profile in cls._PROFILE_LEVELS}
        paths = sorted(
            (p for p in _kpf_pipe_cfg.iterdir() if p.name.endswith("-keywords.csv")),
            key=lambda p: p.name,
        )
        for path in paths:
            stem = path.name[: -len("-keywords.csv")]
            profile, _, extension = stem.rpartition("-")
            if profile not in cls._PROFILE_LEVELS:
                raise ValueError(
                    f"{path.name}: unrecognized keyword-CSV prefix {profile!r}; "
                    f"expected one of {sorted(cls._PROFILE_LEVELS)}"
                )
            level = cls._PROFILE_LEVELS[profile]
            extensions = cls._resolve_extensions(
                extension, manifest_names[profile], path.name
            )
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                descr = (
                    "" if pd.isna(r["Description"]) else str(r["Description"]).strip()
                )
                units = "" if pd.isna(r["Units"]) else str(r["Units"]).strip()
                dtype = "" if pd.isna(r["DataType"]) else str(r["DataType"]).strip()
                populated_by = (
                    "" if pd.isna(r["PopulatedBy"]) else str(r["PopulatedBy"]).strip()
                )
                for keyword in cls._expand_template(str(r["Keyword"]).strip()):
                    for ext in extensions:
                        rows.append(
                            [keyword, descr, ext, dtype, populated_by, level, units]
                        )
                        if ext == "PRIMARY":
                            profile_primary[profile].append(keyword)
        return rows, profile_primary

    def _load_header_map(self):
        """Read ``config/EPRV-header-map.csv``, the WMKO-native -> EPRV map.

        The map is PRIMARY-only by definition (a per-extension EPRV keyword needs
        no map row), so it is also the definition of the EPRV PRIMARY keyword
        set: every ``EPRV_KEY`` must be unique and registered on PRIMARY, or a
        stray key would seed a comment-less, untyped card silently. Runs after
        ``_build_registry`` -- it filters against ``self.allowed``.

        The four site-coordinate keywords take their default from
        ``kpfpipe.OBSERVATORY`` rather than a DEFAULT cell, so the observatory
        config stays their single source.
        """
        raw = pd.read_csv(_kpf_pipe_cfg / "EPRV-header-map.csv")
        keys = raw["EPRV_KEY"].astype(str).str.strip()
        duplicated = sorted(set(keys[keys.duplicated()]))
        if duplicated:
            raise ValueError(
                f"EPRV-header-map.csv has duplicate EPRV_KEY rows: {duplicated}"
            )
        unregistered = sorted(set(keys) - self.allowed.get("PRIMARY", frozenset()))
        if unregistered:
            raise ValueError(
                "EPRV-header-map.csv maps EPRV_KEY values that are not registered "
                f"on PRIMARY: {unregistered}. Register them in the appropriate "
                "config/{prefix}-PRIMARY-keywords.csv before mapping them."
            )
        for keyword, config_key in self._OBSERVATORY_DEFAULTS.items():
            blank = (keys == keyword) & raw["DEFAULT"].isna()
            # str(): DEFAULT is a text column, and _parse_value types it on read.
            raw.loc[blank, "DEFAULT"] = str(OBSERVATORY[config_key])
        self.header_map = raw

    # --- Accessors ------------------------------------------------------------

    def is_structural(self, key):
        """True for a FITS structural / bookkeeping card (never a registered keyword).

        The single structural test, consumed by the checkpoint header validator: a
        card is structural if it is an exact-match bookkeeping card (``structural``)
        or belongs to an enumerated card family (``_STRUCTURAL_PREFIXES``, e.g.
        ``NAXIS2``/``TTYPE3``).
        """
        k = str(key).strip()
        return k in self.structural or k.startswith(self._STRUCTURAL_PREFIXES)

    @staticmethod
    def _compose_comment(description, units):
        """The FITS comment for a registry row: ``Description [Units]``.

        Unit-less rows (blank or the EPRV ``N/A`` placeholder) carry the
        description alone.
        """
        u = str(units).strip()
        if not u or u.lower() == "n/a":
            return description
        return f"{description} [{u}]"

    def comment_for(self, keyword, extension):
        """FITS comment for ``keyword`` on ``extension`` (``Description [Units]``).

        Returns None when the keyword is not registered for that extension -- the
        membership test set_keyword's targeted (``ext=``) path uses; a registered
        keyword with an empty Description returns ``""``, distinct from None.
        """
        return self.comments.get((str(keyword).strip(), extension))

    def datatype_for(self, keyword, extension):
        """Registry ``DataType`` for ``keyword`` on ``extension``.

        Mirrors ``comment_for``: None when the keyword is not registered there,
        ``""`` when it is registered with no declared type (which
        ``_parse_value`` passes through unchanged).
        """
        return self.datatypes.get((str(keyword).strip(), extension))

    @classmethod
    def _parse_value(cls, keyword, datatype, value):
        """Type ``value`` to the registry ``DataType``.

        KPF-owned typing over the CSVs' own vocabulary, matched
        case-insensitively. An empty value is None; a blank ``DataType`` passes
        the value through unchanged. There is deliberately **no**
        ``"UNKNOWN"``/``"UNDEFINED"`` sentinel rule: ``SCI-OBJ``/``SKY-OBJ``
        carry the literal value ``Unknown`` on real frames, and blanking those
        would silently empty ``TRACE1..TRACE4``.

        An unknown ``DataType`` raises -- it is a config error, not frame data. A
        value that will not convert warns and yields None, so one malformed
        native card cannot abort a reduction.
        """
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        if not datatype or (isinstance(datatype, float) and pd.isna(datatype)):
            return value
        parser = cls._TYPE_PARSERS.get(str(datatype).strip().lower())
        if parser is None:
            raise ValueError(
                f"keyword {keyword!r}: unknown DataType {datatype!r}; expected one "
                f"of {sorted(cls._TYPE_PARSERS)}"
            )
        try:
            return parser(value)
        except (TypeError, ValueError, IndexError):
            logger.warning(
                "cannot convert value %r for keyword %r to type %s",
                value,
                keyword,
                datatype,
            )
            return None

    def primary_seed(self, profile):
        """The typed PRIMARY skeleton for ``profile``: ``{keyword: (value, comment)}``.

        Every keyword registered on PRIMARY for that profile, carrying its
        ``EPRV-header-map.csv`` default where it has one and blank where it does
        not, each with its registry comment. Nothing is filtered on ``REQUIRED``:
        that column is a compliance label, so the seed stamps a card for every
        member of every ``#`` family -- the five-trace rule at the header level.

        For ``L0``/``L1``/``L2``/``L4`` the set is the header-map rows at
        ``Level <= n``, cumulative (L0 == L1 == 154 cards, L2 adds
        ``EXTRACT``/``EXSNR#``/``EXSNRW#``, L4 the seven RV rows). For the
        masters profiles it is that master's own ``ML*-PRIMARY-keywords.csv``
        rows: masters are outside EPRV scope and do not inherit the science
        skeleton.
        """
        if profile not in self._PROFILE_LEVELS:
            raise ValueError(
                f"unknown keyword profile {profile!r}; expected one of "
                f"{sorted(self._PROFILE_LEVELS)}"
            )
        if profile.startswith("ML"):
            defaults = [(kw, None) for kw in self._profile_primary[profile]]
        else:
            cap = self._PROFILE_LEVELS[profile]
            defaults = [
                (str(row.EPRV_KEY).strip(), row.DEFAULT)
                for row in self.header_map.itertuples(index=False)
                if self._primary_levels[str(row.EPRV_KEY).strip()] <= cap
            ]
        return {
            keyword: (
                self._parse_value(
                    keyword, self.datatype_for(keyword, "PRIMARY"), default
                ),
                self.comment_for(keyword, "PRIMARY"),
            )
            for keyword, default in defaults
        }

    # --- Derived lookups (all read self.table) -------------------------------

    def _routing_lookup(self):
        """keyword -> home extension name, derived from ``self.table``.

        A keyword registered on PRIMARY routes to PRIMARY (``RVMETHOD`` and
        ``RVGREEN``/``RVRED``/``ERVGREEN``/``ERVRED`` are on PRIMARY *and* on
        ``RV1..5``); otherwise a keyword with exactly one home routes there.
        Everything else -- the per-extension cards that recur on every orderlet
        (``CTYPE1``, ``VELWIDTH``) -- has no single home and is written via
        ``set_keyword``'s targeted ``ext=`` path.
        """
        homes = {}
        for row in self.table.itertuples(index=False):
            homes.setdefault(row.Keyword, set()).add(row.Extension)
        routing = {}
        for keyword, extensions in homes.items():
            if "PRIMARY" in extensions:
                routing[keyword] = "PRIMARY"
            elif len(extensions) == 1:
                routing[keyword] = next(iter(extensions))
        return routing

    def _allowed_lookup(self):
        """extension -> every keyword registered for it (no level gate)."""
        allowed = {}
        for row in self.table.itertuples(index=False):
            allowed.setdefault(row.Extension, set()).add(row.Keyword)
        return allowed

    def _primary_level_lookup(self):
        """keyword -> the lowest Level it is registered on PRIMARY at."""
        levels = {}
        for row in self.table.itertuples(index=False):
            if row.Extension != "PRIMARY":
                continue
            levels[row.Keyword] = min(levels.get(row.Keyword, row.Level), row.Level)
        return levels

    def _qc_flag_sets_lookup(self):
        """QC-flag keyword sets, derived from ``self.table``.

        Returns
        -------
        tuple
            ``(all_flags, by_level)``. ``all_flags`` is every QUALITY_CONTROL row
            tagged by a QC populator (the cross-level L0->L4 set). ``by_level``
            maps a LEVEL tag (one per registered level, e.g. "L0"/"L1"/"L2"/"L4")
            to that level's own ``QCL{n}`` flags (used by the per-level checkpoint).
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
            # "QCL{n}" -> "L{n}", for each registered level
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
        neither auto-creates it nor lists it in EXT_DESCRIPT -- the KPF model
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


# Module singleton -- the one registry instance every consumer reaches through.
keyword_registry = KeywordRegistry()
