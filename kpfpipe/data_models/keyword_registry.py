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
writes an informative value to it yet: the card ships present and blank, and the
blank cell is the machine-readable record of which keywords are still unsourced.
``ExampleValue`` is documentation only and is not read here.

The three use-cases:
  (1) Mapping  -- ``header_map`` (``config/EPRV-header-map.csv``: WMKO-native ->
      EPRV-standard), consumed by ``KPF0.standardize_header_format``; ``datatype_for``
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

import numpy as np
import pandas as pd

from kpfpipe import DETECTOR
from kpfpipe.data_models.extension_manifest import extension_manifest
from kpfpipe.utils.astro import KECK_LOCATION

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
    _DATA_MODEL_LEVELS = {
        "L0": 0,
        "L1": 1,
        "L2": 2,
        "L4": 4,
        "ML1": 1,
        "ML2-flat": 2,
        "ML2-wls": 2,
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
        rows, data_model_primary = self._load_keyword_rows()
        self.table = pd.DataFrame(rows, columns=self._COLUMNS)
        self.registered = frozenset(self.table["Keyword"])
        self.structural = frozenset(self._STRUCTURAL)
        # PRIMARY keywords contributed by each data model's own CSVs -- the seed
        # set for the masters, which are outside EPRV scope.
        self._data_model_primary = MappingProxyType(
            {p: tuple(kws) for p, kws in data_model_primary.items()}
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

        The field either *is* a literal extension name in that data model's
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

        Returns ``(rows, data_model_primary)``; ``data_model_primary`` maps each
        data model to the PRIMARY keywords its own CSVs contribute.
        """
        rows = []
        data_model_primary = {data_model: [] for data_model in cls._DATA_MODEL_LEVELS}
        paths = sorted(
            (p for p in _kpf_pipe_cfg.iterdir() if p.name.endswith("-keywords.csv")),
            key=lambda p: p.name,
        )
        for path in paths:
            stem = path.name[: -len("-keywords.csv")]
            data_model, _, extension = stem.rpartition("-")
            if data_model not in cls._DATA_MODEL_LEVELS:
                raise ValueError(
                    f"{path.name}: unrecognized keyword-CSV prefix {data_model!r}; "
                    f"expected one of {sorted(cls._DATA_MODEL_LEVELS)}"
                )
            level = cls._DATA_MODEL_LEVELS[data_model]
            extensions = cls._resolve_extensions(
                extension, set(extension_manifest.names(data_model)), path.name
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
                            data_model_primary[data_model].append(keyword)
        return rows, data_model_primary

    def _load_header_map(self):
        """Read ``config/EPRV-header-map.csv``, the WMKO-native -> EPRV map.

        The map is PRIMARY-only by definition (a per-extension EPRV keyword needs
        no map row), so it is also the definition of the EPRV PRIMARY keyword
        set: every ``EPRV_KEY`` must be unique and registered on PRIMARY, or a
        stray key would seed a comment-less, untyped card silently. Runs after
        ``_build_registry`` -- it filters against ``self.allowed``.

        The seven site-coordinate keywords take their default from
        ``kpfpipe.KECK_LOCATION`` rather than a DEFAULT cell, so the observatory
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
        # 1e-5 deg is ~1 m, against the ~140 m a 1 cm/s barycentric correction
        # needs (dv = omega * dx); rounding also absorbs astropy's geodetic noise.
        site = {
            "GEOSYS": KECK_LOCATION.ellipsoid,
            "OBSLON": round(KECK_LOCATION.lon.deg, 5),
            "OBSLAT": round(KECK_LOCATION.lat.deg, 5),
            "OBSALT": round(KECK_LOCATION.height.to_value("m"), 3),
            "OBSGEO-X": round(KECK_LOCATION.x.to_value("m"), 3),
            "OBSGEO-Y": round(KECK_LOCATION.y.to_value("m"), 3),
            "OBSGEO-Z": round(KECK_LOCATION.z.to_value("m"), 3),
        }
        for keyword, value in site.items():
            blank = (keys == keyword) & raw["DEFAULT"].isna()
            # str(): DEFAULT is a text column, and _parse_value types it on read.
            raw.loc[blank, "DEFAULT"] = str(value)
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

    @staticmethod
    def coerce(datatype, value):
        """Convert ``value`` to ``datatype``, the CSVs' vocabulary lowercased.

        Floats carry their declared width, so a ``float32`` card is written at
        single precision rather than silently at double. Booleans accept only
        the three spellings a header value ever has -- ``True``/``False`` in
        memory, ``T``/``F`` on disk, ``0``/``1`` by convention -- so a truthy
        string is a bad value, not ``True``. Raises ``KeyError`` on an unknown
        ``datatype`` and ``TypeError``/``ValueError`` on a value that will not
        convert; the callers own what each of those means -- ``_parse_value``
        warns past a bad frame value, ``set_keyword`` raises on a bad write.
        """
        match datatype:
            case "str" | "string":
                return str(value)
            case "int":
                return int(value)
            case "uint":
                number = int(value)
                if number < 0:
                    raise ValueError(f"{value!r} is negative")
                return number
            case "float" | "double":
                return np.float64(value)
            case "float32":
                return np.float32(value)
            case "bool" | "boolean":
                if isinstance(value, (bool, np.bool_)):
                    return bool(value)
                if isinstance(value, str) and value.strip().upper() in ("T", "F"):
                    return value.strip().upper() == "T"
                if isinstance(value, (int, np.integer)) and value in (0, 1):
                    return bool(value)
                raise ValueError(f"{value!r} is not True/False, T/F or 0/1")
            case _:
                raise KeyError(datatype)

    @classmethod
    def _parse_value(cls, keyword, datatype, value):
        """Type ``value`` to the registry ``DataType`` via ``coerce``.

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
        try:
            return cls.coerce(str(datatype).strip().lower(), value)
        except KeyError:
            raise ValueError(
                f"keyword {keyword!r}: unknown DataType {datatype!r}"
            ) from None
        except (TypeError, ValueError):
            logger.warning(
                "cannot convert value %r for keyword %r to type %s",
                value,
                keyword,
                datatype,
            )
            return None

    def primary_seed(self, data_model):
        """``data_model``'s typed PRIMARY skeleton: ``{keyword: (value, comment)}``.

        Every keyword registered on PRIMARY for that data model, carrying its
        ``EPRV-header-map.csv`` default where it has one and blank where it does
        not, each with its registry comment. Nothing is filtered on ``REQUIRED``:
        that column is a compliance label, so the seed stamps a card for every
        member of every ``#`` family -- the five-trace rule at the header level.

        For ``L0``/``L1``/``L2``/``L4`` the set is the header-map rows at
        ``Level <= n``, cumulative (154 cards at L0, L1 adds ``DQLVL1``, L2
        ``EXTRACT``/``EXSNR#``/``EXSNRW#``/``DQLVL2``, L4 the seven RV rows and
        ``DQLVL4``). For the masters it is that master's own
        ``ML*-PRIMARY-keywords.csv`` rows: masters are outside EPRV scope and do
        not inherit the science skeleton.
        """
        if data_model not in self._DATA_MODEL_LEVELS:
            raise ValueError(
                f"unknown data model {data_model!r}; expected one of "
                f"{sorted(self._DATA_MODEL_LEVELS)}"
            )
        if data_model.startswith("ML"):
            defaults = [(kw, None) for kw in self._data_model_primary[data_model]]
        else:
            cap = self._DATA_MODEL_LEVELS[data_model]
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


# Module singleton -- the one registry instance every consumer reaches through.
keyword_registry = KeywordRegistry()
