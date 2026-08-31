"""
Header/extension keyword registry for the KPF data models.

Single home for the header keyword reference data, organized by the three
use-cases it serves (below). ``KeywordRegistry`` builds every lookup once in
``__init__`` from one source-of-truth table; the module exposes the singleton
``keyword_registry``, reached via ``kpf.keyword_registry``.

Source of truth: ``self.table``, unioning every
``config/{prefix}-{EXTENSION}-keywords.csv``. ``prefix`` (``L0 L1 L2 L4 ML1
ML2-flat ML2-wls``) gives the ``Level``; ``EXTENSION`` is the extension name,
literal or a family stem expanded over ``1..DETECTOR["numtrace"]``
(``L2-TRACE_FLUX`` -> ``TRACE1_FLUX``..``TRACE5_FLUX``). ``#`` in a *keyword*
is the same template marker and is reserved; the CSVs carry no comment rows.

Table columns: ``Keyword, Description, Extension, DataType, PopulatedBy,
Level, Units``. A blank ``PopulatedBy`` means the keyword is registered but
unsourced. ``ExampleValue`` is documentation only, not read here.

Three use-cases:
  (1) Mapping -- ``header_map`` (WMKO-native -> EPRV-standard, from
      ``config/header-map.csv``), consumed by
      ``KPF0.standardize_header_format``; defines the EPRV PRIMARY keyword
      set (every ``EPRV_KEY`` must be registered on PRIMARY, or the load
      raises).
  (2) Validation -- ``is_registered``, ``allowed`` (per-extension set),
      ``is_structural`` (FITS bookkeeping cards, permitted everywhere,
      registered nowhere).
  (3) Routing -- ``routing`` (keyword -> home extension), ``comment_for``
      ((keyword, extension) -> FITS comment) for ``set_keyword``'s targeted
      (``ext=``) write.

``comment_for``, ``datatype_for`` and ``is_registered`` all read the one
per-card index, ``cards``: ``(keyword, extension) -> (comment, DataType)``.
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


class KeywordRegistry:
    """Owns the KPF keyword reference data and the lookups derived from it.

    Built once at import; the module exposes the singleton ``keyword_registry``.
    All attributes are read-only reference data (see module docstring).
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

    # Data level per data model -- the one thing a config filename doesn't carry.
    # The vocabulary belongs to ``extension_manifest``; ``__init__`` checks the
    # two agree, so adding a model in only one place is a load-time error.
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

    # FITS structural cards: written by astropy from the HDU's structure, never
    # authored by the pipeline -- permitted anywhere, never registered keywords.
    # Exact matches here; enumerated families in _STRUCTURAL_PREFIXES. Exception:
    # BUNIT carries content but is stamped directly on masters images, not
    # registered (see masters/base.py).
    _STRUCTURAL = frozenset(
        {
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
    )

    # Structural card families (NAXIS*, bintable column descriptors), by prefix.
    # No WCS family: KPF authors no WCS and CTYPE is registered content (rvdata
    # names each extension's axes), so a stray WCS card should be flagged.
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
        drift = set(self._DATA_MODEL_LEVELS) ^ set(extension_manifest.data_models)
        if drift:
            raise ValueError(
                f"data model vocabulary has drifted: {sorted(drift)} is declared "
                "in only one of KeywordRegistry._DATA_MODEL_LEVELS and the "
                "config/{data_model}-extensions.csv set"
            )
        # Order matters: _load_header_map filters against the registry
        # _build_registry produces.
        self._build_registry()
        self._load_header_map()

    def _build_registry(self):
        """Build ``self.table`` and every read-only lookup derived from it.

        One pass over the table feeds them all: ``cards`` is the per-card index
        every accessor reads; the rest regroup the same rows. All lookups are
        frozen (frozenset / MappingProxyType) since the singleton shares them
        process-wide.
        """
        rows, data_model_primary = self._load_keyword_rows()
        self.table = pd.DataFrame(rows, columns=self._COLUMNS)
        self.registered = frozenset(self.table["Keyword"])
        # PRIMARY keywords contributed by each data model's own CSVs -- the seed
        # set for the masters, which are outside EPRV scope.
        self._data_model_primary = MappingProxyType(
            {p: tuple(kws) for p, kws in data_model_primary.items()}
        )

        cards = {}  # (keyword, extension) -> (comment, DataType)
        allowed = {}  # extension -> {keyword}
        homes = {}  # keyword -> {extension it is registered on}
        primary_levels = {}  # keyword -> lowest Level it is on PRIMARY at
        qc_all = set()
        qc_by_level = {}
        for row in self.table.itertuples(index=False):
            cards[(row.Keyword, row.Extension)] = (
                self._compose_comment(row.Description, row.Units),
                row.DataType,
            )
            allowed.setdefault(row.Extension, set()).add(row.Keyword)
            homes.setdefault(row.Keyword, set()).add(row.Extension)
            if row.Extension == "PRIMARY":
                primary_levels[row.Keyword] = min(
                    primary_levels.get(row.Keyword, row.Level), row.Level
                )
            elif (
                row.Extension == "QUALITY_CONTROL"
                and row.PopulatedBy in self._QC_POPULATORS
            ):
                qc_all.add(row.Keyword)
                # "QCL{n}" -> "L{n}", for each registered level
                qc_by_level.setdefault(row.PopulatedBy[2:], set()).add(row.Keyword)

        # The one per-card index: comment and DataType together. Read through
        # comment_for / datatype_for / is_registered.
        self.cards = MappingProxyType(cards)
        self.allowed = MappingProxyType(
            {ext: frozenset(kws) for ext, kws in allowed.items()}
        )
        self.routing = MappingProxyType(self._routing_lookup(homes))
        # The level gate primary_seed applies to the header map.
        self._primary_levels = MappingProxyType(primary_levels)
        self.qc_flag_keywords = frozenset(qc_all)
        self.qc_flag_keywords_by_level = MappingProxyType(
            {lvl: frozenset(kws) for lvl, kws in qc_by_level.items()}
        )

    # --- Source table construction -------------------------------------------

    @classmethod
    def _expand_template(cls, name):
        """Expand a ``#`` template to its ``1..DETECTOR["numtrace"]`` members.

        ``#`` means exactly ``1..numtrace`` wherever it appears (keyword or
        family stem), with no family exceptions. A name without ``#`` expands
        to itself.
        """
        if "#" not in name:
            return [name]
        return [name.replace("#", str(i)) for i in range(1, DETECTOR["numtrace"] + 1)]

    @classmethod
    def _resolve_extensions(cls, extension, names, source):
        """Resolve a keyword CSV's extension field to concrete extension names.

        The field is either a literal extension name, or a family stem: insert
        an index at every position and accept the single candidate whose full
        expansion the manifest contains (``CCF_VAR`` -> ``CCF1_VAR..``, not
        ``CCF_VAR1``). Zero or two qualifying positions raise -- a typo must
        not resolve.
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
        """Read ``config/header-map.csv``, the WMKO-native -> EPRV map.

        PRIMARY-only by definition, so also the definition of the EPRV PRIMARY
        keyword set: every ``EPRV_KEY`` must be unique and registered on
        PRIMARY, or a stray key would seed a comment-less, untyped card
        silently. Runs after ``_build_registry`` -- filters against
        ``self.allowed``.

        The seven site-coordinate keywords default from
        ``kpfpipe.KECK_LOCATION`` rather than a DEFAULT cell, so the
        observatory config stays their single source.
        """
        raw = pd.read_csv(_kpf_pipe_cfg / "header-map.csv")
        keys = raw["EPRV_KEY"].astype(str).str.strip()
        duplicated = sorted(set(keys[keys.duplicated()]))
        if duplicated:
            raise ValueError(
                f"header-map.csv has duplicate EPRV_KEY rows: {duplicated}"
            )
        unregistered = sorted(set(keys) - self.allowed.get("PRIMARY", frozenset()))
        if unregistered:
            raise ValueError(
                "header-map.csv maps EPRV_KEY values that are not registered "
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

        Consumed by the checkpoint header validator: exact-match bookkeeping
        card (``_STRUCTURAL``) or an enumerated card family
        (``_STRUCTURAL_PREFIXES``, e.g. ``NAXIS2``/``TTYPE3``).
        """
        k = str(key).strip()
        return k in self._STRUCTURAL or k.startswith(self._STRUCTURAL_PREFIXES)

    @staticmethod
    def _compose_comment(description, units):
        """The FITS comment for a registry row: ``Description [Units]``.

        A unit-less row leaves ``Units`` blank and carries the description alone.
        """
        u = str(units).strip()
        if not u:
            return description
        return f"{description} [{u}]"

    def is_registered(self, keyword, extension):
        """True if ``keyword`` is registered for ``extension``.

        ``allowed[ext]`` answers the same question set-wise, for the
        checkpoint that sweeps a whole header.
        """
        return (str(keyword).strip(), extension) in self.cards

    def comment_for(self, keyword, extension=None):
        """FITS comment for ``keyword`` on ``extension`` (``Description [Units]``).

        With ``extension`` omitted, the keyword's routed home is used -- what
        QC and diagnostics want when mirroring a comment into their results.

        Returns None when not registered there; a registered keyword with an
        empty Description returns ``""``, distinct from None.
        """
        name = str(keyword).strip()
        if extension is None:
            extension = self.routing.get(name)
        card = self.cards.get((name, extension))
        return None if card is None else card[0]

    def datatype_for(self, keyword, extension):
        """Registry ``DataType`` for ``keyword`` on ``extension``.

        Mirrors ``comment_for``: None when not registered there, ``""`` when
        registered with no declared type (``_parse_value`` passes it through).
        """
        card = self.cards.get((str(keyword).strip(), extension))
        return None if card is None else card[1]

    @staticmethod
    def coerce(datatype, value):
        """Convert ``value`` to ``datatype``, the CSVs' vocabulary lowercased.

        Floats carry their declared width (``float32`` writes single
        precision, not double). Booleans accept only ``True``/``False``,
        ``T``/``F``, ``0``/``1`` -- a truthy string is a bad value, not
        ``True``. Raises ``KeyError`` on an unknown ``datatype``,
        ``TypeError``/``ValueError`` on a value that won't convert; callers
        decide what that means -- ``_parse_value`` warns past a bad frame
        value, ``set_keyword`` raises on a bad write.
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

        Case-insensitive match to the CSVs' vocabulary. An empty value is
        None; a blank ``DataType`` passes the value through unchanged.
        Deliberately **no** ``"UNKNOWN"``/``"UNDEFINED"`` sentinel rule:
        ``SCI-OBJ``/``SKY-OBJ`` carry the literal value ``Unknown`` on real
        frames, and blanking those would silently empty ``TRACE1..TRACE4``.

        An unknown ``DataType`` raises (config error, not frame data). A value
        that won't convert warns and yields None, so one malformed native
        card cannot abort a reduction.
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

        Every keyword registered on PRIMARY for that data model, with its
        ``header-map.csv`` default where it has one and blank otherwise,
        each carrying its registry comment. Nothing is filtered on
        ``REQUIRED`` -- the seed stamps a card for every member of every ``#``
        family (the five-trace rule at the header level).

        For ``L0``/``L1``/``L2``/``L4`` the set is the header-map rows at
        ``Level <= n``, cumulative (156 cards at L0, L1 adds ``DQLVL1``, L2
        ``EXTRACT``/``EXSNR#``/``EXSNRW#``/``DQLVL2``, L4 the seven RV rows and
        ``DQLVL4``). For the masters it is that master's own
        ``ML*-PRIMARY-keywords.csv`` rows -- masters are outside EPRV scope and
        do not inherit the science skeleton.
        """
        extension_manifest.require(data_model)
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

    # --- Derived lookups ------------------------------------------------------

    @staticmethod
    def _routing_lookup(homes):
        """keyword -> home extension name, from ``keyword -> {extension}``.

        A keyword registered on PRIMARY routes to PRIMARY (``RVMETHOD`` and
        ``RVGREEN``/``RVRED``/``ERVGREEN``/``ERVRED`` are on PRIMARY *and*
        ``RV1..5``); otherwise a keyword with exactly one home routes there.
        Everything else -- per-extension cards recurring on every orderlet
        (``CTYPE1``, ``VELWIDTH``) -- has no single home and is written via
        ``set_keyword``'s targeted ``ext=`` path.
        """
        routing = {}
        for keyword, extensions in homes.items():
            if "PRIMARY" in extensions:
                routing[keyword] = "PRIMARY"
            elif len(extensions) == 1:
                routing[keyword] = next(iter(extensions))
        return routing


# Module singleton -- the one registry instance every consumer reaches through.
keyword_registry = KeywordRegistry()
