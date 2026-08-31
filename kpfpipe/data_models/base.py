"""
KPF base data model.

Shared base for every KPF data model (L0, L1, L2, L4). Extends the EPRV Data
Standard ``RVDataModel`` with FITS-header storage, provenance receipts, and
alias-aware data/header access.

Bi-directional extension aliasing lets KPF names substitute for EPRV
order-trace names, e.g. "SCI2_FLUX" aliases "TRACE3_FLUX":

* TRACE1 = SKY
* TRACE2 = SCI1
* TRACE3 = SCI2
* TRACE4 = SCI3
* TRACE5 = CAL
"""

import logging

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from rvdata.core.models.base import RVDataModel
from rvdata.core.models.definitions import (
    BASE_DRP_CONFIG_COLUMNS,
    BASE_RECEIPT_COLUMNS,
)

# Reference-data singletons, re-exported so level2/4 import them via base.
from kpfpipe.data_models.aliased_dict import AliasedOrderedDict
from kpfpipe.data_models.config import PATH as _config_path
from kpfpipe.data_models.extension_manifest import extension_manifest
from kpfpipe.data_models.keyword_registry import keyword_registry
from kpfpipe.utils.io import check_filename_convention, kpf_filename
from kpfpipe.utils.kpf import is_obs_id

# Trace index -> fiber name, and the 1:1 KPF -> EPRV extension synonyms.
TRACE_MAP = pd.read_csv(_config_path / "trace-map.csv")
EXTENSION_ALIASES = pd.read_csv(_config_path / "extension-aliases.csv")

# Data-model conversion/serialization receipts, excluded from DRPSTATU so it
# names the last real pipeline stage. ``from_fits`` is here too: reading a
# product back must not clobber the status the writer stamped.
_INTERNAL_RECEIPTS = frozenset(
    {"to_kpf1", "to_kpf2", "to_kpf4", "to_fits", "from_fits"}
)

# Explicit __all__ marks the registry/manifest re-export as intentional.
__all__ = [
    "KPFDataModel",
    "extension_manifest",
    "keyword_registry",
]

logger = logging.getLogger(__name__)


class KPFDataModel(RVDataModel):
    """Shared base for every KPF data model (L0, L1, L2, L4)."""

    # Surfaced as class attributes so any kpf_obj reaches them via
    # kpf.keyword_registry / kpf.extension_manifest.
    keyword_registry = keyword_registry
    extension_manifest = extension_manifest

    # Per-trace aliases as (canonical template, alias suffix) pairs, e.g.
    # ("TRACE#_FLUX", "FLUX") makes SCI2_FLUX an alias of TRACE3_FLUX. Empty for L0/L1.
    _ALIAS_TEMPLATES = ()

    # Alias-aware data dict class for this level's arrays, or None for L0/L1.
    _DATA_DICT = None

    # Whether construction stamps the registry's PRIMARY skeleton. L0 does not:
    # ``_read`` replaces PRIMARY wholesale, so anything stamped here would be
    # discarded; ``standardize_header_format`` seeds it once, after the read.
    _SEEDS_PRIMARY = True

    def __init__(self):
        super().__init__()
        self.obs_id = None
        self.dirname = None

    def _build(self):
        """Build this level's extensions, tables and PRIMARY skeleton.

        Shared by every level: create manifest extensions, swap in alias-aware
        dicts and register aliases (aliased levels only), fill typed empty
        tables, seed PRIMARY and restamp DATALVL, rebuild EXT_DESCRIPT. A
        concrete model's ``__init__`` sets ``self.level`` and calls this.
        """
        self._create_manifest_extensions()
        if self._DATA_DICT is not None:
            self.extensions = AliasedOrderedDict.from_ordered_dict(self.extensions)
            self.headers = AliasedOrderedDict.from_ordered_dict(self.headers)
            self.data = self._DATA_DICT.from_ordered_dict(self.data)
            self._register_aliases()
        self._fill_typed_empty_tables()
        if self._SEEDS_PRIMARY:
            self._seed_primary()
            # DATALVL's seeded value is the L0 default; restamp it for this level.
            self.set_keyword("DATALVL", f"L{self.level}")
        self._set_ext_descript()

    def _register_aliases(self):
        """Register this level's KPF-friendly extension aliases, from config.

        Two families, both table-driven: 1:1 synonyms in ``extension-aliases.csv`` (e.g.
        CA_HK -> ANCILLARY_SPECTRUM), and per-trace ``_ALIAS_TEMPLATES`` keyed
        off ``trace-map.csv`` (e.g. SCI2_FLUX -> TRACE3_FLUX). An alias whose
        canonical extension this level lacks is skipped.
        """
        for _, row in EXTENSION_ALIASES.iterrows():
            self._register_alias(str(row["KPF"]).strip(), str(row["EPRV"]).strip())
        for _, row in TRACE_MAP.iterrows():
            trace = int(row["Trace"])
            fiber = str(row["Fiber"]).strip()
            for template, suffix in self._ALIAS_TEMPLATES:
                self._register_alias(
                    f"{fiber}_{suffix}", template.replace("#", str(trace))
                )

    def _register_alias(self, alias, canonical):
        """Register ``alias`` for ``canonical`` on all three alias-aware dicts."""
        if canonical in self.extensions:
            for d in (self.extensions, self.headers, self.data):
                d.register_alias(alias, canonical)

    @property
    def _data_model(self):
        """This model's keyword/extension tables, named by its data level.

        The masters override this: their tables are ``ML{level}``.
        """
        return f"L{self.level}"

    def _create_manifest_extensions(self):
        """Create every extension this level's manifest declares.

        No ``Required`` gate: that column is a compliance label, so every
        manifest row is created (empty) here.
        """
        for name in self.extension_manifest.names(self._data_model):
            if name not in self.extensions:
                self.create_extension(
                    name, self.extension_manifest.fits_type(self._data_model, name)
                )

    def _seed_primary(self):
        """Stamp the registry's typed PRIMARY skeleton for this data model.

        Every keyword registered on PRIMARY, with its header-map default where
        it has one and blank otherwise, each carrying the registry comment.
        Later writers overlay what they know. Masters land on their own
        minimal ``ML*`` skeleton, not the science one.
        """
        seed = self.keyword_registry.primary_seed(self._data_model)
        for keyword, value in seed.items():
            self.headers["PRIMARY"][keyword] = value

    def _fill_typed_empty_tables(self):
        """Give the structural extensions their empty typed skeletons.

        INSTRUMENT_HEADER's placeholder array, plus the EPRV RECEIPT and
        DRP_CONFIG column sets. Each is gated on membership, since not every
        level declares all three. RECEIPT is built as an astropy Table
        directly, not via pandas: ``Table.from_pandas`` collapses empty
        columns to float64 regardless of pandas dtype. L2 and L4 extend this
        with their own tables.
        """
        if "INSTRUMENT_HEADER" in self.extensions:
            self.set_data("INSTRUMENT_HEADER", np.zeros((1,), dtype=np.float32))
        if "RECEIPT" in self.extensions:
            self.set_data(
                "RECEIPT",
                Table(
                    {
                        c: np.array([], dtype="U256")
                        for c in BASE_RECEIPT_COLUMNS["Name"]
                    }
                ),
            )
        if "DRP_CONFIG" in self.extensions:
            self.set_data(
                "DRP_CONFIG",
                pd.DataFrame(columns=BASE_DRP_CONFIG_COLUMNS["Name"].tolist()),
            )

    def _set_ext_descript(self):
        """Rebuild EXT_DESCRIPT from the live extension set.

        A no-op where the manifest declares no such extension (L0, L1, ML1).
        """
        if "EXT_DESCRIPT" not in self.extensions:
            return
        self.set_data(
            "EXT_DESCRIPT",
            pd.DataFrame(
                [
                    (name, self.extension_manifest.description(self._data_model, name))
                    for name in self.extensions
                ],
                columns=["Name", "Description"],
            ),
        )

    def read(self, hdul, instrument=None, overwrite=False, **kwargs):
        """Read an EPRV-standard FITS HDUList into this model.

        One read path at every level. ``RVDataModel.read`` dispatches L2/L4 via
        a hardcoded class reference keyed on ``self.level``, bypassing the MRO
        -- this override, not the class bases, is what detaches rvdata's
        name-guessing readers. It also drops rvdata's ``@receipt_logged`` row
        and read-time PRIMARY recast, so a product reads back exactly as
        written.
        """
        if instrument is not None:
            raise ValueError(
                f"{type(self).__name__}.read reads EPRV-standard FITS only; "
                f"instrument={instrument!r} is not supported"
            )
        self._read(hdul)

    def _read(self, hdul):
        """Read every extension from an EPRV-standard FITS HDUList.

        The FITS type comes from the astropy HDU class, and the manifest supplies
        the known-extension gate; an unknown extension raises.
        """
        known = set(self.extension_manifest.names(self._data_model))
        for hdu in hdul:
            ext_name = hdu.name

            if isinstance(hdu, fits.PrimaryHDU):
                fits_type = "PrimaryHDU"
            elif isinstance(hdu, (fits.ImageHDU, fits.CompImageHDU)):
                fits_type = "ImageHDU"
            elif isinstance(hdu, fits.BinTableHDU):
                fits_type = "BinTableHDU"
            else:
                continue

            if ext_name != "PRIMARY" and ext_name not in self.extensions:
                if ext_name not in known:
                    raise ValueError(
                        f"Non-standard extension {ext_name!r} in L{self.level} file"
                    )
                self.create_extension(ext_name, fits_type)

            if ext_name == "PRIMARY":
                pass
            elif ext_name == "RECEIPT":
                df = Table.read(hdu).to_pandas()
                receipt_columns = BASE_RECEIPT_COLUMNS["Name"].tolist()
                if df.empty:
                    df = pd.DataFrame(columns=receipt_columns)
                else:
                    all_cols = df.columns.union(receipt_columns, sort=False)
                    df = df.reindex(columns=all_cols).fillna("")
                self.receipt = df
            elif fits_type == "ImageHDU":
                # np.array (not asarray) materializes the memmapped HDU into RAM
                # before from_fits closes the file; a view would dangle afterward.
                self.set_data(ext_name, np.array(hdu.data))
            elif fits_type == "BinTableHDU":
                self.set_data(ext_name, Table.read(hdu))

            self.set_header(ext_name, hdu.header)

    @classmethod
    def from_fits(cls, fn, instrument=None, **kwargs):
        """Read a data product from FITS, logging the file read (DRP-RUN-08).

        The single read chokepoint: one INFO record per read, then delegate to
        ``read`` -> ``_read``.

        Ensures ``obs_id`` is set after the read. L0 resolves it during
        ``read``; L1/L2/L4 filenames are timestamp-based, so it is recovered
        from the ORIGID provenance card instead (see
        ``_obs_id_from_primary``). The ``to_kpfN`` converters set ``obs_id``
        directly, so this only covers the from_fits path.

        Normalizes CATALOG_RECORD here: astropy's FITS reader marks a missing
        cell masked rather than NaN, so ``np.isnan`` would miss it as present.
        """
        logger.info("reading %s from %s", cls.__name__, fn)
        obj = super().from_fits(fn, instrument=instrument, **kwargs)
        if getattr(obj, "obs_id", None) is None:
            obj.obs_id = obj._obs_id_from_primary()
        table = obj.data.get("CATALOG_RECORD")
        if table is not None and getattr(table, "has_masked_values", False):
            obj.set_data("CATALOG_RECORD", table.filled(np.nan))
        return obj

    def _obs_id_from_primary(self):
        """Recover the obs_id from the ORIGID card on PRIMARY (``None`` if absent
        or invalid, e.g. masters). ORIGID is stamped at L0 and forwarded on the
        PRIMARY header, so it is the obs_id source for a from_fits'd L1/L2/L4."""
        primary = self.headers.get("PRIMARY")
        origid = primary.get("ORIGID") if primary is not None else None
        return origid if is_obs_id(origid) else None

    @staticmethod
    def as_fits_header(src):
        """Return ``src`` as an ``astropy.io.fits.Header``, preserving comments.

        KPF stores every extension header as a ``fits.Header`` so reads/writes
        go through astropy natively. A ``fits.Header`` is returned as a copy
        (so callers can rebuild an HDU without aliasing the stored header); a
        plain mapping -- RVData seeds PRIMARY defaults as an ``OrderedDict``
        of ``(value, comment)`` tuples -- is rebuilt card by card.
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

    def set_keyword(self, key, value, ext=None):
        """Write a registered keyword to its home extension header.

        Looks ``key`` up in the keyword registry and writes ``value`` to its
        registered extension, with the registry comment (``Description
        [Units]``) as the FITS comment -- the single write path, so callers
        never name a comment.

        ``value`` is coerced to the registry ``DataType`` (the header
        counterpart of the manifest ``BitDepth`` check on ``set_data``); a
        value that will not convert raises rather than landing wrong. None
        writes the blank seeded card through unchanged.

        ``ext`` targets a specific extension for EPRV per-extension cards with
        no single routed home (e.g. ``VELSTART`` on ``CCF1..5``, ``RVMETHOD``
        on ``RV1..5``); the keyword must be registered for that extension.
        Aliases (e.g. ``SCI2_CCF``) resolve first.

        Raises
        ------
        KeyError
            ``key`` is registered nowhere (default), or not for ``ext``
            (targeted).
        ValueError
            The target extension does not exist on this object.
        TypeError
            ``value`` does not convert to the registered ``DataType``.
        """
        name = str(key).strip()
        if ext is None:
            ext = self.keyword_registry.routing.get(name)
            if ext is None:
                raise KeyError(
                    f"keyword {name!r} is not registered; add it to the "
                    "appropriate config/{prefix}-{EXTENSION}-keywords.csv before "
                    "writing it"
                )
            # A routed home is a real registry row, so the comment cannot miss.
            comment = self.keyword_registry.comment_for(name, ext)
        else:
            if hasattr(self.extensions, "_resolve"):
                ext = self.extensions._resolve(ext)
            if not self.keyword_registry.is_registered(name, ext):
                raise KeyError(
                    f"keyword {name!r} is not registered for extension {ext!r}; "
                    "register it in the appropriate "
                    "config/{prefix}-{EXTENSION}-keywords.csv before writing it"
                )
            comment = self.keyword_registry.comment_for(name, ext)
        if ext not in self.extensions:
            raise ValueError(
                f"cannot write {name!r}: extension {ext!r} does not exist on "
                f"{type(self).__name__}"
            )
        datatype = self.keyword_registry.datatype_for(name, ext)
        if value is not None and datatype:
            try:
                value = self.keyword_registry.coerce(datatype.lower(), value)
            except (KeyError, TypeError, ValueError) as exc:
                raise TypeError(
                    f"keyword {name!r} on {ext!r} is declared {datatype}; "
                    f"cannot write {value!r}: {exc}"
                ) from None
        self.headers[ext][name] = (value, comment)

    def set_data(self, ext_name, data):
        """Set extension data, resolving KPF aliases first.

        For aliased models (KPF2/KPF4), resolves chip-prefix keys (e.g.
        'GREEN_SCI2_FLUX') and extension aliases before the base class
        ``.keys()`` check. The ``hasattr`` guards make this a no-op
        passthrough for L0/L1.
        """
        if hasattr(self.data, "_chip_split"):
            split = self.data._chip_split(ext_name)
            if split is not None:
                canonical = self.data._resolve(split[0])
                self._assert_bit_depth(canonical, data, ext_name)
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
        self._assert_bit_depth(ext_name, data, ext_name)
        # rvdata's upcast branch is unreachable: gated on its own
        # _get_min_bit_depth, which KPF never overrides.
        super().set_data(ext_name, data)
        # Sync self.receipt when the RECEIPT extension is loaded from FITS.
        if ext_name == "RECEIPT" and isinstance(data, Table):
            self.receipt = data.to_pandas()

    def _assert_bit_depth(self, canonical_ext, data, label):
        """Raise unless ``data`` matches ``canonical_ext``'s declared BitDepth.

        Width only: byte order is irrelevant (FITS round-trip returns ``>f4``,
        still 32 bits), and ``np.bool_`` satisfies the 8-bit master MASK rows.
        A no-op with no declaration, non-array data, or an empty array --
        every extension is born ``np.array([])`` (64-bit), so reading an
        unpopulated product must pass.
        """
        depth = self.extension_manifest.bit_depth(self._data_model, canonical_ext)
        if (
            depth is None
            or not isinstance(data, np.ndarray)
            or data.size == 0
            or data.dtype.itemsize * 8 == depth
        ):
            return
        raise TypeError(
            f"{label}: manifest declares {depth}-bit, got {data.dtype} "
            f"({data.dtype.itemsize * 8}-bit)"
        )

    def set_header(self, ext_name, header):
        """Set an extension header, resolving KPF aliases before the base class
        ``.keys()`` check (a no-op for non-aliased L0/L1)."""
        if hasattr(self.extensions, "_resolve"):
            ext_name = self.extensions._resolve(ext_name)
        super().set_header(ext_name, header)

    def _forward_headers(self, target, ext_names):
        """Forward governed extension headers onto ``target``, card by card.

        Shared by the ``to_kpf{1,2,4}`` conversions. Copies every card with
        its FITS comment (``.items()`` would drop comments), overlaying onto
        the target's header rather than replacing it, so a PRIMARY pre-seeded
        with the EPRV skeleton keeps cards the source lacks. An extension
        absent on either side is skipped.
        """
        for ext in ext_names:
            if ext in self.headers and ext in target.headers:
                for card in self.as_fits_header(self.headers[ext]).cards:
                    target.headers[ext][card.keyword] = (card.value, card.comment)

    def receipt_add_entry(self, function, args, status):
        """Record a processing step, and stamp DRPSTATU for pipeline modules.

        Signature matches rvdata >=0.4.0: ``function`` names the step, ``args``
        is a key=value provenance string (``""`` when not applicable),
        ``status`` is ``"PASS"``/``"FAIL"``.

        DRPSTATU becomes '<Module Name> module complete'; conversion/
        serialization receipts (``_INTERNAL_RECEIPTS``) are skipped so it
        names the last real stage.
        """
        super().receipt_add_entry(function, args, status)
        if (
            status == "PASS"
            and function not in _INTERNAL_RECEIPTS
            and "RECEIPT" in self.extensions
        ):
            label = function.replace("_", " ").title()
            self.set_keyword("DRPSTATU", f"{label} module complete")

    def receipt_read_entry(self, function):
        """The provenance ARGS of the most recent ``function`` receipt row.

        ARGS is a ``", "``-joined list of ``key=value`` fragments; returns them
        as a dict of strings, empty when ``function`` has no row. A fragment
        written from ``None`` reads back as ``None``, so a key recorded without
        a value and a key never recorded both ``get()`` as ``None``.
        """
        if self.receipt is None or self.receipt.empty:
            return {}
        rows = self.receipt[self.receipt["FUNCTION"] == function]
        if rows.empty:
            return {}
        entry = {}
        for token in str(rows.iloc[-1]["ARGS"]).split(", "):
            key, _, value = token.partition("=")
            if value:
                entry[key.strip()] = None if value == "None" else value.strip()
        return entry

    def _create_hdul(self):
        """Sync ``self.receipt`` into the RECEIPT extension before writing
        (rvdata serializes ``self.data["RECEIPT"]``, not ``self.receipt``),
        creating the extension if L0/L1 omitted it.
        """
        if self.receipt is not None and not self.receipt.empty:
            if "RECEIPT" not in self.extensions:
                self.create_extension("RECEIPT", "BinTableHDU")
            self._sync_receipt_to_extension()
        return super()._create_hdul()

    def to_fits(self, fn=None):
        """Write this product to ``fn``, defaulting to its standard filename.

        The single write path for every level. Bridges KPF's one-filepath
        signature to rvdata >=0.4.0's ``out_filename`` parameter; rvdata does
        the rest -- receipt row, warn-only filename check, FILENAME stamp,
        ``_create_hdul``, then ``makedirs``/``writeto``.
        """
        if fn is None:
            fn = self.generate_standard_filename()
        out_path = super().to_fits(out_filename=fn)
        logger.info("wrote %s to %s", type(self).__name__, out_path)
        return out_path

    def info(self):
        """Print a summary of this product's extensions.

        One row per extension -- name, type, shape/size -- with an Aliases column
        for the levels whose extensions carry KPF-friendly synonyms (L2/L4).
        """
        if self.filename:
            print(f"KPF L{self.level}: {self.filename}")
        else:
            print(f"Empty {type(self).__name__} data product")
        if self.obs_id:
            print(f"Obs ID: {self.obs_id}")

        aliased = hasattr(self.extensions, "aliases_for")
        width = 25 if aliased else 20
        header = f"\n{'Extension':<{width}s} "
        if aliased:
            header += f"{'Aliases':<25s} "
        print(f"{header}{'Type':<15s} {'Shape/Size':<20s}")
        print("=" * (85 if aliased else 55))

        for name, ext_type in self.extensions.items():
            if name == "PRIMARY":
                kind = "header"
                size = f"{len(self.headers.get(name, {}))} cards"
            else:
                ext = self.data.get(name)
                if isinstance(ext, np.ndarray) and ext.size > 0:
                    kind, size = "array", str(ext.shape)
                elif hasattr(ext, "__len__") and len(ext) > 0:
                    kind, size = "table", f"{len(ext)} rows"
                else:
                    kind, size = ext_type, "(empty)"
            row = f"{name:<{width}s} "
            if aliased:
                aliases = () if name == "PRIMARY" else self.extensions.aliases_for(name)
                row += f"{', '.join(sorted(aliases)):<25s} "
            print(f"{row}{kind:<15s} {size:<20s}")

    def generate_standard_filename(self):
        """This product's standard basename, built from ``obs_id`` and its level.

        Raises
        ------
        ValueError
            If ``obs_id`` is unset or invalid.
        """
        return kpf_filename(self.obs_id, f"L{self.level}")

    def check_filename_convention(self, filename):
        """Warn-only check of ``filename`` against this level's naming rule."""
        return check_filename_convention(filename, f"L{self.level}")
