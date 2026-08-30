"""
KPF base data model.

Shared base for every KPF data model (L0, L1, L2, L4). Extends the EPRV Data
Standard ``RVDataModel`` with the behavior common to all KPF products: FITS-header
storage, provenance receipts, and alias-aware data and header access.

Bi-directional extension aliasing allows use of either EPRV Data Standard
or KPF naming conventions for order traces:

* TRACE1 = SKY
* TRACE2 = SCI1
* TRACE3 = SCI2
* TRACE4 = SCI3
* TRACE5 = CAL

For example, "SCI2_FLUX" is registered as an alias for "TRACE3_FLUX", so reading
``d["SCI2_FLUX"]`` returns the same object stored under "TRACE3_FLUX".
"""

import importlib.resources
import logging

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from rvdata.core.models.base import RVDataModel
from rvdata.core.models.definitions import BASE_RECEIPT_COLUMNS

# The keyword registry lives in its own module as a single KeywordRegistry
# instance; base.py is its only importer, and re-exports it (below) so sibling
# data_models files (level2/4) import the same singleton from base.
from kpfpipe.data_models.keyword_registry import keyword_registry
from kpfpipe.utils.kpf import is_obs_id

# Receipt names that are data-model conversions / serialization rather than
# pipeline modules -- excluded from DRPSTATU so it names the last real stage.
# ``from_fits`` is here too: reading a product back must not clobber the status
# the writer stamped.
_INTERNAL_RECEIPTS = frozenset(
    {"to_kpf1", "to_kpf2", "to_kpf4", "to_fits", "from_fits"}
)

_config_path = importlib.resources.files("kpfpipe.data_models.config")

# The authoritative extension manifest for each level: the complete, literal
# statement of that level's shape. Every row is created (there is no Required
# gate), so no model class declares its own manifest.
_MANIFESTS = {
    level: pd.read_csv(_config_path / f"{level}-extensions.csv")
    for level in ("L0", "L1", "L2", "L4", "ML1", "ML2-flat", "ML2-wls")
}

# Re-exported so anything importing the base also reaches the one registry
# instance. Listed in __all__ so the re-export is intentional, not an accident
# of import.
__all__ = [
    "KPFDataModel",
    "keyword_registry",
]

logger = logging.getLogger(__name__)


class KPFDataModel(RVDataModel):
    """Shared base for every KPF data model (L0, L1, L2, L4)."""

    # The keyword registry singleton, surfaced as a class attribute so anything
    # handed a KPF data model (the checkpoints validator, the WMKO->EPRV
    # standardization, tests) reaches it via kpf.keyword_registry.
    keyword_registry = keyword_registry

    # Unknown extension on read raises, matching rvdata's read contract.
    _strict_read = True

    def __init__(self):
        super().__init__()
        self.obs_id = None
        self.dirname = None

    @property
    def _manifest(self):
        """This model's extension manifest, named by its data level.

        The masters override this: their tables are ``ML{level}``.
        """
        return _MANIFESTS[f"L{self.level}"]

    def _create_manifest_extensions(self):
        """Create every extension this level's manifest declares.

        No ``Required`` gate: that column is a compliance label, so the manifest
        is a complete and literal statement of the level's shape and every row is
        created (empty) here.
        """
        for _, row in self._manifest.iterrows():
            if row["Name"] not in self.extensions:
                self.create_extension(row["Name"], row["DataType"])

    def _seed_primary(self):
        """Stamp the registry's typed PRIMARY skeleton for this data model.

        Every keyword registered on PRIMARY for this level, with its header-map
        default where it has one and blank where it does not, each carrying the
        registry comment. Later writers overlay what they know. Science levels
        only: the masters carry their own minimal PRIMARY and never seed.
        """
        seed = self.keyword_registry.primary_seed(f"L{self.level}")
        for keyword, value in seed.items():
            self.headers["PRIMARY"][keyword] = value

    def _set_ext_descript(self):
        """Rebuild EXT_DESCRIPT from the live extension set.

        A no-op for the levels whose manifest declares no such extension (L0, L1
        and the ML1 masters), so it can be called last in every ``__init__``.
        """
        if "EXT_DESCRIPT" not in self.extensions:
            return
        descriptions = dict(
            zip(self._manifest["Name"], self._manifest["Description"], strict=True)
        )
        self.set_data(
            "EXT_DESCRIPT",
            pd.DataFrame(
                [(name, descriptions.get(name, "")) for name in self.extensions],
                columns=["Name", "Description"],
            ),
        )

    def _bit_depth(self, ext_name):
        """The manifest's declared BitDepth for ``ext_name``, or None.

        None when the cell is blank or the extension is absent from the manifest.
        Named off ``_get_min_bit_depth`` deliberately: that is the hook rvdata's
        ``set_data`` calls to upcast silently, and leaving it at the inherited
        ``None`` makes ``_assert_bit_depth`` the only enforcement.
        """
        row = self._manifest[self._manifest["Name"] == ext_name]
        if row.empty:
            return None
        value = row.iloc[0]["BitDepth"]
        return None if pd.isna(value) else int(value)

    def read(self, hdul, instrument=None, overwrite=False, **kwargs):
        """Read an EPRV-standard FITS HDUList into this model.

        One read path at every level. ``RVDataModel.read`` dispatches L2/L4 to
        ``RV2._read``/``RV4._read`` through a hardcoded class reference keyed on
        ``self.level`` rather than through the MRO, so this override -- not the
        class bases -- is what detaches rvdata's name-guessing readers. It also
        drops rvdata's ``@receipt_logged`` ``read`` row and its read-time PRIMARY
        recast, so a product reads back exactly as written.
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
        the known-extension gate; an unknown extension raises when
        ``_strict_read`` (the default) and warns otherwise.
        """
        known = set(self._manifest["Name"])
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
                    message = (
                        f"Non-standard extension {ext_name!r} in L{self.level} file"
                    )
                    if self._strict_read:
                        raise ValueError(message)
                    logger.warning("%s", message)
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

        The single read chokepoint for every KPF data model: one INFO record
        per FITS read, naming the concrete class and the path, then delegate
        to ``read`` -> ``_read``, the one manifest-driven reader.

        Ensures ``obs_id`` is carried in memory after the read. L0 (and any
        product whose filename embeds the obs_id) resolves it during ``read``;
        L1/L2/L4 filenames are timestamp-based, so a read there recovers the
        obs_id from the ORIGID provenance card instead (see ``_obs_id_from_receipt``).
        The ``to_kpfN`` converters set ``obs_id`` directly, so this only fills the
        from_fits path.

        CATALOG_RECORD is normalized here for every level: astropy's FITS reader
        returns a missing (NaN) cell masked, and a masked cell is not NaN
        (``np.isnan`` on one is falsy), so a missing-value check would read it as
        present. It sits at the chokepoint, after the read, rather than inside
        ``_read``.
        """
        logger.info("reading %s from %s", cls.__name__, fn)
        obj = super().from_fits(fn, instrument=instrument, **kwargs)
        if getattr(obj, "obs_id", None) is None:
            obj.obs_id = obj._obs_id_from_receipt()
        table = obj.data.get("CATALOG_RECORD")
        if table is not None and getattr(table, "has_masked_values", False):
            obj.set_data("CATALOG_RECORD", table.filled(np.nan))
        return obj

    def _obs_id_from_receipt(self):
        """Recover the obs_id from the ORIGID card on RECEIPT (``None`` if absent
        or invalid, e.g. masters). ORIGID is stamped at L0 and forwarded on the
        RECEIPT header, so it is the obs_id source for a from_fits'd L1/L2/L4."""
        receipt = self.headers.get("RECEIPT")
        origid = receipt.get("ORIGID") if receipt is not None else None
        return origid if is_obs_id(origid) else None

    @staticmethod
    def as_fits_header(src):
        """Return ``src`` as an ``astropy.io.fits.Header``, preserving comments.

        KPF stores every extension header as a ``fits.Header`` so reads and writes
        go through astropy natively, with no value-vs-``(value, comment)``
        ambiguity. A ``fits.Header`` is returned as a copy (so callers can rebuild
        an HDU without aliasing the stored header); a plain mapping -- RVData seeds
        PRIMARY defaults as an ``OrderedDict`` of ``(value, comment)`` tuples -- is
        rebuilt card by card, setting each value and comment together.
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

        Looks ``key`` up in the keyword registry (the
        ``config/{prefix}-{EXTENSION}-keywords.csv`` tables) and writes ``value``
        to the extension named there, with the registry comment
        (``Description [Units]``) as the FITS comment. This is the single write
        path for registered keywords, so a keyword always lands on the same
        extension with the same comment -- callers never name a comment.

        ``ext`` targets a specific extension for EPRV per-extension cards that
        have no single routed home because they recur on every orderlet's
        extension (e.g. ``VELSTART`` on ``CCF1..5``, ``RVMETHOD`` on ``RV1..5``).
        The keyword must be registered *for that extension*; the comment still
        comes from the registry. Aliases (e.g. ``SCI2_CCF``) resolve first. When
        ``ext`` is None the keyword routes to its registered home as usual.

        Raises
        ------
        KeyError
            If ``key`` is registered nowhere (default), or not registered for
            ``ext`` (targeted); register it in the appropriate
            ``config/{prefix}-{EXTENSION}-keywords.csv`` before writing it.
        ValueError
            If the target extension does not exist on this object (a config
            error -- the extension must be created before the write).
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
            comment = self.keyword_registry.comment_for(name, ext)
            if comment is None:
                raise KeyError(
                    f"keyword {name!r} is not registered for extension {ext!r}; "
                    "register it in the appropriate "
                    "config/{prefix}-{EXTENSION}-keywords.csv before writing it"
                )
        if ext not in self.extensions:
            raise ValueError(
                f"cannot write {name!r}: extension {ext!r} does not exist on "
                f"{type(self).__name__}"
            )
        self.headers[ext][name] = (value, comment)

    def set_data(self, ext_name, data):
        """Set extension data, resolving KPF aliases first.

        For aliased models (KPF2/KPF4) this resolves chip-prefix keys (e.g.
        'GREEN_SCI2_FLUX', routed through the data dict's ``__setitem__``) and
        extension aliases before the base class ``.keys()`` check. The
        ``hasattr`` guards make it a no-op passthrough for non-aliased L0/L1.
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
        super().set_data(ext_name, data)
        # Sync self.receipt when the RECEIPT extension is loaded from FITS.
        if ext_name == "RECEIPT" and isinstance(data, Table):
            self.receipt = data.to_pandas()

    def _assert_bit_depth(self, canonical_ext, data, label):
        """Raise unless ``data`` matches ``canonical_ext``'s declared BitDepth.

        Width only: byte order is irrelevant (a FITS round-trip returns ``>f4``,
        still 32 bits) and ``np.bool_`` satisfies the 8-bit master MASK rows it is
        held as in memory. A no-op with no declaration, non-array data, or an
        empty array -- every extension is born ``np.array([])``, which is 64-bit,
        and reading an unpopulated product feeds that straight back through here.
        """
        depth = self._bit_depth(canonical_ext)
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

        Shared by the ``to_kpf{1,2,4}`` conversions: for each name present on both
        sides, copy every card with its FITS comment (``.items()`` would drop
        comments). Copying overlays onto the target's header rather than replacing
        it, so a PRIMARY pre-seeded with the EPRV skeleton keeps cards the source
        lacks (native values win). An extension absent on either side is skipped.
        """
        for ext in ext_names:
            if ext in self.headers and ext in target.headers:
                for card in self.as_fits_header(self.headers[ext]).cards:
                    target.headers[ext][card.keyword] = (card.value, card.comment)

    def receipt_add_entry(self, function, args, status):
        """Record a processing step, and update DRPSTATU for pipeline modules.

        Signature matches rvdata >=0.4.0: ``function`` names the step, ``args``
        is a key=value provenance string (``""`` when not applicable), ``status``
        is ``"PASS"``/``"FAIL"``.
        """
        super().receipt_add_entry(function, args, status)
        if status == "PASS":
            self._update_drpstatus(function)

    def _update_drpstatus(self, function):
        """Stamp DRPSTATU = '<Module Name> module complete' for a completed module.

        Called from ``receipt_add_entry``; conversions/serialization receipts
        (``_INTERNAL_RECEIPTS``) are skipped so DRPSTATU names the last real stage.
        DRPSTATU's registry home is RECEIPT, so ``set_keyword`` routes it there.
        """
        if function in _INTERNAL_RECEIPTS:
            return
        if "RECEIPT" not in self.extensions:
            return
        label = function.replace("_", " ").title()
        self.set_keyword("DRPSTATU", f"{label} module complete")

    def _create_hdul(self):
        """Sync ``self.receipt`` into the RECEIPT extension before writing (rvdata
        serializes ``self.data["RECEIPT"]``, not ``self.receipt``), creating the
        extension if L0/L1 omitted it. PRIMARY comments are preserved by rvdata's
        own ``_create_hdul``, so no PRIMARY rebuild is needed."""
        if self.receipt is not None and not self.receipt.empty:
            if "RECEIPT" not in self.extensions:
                self.create_extension("RECEIPT", "BinTableHDU")
            self._sync_receipt_to_extension()
        return super()._create_hdul()

    def generate_standard_filename(self):
        """Abstract: every concrete KPF model builds its own standard filename.

        KPFDataModel is never instantiated directly -- only inherited -- so reaching
        this means a subclass failed to define the method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define generate_standard_filename"
        )

    def check_filename_convention(self, filename):
        """Abstract: every concrete KPF model declares its own filename convention.

        KPFDataModel is never instantiated directly -- only inherited -- so reaching
        this means a subclass failed to define the method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define check_filename_convention"
        )
