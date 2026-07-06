"""
KPF-specific base data model.

Thin layer on top of RVDataModel that adds KPF-specific attributes and the
behaviour shared by every KPF data model: fits.Header storage, the DRPSTATU
receipt stamp, alias-aware set_data/set_header, and lossless PRIMARY
serialization. All four KPF models inherit it — L0/L1 directly, and L2/L4 via
multiple inheritance alongside rvdata's RV2/RV4 (KPFDataModel listed first so
its overrides win while RV2/RV4 remain reachable through ``super()``).
"""

import logging
import warnings

import numpy as np
from astropy.io import fits
from astropy.table import Table
from rvdata.core.models.base import RVDataModel

# The header/extension keyword registry lives in its own module as a single
# KeywordRegistry instance; base.py is its only importer. The instance is
# surfaced as a KPFDataModel class attribute (below) so consumers handed a
# kpf_obj (the checkpoints validator, level0's WMKO->EPRV mapping, tests) reach
# it through kpf.keyword_registry, and re-exported so sibling data_models files
# (level2/4) import the same singleton from base.
from kpfpipe.data_models.keyword_registry import keyword_registry
from kpfpipe.utils.kpf import is_obs_id

# Receipt names that are data-model conversions / serialization rather than
# pipeline modules — excluded from DRPSTATU so it names the last real stage.
# ``read``/``from_fits`` are here too: reading a product back must not clobber
# the status the writer stamped (rvdata >=0.4.0 logs a ``read`` receipt via its
# ``@receipt_logged`` decorator on ``RVDataModel.read``).
_INTERNAL_RECEIPTS = frozenset(
    {"to_kpf1", "to_kpf2", "to_kpf4", "to_fits", "from_fits", "read"}
)

# Re-exported for sibling data_models modules: KPF2/KPF4 call
# keyword_registry.register_rvdata_extension at import. Listed in __all__ so the
# re-export is intentional, not an accident of import.
__all__ = [
    "KPFDataModel",
    "keyword_registry",
]

logger = logging.getLogger(__name__)


class KPFDataModel(RVDataModel):
    """Shared base for every KPF data model (L0, L1, and — multiply-inherited
    with RV2/RV4 — L2, L4)."""

    # The keyword registry singleton, surfaced as a class attribute so anything
    # handed a KPF data model (the checkpoints header validator, level0's
    # WMKO->EPRV mapping, tests) reaches it via kpf.keyword_registry — keeping
    # data_models/keyword_registry imported only by base.py. set_keyword uses
    # .routing; the validator uses .allowed / .required / .structural; .table is
    # the source table and .registered the allowlist.
    keyword_registry = keyword_registry

    def __init__(self):
        super().__init__()
        self.obs_id = None
        self.dirname = None

    @classmethod
    def from_fits(cls, fn, instrument=None, **kwargs):
        """Read a data product from FITS, logging the file read (DRP-RUN-08).

        The single read chokepoint for every KPF data model: one INFO record
        per FITS read, naming the concrete class and the path, then delegate
        to the inherited reader (rvdata's for L2/L4 via RV2/RV4 in the MRO).

        Ensures ``obs_id`` is carried in memory after the read. L0 (and any
        product whose filename embeds the obs_id) resolves it during ``read``;
        L1/L2/L4 filenames are timestamp-based, so a read there recovers the
        obs_id from the ORIGID provenance card instead (see ``_obs_id_from_receipt``).
        The ``to_kpfN`` converters set ``obs_id`` directly, so this only fills the
        from_fits path.
        """
        logger.info("reading %s from %s", cls.__name__, fn)
        obj = super().from_fits(fn, instrument=instrument, **kwargs)
        if getattr(obj, "obs_id", None) is None:
            obj.obs_id = obj._obs_id_from_receipt()
        return obj

    def _obs_id_from_receipt(self):
        """Recover the obs_id from the ORIGID provenance card on RECEIPT.

        ORIGID is the original L0 obs_id, stamped by ``KPF0`` and forwarded on the
        RECEIPT header through every level, so it is the obs_id source for a
        from_fits'd L1/L2/L4 product (whose own filename does not embed it).
        Returns ``None`` when RECEIPT/ORIGID is absent or not a valid obs_id
        (e.g. masters, which carry no ORIGID).
        """
        receipt = self.headers.get("RECEIPT")
        origid = receipt.get("ORIGID") if receipt is not None else None
        return origid if is_obs_id(origid) else None

    @staticmethod
    def as_fits_header(src):
        """Return ``src`` as an ``astropy.io.fits.Header``, preserving comments.

        KPF stores every extension header as a ``fits.Header`` so reads and writes
        go through astropy natively, with no value-vs-``(value, comment)``
        ambiguity. This is the single bridge from the two legacy in-memory forms:

        - a ``fits.Header`` (e.g. from ``from_fits``) is returned as a copy, so a
          caller can rebuild an HDU without aliasing the stored header;
        - a plain mapping (RVData seeds PRIMARY defaults as an ``OrderedDict`` whose
          entries are ``(value, comment)`` tuples) is rebuilt card by card —
          ``head[kw] = (value, comment)`` sets the value and comment together.
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

        Looks ``key`` up in the merged KPF + EPRV keyword registry
        (``config/L{0,1,2,4}-headers.csv`` plus the EPRV PRIMARY keywords) and
        writes ``value`` to the extension named there, with the registry
        Description as the FITS comment. This is the single write path for
        registered keywords, so a keyword always lands on the same extension with
        the same comment — callers never name a comment.

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
            ``config/L{level}-headers.csv`` before writing it.
        ValueError
            If the target extension does not exist on this object (a config
            error — the extension must be created before the write).
        """
        name = str(key).strip()
        if ext is None:
            route = self.keyword_registry.routing.get(name)
            if route is None:
                raise KeyError(
                    f"keyword {name!r} is not registered; add it to "
                    "config/L{0,1,2,4}-headers.csv before writing it"
                )
            ext, comment = route
        else:
            if hasattr(self.extensions, "_resolve"):
                ext = self.extensions._resolve(ext)
            comment = self.keyword_registry.comment_for(name, ext)
            if comment is None:
                raise KeyError(
                    f"keyword {name!r} is not registered for extension {ext!r}; "
                    "register it in the appropriate config/L{0,1,2,4}-headers.csv "
                    "before writing it"
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
                # The chip-prefix write goes straight into the underlying
                # canonical array (sized from the first write's dtype), bypassing
                # rvdata's MinBitDepth check in super().set_data. Enforce it here
                # so both write paths honor the born-64 WAVE / 8-bit policy.
                canonical = self.data._resolve(split[0])
                data = self._coerce_to_min_bit_depth(canonical, data, ext_name)
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
        super().set_data(ext_name, data)
        # Sync self.receipt when the RECEIPT extension is loaded from FITS.
        if ext_name == "RECEIPT" and isinstance(data, Table):
            self.receipt = data.to_pandas()

    def _coerce_to_min_bit_depth(self, canonical_ext, data, label):
        """Upcast ``data`` to satisfy ``canonical_ext``'s MinBitDepth, mirroring
        rvdata's base ``set_data`` enforcement (incl. its warning) for the
        chip-prefix write path that bypasses it. ``canonical_ext`` is the resolved
        extension whose requirement applies (e.g. ``TRACE3_WAVE``); ``label`` is
        the name shown in the warning (the chip-prefix key the caller passed).
        A no-op when there is no requirement, the array is empty, or it already
        meets the depth (and for KPF4, whose ``_get_min_bit_depth`` returns None).
        """
        min_depth = self._get_min_bit_depth(canonical_ext)
        if (
            min_depth is None
            or not isinstance(data, np.ndarray)
            or data.size == 0
            or data.dtype.itemsize * 8 >= min_depth
        ):
            return data
        if np.issubdtype(data.dtype, np.floating):
            target = np.dtype(f"float{min_depth}")
        elif np.issubdtype(data.dtype, np.unsignedinteger):
            target = np.dtype(f"uint{min_depth}")
        else:
            target = np.dtype(f"int{min_depth}")
        warnings.warn(
            f"Extension '{label}' has dtype {data.dtype} "
            f"({data.dtype.itemsize * 8}-bit) but MinBitDepth={min_depth}. "
            f"Upcasting to {target.name}.",
            UserWarning,
            stacklevel=2,
        )
        return data.astype(target)

    def set_header(self, ext_name, header):
        """Set an extension header, resolving KPF aliases before the base class
        ``.keys()`` check (a no-op for non-aliased L0/L1)."""
        if hasattr(self.extensions, "_resolve"):
            ext_name = self.extensions._resolve(ext_name)
        super().set_header(ext_name, header)

    def _forward_headers(self, target, ext_names):
        """Forward governed extension headers onto ``target``, card by card.

        The single home for the header carry-over shared by the
        ``to_kpf{1,2,4}`` level-up conversions. For each name present on both
        sides, every card is copied with its FITS comment (iterating
        ``.items()`` would drop comments). Copying *overlays* onto the target's
        existing header rather than replacing it, so a PRIMARY pre-seeded with
        the EPRV skeleton keeps cards the source lacks (native values win), and
        INSTRUMENT_HEADER / QUALITY_CONTROL / RECEIPT — created empty on the
        target — receive a verbatim, comment-preserving copy. An extension
        absent on either side is skipped.
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
        """Sync ``self.receipt`` into the RECEIPT extension before writing; rvdata
        serializes ``self.data["RECEIPT"]``, not ``self.receipt``. L0/L1 omit
        RECEIPT from their default extensions, so create it if absent. PRIMARY
        keyword comments are preserved by rvdata's own ``_create_hdul`` (it copies
        a ``fits.Header`` directly) as of rvdata >=0.4.0, so no PRIMARY rebuild is
        needed here."""
        if self.receipt is not None and not self.receipt.empty:
            if "RECEIPT" not in self.extensions:
                self.create_extension("RECEIPT", "BinTableHDU")
            self._sync_receipt_to_extension()
        return super()._create_hdul()

    def generate_standard_filename(self):
        """Abstract: every concrete KPF model builds its own standard filename.

        KPFDataModel is never instantiated directly — only inherited — so reaching
        this means a subclass failed to define the method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define generate_standard_filename"
        )

    def check_filename_convention(self, filename):
        """Abstract: every concrete KPF model declares its own filename convention.

        KPFDataModel is never instantiated directly — only inherited — so reaching
        this means a subclass failed to define the method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define check_filename_convention"
        )
