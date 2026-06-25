"""
KPF-specific base data model.

Thin layer on top of RVDataModel that adds KPF-specific attributes and the
behaviour shared by every KPF data model: fits.Header storage, the DRPSTATU
receipt stamp, alias-aware set_data/set_header, and lossless PRIMARY
serialization. All four KPF models inherit it — L0/L1 directly, and L2/L4 via
multiple inheritance alongside rvdata's RV2/RV4 (KPFDataModel listed first so
its overrides win while RV2/RV4 remain reachable through ``super()``).
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table
from rvdata.core.models.base import RVDataModel

from kpfpipe.utils.kpf import _DATECODE_PATTERN, _OBS_ID_PATTERN

# Receipt names that are data-model conversions / serialization rather than
# pipeline modules — excluded from DRPSTATU so it names the last real stage.
# ``from_fits`` is here too: reading a product back must not clobber the status
# the writer stamped.
_INTERNAL_RECEIPTS = frozenset({"to_l1", "to_kpf2", "to_kpf4", "to_fits", "from_fits"})


class KPFDataModel(RVDataModel):
    """Shared base for every KPF data model (L0, L1, and — multiply-inherited
    with RV2/RV4 — L2, L4)."""

    OBS_ID_PATTERN = _OBS_ID_PATTERN
    DATECODE_PATTERN = _DATECODE_PATTERN

    def __init__(self):
        super().__init__()
        self.obs_id = None

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

    def set_data(self, ext_name, data):
        """Set extension data, resolving KPF aliases first.

        For aliased models (KPF2/KPF4) this resolves chip-prefix keys (e.g.
        'GREEN_SCI2_FLUX', routed through the data dict's ``__setitem__``) and
        extension aliases before the base class ``.keys()`` check. The
        ``hasattr`` guards make it a no-op passthrough for non-aliased L0/L1.
        """
        if (
            hasattr(self.data, "_chip_split")
            and self.data._chip_split(ext_name) is not None
        ):
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

    def set_header(self, ext_name, header):
        """Set an extension header, resolving KPF aliases before the base class
        ``.keys()`` check (a no-op for non-aliased L0/L1)."""
        if hasattr(self.extensions, "_resolve"):
            ext_name = self.extensions._resolve(ext_name)
        super().set_header(ext_name, header)

    def receipt_add_entry(self, module, status):
        """Record a processing step, and update DRPSTATU for pipeline modules."""
        super().receipt_add_entry(module, status)
        if status == "PASS":
            self._update_drpstatus(module)

    def _update_drpstatus(self, module):
        """Stamp DRPSTATU = '<Module Name> module complete' for a completed module.

        Called from ``receipt_add_entry``; conversions/serialization receipts
        (``_INTERNAL_RECEIPTS``) are skipped so DRPSTATU names the last real stage.
        """
        if module in _INTERNAL_RECEIPTS:
            return
        primary = self.headers.get("PRIMARY")
        if primary is None:
            return
        label = module.replace("_", " ").title()
        primary["DRPSTATU"] = (
            f"{label} module complete",
            "DRP reduction status (DRP-RUN-20)",
        )

    def _create_hdul(self):
        """Sync self.receipt into the RECEIPT extension before writing; rvdata
        serializes self.data["RECEIPT"], not self.receipt. L0/L1 omit RECEIPT
        from their default extensions, so create it if absent."""
        if self.receipt is not None and not self.receipt.empty:
            if "RECEIPT" not in self.extensions:
                self.create_extension("RECEIPT", "BinTableHDU")
            self.data["RECEIPT"] = Table.from_pandas(self.receipt)
        return self._restore_primary_comments(super()._create_hdul())

    def _restore_primary_comments(self, hdu_list):
        """Rebuild the PRIMARY HDU so its keyword comments survive serialization.

        RVData's ``_create_hdul`` builds PRIMARY by iterating
        ``headers["PRIMARY"].items()``, which on a ``fits.Header`` yields only
        ``(key, value)`` and silently drops the comments. Replace that one HDU with
        a fresh ``PrimaryHDU`` built from the stored header (a ``fits.Header`` that
        holds the comments). The ``PrimaryHDU`` constructor re-adds the structural
        cards.
        """
        primary = self.headers.get("PRIMARY")
        if primary is None:
            return hdu_list
        for i, hdu in enumerate(hdu_list):
            if isinstance(hdu, fits.PrimaryHDU):
                hdu_list[i] = fits.PrimaryHDU(header=self.as_fits_header(primary))
                break
        return hdu_list

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
