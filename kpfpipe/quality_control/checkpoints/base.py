"""Checkpoint framework base class.

The third quality-control stage (after Diagnostics and QC). A Checkpoint
subclass READS the 0/1 QC flags written to QUALITY_CONTROL and the product
headers, then emits warnings or raises errors -- it never writes keywords. A
method opts in by setting ``_checkpoint_name`` on the function object; ``run()``
walks all such methods (MRO order) and calls each. The pipeline order is:
science modules -> Diagnostics -> QC -> Checkpoints.

Two responsibilities, both inherited base checkpoints:
  - ``unregistered_keywords`` -- structural header validation: any non-structural
    card not registered for its extension raises ``ValueError`` (this logic used
    to live in ``QC._validate_headers``).
  - ``qc_flags`` -- read each 0/1 QC flag; a failed (0) flag named in the
    subclass's ``RAISE_FLAGS`` raises, every other failed flag warns.

Severity policy lives in the per-level subclasses (their ``RAISE_FLAGS``).
"""

import warnings

# Bookkeeping/structural cards astropy adds to a serialized extension header
# (BinTable column descriptors, image WCS); always permitted on any governed
# extension and never treated as unregistered. These supplement the registry's
# structural set (the PRIMARY/bookkeeping cards). Moved here from QC when header
# validation became a Checkpoint responsibility.
_EXT_STRUCTURAL_PREFIXES = (
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
_EXT_STRUCTURAL_EXTRA = {"EXTNAME", "TFIELDS", "PCOUNT", "GCOUNT"}

# Registry ``PopulatedBy`` values that mark a keyword as a 0/1 QC flag (so the
# qc_flags checkpoint can find them without a hardcoded list).
_QC_POPULATORS = {"QC", "QCL0", "QCL1", "QCL2"}


class Checkpoint:
    """Base runner for per-level checkpoint methods.

    Parameters
    ----------
    kpf_obj : KPFDataModel
        Finished data product whose flags/headers are read (never written).
    """

    LEVEL = None  # Subclasses set the level tag ("L0", "L1", "L2").
    RAISE_FLAGS = ()  # QC keywords whose failure (0) raises; every other 0 warns.

    def __init__(self, kpf_obj):
        self.kpf_obj = kpf_obj

    def run(self):
        """Run every checkpoint method; each warns or raises (no return value)."""
        for _name, fn in self._iter_checkpoints():
            fn()

    def unregistered_keywords(self):
        """Raise on any non-structural card not registered for its extension.

        For each registry-governed extension present on the product (PRIMARY only
        where it is EPRV-standard -- L1 onward; the raw WMKO L0 PRIMARY is
        skipped), a card that is neither structural nor registered (in
        ``keyword_registry.allowed[ext]``) raises ``ValueError`` and names it.
        This subsumes the old WMKO-native leak check: a raw instrument keyword
        kept in INSTRUMENT_HEADER is simply unregistered for an EPRV PRIMARY.
        """
        reg = self.kpf_obj.keyword_registry
        skip_primary = str(self.LEVEL).upper() in ("L0", "NONE")
        for ext, allowed in reg.allowed.items():
            if ext not in self.kpf_obj.extensions:
                continue
            if ext == "PRIMARY" and skip_primary:
                continue
            header = self.kpf_obj.headers.get(ext)
            if header is None:
                continue
            for raw_key in list(header):
                key = str(raw_key).strip()
                if self._is_structural(key) or key in allowed:
                    continue
                raise ValueError(
                    f"unregistered keyword {key!r} on {ext}; add it to "
                    "config/L{0,1,2,4}-headers.csv or fix the writer"
                )

    unregistered_keywords._checkpoint_name = "unregistered_keywords"

    def qc_flags(self):
        """Read each 0/1 QC flag; raise a RAISE_FLAGS failure, warn the rest.

        QC-flag keywords are the QUALITY_CONTROL rows the registry attributes to a
        QC populator. A flag absent from the header is skipped (the check did not
        run); a flag equal to 0 raises if it is in ``RAISE_FLAGS``, else warns.
        """
        header = self.kpf_obj.headers.get("QUALITY_CONTROL")
        if header is None:
            return
        reg = self.kpf_obj.keyword_registry
        flag_keys = {
            row.Keyword
            for row in reg.table.itertuples(index=False)
            if row.Extension == "QUALITY_CONTROL" and row.PopulatedBy in _QC_POPULATORS
        }
        for key in sorted(flag_keys):
            value = header.get(key)
            if value is None or value != 0:
                continue
            if key in self.RAISE_FLAGS:
                raise ValueError(f"QC checkpoint failed: {key} = 0 ({self.LEVEL})")
            warnings.warn(
                f"QC checkpoint flagged: {key} = 0 ({self.LEVEL})",
                UserWarning,
                stacklevel=2,
            )

    qc_flags._checkpoint_name = "qc_flags"

    def _is_structural(self, key):
        """True for a FITS structural/bookkeeping card (not a registered keyword).

        Combines the registry's structural cards (PRIMARY/bookkeeping) with the
        extension-table/WCS cards astropy adds at serialization.
        """
        return (
            key in self.kpf_obj.keyword_registry.structural
            or key in _EXT_STRUCTURAL_EXTRA
            or key.startswith(_EXT_STRUCTURAL_PREFIXES)
        )

    def _iter_checkpoints(self):
        """Yield each checkpoint method tagged with `_checkpoint_name`.

        Walks the MRO so subclass methods come before the base class and ordering
        is stable across runs.

        Yields
        ------
        tuple
            ``(name, bound_method)`` for each tagged checkpoint method.
        """
        seen = set()
        for cls in type(self).__mro__:
            for name, attr in cls.__dict__.items():
                if name in seen:
                    continue
                if not callable(attr):
                    continue
                if getattr(attr, "_checkpoint_name", None) is None:
                    continue
                seen.add(name)
                yield name, getattr(self, name)
