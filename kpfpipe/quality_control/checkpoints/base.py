"""Checkpoint framework base class.

The third quality-control stage (after Diagnostics and QC). A Checkpoint's
checkpoint methods READ the 0/1 QC flags written to QUALITY_CONTROL and the
product headers, then emit warnings or raise errors -- they never write
keywords. A method opts in by setting ``_checkpoint_name`` on the function
object; ``run()`` walks all such methods (MRO order) and calls each.

``run()`` also folds in the two upstream read-only stages: it first runs the
subclass's paired Diagnostics and QC classes (``DIAGNOSTICS``/``QC``) -- which
write the metrics and 0/1 flags -- then the checkpoint methods. So the recipe
drives the whole pipeline order science modules -> Diagnostics -> QC ->
Checkpoints through one ``CheckpointL{n}(obj).run()`` call.

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


class Checkpoint:
    """Base runner for per-level checkpoint methods.

    Parameters
    ----------
    kpf_obj : KPFDataModel
        Finished data product whose flags/headers are read (never written).
    """

    LEVEL = None  # Subclasses set the level tag ("L0", "L1", "L2").
    RAISE_FLAGS = ()  # QC keywords whose failure (0) raises; every other 0 warns.
    DIAGNOSTICS = None  # Paired Diagnostics class, run first by run() (None = skip).
    QC = None  # Paired QC class, run second; its results land in self.qc_results.

    def __init__(self, kpf_obj):
        self.kpf_obj = kpf_obj
        self.qc_results = {}  # pass/fail dict from the folded QC.run() (empty if none)

    def run(self):
        """Run the paired Diagnostics and QC, then every checkpoint method.

        Folds in the two upstream read-only stages: Diagnostics writes its
        metrics, QC writes the 0/1 flags + ISGOOD (captured in ``self.qc_results``
        for callers that report them), then each checkpoint method warns or
        raises. A level with no paired ``DIAGNOSTICS``/``QC`` skips that stage.
        The checkpoint methods themselves never write (no return value).
        """
        if self.DIAGNOSTICS is not None:
            self.DIAGNOSTICS(self.kpf_obj).run()
        if self.QC is not None:
            self.qc_results = self.QC(self.kpf_obj).run()
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
        """Read this level's 0/1 QC flags; raise a RAISE_FLAGS failure, warn the rest.

        Scoped to **this level's own** checks (the registry's
        ``qc_flag_keywords_by_level[LEVEL]`` -- the ``QCL{n}`` flags), NOT the
        flags propagated from lower levels: QUALITY_CONTROL accumulates the whole
        L0->L1->L2->L4 history, but a lower-level flag was already surfaced at its
        own level's checkpoint, so re-warning it here would just be noise. A flag
        absent from the header is skipped (the check did not run); a flag equal to
        0 raises if it is in ``RAISE_FLAGS``, else warns.
        """
        header = self.kpf_obj.headers.get("QUALITY_CONTROL")
        if header is None:
            return
        reg = self.kpf_obj.keyword_registry
        flag_keys = reg.qc_flag_keywords_by_level.get(self.LEVEL, frozenset())
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
