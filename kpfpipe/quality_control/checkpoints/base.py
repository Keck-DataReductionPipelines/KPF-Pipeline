"""Checkpoint framework base class.

The third and final read-only quality-control stage (after Diagnostics and QC).
A Checkpoint reads the 0/1 QC flags and the product headers and emits warnings
or raises errors -- it never writes keywords. ``run()`` also folds in the paired
Diagnostics and QC classes first, so the recipe drives the whole
Diagnostics -> QC -> Checkpoints sequence through one ``CheckpointL{n}(obj).run()``
call. Two base checkpoints are inherited by every level: ``unregistered_keywords``
(structural header validation) and ``qc_flags`` (raise on a failed flag named in
the subclass's ``RAISE_FLAGS``, else summarize the flags behind an ISGOOD=0).
"""

import logging

logger = logging.getLogger(__name__)


class Checkpoint:
    """Base runner for per-level checkpoint methods.

    Parameters
    ----------
    kpf_obj : KPFDataModel
        Finished data product whose flags/headers are read (never written).
    """

    LEVEL = None  # Subclasses set the level tag ("L0", "L1", "L2", "L4").
    RAISE_FLAGS = ()  # QC keywords whose failure (0) raises; every other 0 warns.
    DIAGNOSTICS = None  # Paired Diagnostics class, run first by run() (None = skip).
    QC = None  # Paired QC class, run second; its results land in self.qc_results.

    def __init__(self, kpf_obj):
        self.kpf_obj = kpf_obj
        self.qc_results = {}  # pass/fail dict from the folded QC.run() (empty if none)

    def run(self):
        """Run the paired Diagnostics and QC, then every checkpoint method.

        Folds in the two upstream stages: Diagnostics writes its
        metrics, QC writes the 0/1 flags + ISGOOD (captured in ``self.qc_results``
        for callers that report them), then each checkpoint method warns or
        raises. A level with no paired ``DIAGNOSTICS``/``QC`` skips that stage.
        The checkpoint methods themselves never write (no return value).
        """
        if self.DIAGNOSTICS is not None:
            self.DIAGNOSTICS(self.kpf_obj).run()
        if self.QC is not None:
            self.qc_results = self.QC(self.kpf_obj).run()
        for name, fn in self._iter_checkpoints():
            try:
                fn()
            except Exception as e:
                logger.error("%s checkpoint %r raised: %s", self.LEVEL, name, e)
                raise
        qc_hdr = self.kpf_obj.headers.get("QUALITY_CONTROL")
        isgood = qc_hdr.get("ISGOOD") if qc_hdr is not None else None
        logger.info(
            "%s checkpoints passed (%d QC flag(s), ISGOOD=%s)",
            self.LEVEL,
            len(self.qc_results),
            isgood,
        )

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
                if reg.is_structural(key) or key in allowed:
                    continue
                raise ValueError(
                    f"unregistered keyword {key!r} on {ext}; add it to "
                    "config/L{0,1,2,4}-headers.csv or fix the writer"
                )

    unregistered_keywords._checkpoint_name = "unregistered_keywords"

    def qc_flags(self):
        """Raise on a fatal flag failure; summarize the flags behind an ISGOOD=0.

        The fatal check is scoped to **this level's own** ``RAISE_FLAGS``: a 0
        there raises. The summary then names every failing QC flag on
        QUALITY_CONTROL (the cross-level L0->L4 accumulation, ISGOOD excluded) by
        bare keyword -- each was already logged with its comment by the QC stage
        as the flag was written, so the names alone suffice here. A flag absent
        from the header is skipped (its check did not run).
        """
        header = self.kpf_obj.headers.get("QUALITY_CONTROL")
        if header is None:
            return
        reg = self.kpf_obj.keyword_registry
        for key in sorted(reg.qc_flag_keywords_by_level.get(self.LEVEL, frozenset())):
            if key in self.RAISE_FLAGS and header.get(key) == 0:
                raise ValueError(f"QC checkpoint failed: {key} = 0 ({self.LEVEL})")
        failing = sorted(
            key for key in reg.qc_flag_keywords - {"ISGOOD"} if header.get(key) == 0
        )
        if failing:
            logger.warning(
                "%s ISGOOD=0; failing QC flags: %s", self.LEVEL, ", ".join(failing)
            )

    qc_flags._checkpoint_name = "qc_flags"

    def _iter_checkpoints(self):
        """Yield each ``(name, method)`` tagged ``_checkpoint_name``.

        MRO-walk discovery: walk ``type(self).__mro__``, collect tagged methods,
        subclass first.
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
