"""Checkpoint framework base class.

The third and final read-only quality-control stage (after Diagnostics and QC).
A Checkpoint reads the 0/1 QC flags and the product headers and emits warnings
or raises errors -- it never writes keywords. ``run()`` also folds in the paired
Diagnostics and QC classes first, so the recipe drives the whole
Diagnostics -> QC -> Checkpoints sequence through one ``CheckpointL{n}(obj).run()``
call. Two base checkpoints are inherited by every level:
``raise_on_unregistered_keyword`` and ``raise_on_fatal_qc_flag``.
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
    DIAGNOSTICS = ()  # Paired Diagnostics classes, run first by run(), in order.
    QC = None  # Paired QC class, run second; its results land in self.qc_results.

    def __init__(self, kpf_obj):
        self.kpf_obj = kpf_obj
        self.qc_results = {}  # pass/fail dict from the folded QC.run() (empty if none)

    def run(self):
        """Run the paired Diagnostics and QC, then every checkpoint method.

        Folds in the two upstream stages: Diagnostics writes its
        metrics, QC writes the 0/1 flags (captured in ``self.qc_results``
        for callers that report them), then each checkpoint method warns or
        raises. A level with no paired ``DIAGNOSTICS``/``QC`` skips that stage.
        The checkpoint methods themselves never write (no return value).
        """
        for diagnostics in self.DIAGNOSTICS:
            diagnostics(self.kpf_obj).run()
        if self.QC is not None:
            self.qc_results = self.QC(self.kpf_obj).run()
        for name, fn in self._iter_checkpoints():
            try:
                fn()
            except Exception as e:
                logger.error("%s checkpoint %r raised: %s", self.LEVEL, name, e)
                raise
        logger.info(
            "%s checkpoints passed (%d QC flag(s))", self.LEVEL, len(self.qc_results)
        )

    def raise_on_unregistered_keyword(self):
        """Raise on a card that is neither structural nor registered for its extension.

        PRIMARY is checked at every level, L0 included, since standardize_headers
        runs at load: a leaked WMKO-native keyword is unregistered for an EPRV
        PRIMARY and so is caught here.
        """
        reg = self.kpf_obj.keyword_registry
        for ext, allowed in reg.allowed.items():
            if ext not in self.kpf_obj.extensions:
                continue
            header = self.kpf_obj.headers[ext]
            for raw_key in list(header):
                key = str(raw_key).strip()
                if reg.is_structural(key) or key in allowed:
                    continue
                raise ValueError(
                    f"unregistered keyword {key!r} on {ext}; add it to the "
                    "appropriate config/{prefix}-{EXTENSION}-keywords.csv or fix "
                    "the writer"
                )

    raise_on_unregistered_keyword._checkpoint_name = "raise_on_unregistered_keyword"

    def raise_on_fatal_qc_flag(self):
        """Raise on a failed flag in this level's ``RAISE_FLAGS``; warn on any other.

        The warning names every failing flag on QUALITY_CONTROL, which accumulates
        L0->L4; the QC stage already logged each one with its comment. A flag the
        header lacks did not run, so it is not a failure.
        """
        header = self.kpf_obj.headers["QUALITY_CONTROL"]
        reg = self.kpf_obj.keyword_registry
        for key in sorted(self.RAISE_FLAGS):
            if header.get(key) == 0:
                raise ValueError(f"QC checkpoint failed: {key} = 0 ({self.LEVEL})")
        failing = sorted(key for key in reg.qc_flag_keywords if header.get(key) == 0)
        if failing:
            logger.warning("%s failing QC flags: %s", self.LEVEL, ", ".join(failing))

    raise_on_fatal_qc_flag._checkpoint_name = "raise_on_fatal_qc_flag"

    def _iter_checkpoints(self):
        """Yield each ``(name, method)`` tagged ``_checkpoint_name``, subclass first."""
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
