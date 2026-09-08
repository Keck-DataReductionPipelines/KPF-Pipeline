"""Diagnostics framework base class.

The first of the three quality-control stages (Diagnostics -> QC -> Checkpoints).
Each Diagnostics subclass computes metrics from a finished data product and writes
them to the product headers via ``set_keyword``; it never modifies the science
extensions. QC then reads those metrics and applies pass/fail thresholds.
"""

import logging

from kpfpipe.quality_control.applicability import applicability

logger = logging.getLogger(__name__)


class Diagnostics:
    """Base runner for per-level diagnostic metric methods.

    Parameters
    ----------
    kpf_obj : KPFDataModel
        Finished data product whose headers receive the metrics (via
        ``set_keyword``, routed to each keyword's registry-home extension).
    """

    LEVEL = None  # Subclasses set the level tag ("L0", "L1", "L2", "L4").

    def __init__(self, kpf_obj):
        self.kpf_obj = kpf_obj
        self.results = {}  # Populated by run(): maps keyword to (value, comment).

    def run(self):
        """Run all diagnostic methods, writing each result via set_keyword.

        Resets ``self.results`` at the start so calling ``run()`` repeatedly
        is deterministic. A method the applicability table does not declare for
        this frame type is skipped before it runs, emitting no keyword.
        A method that raises is logged at ERROR (naming it) and
        skipped: this layer is informational and never aborts the pipeline, so its
        keywords are simply not written. Halting is the checkpoint layer's role.

        A keyword the header rejects is skipped on its own, so it takes neither
        the siblings its method computed nor its own ``self.results`` entry.

        Returns
        -------
        dict
            Maps each FITS keyword to its ``(value, comment)`` pair.
        """
        self.results = {}
        frame = applicability.frame_type(self.kpf_obj)

        for name, fn in self._iter_methods():
            if not applicability.applies(type(self).__name__, name, frame):
                logger.debug(
                    "%s diagnostic %r does not apply to a %s frame; skipped",
                    self.LEVEL,
                    name,
                    frame,
                )
                continue
            try:
                output = list(fn().items())
            except Exception as e:
                logger.error("%s diagnostic %r raised: %s", self.LEVEL, name, e)
                continue
            for kw, value in output:
                try:
                    self.kpf_obj.set_keyword(kw, value)
                except Exception as e:
                    logger.error(
                        "%s diagnostic %r keyword %r rejected: %s",
                        self.LEVEL,
                        name,
                        kw,
                        e,
                    )
                    continue
                self.results[kw] = (
                    value,
                    self.kpf_obj.keyword_registry.comment_for(kw),
                )

        for kw, (value, comment) in self.results.items():
            logger.debug("%s %s = %s — %s", self.LEVEL, kw, value, comment)
        return self.results

    def _iter_methods(self):
        """Yield each ``(name, method)`` tagged ``_diag_name``.

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
                if getattr(attr, "_diag_name", None) is None:
                    continue
                seen.add(name)
                yield name, getattr(self, name)
