"""Diagnostics framework base class.

The first of the three quality-control stages (Diagnostics -> QC -> Checkpoints).
Each Diagnostics subclass computes metrics from a finished data product and writes
them to the product headers via ``set_keyword``; it never modifies the science
extensions. QC then reads those metrics and applies pass/fail thresholds.
"""

import logging

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

    def _tag(self, **values):
        """Pair each ``keyword=value`` with its registry-sourced FITS comment.

        Sources the comment from the keyword registry (single source of truth) so
        ``self.results`` stays in sync with the ``set_keyword`` header write. Every
        emitted keyword must be registered; an unregistered one raises rather than
        getting a blank comment.
        """
        routing = self.kpf_obj.keyword_registry.routing
        return {kw: (value, routing[kw][1]) for kw, value in values.items()}

    def run(self):
        """Run all diagnostic methods, writing each result via set_keyword.

        Resets ``self.results`` at the start so calling ``run()`` repeatedly
        is deterministic. A method that raises is logged at ERROR (naming it) and
        re-raised unchanged -- fail-fast; halting is the checkpoint layer's role.

        Returns
        -------
        dict
            Maps each FITS keyword to its ``(value, comment)`` pair.
        """
        self.results = {}

        for name, fn in self._iter_methods():
            try:
                output = fn()
                if not output:
                    continue
                for kw, (value, comment) in output.items():
                    self.results[kw] = (value, comment)
                    # set_keyword routes each metric to its registry home; the FITS
                    # comment is the registry Description (the metric-dict comment is
                    # retained in self.results only).
                    self.kpf_obj.set_keyword(kw, value)
            except Exception as e:
                logger.error("%s diagnostic %r raised: %s", self.LEVEL, name, e)
                raise

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
