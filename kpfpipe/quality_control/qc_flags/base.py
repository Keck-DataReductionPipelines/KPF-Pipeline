"""QC framework base class.

The second of three quality-control stages (Diagnostics -> QC -> Checkpoints).
Each QC subclass runs pass/fail check methods, writing a 0/1 flag per check to
QUALITY_CONTROL via ``set_keyword``. Header validation and raising live in the
separate Checkpoints layer.
"""

import logging

from kpfpipe.quality_control.applicability import applicability

logger = logging.getLogger(__name__)


class QC:
    """Base runner for per-level pass/fail QC check methods.

    Parameters
    ----------
    kpf_obj : KPFDataModel
        Finished data product whose QUALITY_CONTROL header receives the 0/1 flags.
    """

    LEVEL = None  # Subclasses set the level tag ("L0", "L1", "L2", "L4").

    def __init__(self, kpf_obj):
        self.kpf_obj = kpf_obj
        self.results = {}  # Populated by run(): maps keyword to (passed, comment).

    def run(self):
        """Run all checks and write each 0/1 result.

        Resets ``self.results`` at the start so repeated calls are deterministic.
        A check the applicability table does not declare for this frame type is
        skipped before it runs, as is one raising ``NotImplementedError``; neither
        writes a flag. Any other exception counts as a fail -- this layer never
        aborts; halting is the checkpoint layer's role.

        Returns
        -------
        dict
            Maps each FITS keyword to its ``(passed, comment)`` pair (this level's
            checks only).
        """
        self.results = {}
        frame = applicability.frame_type(self.kpf_obj)

        for name, fn in self._iter_checks():
            if not applicability.applies(type(self).__name__, name, frame):
                logger.debug(
                    "%s QC check %r does not apply to a %s frame; skipped",
                    self.LEVEL,
                    name,
                    frame,
                )
                continue
            kw = fn._qc_key
            comment = self.kpf_obj.keyword_registry.comment_for(kw)
            try:
                passed = fn()
            except NotImplementedError:
                logger.info(
                    "%s QC check %r is not implemented; skipped", self.LEVEL, name
                )
                continue
            except Exception as e:
                logger.error("%s QC check %r raised: %s", self.LEVEL, name, e)
                passed = False
            self.results[kw] = (passed, comment)
            self.kpf_obj.set_keyword(kw, 1 if passed else 0)
            logger.log(
                logging.DEBUG if passed else logging.WARNING,
                "%s %s = %s — %s",
                self.LEVEL,
                kw,
                1 if passed else 0,
                comment,
            )

        return self.results

    def _primary_keywords_populated(self):
        """Every PRIMARY keyword an upstream stage owes this level carries a value.

        The seed stamps every card at standardization, so a blank -- not a missing
        key -- is what this reports, and it is cumulative (L0 through ``LEVEL``),
        so a card an upstream stage left blank fails here too.

        Cards the quality-control suite writes itself are exempt: those stages run
        beside these checks under the same applicability tables, so requiring one
        would fail every frame its writer is not declared for -- every solar frame,
        for the pointing and Sun/Moon cards.

        Known gap: eight L0 cards (DQLVL0, FULLCOMP, the six *FLAG summaries) have
        no writer yet, so this fails on every frame it runs on.
        """
        registry = self.kpf_obj.keyword_registry
        header = self.kpf_obj.headers["PRIMARY"]
        for keyword in registry.primary_seed(self.LEVEL):
            if registry.populated_by(keyword, "PRIMARY") in applicability.classes:
                continue
            value = header.get(keyword)
            if value is None or (isinstance(value, str) and not value.strip()):
                return False
        return True

    def _iter_checks(self):
        """Yield each ``(name, method)`` tagged ``_qc_key``, subclass first."""
        seen = set()
        for cls in type(self).__mro__:
            for name, attr in cls.__dict__.items():
                if name in seen:
                    continue
                if not callable(attr):
                    continue
                if getattr(attr, "_qc_key", None) is None:
                    continue
                seen.add(name)
                yield name, getattr(self, name)
