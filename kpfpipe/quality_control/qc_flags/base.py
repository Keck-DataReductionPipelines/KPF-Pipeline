"""QC framework base class.

The second of three quality-control stages (Diagnostics -> QC -> Checkpoints).
Each QC subclass runs pass/fail check methods, writing a 0/1 flag per check to
QUALITY_CONTROL via ``set_keyword``. Header validation and raising live in the
separate Checkpoints layer.
"""

import logging

logger = logging.getLogger(__name__)


class QC:
    """Base runner for per-level pass/fail QC check methods.

    Every level carries a required-PRIMARY-keyword placeholder check (e.g. L0's
    KWRDPRL0): REQUIRED is now a compliance label, not a decision about what must
    be on a product, so the registry-derived notion of "required" these checks
    would read is gone. Until a KPF-owned definition replaces it, each raises
    ``NotImplementedError`` (writing no flag, per ``run``) while its registry row
    stays so the comment lookup still resolves.

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

        Each result is logged as it is written: DEBUG on a pass, WARNING on a
        fail, ERROR on a check that raised (counted as a fail -- this layer never
        aborts; halting is the checkpoint layer's role). ``NotImplementedError``
        from a placeholder check writes no flag.
        ``self.results`` is reset at the start so repeated calls are deterministic.

        Returns
        -------
        dict
            Maps each FITS keyword to its ``(passed, comment)`` pair (this level's
            checks only).
        """
        self.results = {}

        for name, fn in self._iter_checks():
            kw = fn._qc_key
            # Mirror the registry Description into results (the FITS comment
            # source; see ``_tag``). The _qc_key must be registered.
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

    def _iter_checks(self):
        """Yield each ``(name, method)`` tagged ``_qc_key``.

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
                if getattr(attr, "_qc_key", None) is None:
                    continue
                seen.add(name)
                yield name, getattr(self, name)
