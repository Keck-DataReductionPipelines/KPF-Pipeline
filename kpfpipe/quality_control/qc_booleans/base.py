"""QC framework base class.

Each QC subclass defines check methods. A check is a method whose function
object has `_qc_key` (8-char FITS keyword) and `_qc_comment` (short string)
attributes. The runner walks all such methods, calls each one, writes a
0/1 result to PRIMARY header, and aggregates ISGOOD = AND of all checks.
If any check raises, run() raises (loud failure, no silent suppression).
"""


class QC:
    """Base runner for per-level pass/fail QC check methods.

    Parameters
    ----------
    kpf_obj : KPFDataModel
        Finished data product whose PRIMARY header receives the QC flags.
    """

    LEVEL = None  # Subclasses set the level tag ("L0", "L1", "L2").

    def __init__(self, kpf_obj):
        self.kpf = kpf_obj
        self.results = {}  # Populated by run(): maps keyword to (passed, comment).

    def run(self):
        """Run all checks, write each result to PRIMARY, and aggregate ISGOOD.

        Resets ``self.results`` at the start so calling ``run()`` repeatedly
        on the same instance is deterministic.

        Returns
        -------
        dict
            Maps each FITS keyword to its ``(passed, comment)`` pair.
        """
        self.results = {}

        for name, fn in self._iter_checks():
            try:
                passed = fn()
            except Exception as e:
                raise RuntimeError(f"QC check {name!r} raised: {e}") from e

            kw = fn._qc_key
            comment = fn._qc_comment
            self.results[kw] = (passed, comment)
            self.kpf.headers["PRIMARY"][kw] = (1 if passed else 0, comment)

        is_good = all(p for p, _ in self.results.values())
        is_good_comment = (
            "QC: all checks pass" if is_good else "QC: one or more checks failed"
        )
        self.kpf.headers["PRIMARY"]["ISGOOD"] = (1 if is_good else 0, is_good_comment)
        return self.results

    def _iter_checks(self):
        """Yield each check method tagged with `_qc_key`.

        Iterates in source order via ``__dict__`` plus an MRO walk so check
        ordering is stable.

        Yields
        ------
        tuple
            ``(name, bound_method)`` for each tagged check method.
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
