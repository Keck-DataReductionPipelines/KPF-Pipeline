"""QC framework base class.

Each QC subclass defines check methods. A check is a method whose function
object has `_qc_key` (8-char FITS keyword) and `_qc_comment` (short string)
attributes. The runner walks all such methods, calls each one, writes a
0/1 result to PRIMARY header, and aggregates ISGOOD = AND of all checks.
If any check raises, run() raises (loud failure, no silent suppression).
"""


class QC:
    LEVEL = None  # "L0", "L1", "L2"

    def __init__(self, kpf_obj):
        self.kpf = kpf_obj
        self.results = {}

    def run(self):
        """Run all checks, write each result to PRIMARY, aggregate ISGOOD.

        Returns dict {keyword: (passed, comment)}.
        """
        for name, fn in self._iter_checks():
            try:
                passed = fn()
            except Exception as e:
                raise RuntimeError(
                    f"QC check {name!r} raised: {e}"
                ) from e

            kw = fn._qc_key
            comment = fn._qc_comment
            self.results[kw] = (passed, comment)
            self.kpf.headers["PRIMARY"][kw] = (1 if passed else 0, comment)

        is_good = all(p for p, _ in self.results.values())
        self.kpf.headers["PRIMARY"]["ISGOOD"] = (
            1 if is_good else 0,
            "QC: all checks pass",
        )
        return self.results

    def _iter_checks(self):
        """Yield (name, bound_method) for each method tagged with _qc_key.

        Iterates in source order via __dict__ + MRO walk so check ordering
        is stable.
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
