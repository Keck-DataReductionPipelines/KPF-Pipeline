"""Diagnostics framework base class.

Each Diagnostics subclass defines methods that compute metrics from a
finished data product and return a dict of {keyword: (value, comment)}
to write to the product's PRIMARY header. The runner walks all such
methods, writes their results, and stores them on self.results.

A method opts in by setting `_diag_name` on the function object.
If a method raises, run() raises (loud failure, no silent suppression).

Diagnostics is read-only with respect to the data extensions — it only
adds keywords to the PRIMARY header. Pair with QC, which reads the
metrics Diagnostics writes and applies pass/fail thresholds.
"""


class Diagnostics:
    LEVEL = None  # "L0", "L1", "L2"

    def __init__(self, kpf_obj):
        self.kpf = kpf_obj
        self.results = {}  # {keyword: (value, comment)}

    def run(self):
        """Run all diagnostic methods, write each result to PRIMARY.

        Returns dict {keyword: (value, comment)}. Resets self.results at
        the start so calling run() repeatedly is deterministic.
        """
        self.results = {}

        for name, fn in self._iter_methods():
            try:
                output = fn()
            except Exception as e:
                raise RuntimeError(
                    f"Diagnostic {name!r} raised: {e}"
                ) from e

            if not output:
                continue
            for kw, (value, comment) in output.items():
                self.results[kw] = (value, comment)
                self.kpf.headers["PRIMARY"][kw] = (value, comment)

        return self.results

    def _iter_methods(self):
        """Yield (name, bound_method) for each method tagged with _diag_name.

        Walks the MRO so subclasses' methods come before the base class
        and ordering is stable across runs.
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
