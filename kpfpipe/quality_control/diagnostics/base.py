"""Diagnostics framework base class.

Each Diagnostics subclass defines methods that compute metrics from a
finished data product and return a dict of {keyword: (value, comment)}.
The runner writes each via ``set_keyword`` (which routes the keyword to its
registry-home extension — DiagL2 metrics land on QUALITY_CONTROL) and stores
them on self.results.

A method opts in by setting `_diag_name` on the function object.
If a method raises, run() raises (loud failure, no silent suppression).

Diagnostics is read-only with respect to the data extensions — it only
adds header keywords via ``set_keyword``. Pair with QC, which reads the
metrics Diagnostics writes and applies pass/fail thresholds.
"""


class Diagnostics:
    """Base runner for per-level diagnostic metric methods.

    Parameters
    ----------
    kpf_obj : KPFDataModel
        Finished data product whose headers receive the metrics (via
        ``set_keyword``, routed to each keyword's registry-home extension).
    """

    LEVEL = None  # Subclasses set the level tag ("L0", "L1", "L2").

    def __init__(self, kpf_obj):
        self.kpf_obj = kpf_obj
        self.results = {}  # Populated by run(): maps keyword to (value, comment).

    def _tag(self, **values):
        """Pair each ``keyword=value`` with its registry-sourced FITS comment.

        Diagnostic methods build their result dict with this rather than
        hardcoding comment strings, so the FITS comment has a single source of
        truth (the keyword registry / ``config/L{n}-headers.csv`` Description)
        and never drifts from it. ``set_keyword`` already uses the registry
        Description for the header write; this keeps ``self.results`` in sync.
        """
        routing = self.kpf_obj.keyword_registry.routing
        out = {}
        for kw, value in values.items():
            route = routing.get(kw)
            out[kw] = (value, route[1] if route is not None else "")
        return out

    def run(self):
        """Run all diagnostic methods, writing each result via set_keyword.

        Resets ``self.results`` at the start so calling ``run()`` repeatedly
        is deterministic.

        Returns
        -------
        dict
            Maps each FITS keyword to its ``(value, comment)`` pair.
        """
        self.results = {}

        for name, fn in self._iter_methods():
            try:
                output = fn()
            except Exception as e:
                raise RuntimeError(f"Diagnostic {name!r} raised: {e}") from e

            if not output:
                continue
            for kw, (value, comment) in output.items():
                self.results[kw] = (value, comment)
                # set_keyword routes each metric to its registry home; the FITS
                # comment is the registry Description (the metric-dict comment is
                # retained in self.results only).
                self.kpf_obj.set_keyword(kw, value)

        return self.results

    def _iter_methods(self):
        """Yield each method tagged with `_diag_name`.

        Walks the MRO so subclasses' methods come before the base class
        and ordering is stable across runs.

        Yields
        ------
        tuple
            ``(name, bound_method)`` for each tagged diagnostic method.
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
