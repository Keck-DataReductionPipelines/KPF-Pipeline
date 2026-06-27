"""QC framework base class.

Each QC subclass defines check methods. A check is a method whose function
object has a `_qc_key` (8-char FITS keyword) attribute. The runner walks all
such methods, calls each one, writes a 0/1 result via ``set_keyword`` (which
routes it to its registry home, QUALITY_CONTROL, with the registry ``Description``
as the FITS comment), and aggregates ISGOOD = AND of all checks. The per-check
comment lives once — in the registry ``Description`` — not on the method. If any
check raises, run() raises (loud failure, no silent suppression).

QC writes only 0/1 keywords. Header validation (unregistered cards, missing
required keywords) is NOT done here -- it lives in the separate ``checkpoints``
layer, which reads these flags plus the product headers and emits warnings or
raises. The pipeline order is: science modules -> Diagnostics -> QC -> Checkpoints.
"""

# Level -> the EPRV PRIMARY Level a product is held to. L1 PRIMARY is already
# EPRV-L2-standard, so it caps at 2 like L2. L0 (or an untagged subclass) is
# absent here: its PRIMARY is raw WMKO, not registry-governed.
_LEVEL_CAP = {"L1": 2, "L2": 2, "L4": 4}


class QC:
    """Base runner for per-level pass/fail QC check methods.

    Parameters
    ----------
    kpf_obj : KPFDataModel
        Finished data product whose QUALITY_CONTROL header receives the 0/1 flags.
    """

    LEVEL = None  # Subclasses set the level tag ("L0", "L1", "L2").

    def __init__(self, kpf_obj):
        self.kpf_obj = kpf_obj
        self.results = {}  # Populated by run(): maps keyword to (passed, comment).

    def run(self):
        """Run all checks, write each 0/1 result, and aggregate ISGOOD.

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
            # The comment is the registry Description (the single source) — the same
            # string set_keyword writes as the FITS comment; mirror it into results.
            comment = self.kpf_obj.keyword_registry.routing.get(kw, (None, ""))[1]
            self.results[kw] = (passed, comment)
            self.kpf_obj.set_keyword(kw, 1 if passed else 0)

        is_good = all(p for p, _ in self.results.values())
        self.kpf_obj.set_keyword("ISGOOD", 1 if is_good else 0)
        return self.results

    def _required_primary_keywords(self):
        """Keywords a level-N product must carry on PRIMARY (a presence set).

        The registry's EPRV ``Required`` PRIMARY keywords at or below this
        product's level, unioned with the KPF-pipeline keywords routed to PRIMARY
        (the provenance cards). Read off the validated model's registry singleton
        (``self.kpf_obj.keyword_registry``), so qc_flags imports nothing from
        data_models. L0 (or an untagged subclass) returns the empty set -- raw
        WMKO L0 PRIMARY is not registry-governed.
        """
        cap = _LEVEL_CAP.get(str(self.LEVEL).upper())
        if cap is None:
            return set()
        reg = self.kpf_obj.keyword_registry
        required = {
            k for k, lvl in reg.required.get("PRIMARY", {}).items() if lvl <= cap
        }
        # KPF-routed PRIMARY keywords are the registry's PRIMARY rows that are not
        # EPRV-sourced (PopulatedBy != the "EPRV" discriminator) -- the provenance
        # cards; they are not flagged EPRV-Required, so add them explicitly.
        kpf_primary = {
            row.Keyword
            for row in reg.table.itertuples(index=False)
            if row.Extension == "PRIMARY"
            and row.PopulatedBy != "EPRV"
            and row.Level <= cap
        }
        return required | kpf_primary

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
