"""QC framework base class.

The second of three quality-control stages (Diagnostics -> QC -> Checkpoints).
Each QC subclass runs pass/fail check methods, writing a 0/1 flag per check to
QUALITY_CONTROL via ``set_keyword`` and aggregating ISGOOD as the AND of all
checks. Header validation and raising live in the separate Checkpoints layer.
"""


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
            Maps each FITS keyword to its ``(passed, comment)`` pair (this level's
            checks only). ``ISGOOD`` is the cross-level aggregate (see below).
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

        # ISGOOD is the running aggregate: AND over every QC flag now on
        # QUALITY_CONTROL — the flags this level just wrote PLUS those propagated
        # from lower levels (QUALITY_CONTROL accumulates L0->L1->L2->L4). Reading
        # the accumulated header makes it level-agnostic; exclude ISGOOD itself.
        hdr = self.kpf_obj.headers["QUALITY_CONTROL"]
        flags = self.kpf_obj.keyword_registry.qc_flag_keywords - {"ISGOOD"}
        present = [hdr.get(kw) for kw in flags if hdr.get(kw) is not None]
        is_good = all(bool(v) for v in present)
        self.kpf_obj.set_keyword("ISGOOD", 1 if is_good else 0)
        return self.results

    def _required_primary_keywords(self):
        """Keywords a level-N product must carry on PRIMARY (a presence set).

        The registry's EPRV ``Required`` PRIMARY keywords at or below this
        product's own level -- the EPRV L2 PRIMARY set is tagged Level 1 in the
        registry (KPF holds the L1 PRIMARY to the EPRV L2 spec; see
        ``keyword_registry._build_rows``), so the check needs no L1->L2 cap: the
        level cap *is* the level. PRIMARY now holds EPRV-registered keywords only
        (the DRP provenance cards moved to RECEIPT), so there is no
        KPF-routed-PRIMARY set to union in. Read off the validated model's
        registry singleton (``self.kpf_obj.keyword_registry``), so qc_flags
        imports nothing from data_models. L0 -> Level 0 yields the empty set (no
        PRIMARY keyword is Required there -- raw WMKO L0 PRIMARY is not
        registry-governed); an untagged subclass (``LEVEL`` None) also gets it.
        """
        level = str(self.LEVEL or "")
        if not (level[:1].upper() == "L" and level[1:].isdigit()):
            return set()
        cap = int(level[1:])
        reg = self.kpf_obj.keyword_registry
        return {k for k, lvl in reg.required.get("PRIMARY", {}).items() if lvl <= cap}

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
