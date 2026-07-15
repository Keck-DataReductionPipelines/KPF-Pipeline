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

    LEVEL = None  # Subclasses set the level tag ("L0", "L1", "L2", "L4").

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
            # Mirror the registry Description into results (the FITS comment
            # source; see ``_tag`` and style guide §11).
            comment = self.kpf_obj.keyword_registry.routing.get(kw, (None, ""))[1]
            self.results[kw] = (passed, comment)
            self.kpf_obj.set_keyword(kw, 1 if passed else 0)

        # ISGOOD is the running aggregate: AND over every QC flag now on
        # QUALITY_CONTROL -- the flags this level just wrote PLUS those propagated
        # from lower levels (QUALITY_CONTROL accumulates L0->L1->L2->L4). Reading
        # the accumulated header makes it level-agnostic; exclude ISGOOD itself.
        hdr = self.kpf_obj.headers["QUALITY_CONTROL"]
        flags = self.kpf_obj.keyword_registry.qc_flag_keywords - {"ISGOOD"}
        present = [hdr.get(kw) for kw in flags if hdr.get(kw) is not None]
        is_good = all(bool(v) for v in present)
        self.kpf_obj.set_keyword("ISGOOD", 1 if is_good else 0)
        return self.results

    def _required_primary_keywords(self):
        """Registry EPRV ``Required`` PRIMARY keywords at or below this level.

        The level cap is the level's own number, so this runs unchanged for L1,
        L2, and L4 -- each returns the required PRIMARY keywords tagged at or
        below its own level. Read off the model's registry singleton so qc_flags
        imports nothing from data_models. L0 (and an untagged ``LEVEL`` None)
        yields the empty set -- raw WMKO L0 PRIMARY is not registry-governed.
        """
        level = str(self.LEVEL or "")
        if not (level[:1].upper() == "L" and level[1:].isdigit()):
            return set()
        cap = int(level[1:])
        reg = self.kpf_obj.keyword_registry
        return {k for k, lvl in reg.required.get("PRIMARY", {}).items() if lvl <= cap}

    def _iter_checks(self):
        """Yield each ``(name, method)`` tagged ``_qc_key``.

        MRO-walk discovery; the mechanism is documented in style guide §11.
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
