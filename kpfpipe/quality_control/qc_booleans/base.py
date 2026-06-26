"""QC framework base class.

Each QC subclass defines check methods. A check is a method whose function
object has `_qc_key` (8-char FITS keyword) and `_qc_comment` (short string)
attributes. The runner walks all such methods, calls each one, writes a
0/1 result via ``set_keyword`` (which routes it to its registry home,
QUALITY_CONTROL), and aggregates ISGOOD = AND of all checks. If any check
raises, run() raises (loud failure, no silent suppression).

After the checks, the runner validates every registry-governed extension
header against the keyword registry (``_validate_headers``): an unexpected
keyword raises, a missing required EPRV keyword warns. This is the single
home for header validation (it used to live on ``KPFDataModel`` and run
inside the level transforms).
"""

import warnings

# Header-registry lookups are read off the validated kpf_obj (e.g.
# self.kpf.EXT_ALLOWED / EXT_REQUIRED), so qc_booleans does not import from
# data_models — the registry stays reachable purely through the data model. The
# structural-card policy below is validator-specific and lives here.

# Bookkeeping/structural cards astropy adds to a serialized extension header
# (BinTable column descriptors, image WCS); always permitted on any governed
# extension and never counted as a "missing required" keyword. These supplement
# the model's STRUCTURAL_KEYS (the PRIMARY/bookkeeping set).
_EXT_STRUCTURAL_PREFIXES = (
    "NAXIS",
    "TTYPE",
    "TFORM",
    "TUNIT",
    "TDIM",
    "TDISP",
    "TNULL",
    "TSCAL",
    "TZERO",
    "CTYPE",
    "CUNIT",
    "CRPIX",
    "CRVAL",
    "CDELT",
    "CROTA",
)
_EXT_STRUCTURAL_EXTRA = {"EXTNAME", "TFIELDS", "PCOUNT", "GCOUNT"}


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
            # set_keyword routes the flag to its registry home (QUALITY_CONTROL)
            # with the registry comment; fn._qc_comment is kept for self.results.
            self.kpf.set_keyword(kw, 1 if passed else 0)

        is_good = all(p for p, _ in self.results.values())
        self.kpf.set_keyword("ISGOOD", 1 if is_good else 0)

        # Validate every governed extension header now that all keywords (this
        # level's QC flags included) have been written.
        self._validate_headers()
        return self.results

    def _validate_headers(self):
        """Validate every registry-governed extension header (fail loud).

        For each governed extension present on the product, every card must be a
        registered keyword for that extension (KPF registry + EPRV per-extension
        keywords) or a structural/bookkeeping card; an unexpected card raises
        ValueError. A Required EPRV keyword that is absent emits a warning.
        PRIMARY is validated only where it is EPRV-standard (L1/L2/L4); the raw
        WMKO L0 PRIMARY is skipped.
        """
        level = str(self.LEVEL).upper()
        # Registry lookups are read off the validated model (kpf_obj), so
        # data_models/_registry stays imported only by base.py. PRIMARY allowed is
        # NOT level-gated (the registry already lists every EPRV + KPF PRIMARY
        # keyword); required-warnings ARE level-aware. L1 PRIMARY is already
        # EPRV-L2-standard, so it caps at level 2 like L2.
        kpf = self.kpf
        cap = {"L1": 2, "L2": 2, "L4": 4}.get(level)

        def required_at(ext):
            """Keywords required for ``ext`` at or below this product's level."""
            if cap is None:  # L0 (or an untagged QC subclass): no required-warnings.
                return set()
            return {k for k, lvl in kpf.EXT_REQUIRED.get(ext, {}).items() if lvl <= cap}

        # PRIMARY is EPRV-standard from L1 onward; the raw WMKO L0 PRIMARY is not.
        if cap is not None:
            self._validate_extension(
                "PRIMARY",
                kpf.EXT_ALLOWED.get("PRIMARY", set()),
                required_at("PRIMARY"),
            )

        # KPF/EPRV governed extensions (QUALITY_CONTROL, RECEIPT, BARYCORR_*,
        # BJD_TDB, RV#/CCF#): validate each that exists on this product.
        for ext, allowed in kpf.EXT_ALLOWED.items():
            if ext == "PRIMARY" or ext not in kpf.extensions:
                continue
            self._validate_extension(ext, allowed, required_at(ext))

    def _is_structural(self, key):
        """True for a FITS structural/bookkeeping card (not a registered keyword).

        Combines the model's STRUCTURAL_KEYS (PRIMARY/bookkeeping cards) with the
        extension-table/WCS cards astropy adds at serialization.
        """
        return (
            key in self.kpf.STRUCTURAL_KEYS
            or key in _EXT_STRUCTURAL_EXTRA
            or key.startswith(_EXT_STRUCTURAL_PREFIXES)
        )

    def _validate_extension(self, ext, allowed, required):
        """Check one extension header: raise on an unexpected card, warn on an
        absent required (non-structural) keyword.

        Any card that is neither structural nor a registered keyword for ``ext``
        raises -- this subsumes the old WMKO-native leak check, since a raw
        instrument keyword (kept in INSTRUMENT_HEADER, never registered for an
        EPRV PRIMARY) is simply unregistered here.
        """
        header = self.kpf.headers.get(ext)
        if header is None:
            return
        present = set()
        for raw_key in list(header):
            key = str(raw_key).strip()
            present.add(key)
            if self._is_structural(key) or key in allowed:
                continue
            raise ValueError(
                f"unregistered keyword {key!r} on {ext}; add it to "
                "config/L{0,1,2,4}-headers.csv or fix the writer"
            )

        missing = sorted(
            k for k in required if not self._is_structural(k) and k not in present
        )
        if missing:
            warnings.warn(
                f"missing required keyword(s) on {ext}: {missing}",
                UserWarning,
                stacklevel=2,
            )

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
