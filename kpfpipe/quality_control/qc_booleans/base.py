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

from kpfpipe.data_models.base import (
    EPRV_L2_KEYS,
    EPRV_L4_KEYS,
    EXT_ALLOWED_KEYS,
    EXT_REQUIRED_KEYS,
    HEADERMAP_STANDARD_KEYS,
    REQUIRED_L2_KEYS,
    REQUIRED_L4_KEYS,
    STRUCTURAL_KEYS,
    WMKO_PRIMARY_KEYS,
)

# Bookkeeping/structural cards astropy adds to a serialized extension header
# (BinTable column descriptors, image WCS); always permitted on any governed
# extension and never counted as a "missing required" keyword.
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
_EXT_STRUCTURAL_KEYS = STRUCTURAL_KEYS | {"EXTNAME", "TFIELDS", "PCOUNT", "GCOUNT"}


def _is_structural(key):
    """True for a FITS structural/bookkeeping card (not a registered keyword)."""
    return key in _EXT_STRUCTURAL_KEYS or key.startswith(_EXT_STRUCTURAL_PREFIXES)


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

        # PRIMARY is EPRV-standard from L1 onward; the raw WMKO L0 PRIMARY is not.
        if level in ("L1", "L2", "L4"):
            eprv_keys = EPRV_L4_KEYS if level == "L4" else EPRV_L2_KEYS
            required = REQUIRED_L4_KEYS if level == "L4" else REQUIRED_L2_KEYS
            allowed = (
                eprv_keys
                | HEADERMAP_STANDARD_KEYS
                | EXT_ALLOWED_KEYS.get("PRIMARY", set())
            )
            self._validate_extension("PRIMARY", allowed, required, wmko_check=True)

        # KPF/EPRV governed extensions (QUALITY_CONTROL, RECEIPT, BARYCORR_*,
        # BJD_TDB, RV#/CCF#): validate each that exists on this product.
        for ext, allowed in EXT_ALLOWED_KEYS.items():
            if ext == "PRIMARY" or ext not in self.kpf.extensions:
                continue
            self._validate_extension(
                ext, allowed, EXT_REQUIRED_KEYS.get(ext, set()), wmko_check=False
            )

    def _validate_extension(self, ext, allowed, required, *, wmko_check):
        """Check one extension header: raise on an unexpected card, warn on an
        absent required (non-structural) keyword."""
        header = self.kpf.headers.get(ext)
        if header is None:
            return
        present = set()
        for raw_key in list(header):
            key = str(raw_key).strip()
            present.add(key)
            if _is_structural(key) or key in allowed:
                continue
            if wmko_check and key in WMKO_PRIMARY_KEYS:
                raise ValueError(
                    f"native WMKO keyword {key!r} found on {ext}; it must be "
                    "converted to its EPRV name or kept in INSTRUMENT_HEADER"
                )
            raise ValueError(
                f"unregistered keyword {key!r} on {ext}; add it to "
                "config/L{0,1,2,4}-headers.csv or fix the writer"
            )

        missing = sorted(
            k for k in required if not _is_structural(k) and k not in present
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
