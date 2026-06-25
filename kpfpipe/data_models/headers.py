"""
FITS header parsing and WMKO ↔ EPRV ↔ KPF PRIMARY conversion.

Two classes own all KPF header behaviour:

- :class:`HeaderParser` — read/write a keyword without worrying about whether a
  value is stored bare or as a ``(value, comment)`` tuple. The single place that
  bridges the two in-memory representations (``fits.Header`` vs. ``OrderedDict``).
- :class:`HeaderConverter` — the single source of truth for the native→EPRV
  keyword mapping (rvdata's ``header_map.csv``) and for what may legitimately
  appear on a KPF EPRV PRIMARY header:

  - ``KPF0.to_kpf1`` calls :meth:`HeaderConverter.native_to_eprv` to build an
    EPRV-standard L1 PRIMARY from the raw WMKO L0 PRIMARY, and
    :meth:`HeaderConverter.build_instrument_header` to preserve the verbatim raw
    L0 PRIMARY in an immutable ``INSTRUMENT_HEADER`` extension.
  - ``KPF1.to_kpf2`` / ``KPF2.to_kpf4`` call
    :meth:`HeaderConverter.validate_eprv_primary` to fail loudly if a raw native
    keyword leaked onto PRIMARY, an unregistered keyword appears, or a required
    EPRV keyword is missing.

KPF-pipeline keywords that are written directly to PRIMARY (read noise, QC
booleans, diagnostics, RV/barycentric cards, …) are catalogued in the packaged
registry (``config/L{0,1,2,4}-headers.csv``) and allowed by the validator.
"""

import importlib.metadata
import importlib.resources

import pandas as pd
from astropy.time import Time

from kpfpipe import DETECTOR

# --- Authoritative configs -------------------------------------------------
# rvdata (installed package): the native↔EPRV map and the EPRV PRIMARY keyword
# definitions. KPF consumes these rather than re-encoding the standard.
_kpf_cfg = importlib.resources.files("rvdata.instruments.kpf.config")
_rv_cfg = importlib.resources.files("rvdata.core.models.config")
# This repo: the KPF-pipeline keyword registry (validator allowlist).
_kpf_data_cfg = importlib.resources.files("kpfpipe.data_models.config")

HEADER_MAP = pd.read_csv(_kpf_cfg / "header_map.csv")


def _load_keyword_csv(handle):
    """Read an rvdata PRIMARY-keyword CSV, stripping any UTF-8 BOM on column 1."""
    df = pd.read_csv(handle)
    df.columns = [str(c).lstrip("﻿").strip() for c in df.columns]
    return df


_L2_KW = _load_keyword_csv(_rv_cfg / "L2-PRIMARY-keywords.csv")
_L4_KW = _load_keyword_csv(_rv_cfg / "L4-PRIMARY-keywords.csv")
# KPF-pipeline keyword registry, split by the level that first writes the
# keyword (config/L{0,1,2,4}-headers.csv). All levels are combined into one
# allowlist, since a keyword written at an earlier level persists on PRIMARY.
_KPFPIPE_KW = pd.concat(
    [
        pd.read_csv(_kpf_data_cfg / f"L{level}-headers.csv")
        for level in ("0", "1", "2", "4")
    ],
    ignore_index=True,
)


# The EPRV keyword CSVs encode per-target/per-telescope families as a
# "BASE1 ... BASE#" template (e.g. "CDEC1 ... CDEC#"). rvdata seeds these
# expanded at index 1 (CDEC1); a producer may add CDEC2.. up to NUMTRACE/NUMTEL.
def _family_base(first_token):
    """'CDEC1' -> 'CDEC' (strip the trailing index)."""
    return first_token.rstrip("0123456789")


def _keyword_literals(df):
    """All EPRV keyword names, expanding 'BASE1 ... BASE#' families to BASE1-9."""
    out = set()
    for kw in df["Keyword"]:
        name = str(kw).strip()
        if not name:
            continue
        if "..." in name:
            base = _family_base(name.split("...")[0].strip())
            out.update(f"{base}{i}" for i in range(1, 10))
        else:
            out.add(name)
    return out


def _required_literals(df):
    """Required EPRV keyword names; families require only index 1 (rvdata's seed)."""
    flags = df["Required"].astype(str).str.strip().str.lower() == "true"
    out = set()
    for kw in df.loc[flags, "Keyword"]:
        name = str(kw).strip()
        if not name:
            continue
        if "..." in name:
            out.add(f"{_family_base(name.split('...')[0].strip())}1")
        else:
            out.add(name)
    return out


# --- Derived keyword sets --------------------------------------------------
# Every EPRV-standard PRIMARY keyword (L4 = L2 plus the L4-only RV summary keys).
EPRV_L2_KEYS = _keyword_literals(_L2_KW)
EPRV_L4_KEYS = EPRV_L2_KEYS | _keyword_literals(_L4_KW)
REQUIRED_L2_KEYS = _required_literals(_L2_KW)
REQUIRED_L4_KEYS = REQUIRED_L2_KEYS | _required_literals(_L4_KW)

# header_map STANDARD names are the EPRV-side targets the conversion produces.
HEADERMAP_STANDARD_KEYS = {
    str(r["STANDARD"]).strip()
    for _, r in HEADER_MAP.iterrows()
    if pd.notna(r["STANDARD"])
}
# header_map INSTRUMENT names that differ from their STANDARD target are the raw
# WMKO names that must NOT remain on a converted PRIMARY (they belong in
# INSTRUMENT_HEADER). Used for a precise "native leak" diagnostic.
NATIVE_PRIMARY_KEYS = {
    str(r["INSTRUMENT"]).strip()
    for _, r in HEADER_MAP.iterrows()
    if pd.notna(r["INSTRUMENT"])
    and str(r["INSTRUMENT"]).strip()
    and str(r["INSTRUMENT"]).strip() != str(r["STANDARD"]).strip()
}

# KPF-pipeline keywords legitimately written to PRIMARY. Each registry entry
# is an explicit 8-character-max FITS keyword (no wildcards); families are
# enumerated per member (e.g. RNGREEN1-4, CCD1BJD/CCD2BJD).
KPFPIPE_PRIMARY_KEYS = {str(k).strip() for k in _KPFPIPE_KW["keyword"]}

# FITS structural / bookkeeping cards plus KPF-internal PRIMARY keys that are
# neither EPRV nor in the registry but are always permitted.
_STRUCTURAL_KEYS = {
    "SIMPLE",
    "BITPIX",
    "EXTEND",
    "XTENSION",
    "PCOUNT",
    "GCOUNT",
    "BSCALE",
    "BZERO",
    "COMMENT",
    "HISTORY",
    "CONTINUE",
    "CHECKSUM",
    "DATASUM",
    "",
    "DATALVL",
    "ORIGID",
    "FILENAME",
    "DATE",
}

# Detector truth: KPF has 35 green + 32 red echelle orders (rvdata's header_map
# default of 65 is wrong — see notes/header_audit.md A2).
_NUMORDER = int(DETECTOR["norder"]["GREEN"]) + int(DETECTOR["norder"]["RED"])

# Initial DRPSTATU value on the L1 EPRV PRIMARY, before any pipeline module runs.
# Each module overwrites it via the receipt_add_entry override (see base.py).
_DRPSTATU_DEFAULT = "File ingested into KPF-DRP"


class HeaderParser:
    """Read and write FITS header keywords without minding the storage form.

    A KPF extension header is either an astropy ``fits.Header`` (scalar values,
    comments held separately) or a plain ``OrderedDict`` whose PRIMARY keys may
    hold ``(value, comment)`` tuples. These static helpers are the single place
    that bridges the two, so callers never hand-roll a tuple unwrap.
    """

    @staticmethod
    def _unwrap(value):
        """Drop a ``(value, comment)`` tuple wrapper, returning the bare value."""
        return value[0] if isinstance(value, tuple) else value

    @staticmethod
    def get(header, key, default=None):
        """Return the scalar value of ``key``, unwrapping a ``(value, comment)``
        tuple if present (a ``fits.Header`` already yields the scalar).

        ``default`` is returned only when ``key`` is **absent**; a stored ``None``
        value is returned as ``None``.
        """
        if key not in header:
            return default
        return HeaderParser._unwrap(header[key])

    @staticmethod
    def set(header, key, value, comment=None):
        """Write ``key``: as a ``(value, comment)`` card when ``comment`` is
        given, else as the bare value. The single documented write path.

        A commented write round-trips on PRIMARY and on any ``fits.Header``
        extension; on a plain-dict **non-PRIMARY** extension rvdata's serializer
        rejects tuples, so comments there must go through a real ``fits.Header``.
        """
        header[key] = (value, comment) if comment is not None else value


class HeaderConverter:
    """Convert and validate KPF PRIMARY headers across the WMKO-native, EPRV,
    and KPF-pipeline conventions.

    Single source of truth for the native→EPRV mapping (rvdata's
    ``header_map.csv``) and the EPRV PRIMARY allowlist. Stateless: the reference
    data (``HEADER_MAP`` and the derived keyword sets) is module-level, loaded
    once at import.
    """

    @staticmethod
    def _drp_version():
        """Exact DRP version (WMKO DRP-RUN-11), from installed package metadata."""
        return importlib.metadata.version("kpfpipe")

    @staticmethod
    def _full_jd_utc(native_primary):
        """Full Julian Date of the exposure start (EPRV JD_UTC), or None.

        The rvdata header_map maps JD_UTC <- MJD-OBS but omits the epoch offset,
        leaving a raw MJD (notes/header_audit.md A1). KPF's canonical exposure
        time is MJD-OBS — legacy code keys off it, and KPF's native DATE-OBS is
        date-only (e.g. '2024-04-05'), so it cannot supply the time of day. Hence
        JD_UTC = MJD-OBS + 2400000.5, which equals the JD of the precise ISO
        DATE-BEG. Fall back to DATE-BEG, then DATE-OBS, only if MJD-OBS is absent.
        """
        mjd = HeaderParser.get(native_primary, "MJD-OBS")
        if mjd not in (None, "", "UNKNOWN"):
            return float(mjd) + 2400000.5
        for key in ("DATE-BEG", "DATE-OBS"):
            val = HeaderParser.get(native_primary, key)
            if val not in (None, "", "UNKNOWN"):
                return float(Time(str(val), format="isot", scale="utc").jd)
        return None

    @classmethod
    def native_to_eprv(cls, native_primary):
        """Map a raw WMKO PRIMARY header to an EPRV-standard PRIMARY dict.

        For each header_map row, take the instrument (native) value when present,
        else the row default. Then apply the value corrections the installed
        header_map gets wrong (NUMORDER, JD_UTC) and stamp the DRP version
        (DRPTAG for EPRV, DRPVERNO for WMKO DRP-RUN-11).

        Parameters
        ----------
        native_primary : Mapping
            The raw L0 PRIMARY header (astropy ``fits.Header`` or dict).

        Returns
        -------
        dict
            EPRV-standard PRIMARY keyword → value (a few as ``(value, comment)``).
        """
        out = {}
        for _, row in HEADER_MAP.iterrows():
            standard_key = str(row["STANDARD"]).strip()
            instrument_key = (
                str(row["INSTRUMENT"]).strip() if pd.notna(row["INSTRUMENT"]) else ""
            )
            default_val = row["DEFAULT"] if pd.notna(row["DEFAULT"]) else None

            if instrument_key and instrument_key in native_primary:
                out[standard_key] = HeaderParser.get(native_primary, instrument_key)
            elif default_val is not None and str(default_val).strip():
                out[standard_key] = default_val

        # --- Value corrections (see notes/header_audit.md A1/A2/A3) ---
        out["NUMORDER"] = (_NUMORDER, "Number of echelle orders (green+red)")
        jd_utc = cls._full_jd_utc(native_primary)
        if jd_utc is not None:
            out["JD_UTC"] = (jd_utc, "[day] Julian date of exposure start")
        version = cls._drp_version()
        out["DRPTAG"] = (version, "DRP version")
        out["DRPVERNO"] = (version, "DRP version (WMKO DRP-RUN-11)")

        # WMKO provenance (DRP-RUN-19): native PROGID/KOAID, or 'UNKNOWN' if absent.
        out["PROGID"] = (
            HeaderParser.get(native_primary, "PROGID") or "UNKNOWN",
            "WMKO program ID",
        )
        out["KOAID"] = (
            HeaderParser.get(native_primary, "KOAID") or "UNKNOWN",
            "KOA archive ID",
        )
        # Initial reduction status (DRP-RUN-20); modules overwrite it as they run.
        out["DRPSTATU"] = (_DRPSTATU_DEFAULT, "DRP reduction status")
        return out

    @staticmethod
    def build_instrument_header(native_primary):
        """Verbatim scalar copy of the raw WMKO PRIMARY for INSTRUMENT_HEADER.

        INSTRUMENT_HEADER is an ImageHDU header (scalar values only) and is an
        immutable, pure pass-through of the raw instrument PRIMARY — nothing
        writes to it after ``to_kpf1``.

        Parameters
        ----------
        native_primary : Mapping
            The raw L0 PRIMARY header.

        Returns
        -------
        dict
            keyword → scalar value.
        """
        # Iterate .items() (not keyed get): for a fits.Header this yields each
        # commentary card's scalar string, where header["COMMENT"] would instead
        # return an astropy commentary-card object that fits.Header(dict) cannot
        # re-serialize.
        return {
            key: HeaderParser._unwrap(value) for key, value in native_primary.items()
        }

    @staticmethod
    def validate_eprv_primary(header, level):
        """Validate a converted EPRV PRIMARY header, raising on any inconsistency.

        Fail-loud guard (no silent fallback) for the L2/L4 boundary. Three rules:

        1. **Native leak** — a raw WMKO keyword name (header_map INSTRUMENT name
           differing from its EPRV target) is present on PRIMARY; it should have
           been converted or left in INSTRUMENT_HEADER.
        2. **Unregistered keyword** — a card that is neither an EPRV-standard
           keyword, a header_map STANDARD target, a registered KPF-pipeline
           keyword (``config/L{0,1,2,4}-headers.csv``), nor a structural card.
        3. **Missing required** — a Required EPRV PRIMARY keyword is absent.

        Parameters
        ----------
        header : Mapping
            The PRIMARY header to validate.
        level : str
            ``"L2"`` or ``"L4"`` (selects the EPRV keyword set).

        Raises
        ------
        ValueError
            On the first inconsistency found.
        """
        level = str(level).upper()
        eprv_keys = EPRV_L4_KEYS if level == "L4" else EPRV_L2_KEYS
        required_keys = REQUIRED_L4_KEYS if level == "L4" else REQUIRED_L2_KEYS

        for raw_key in list(header):
            key = str(raw_key).strip()
            if key in _STRUCTURAL_KEYS or key.startswith("NAXIS"):
                continue
            if (
                key in eprv_keys
                or key in HEADERMAP_STANDARD_KEYS
                or key in KPFPIPE_PRIMARY_KEYS
            ):
                continue
            if key in NATIVE_PRIMARY_KEYS:
                raise ValueError(
                    f"native WMKO keyword {key!r} found on {level} PRIMARY; it must "
                    "be converted to its EPRV name or kept in INSTRUMENT_HEADER"
                )
            raise ValueError(
                f"unregistered keyword {key!r} on {level} PRIMARY; add it to "
                "config/L{0,1,2,4}-headers.csv or fix the writer"
            )

        missing = sorted(k for k in required_keys if k not in header)
        if missing:
            raise ValueError(
                f"missing required EPRV PRIMARY keyword(s) on {level}: {missing}"
            )
