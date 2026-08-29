"""EPRV-standard compliance: the KPF keyword tables against rvdata's own.

The KPF registry is now the sole authority for what a KPF product's headers
contain -- nothing is derived from ``rvdata`` at runtime any more. This module is
what keeps that authority honest: it asserts the KPF tables against the pinned
upstream release, vendored byte-for-byte under
``kpfpipe/data_models/config/rvdata-0.4.0/``, plus the installed package's
instrument ``header_map.csv``.

Every known divergence is one commented entry in ``EXCEPTIONS``. That dict is the
single home for rvdata's known defects and KPF's deliberate deviations: adding a
row is a decision, and an undocumented divergence fails.

The classes are grouped by what they assert against: the pinned upstream release
and the installed header map first, then the science extension manifests, then
the masters', then the file-level checks that keep this module itself honest.
"""

import importlib.metadata
import importlib.resources
import re
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from kpfpipe import DETECTOR
from kpfpipe.data_models.level1 import KPF1

_CFG = importlib.resources.files("kpfpipe.data_models.config")
_VENDORED = _CFG / "rvdata-0.4.0"
_REPO_ROOT = Path(__file__).resolve().parents[2]

_KEYWORD_COLUMNS = [
    "Keyword",
    "Description",
    "Units",
    "DataType",
    "ExampleValue",
    "PopulatedBy",
]
_MANIFEST_COLUMNS = [
    "HDU",
    "Name",
    "DataType",
    "MinBitDepth",
    "Required",
    "Description",
]

_PROFILES = ("L0", "L1", "L2", "L4", "ML1", "ML2-flat", "ML2-wls")

_INDICES = range(1, DETECTOR["numtrace"] + 1)


# Every divergence from the pinned rvdata release, with the reason. An entry is a
# decision; anything not listed here must match.
EXCEPTIONS = {
    # -- rvdata defects ------------------------------------------------------
    # The order count is wrong: KPF reads 35 green + 32 red = 67, not 65.
    "NUMORDER": "rvdata header_map default 65; KPF is 35 green + 32 red = 67",
    # OBSALT is mapped from the native OBSLAT card.
    "OBSALT": "rvdata maps OBSALT <- native OBSLAT; KPF uses OBSERVATORY config",
    # Per-trace keywords are numbered CAL-first (1=CAL..5=SKY), the stale
    # translator convention. KPF is SKY-first, per trace-map.csv.
    "TRACE": "rvdata header_map is CAL-first; KPF is SKY-first (trace-map.csv)",
    "CLSRC": "rvdata header_map is CAL-first; KPF is SKY-first (trace-map.csv)",
    # The RVMETHOD row puts "CCF" one column past DEFAULT, so its default is dead
    # there; RVMETHOD is an L4 value RadialVelocity writes.
    "RVMETHOD": "rvdata header_map column shift; written by RadialVelocity",
    # PARANG/PARANG2 are not EPRV keywords at all, but the map names them.
    "PARANG": "not an EPRV keyword; rvdata header_map names it anyway",
    "PARANG2": "not an EPRV keyword; rvdata header_map names it anyway",
    # ISSOLAR is Boolean, but its default is the string "UNKNOWN".
    "ISSOLAR": 'rvdata default "UNKNOWN" is not a Boolean; KPF ships it blank',
    # TTIME is the TCS data timestamp; rvdata maps it from DATE, the file write
    # time. KPF maps DATE-MID, the mid-exposure time the TCS block describes.
    "TTIME": "rvdata maps TTIME <- DATE (file write time); KPF uses DATE-MID",
    # -- KPF-owned values, not static native mappings ------------------------
    "OBSLON": "KPF takes the site longitude from OBSERVATORY config",
    "OBSLAT": "KPF takes the site latitude from OBSERVATORY config",
    "GEOSYS": "KPF takes the reference frame from OBSERVATORY config",
    "JD_UTC": "float(MJD-OBS) + 2400000.5, applied in StandardizeDataFormat",
    "DRPTAG": "stamped from the kpfpipe version at standardization",
    "EPRVTAG": "stamped from the rv-data-standard pin at standardization",
    "VOCLASS": "stamped from the rv-data-standard release month",
    "DRPHASH": "stamped from the git commit hash at standardization",
    "DATALVL": "stamped from the model level, not defaulted",
    "INSTERA": "derived from JD_UTC against reference/instrument_eras.csv",
    "FILENAME": "stamped by to_fits from the path actually written",
    "SEEING": "sourced from the DiagL0 GDRSEEV metric",
    "SUNEL": "sourced from the DiagL0 TCSSUN metric",
    "MOONANG": "sourced from the DiagL0 TCSMOON metric",
    "EXSNR": "measured by DiagL2, not mapped from a native card",
    # rvdata maps the L4 summary from the previous DRP's CCF* natives; the KPF
    # pipeline recomputes all four at L4.
    "BJDTDB": "recomputed by BarycentricCorrection/RadialVelocity, not mapped",
    "RV": "recomputed by RadialVelocity, not mapped from the previous DRP",
    "RVERR": "recomputed by RadialVelocity, not mapped from the previous DRP",
    "BERV": "recomputed by BarycentricCorrection, not mapped",
    # -- CATALOG_RECORD is the sole writer of the SCI-fiber astrometry --------
    "CSRC": "CATALOG_RECORD is the sole writer of the SCI-fiber astrometry",
    "CID": "CATALOG_RECORD is the sole writer of the SCI-fiber astrometry",
    "CRA": "CATALOG_RECORD is the sole writer of the SCI-fiber astrometry",
    "CDEC": "CATALOG_RECORD is the sole writer of the SCI-fiber astrometry",
    "CEQNX": "CATALOG_RECORD is the sole writer of the SCI-fiber astrometry",
    "CEPCH": "CATALOG_RECORD is the sole writer of the SCI-fiber astrometry",
    "CPLX": "CATALOG_RECORD is the sole writer of the SCI-fiber astrometry",
    "CPMR": "CATALOG_RECORD is the sole writer of the SCI-fiber astrometry",
    "CPMD": "CATALOG_RECORD is the sole writer of the SCI-fiber astrometry",
    "CRV": "CATALOG_RECORD is the sole writer of the SCI-fiber astrometry",
    "CZ": "CATALOG_RECORD is the sole writer of the SCI-fiber astrometry",
    "CCLR": "CATALOG_RECORD is the sole writer of the SCI-fiber astrometry",
    # -- Cards that ship blank until something populates them ----------------
    # Each is a native mapping rvdata supplies that KPF has dropped. The card is
    # present and blank rather than absent, so an absent card can never be
    # mistaken for "no value"; populating them is follow-up work.
    "ORGANIZA": "blank until the licensing owner is decided",
    "OBSERVER": "blank: GROBSERV is the guider observer, not the KPF observer",
    "PROGRAM": "blank: GRPROGNA is the guider program, not the KPF program",
    "READMODE": "written by ImageAssembly from the ACF waveform, not READSPED",
    "ALIASES": "blank: TARGNAME is the pointing name, not a curated alias list",
    "AIRMASS": "blank until a KPF-computed airmass exists",
    "INHUM": "blank until the dome humidity telemetry is extracted",
    "INHUMT": "blank until the dome humidity telemetry is extracted",
    "SYSVEL": "blank: no systemic velocity is subtracted from KPF RVs yet",
    # -- KPF is a single-telescope instrument --------------------------------
    # rvdata's "BASE1 ... BASE#" telescope families expand to NUMTEL members;
    # KPF has one telescope, so only index 1 is registered.
    "TELEID": "single telescope: only index 1 is registered",
    "TLST": "single telescope: only index 1 is registered",
    "TRA": "single telescope: only index 1 is registered",
    "TDEC": "single telescope: only index 1 is registered",
    "TEL": "single telescope: only index 1 is registered",
    "TZA": "single telescope: only index 1 is registered",
    "TAZ": "single telescope: only index 1 is registered",
    "THA": "single telescope: only index 1 is registered",
    # PARST/PAREND are deliberately absent: rvdata's instrument header_map has no
    # row for either, so there is no upstream claim to except. Their single-
    # telescope expansion is encoded in _TELESCOPE_BASES, which _expand consumes.
    # CCLRN is absent for the same reason -- header_map names CCLR1..5 but no
    # CCLRN. TestExceptionsCoverage is what keeps such dead entries out.
    # -- Out of scope --------------------------------------------------------
    # The parametric wavelength-solution coefficients are a variable-length
    # family; KPF registers PVN_0/PVN_1 and writes none of them yet.
    "PVN_#": "variable-length coefficient family; KPF writes no parametric WLS",
}

# Every EXCEPTIONS key some assertion actually consulted, populated by _excepted()
# below and checked by TestExceptionsCoverage. An entry no assertion consumes is a
# claim about rvdata that nothing checks: it outlives the defect it documents and
# misleads the next reader.
_CONSUMED: set[str] = set()


def _excepted(key):
    """True if ``key`` is a recorded divergence, noting that it was consulted."""
    if key in EXCEPTIONS:
        _CONSUMED.add(key)
        return True
    return False


# The telescope-indexed families above, as bare bases -- KPF registers index 1
# only, so an rvdata "BASE1 ... BASE#" template expands to one member here.
_TELESCOPE_BASES = frozenset(
    {
        "TELEID",
        "TLST",
        "TRA",
        "TDEC",
        "TEL",
        "TZA",
        "TAZ",
        "THA",
        "PARST",
        "PAREND",
    }
)


def _text(value):
    """A CSV cell as text; a blank/NaN cell is the empty string.

    pandas reads the EPRV ``N/A`` units placeholder as NaN, so both sides of
    every comparison have to be normalized the same way.
    """
    return "" if pd.isna(value) else str(value).strip()


def _vendored(name):
    return pd.read_csv(_VENDORED / name)


def _kpf_keyword_files():
    return sorted(
        (p for p in _CFG.iterdir() if p.name.endswith("-keywords.csv")),
        key=lambda p: p.name,
    )


def _expand(name):
    """Expand a ``#`` template, and an rvdata ``BASE1 ... BASE#`` template.

    Returns ``(members, base)``. A telescope-indexed family stops at index 1:
    KPF observes on one telescope, so those keywords are written literally.
    """
    name = str(name).strip()
    if "..." in name:
        base = name.split("...")[0].strip().rstrip("0123456789")
        indices = [1] if base in _TELESCOPE_BASES else _INDICES
        return [f"{base}{i}" for i in indices], base
    if "#" in name:
        return [name.replace("#", str(i)) for i in _INDICES], name.rstrip("#")
    return [name], name


class TestPinnedRelease:
    """The exception list can only ever be applied to the release it was written
    against, so the pin is asserted before anything else."""

    def test_installed_release_matches_the_pin(self):
        pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
        pins = [
            d
            for d in pyproject["project"]["dependencies"]
            if d.startswith("rv-data-standard")
        ]
        assert pins == ["rv-data-standard==0.4.0"]
        assert importlib.metadata.version("rv-data-standard") == "0.4.0"

    def test_the_vendored_snapshot_is_that_release(self):
        installed = importlib.resources.files("rvdata.core.models.config")
        for name in ("L2-PRIMARY-keywords.csv", "L2-extensions.csv"):
            assert (_VENDORED / name).read_bytes() == (installed / name).read_bytes()


class TestTableSchemas:
    """Every KPF table carries exactly its schema -- the filename-driven layout
    only works if no file quietly grows or loses a column."""

    def test_keyword_files_have_the_keyword_schema(self):
        for path in _kpf_keyword_files():
            columns = list(pd.read_csv(path).columns)
            assert columns == _KEYWORD_COLUMNS, path.name

    def test_every_profile_has_a_manifest_with_the_manifest_schema(self):
        for profile in _PROFILES:
            columns = list(pd.read_csv(_CFG / f"{profile}-extensions.csv").columns)
            assert columns == _MANIFEST_COLUMNS, profile

    def test_keyword_filenames_use_a_known_profile(self):
        for path in _kpf_keyword_files():
            profile, _, extension = path.name[: -len("-keywords.csv")].rpartition("-")
            assert profile in _PROFILES, path.name
            assert extension, path.name

    def test_no_structural_card_is_registered(self):
        # rvdata declares XTENSION/EXTNAME as per-extension keywords; they are
        # written by astropy from the HDU structure and must not be carried over.
        registry = KPF1.keyword_registry
        assert not [k for k in registry.registered if registry.is_structural(k)]

    def test_the_deleted_header_csvs_are_gone(self):
        names = {p.name for p in _CFG.iterdir()}
        assert not [n for n in names if n.endswith("-headers.csv")]


class TestHeaderMapIsTheEprvPrimarySet:
    """``EPRV-header-map.csv`` and the keyword CSVs are two authorities over one
    set, so they are asserted equal rather than left to drift."""

    @staticmethod
    def _map():
        return pd.read_csv(_CFG / "EPRV-header-map.csv")

    def test_keys_are_unique(self):
        keys = self._map()["EPRV_KEY"].astype(str).str.strip()
        assert not list(keys[keys.duplicated()])

    def test_keys_are_exactly_the_registered_eprv_primary_set(self):
        mapped = set(self._map()["EPRV_KEY"].astype(str).str.strip())
        # Scoped to the science chain: the masters PRIMARY keywords (MASTYPE, the
        # WLS metadata) are registered but outside EPRV scope, so unmapped by
        # construction.
        registered = set()
        for level in ("L0", "L2", "L4"):
            for _, row in pd.read_csv(
                _CFG / f"{level}-PRIMARY-keywords.csv"
            ).iterrows():
                registered.update(_expand(row["Keyword"])[0])
        assert mapped <= registered
        # The only registered-but-unmapped science keywords are KPF's own.
        assert registered - mapped == {"RVGREEN", "RVRED", "ERVGREEN", "ERVRED"}

    def test_kpf_ext_is_primary_or_quality_control(self):
        for value in self._map()["KPF_EXT"].dropna().astype(str).str.strip():
            assert value in ("", "PRIMARY", "QUALITY_CONTROL"), value

    def test_required_matches_rvdata(self):
        # REQUIRED is a compliance label transcribed from rvdata, with rvdata's
        # own index-1-only family rule applied. Nothing in the pipeline gates on
        # it, so this assertion is the only thing keeping it true.
        expected = {}
        for name in ("L2-PRIMARY-keywords.csv", "L4-PRIMARY-keywords.csv"):
            for _, row in _vendored(name).iterrows():
                members, _base = _expand(row["Keyword"])
                required = str(row["Required"]).strip().lower() == "true"
                for i, member in enumerate(members):
                    expected[member] = required and i == 0

        mismatches = []
        for _, row in self._map().iterrows():
            key = str(row["EPRV_KEY"]).strip()
            if key not in expected:
                continue
            got = str(row["REQUIRED"]).strip().lower() == "true"
            if got != expected[key]:
                mismatches.append((key, expected[key], got))
        assert not mismatches


# The vendored per-extension keyword tables, and the extension each describes.
# Module-level so TestExceptionsCoverage can drive the same set; see there.
_PER_EXTENSION_TABLES = [
    ("L2-TRACE_FLUX-keywords.csv", "TRACE1_FLUX"),
    ("L2-TRACE_VAR-keywords.csv", "TRACE1_VAR"),
    ("L2-TRACE_BLAZE-keywords.csv", "TRACE1_BLAZE"),
    ("L2-TRACE_WAVE-keywords.csv", "TRACE1_WAVE"),
    ("L2-BJD_TDB-keywords.csv", "BJD_TDB"),
    ("L2-BARYCORR_KMS-keywords.csv", "BARYCORR_KMS"),
    ("L2-BARYCORR_Z-keywords.csv", "BARYCORR_Z"),
    ("L4-CCF1-keywords.csv", "CCF1"),
    ("L4-RV1-keywords.csv", "RV1"),
]


class TestKeywordsMatchRvdata:
    """The KPF EPRV rows against the vendored upstream tables."""

    @staticmethod
    def _rvdata_primary():
        """``{keyword: (Units, DataType)}`` from the vendored L2/L4 PRIMARY tables."""
        out = {}
        for name in ("L2-PRIMARY-keywords.csv", "L4-PRIMARY-keywords.csv"):
            for _, row in _vendored(name).iterrows():
                members, _base = _expand(row["Keyword"])
                for member in members:
                    out.setdefault(
                        member, (_text(row["Units"]), _text(row["DataType"]))
                    )
        return out

    def test_every_eprv_primary_keyword_is_registered(self):
        registered = KPF1.keyword_registry.allowed["PRIMARY"]
        missing = sorted(set(self._rvdata_primary()) - registered)
        assert not missing

    def test_units_and_datatype_are_transcribed_verbatim(self):
        registry = KPF1.keyword_registry
        table = registry.table
        primary = table[table["Extension"] == "PRIMARY"].set_index("Keyword")
        mismatches = []
        for keyword, (units, datatype) in self._rvdata_primary().items():
            if keyword not in primary.index:
                continue
            row = primary.loc[keyword]
            got = (_text(row["Units"]), _text(row["DataType"]))
            if got != (units, datatype):
                mismatches.append((keyword, (units, datatype), got))
        assert not mismatches

    @pytest.mark.parametrize(("filename", "extension"), _PER_EXTENSION_TABLES)
    def test_per_extension_keywords_are_registered(self, filename, extension):
        registry = KPF1.keyword_registry
        allowed = registry.allowed[extension]
        for _, row in _vendored(filename).iterrows():
            keyword = str(row["Keyword"]).strip()
            if registry.is_structural(keyword):
                continue
            if _excepted(keyword):
                continue
            assert keyword in allowed, (extension, keyword)
            assert (
                registry.datatype_for(keyword, extension)
                == str(row["DataType"]).strip()
            )


class TestInstalledHeaderMap:
    """Against the installed package's instrument map, which the vendored
    snapshot does not ship."""

    @staticmethod
    def _rvdata_map():
        cfg = importlib.resources.files("rvdata.instruments.kpf.config")
        return pd.read_csv(cfg / "header_map.csv")

    def test_native_sources_agree_outside_the_exception_list(self):
        kpf = pd.read_csv(_CFG / "EPRV-header-map.csv")
        kpf_source = {
            str(row["EPRV_KEY"]).strip(): _text(row["KPF_KEY"])
            for _, row in kpf.iterrows()
        }

        mismatches = []
        for _, row in self._rvdata_map().iterrows():
            key = str(row["STANDARD"]).strip()
            base = key.rstrip("0123456789") or key
            if _excepted(key) or _excepted(base):
                continue
            if key not in kpf_source:
                mismatches.append((key, "unregistered in KPF"))
                continue
            want = _text(row["INSTRUMENT"])
            if kpf_source[key] != want:
                mismatches.append((key, want, kpf_source[key]))
        assert not mismatches

    def test_defaults_agree_outside_the_exception_list(self):
        kpf = pd.read_csv(_CFG / "EPRV-header-map.csv")
        kpf_default = {
            str(row["EPRV_KEY"]).strip(): _text(row["DEFAULT"])
            for _, row in kpf.iterrows()
        }

        mismatches = []
        for _, row in self._rvdata_map().iterrows():
            key = str(row["STANDARD"]).strip()
            base = key.rstrip("0123456789") or key
            if _excepted(key) or _excepted(base) or key not in kpf_default:
                continue
            want = _text(row["DEFAULT"])
            if kpf_default[key] != want:
                mismatches.append((key, want, kpf_default[key]))
        assert not mismatches


class TestRoutingIsUnchanged:
    """The refactor must not silently rehome a keyword.

    The one intended change is the L4 per-fiber RV set: the vestigial
    ``{GRN,RED}{,E}RV{SK,S1,S2,S3,CL}`` names are replaced by
    ``RVGREEN``/``RVRED``/``ERVGREEN``/``ERVRED`` on ``RV1..5``.
    """

    _RETIRED = re.compile(r"^(GRN|RED)E?RV(SK|S1|S2|S3|CL)$")

    def test_no_retired_l4_rv_keyword_is_registered(self):
        registered = KPF1.keyword_registry.registered
        assert not [k for k in registered if self._RETIRED.match(k)]

    def test_the_renamed_set_is_registered_on_every_rv_extension(self):
        registry = KPF1.keyword_registry
        for i in _INDICES:
            for keyword in ("RVGREEN", "RVRED", "ERVGREEN", "ERVRED"):
                assert keyword in registry.allowed[f"RV{i}"]

    def test_routing_prefers_primary(self):
        # PRIMARY-preference is what keeps RVMETHOD and the combined-RV keywords
        # routable while they also live on RV1..5.
        routing = KPF1.keyword_registry.routing
        for keyword in ("RVMETHOD", "RVGREEN", "RVRED", "ERVGREEN", "ERVRED"):
            assert routing[keyword] == "PRIMARY"

    def test_multi_home_keywords_stay_unrouted(self):
        routing = KPF1.keyword_registry.routing
        for keyword in ("CTYPE1", "CTYPE2", "VELSTART", "VELWIDTH", "SKYRMVD"):
            assert keyword not in routing


class TestMinBitDepth:
    """The manifest declares the dtype floor; ``set_data`` enforces it.

    Photon-derived quantities (counts, ADU, electrons) are float32; time,
    velocity and wavelength quantities are float64, because 10 cm/s RV accuracy
    does not survive float32. L0 declares no floor at all: its arrays are raw
    native integers that must not be upcast.
    """

    @pytest.mark.parametrize(
        ("model", "extension", "depth"),
        [
            ("L1", "GREEN_CCD", 32),
            ("L1", "RED_VAR", 32),
            ("L1", "CA_HK", None),
            ("L1", "QUALITY_CONTROL", None),
            ("L2", "TRACE1_WAVE", 64),
            ("L2", "TRACE5_WAVE", 64),
            ("L2", "BJD_TDB", 64),
            ("L2", "BARYCORR_KMS", 64),
            ("L2", "BARYCORR_Z", 64),
            ("L2", "TRACE3_FLUX", 32),
            ("L2", "TRACE3_BLAZE", 32),
            ("L2", "CATALOG_RECORD", None),
            ("L4", "CCF1", 64),
            ("L4", "CCF_VAR5", 64),
            ("L4", "RV3", None),
            ("ML1", "GREEN_IMG", 32),
            ("ML1", "GREEN_MASK", 8),
            ("ML1", "RED_MASK", 8),
            ("ML2-wls", "TRACE1_WAVE", 64),
            ("ML2-wls", "GREEN_WLS_COEFFS", 64),
            ("ML2-flat", "TRACE2_FLUX", 32),
            ("L0", "GREEN_AMP1", None),
        ],
    )
    def test_declared_depth(self, model, extension, depth):
        manifest = pd.read_csv(_CFG / f"{model}-extensions.csv").set_index("Name")
        value = manifest.loc[extension, "MinBitDepth"]
        assert (None if pd.isna(value) else int(value)) == depth

    def test_the_model_reads_its_own_manifest(self):
        from kpfpipe.data_models.level2 import KPF2
        from kpfpipe.data_models.level4 import KPF4
        from kpfpipe.data_models.masters.level2 import KPFMasterL2

        assert KPF2()._get_min_bit_depth("TRACE1_WAVE") == 64
        assert KPF2()._get_min_bit_depth("TRACE3_FLUX") == 32
        assert KPF2()._get_min_bit_depth("CATALOG_RECORD") is None
        assert KPF4()._get_min_bit_depth("CCF3") == 64
        assert KPF4()._get_min_bit_depth("CCF_VAR3") == 64
        assert KPFMasterL2(kind="wls")._get_min_bit_depth("TRACE1_WAVE") == 64
        assert KPFMasterL2(kind="flat")._get_min_bit_depth("TRACE1_FLUX") == 32
        # An extension the manifest does not declare has no floor.
        assert KPF1()._get_min_bit_depth("NOT_AN_EXTENSION") is None


# --- the science L2/L4 extension manifests ---------------------------------

# rvdata rows KPF deliberately does not build. The EPRV-optional set rvdata's
# own readers name-guessed their way through; KPF's manifest-driven read declares
# what it accepts, so an undeclared extension is rejected rather than inferred.
_UNBUILT = {
    "L2": {
        "IMAGE": "EPRV-optional; KPF ships no whole-detector image at L2",
        "TRACE1_DRIFT": "EPRV-optional; KPF has no drift product",
        "TRACE1_QUALITY": "EPRV-optional; KPF carries QC in QUALITY_CONTROL",
        "TRACE1_SKYMODEL": "EPRV-optional; KPF has no sky model",
        "TRACE1_TELLURIC": "EPRV-optional; KPF does no telluric correction",
        "CUSTOM1_TRACE1_FLUX": "EPRV CUSTOM slot; unused by KPF",
        "CUSTOM1_TRACE1_VAR": "EPRV CUSTOM slot; unused by KPF",
        "CUSTOM1_TRACE1_WAVE": "EPRV CUSTOM slot; unused by KPF",
    },
    "L4": {
        "DIAGNOSTICS1": "EPRV-optional; KPF carries diagnostics in QUALITY_CONTROL",
        "CUSTOM_CCF1": "EPRV CUSTOM slot; unused by KPF",
        "CUSTOM_RV1": "EPRV CUSTOM slot; unused by KPF",
    },
}

# Extensions whose DataType deliberately differs from the standard's.
_DATATYPE_DEVIATIONS = {
    "ANCILLARY_SPECTRUM": (
        "EPRV says ImageHDU; KPF ships Ca H&K as a BinTableHDU placeholder while "
        "extraction is WIP, and existing products encode it that way"
    ),
}


def _science_manifests(level):
    return (
        pd.read_csv(_VENDORED / f"{level}-extensions.csv"),
        pd.read_csv(_CFG / f"{level}-extensions.csv"),
    )


class TestScienceManifestsMatchRvdata:
    """The L2/L4 manifests are now the sole authority for what KPF builds.

    Until the manifests landed, the KPF models could not under-declare: rvdata's
    ``RV2.__init__``/``RV4.__init__`` created every ``Required`` row first,
    whatever KPF asked for. Dropping those bases removes that enforcer, so the
    guarantee has to be asserted here instead of inherited.
    """

    @pytest.mark.parametrize("level", ("L2", "L4"))
    def test_every_rvdata_required_extension_is_declared(self, level):
        rvdata, kpf = _science_manifests(level)
        required = set(rvdata[rvdata["Required"]]["Name"])
        assert required <= set(kpf["Name"])

    @pytest.mark.parametrize("level", ("L2", "L4"))
    def test_shared_extensions_agree_on_datatype(self, level):
        rvdata, kpf = _science_manifests(level)
        theirs = dict(zip(rvdata["Name"], rvdata["DataType"], strict=True))
        ours = dict(zip(kpf["Name"], kpf["DataType"], strict=True))
        differing = {
            name for name in set(theirs) & set(ours) if theirs[name] != ours[name]
        }
        assert differing <= set(_DATATYPE_DEVIATIONS)

    @pytest.mark.parametrize("level", ("L2", "L4"))
    def test_undeclared_rvdata_extensions_are_listed(self, level):
        rvdata, kpf = _science_manifests(level)
        assert set(rvdata["Name"]) - set(kpf["Name"]) == set(_UNBUILT[level])

    def test_the_models_build_their_whole_manifest(self):
        from kpfpipe.data_models.level2 import KPF2
        from kpfpipe.data_models.level4 import KPF4

        for model in (KPF2(), KPF4()):
            assert set(model.extensions) == set(model._manifest["Name"])

    def test_min_bit_depth_is_at_least_rvdatas(self):
        # L2 only: rvdata's L4 manifest has no MinBitDepth column. KPF's is a
        # superset policy, not a copy -- it adds the float32 and 8-bit rows and
        # the whole L4 column -- so the assertion is a floor, not equality.
        rvdata, kpf = _science_manifests("L2")
        ours = dict(zip(kpf["Name"], kpf["MinBitDepth"], strict=True))
        declared = rvdata[rvdata["MinBitDepth"].notna()]
        checked = 0
        for _, row in declared.iterrows():
            if row["Name"] in _UNBUILT["L2"]:
                continue
            assert ours[row["Name"]] >= row["MinBitDepth"]
            checked += 1
        assert checked, "no shared MinBitDepth rows left to check"


# --- the master extension manifests ----------------------------------------

# The MinBitDepth floor each master manifest declares. config/rvdata-0.4.0/
# vendors no master manifests -- masters are outside EPRV scope -- so there is
# no upstream table to compare against; this is the statement of record.
_MASTER_DEPTHS = {
    "ML1": {
        f"{chip}_{kind}": 32 for chip in ("GREEN", "RED") for kind in ("IMG", "SNR")
    }
    | {f"{chip}_MASK": 8 for chip in ("GREEN", "RED")},
    "ML2-wls": {f"TRACE{n}_WAVE": 64 for n in _INDICES}
    | {f"{chip}_WLS_COEFFS": 64 for chip in ("GREEN", "RED")},
    "ML2-flat": {
        f"TRACE{n}_{kind}": 32 for n in _INDICES for kind in ("FLUX", "VAR", "BLAZE")
    },
}

# Which science manifest each master profile shares its rows with.
_MASTER_LEVELS = {"ML1": "L1", "ML2-wls": "L2", "ML2-flat": "L2"}


def _master_model(profile):
    from kpfpipe.data_models.masters.level1 import KPFMasterL1
    from kpfpipe.data_models.masters.level2 import KPFMasterL2

    if profile == "ML1":
        return KPFMasterL1()
    return KPFMasterL2(kind=profile.removeprefix("ML2-"))


class TestMasterManifests:
    """The master manifests are the masters' sole authority, as the science
    manifests are the science levels'.

    Masters are outside EPRV scope, so nothing upstream constrains these tables:
    the floors below and the science-manifest agreement are what keep a master's
    shape honest now that it is built from its own manifest rather than
    subtracted from the science one.
    """

    @pytest.mark.parametrize("profile", tuple(_MASTER_DEPTHS))
    def test_declared_depths(self, profile):
        manifest = pd.read_csv(_CFG / f"{profile}-extensions.csv")
        declared = {
            row["Name"]: int(row["MinBitDepth"])
            for _, row in manifest.iterrows()
            if not pd.isna(row["MinBitDepth"])
        }
        assert declared == _MASTER_DEPTHS[profile]

    @pytest.mark.parametrize("profile", tuple(_MASTER_DEPTHS))
    def test_the_master_reads_its_own_manifest(self, profile):
        master = _master_model(profile)
        for name, depth in _MASTER_DEPTHS[profile].items():
            assert master._get_min_bit_depth(name) == depth
        assert master._get_min_bit_depth("QUALITY_CONTROL") is None

    @pytest.mark.parametrize("profile", tuple(_MASTER_DEPTHS))
    def test_the_master_builds_its_whole_manifest(self, profile):
        master = _master_model(profile)
        assert set(master.extensions) == set(master._manifest["Name"])

    @pytest.mark.parametrize("profile", tuple(_MASTER_DEPTHS))
    def test_no_master_carries_an_instrument_header(self, profile):
        # A master is not a translation of a native instrument product, so it
        # carries no verbatim instrument header. Asserted in both forms: were a
        # manifest to gain the row, the model would start shipping the HDU.
        manifest = pd.read_csv(_CFG / f"{profile}-extensions.csv")
        assert "INSTRUMENT_HEADER" not in set(manifest["Name"])
        assert "INSTRUMENT_HEADER" not in _master_model(profile).extensions

    @pytest.mark.parametrize("profile", tuple(_MASTER_LEVELS))
    def test_shared_rows_agree_with_the_science_manifest(self, profile):
        # The duplication between a master manifest and its science level's is
        # intentional -- each stays a complete, readable spec of one product --
        # so a shared row must not disagree. Building from the master manifest
        # reads these cells, so a disagreement would ship a master whose
        # TRACE1_WAVE differed from L2's.
        master = pd.read_csv(_CFG / f"{profile}-extensions.csv").set_index("Name")
        science = pd.read_csv(
            _CFG / f"{_MASTER_LEVELS[profile]}-extensions.csv"
        ).set_index("Name")
        shared = set(master.index) & set(science.index)
        assert shared, f"{profile} shares no rows with {_MASTER_LEVELS[profile]}"
        for name in shared:
            assert master.loc[name, "DataType"] == science.loc[name, "DataType"]
            ours, theirs = (
                master.loc[name, "MinBitDepth"],
                science.loc[name, "MinBitDepth"],
            )
            assert (pd.isna(ours) and pd.isna(theirs)) or ours == theirs


# --- file-level meta-assertions ---------------------------------------------


class TestPackaging:
    """The `pyproject.toml` package-data globs reach every shipped config table.

    `config/rvdata-0.4.0/` is the vendored pinned snapshot this whole file asserts
    against. The original glob matched only the top level of `config/`, so the
    snapshot was absent from a non-editable install and every assertion here would
    fail there -- on the install path, not in the repo. This check is static: a
    config subdirectory added later without a matching glob fails here rather than
    in someone's wheel.
    """

    @staticmethod
    def _package_data():
        with open(_REPO_ROOT / "pyproject.toml", "rb") as fh:
            data = tomllib.load(fh)
        return data["tool"]["setuptools"]["package-data"]["kpfpipe.data_models.config"]

    def test_top_level_csvs_are_packaged(self):
        assert "*.csv" in self._package_data()

    def test_every_config_subdirectory_with_csvs_has_a_glob(self):
        globs = set(self._package_data())
        config = Path(str(_CFG))
        missing = [
            f"{d.name}/*.csv"
            for d in sorted(config.iterdir())
            if d.is_dir() and any(d.glob("*.csv")) and f"{d.name}/*.csv" not in globs
        ]
        assert not missing, f"config subdirectories not packaged: {missing}"


class TestExceptionsCoverage:
    """Every `EXCEPTIONS` entry is still exercised by some assertion above.

    An exception entry that no assertion consumes is a claim about rvdata that
    nothing checks: it outlives the defect it documents and misleads the next
    reader into thinking a divergence is still being watched. `_excepted()` records
    each consultation, and the test drives every consumer itself rather than
    trusting collection order -- see `_drive_every_consumer`.

    Three entries were removed when this check landed -- `CCLRN`, `PARST`, `PAREND`
    -- because rvdata's instrument `header_map.csv` carries no row for any of them,
    so the assertions that consume `EXCEPTIONS` could never reach them. The
    single-telescope decision they described lives in `_TELESCOPE_BASES`.
    """

    @staticmethod
    def _drive_every_consumer():
        """Run every assertion that consults EXCEPTIONS, in this process.

        Deliberately not a reliance on collection order. `_CONSUMED` is
        per-process state, and `make test` runs `-n auto --dist loadscope`, which
        can place the consuming classes on a different xdist worker than this one
        -- under which a passive check sees a partial set and fails on entries
        that are consumed perfectly well elsewhere. Driving them here makes the
        coverage assertion deterministic under any distribution, and under a
        targeted `-k` subset too.
        """
        installed = TestInstalledHeaderMap()
        installed.test_native_sources_agree_outside_the_exception_list()
        installed.test_defaults_agree_outside_the_exception_list()
        keywords = TestKeywordsMatchRvdata()
        for filename, extension in _PER_EXTENSION_TABLES:
            keywords.test_per_extension_keywords_are_registered(filename, extension)

    def test_every_exception_is_consumed(self):
        self._drive_every_consumer()
        assert _CONSUMED, "no consumer consulted EXCEPTIONS at all"
        assert set(EXCEPTIONS) == _CONSUMED, {
            "never consulted": sorted(set(EXCEPTIONS) - _CONSUMED),
            "consulted but unlisted": sorted(_CONSUMED - set(EXCEPTIONS)),
        }
