"""Tests for shared KPFDataModel behaviour (data_models/base.py).

KPF1 is the vehicle for the contract every model inherits: extension headers are
``fits.Header`` from construction onward, and a commented PRIMARY card survives
``to_fits`` -> ``from_fits`` (rvdata >=0.4.0 copies the ``fits.Header`` directly).
KPF2/KPF4 round-trip guards live in test_data_models_l{2,4}.py.
"""

import importlib.metadata
import importlib.resources
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from kpfpipe import DETECTOR
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.data_models.masters.level1 import KPFMasterL1
from kpfpipe.data_models.masters.level2 import KPFMasterL2

from ._eprv import expand, kpf_table
from ._registry import expected_comment, expected_routing


class TestHeaderStorage:
    """Every extension header is a fits.Header (not rvdata's OrderedDict)."""

    def test_fresh_headers_are_fits_headers(self):
        l1 = KPF1()
        assert isinstance(l1.headers["PRIMARY"], fits.Header)


class TestPrimaryCommentRoundTrip:
    def test_primary_comment_round_trips(self, tmp_path):
        l1 = KPF1()
        l1.headers["PRIMARY"]["DATE-OBS"] = "2024-01-13T10:26:56"
        l1.headers["PRIMARY"]["BIASAGE"] = (-0.5, "[day] bias age")

        fn = str(tmp_path / "kpf_L1_20240113T102656.fits")
        l1.to_fits(fn)

        prim = KPF1.from_fits(fn).headers["PRIMARY"]
        assert prim.get("BIASAGE") == -0.5
        assert prim.comments["BIASAGE"] == "[day] bias age"


class TestUndefinedRoundTrip:
    """A None value (non-finite RV) becomes a FITS UNDEFINED card, not a crash."""

    def test_none_value_round_trips_as_undefined(self, tmp_path):
        l1 = KPF1()
        l1.headers["PRIMARY"]["DATE-OBS"] = "2024-01-13T10:26:56"
        # radial_velocity writes None for a non-finite fit; it must round-trip as
        # an undefined card, never as a finite number.
        l1.headers["PRIMARY"]["RV"] = None

        fn = str(tmp_path / "kpf_L1_20240113T102656.fits")
        l1.to_fits(fn)

        prim = KPF1.from_fits(fn).headers["PRIMARY"]
        assert "RV" in prim
        val = prim["RV"]
        assert val is None or isinstance(val, fits.card.Undefined)


class TestSetKeyword:
    """KPFDataModel.set_keyword routes to the registry extension with its comment."""

    def test_routes_to_quality_control(self):
        l1 = KPF1()
        l1.set_keyword("RNGREEN1", 3.5)
        assert l1.headers["QUALITY_CONTROL"]["RNGREEN1"] == 3.5
        # The comment comes from the registry Description, not the caller.
        assert l1.headers["QUALITY_CONTROL"].comments["RNGREEN1"] == (
            "Read noise GREEN amp 1 [e-]"
        )
        assert "RNGREEN1" not in l1.headers["PRIMARY"]

    def test_routes_to_receipt(self):
        l1 = KPF1()
        l1.set_keyword("BIASFILE", "/path/to/master_bias_L1.fits")
        assert l1.headers["RECEIPT"]["BIASFILE"] == "/path/to/master_bias_L1.fits"
        assert "BIASFILE" not in l1.headers["PRIMARY"]

    def test_routes_to_barycorr_and_bjd_extensions(self):
        l2 = KPF2()
        l2.set_keyword("BVGREEN", -12.3)
        l2.set_keyword("BJDGREEN", 2460000.5)
        assert l2.headers["BARYCORR_KMS"]["BVGREEN"] == -12.3
        assert l2.headers["BJD_TDB"]["BJDGREEN"] == 2460000.5

    def test_routes_l4_orderlet_rv_to_rv_table(self):
        # RVGREEN is homed on both PRIMARY and every RV# table, so the bare write
        # takes the PRIMARY-preferring route and ext= reaches the orderlet's own.
        l4 = KPF4()
        l4.set_keyword("RVGREEN", 1.2345, ext="RV2")  # GREEN SCI1 -> RV2
        l4.set_keyword("RVGREEN", 6.789)  # GREEN SCI-combined -> PRIMARY
        assert l4.headers["RV2"]["RVGREEN"] == 1.2345
        assert l4.headers["PRIMARY"]["RVGREEN"] == 6.789

    def test_targeted_ext_writes_eprv_per_extension_card(self):
        # EPRV per-extension cards (VELSTART on every CCF#, RVMETHOD on every RV#)
        # have no single routed home, so ext= picks one, alias-resolved.
        l4 = KPF4()
        l4.set_keyword("VELSTART", -100.0, ext="SCI2_CCF")  # SCI2_CCF -> CCF3
        l4.set_keyword("RVMETHOD", "CCF", ext="SCI2_RV")  # SCI2_RV -> RV3
        assert l4.headers["CCF3"]["VELSTART"] == -100.0
        assert l4.headers["CCF3"].comments["VELSTART"] == "Velocity grid start [km/s]"
        assert l4.headers["RV3"]["RVMETHOD"] == "CCF"

    def test_targeted_ext_writes_kpf_multi_home_card(self):
        # VELWIDTH is registered as CCF* -> CCF1..5, so like the EPRV per-extension
        # cards it has no routed home and is ext=-only.
        l4 = KPF4()
        l4.set_keyword("VELWIDTH", 1.0, ext="SCI2_CCF")  # SCI2_CCF -> CCF3
        assert l4.headers["CCF3"]["VELWIDTH"] == 1.0
        assert (
            l4.headers["CCF3"].comments["VELWIDTH"] == "CCF mask hole full width [km/s]"
        )
        with pytest.raises(KeyError, match="not registered"):
            l4.set_keyword("VELWIDTH", 1.0)  # no home extension without ext=

    def test_targeted_ext_unregistered_for_extension_raises(self):
        l4 = KPF4()
        with pytest.raises(KeyError, match="not registered for extension"):
            l4.set_keyword("VELSTART", 1.0, ext="SCI2_RV")  # VELSTART is CCF-only

    def test_unregistered_keyword_raises_keyerror(self):
        l1 = KPF1()
        with pytest.raises(KeyError, match="not registered"):
            l1.set_keyword("BOGUSKEY", 1)

    def test_missing_extension_raises_valueerror(self):
        # KPF1 has no BARYCORR_KMS extension for BVGREEN to route to.
        l1 = KPF1()
        with pytest.raises(ValueError, match="does not exist"):
            l1.set_keyword("BVGREEN", 1.0)

    def test_value_is_coerced_to_the_registered_datatype(self):
        l2 = KPF2()
        l2.set_keyword("NUMORDER", "35")  # Uint
        l2.set_keyword("ISSOLAR", "F")  # Boolean
        l2.set_keyword("DATAPRL0", 1.0, ext="QUALITY_CONTROL")  # int
        l2.set_keyword("CRVALN", 1.5, ext="TRACE1_WAVE")  # float32
        assert l2.headers["PRIMARY"]["NUMORDER"] == 35
        assert l2.headers["PRIMARY"]["ISSOLAR"] is False
        assert l2.headers["QUALITY_CONTROL"]["DATAPRL0"] == 1
        # The declared width, so the card is written at single precision.
        assert isinstance(l2.headers["TRACE1_WAVE"]["CRVALN"], np.float32)

    def test_unconvertible_value_raises_typeerror(self):
        l1 = KPF1()
        with pytest.raises(TypeError, match="declared Uint"):
            l1.set_keyword("NUMORDER", -1)
        with pytest.raises(TypeError, match="declared Boolean"):
            l1.set_keyword("ISSOLAR", "yes")

    def test_none_writes_the_blank_card_through(self):
        l1 = KPF1()
        l1.set_keyword("NUMORDER", None)
        assert l1.headers["PRIMARY"]["NUMORDER"] is None


class TestRegistryConformance:
    """The routing table the model exposes agrees with the registry CSVs.

    Oracle: ``_registry`` reads config/*-keywords.csv directly and re-derives the
    routing rule by hand, independent of the code under test.
    """

    def test_routing_matches_the_csv_homes(self):
        routing = KPF1.keyword_registry.routing
        expected = expected_routing()
        mismatches = [
            (kw, want, routing.get(kw))
            for kw, want in expected.items()
            if routing.get(kw) != want
        ]
        assert not mismatches, f"routing != CSV homes: {mismatches}"

    def test_multi_home_keywords_are_not_routed(self):
        # A keyword on several extensions and on none of them PRIMARY (CTYPE1 on
        # every trace, VELWIDTH on every CCF#) is ext=-only.
        registry = KPF1.keyword_registry
        routing = registry.routing
        homes = registry.table.groupby("Keyword")["Extension"].apply(set)
        for keyword, extensions in homes.items():
            if len(extensions) > 1 and "PRIMARY" not in extensions:
                assert keyword not in routing, f"multi-home {keyword} must not route"
            else:
                assert keyword in routing, f"{keyword} missing from routing table"

    def test_comment_is_description_and_units(self):
        # The derived comments lookup against the table it is built from.
        reg = KPF1.keyword_registry
        for row in reg.table.itertuples(index=False):
            got = reg.comment_for(row.Keyword, row.Extension)
            assert got == expected_comment(row.Description, row.Units)

    def test_datatype_matches_the_table(self):
        reg = KPF1.keyword_registry
        for row in reg.table.itertuples(index=False):
            assert reg.datatype_for(row.Keyword, row.Extension) == row.DataType

    def test_qc_flag_keyword_sets_are_scoped_by_level(self):
        # Registry scoping, not Checkpoint behaviour; reached through the model
        # rather than importing data_models.keyword_registry (see _registry.py).
        reg = KPF1.keyword_registry
        # Representative checks live under their own level.
        assert "RNOK" in reg.qc_flag_keywords_by_level["L1"]
        assert "DATAPRL2" in reg.qc_flag_keywords_by_level["L2"]
        assert "RNOK" not in reg.qc_flag_keywords_by_level["L2"]

    def test_per_extension_keyword_expands_across_orderlets(self):
        # A "CCF*" Extension registers the keyword on CCF1..CCF5 while staying
        # out of routing.
        reg = KPF1.keyword_registry
        for n in range(1, 6):
            assert "VELWIDTH" in reg.allowed[f"CCF{n}"]
            assert (
                reg.comment_for("VELWIDTH", f"CCF{n}")
                == "CCF mask hole full width [km/s]"
            )
        assert "VELWIDTH" not in reg.routing


class TestKeywordRegistry:
    """The unified registry table and its derived lookups."""

    def test_columns(self):
        assert list(KPF1.keyword_registry.table.columns) == [
            "Keyword",
            "Description",
            "Extension",
            "DataType",
            "PopulatedBy",
            "Level",
            "Units",
        ]

    def test_unions_every_data_model(self):
        keys = KPF1.keyword_registry.registered
        assert "RNGREEN1" in keys and "RV" in keys and "MASTYPE" in keys

    def test_unregistered_header_map_targets_are_absent(self):
        # PARANG/PARANG2 are rvdata header_map names KPF does not carry; the KPF
        # header map must not name them either, and _load_header_map would raise
        # if it did.
        assert "PARANG" not in KPF1.keyword_registry.registered
        assert "PARANG2" not in KPF1.keyword_registry.registered

    def test_primary_allowed_not_level_gated(self):
        # An L4 keyword (RV) is allowed on PRIMARY regardless of product level.
        assert "RV" in KPF1.keyword_registry.allowed["PRIMARY"]

    def test_is_structural_only_fits_cards(self):
        # Structural = FITS cards astropy writes from the HDU structure, never
        # authored by the pipeline (exact + bintable/array prefixes).
        reg = KPF1.keyword_registry
        for card in ("SIMPLE", "EXTNAME", "XTENSION", "NAXIS2", "TTYPE1", "TFORM3"):
            assert reg.is_structural(card), card
        # Content keywords (pipeline-authored) are NOT structural -- including CTYPE
        # (axis meaning), DATE/FILENAME, and the pruned WCS family (KPF writes no WCS).
        for kw in (
            "DATALVL",
            "ORIGID",
            "PROGID",
            "DRPTAG",
            "RV",
            "CTYPE1",
            "DATE",
            "FILENAME",
            "CUNIT1",
            "CRVAL1",
            "CDELT1",
        ):
            assert not reg.is_structural(kw), kw

    def test_registered_and_structural_are_disjoint(self):
        # The KPF CSVs register no structural card: rvdata declares XTENSION and
        # EXTNAME as per-extension keywords, and they are not carried over.
        reg = KPF1.keyword_registry
        assert not [k for k in reg.registered if reg.is_structural(k)]
        assert "XTENSION" not in reg.registered
        assert "EXTNAME" not in reg.registered

    def test_units_populated_for_eprv_rows_blank_for_unitless(self):
        table = KPF1.keyword_registry.table

        def units(keyword, extension):
            match = table[
                (table["Keyword"] == keyword) & (table["Extension"] == extension)
            ]
            return match.iloc[0]["Units"]

        assert units("EXPTIME", "PRIMARY") == "s"
        assert units("RNGREEN1", "QUALITY_CONTROL") == "e-"
        assert units("DATAPRL1", "QUALITY_CONTROL") == ""

    def test_primary_seed_is_cumulative_by_level(self):
        # Each level adds its own DQLVL bitfield; L1 also READMODE, L2 the
        # extraction SNR cards and L4 the RV summary.
        reg = KPF1.keyword_registry
        l0, l1, l2, l4 = (reg.primary_seed(p) for p in ("L0", "L1", "L2", "L4"))
        assert set(l0) < set(l1) < set(l2) < set(l4)
        assert set(l1) - set(l0) == {"DQLVL1", "READMODE"}
        assert set(l2) - set(l1) == {"EXTRACT", "DQLVL2"} | {
            f"EXSNR{i}" for i in range(1, 6)
        } | {f"EXSNRW{i}" for i in range(1, 6)}
        assert set(l4) - set(l2) == {
            "BJDTDB",
            "RV",
            "RVERR",
            "BERV",
            "RVMETHOD",
            "SYSVEL",
            "SYSACC",
            "DQLVL4",
        }

    def test_primary_seed_is_typed_and_commented(self):
        seed = KPF1.keyword_registry.primary_seed("L0")
        assert seed["NUMTRACE"] == (5, "Number of object-indexed keyword families")
        assert seed["OBSALT"][0] == 4160.0  # injected from kpfpipe.KECK_LOCATION
        assert seed["AIRMASS"] == (None, "Airmass at start of exposure [secZ]")

    def test_primary_seed_covers_every_member_of_a_family(self):
        # The five-trace rule at the header level: no Required filter, so every
        # index of every # family is seeded.
        seed = KPF1.keyword_registry.primary_seed("L0")
        for base in ("TRACE", "CRA", "CDEC", "CSRC", "CID", "CZ", "CCLR"):
            for i in range(1, 6):
                assert f"{base}{i}" in seed

    def test_masters_seed_is_the_master_s_own_rows(self):
        # Masters are outside EPRV scope: they do not inherit the science skeleton.
        reg = KPF1.keyword_registry
        assert set(reg.primary_seed("ML1")) == {"MASTYPE"}
        assert "POLYDEGX" in reg.primary_seed("ML2-wls")
        assert "INSTRUME" not in reg.primary_seed("ML2-wls")

    def test_unknown_data_model_raises(self):
        with pytest.raises(ValueError, match="unknown data model"):
            KPF1.keyword_registry.primary_seed("L3")

    def test_datatype_for_scopes_by_extension(self):
        reg = KPF1.keyword_registry
        assert reg.datatype_for("INSTRUME", "PRIMARY") == "String"
        assert reg.datatype_for("NUMTRACE", "PRIMARY") == "UInt"
        assert reg.datatype_for("INSTRUME", "QUALITY_CONTROL") is None

    def test_header_map_keys_are_registered_on_primary(self):
        # The two authorities cannot drift: _load_header_map raises otherwise.
        reg = KPF1.keyword_registry
        keys = set(reg.header_map["EPRV_KEY"].astype(str).str.strip())
        assert keys <= reg.allowed["PRIMARY"]
        assert len(keys) == len(reg.header_map)

    def test_parse_value_has_no_sentinel_rule(self):
        # SCI-OBJ/SKY-OBJ carry the literal value "Unknown" on real frames;
        # blanking it would silently empty TRACE1..TRACE4.
        reg = KPF1.keyword_registry
        assert reg._parse_value("TRACE2", "String", "Unknown") == "Unknown"
        assert reg._parse_value("TRACE2", "String", "") is None
        assert reg._parse_value("ISSOLAR", "Boolean", "F") is False
        assert reg._parse_value("NUMTRACE", "UInt", "5") == 5
        # A blank DataType passes the value through unchanged.
        assert reg._parse_value("ANY", "", "as-is") == "as-is"

    def test_parse_value_rejects_an_unknown_datatype(self):
        with pytest.raises(ValueError, match="unknown DataType"):
            KPF1.keyword_registry._parse_value("FOO", "complex", "1")


class TestQualityControlPropagation:
    """QUALITY_CONTROL + RECEIPT header cards survive to_fits and L0->L1->L2."""

    def test_l1_quality_control_receipt_roundtrip(self, tmp_path):
        l1 = KPF1()
        l1.set_keyword("RNGREEN1", 4.2)
        l1.set_keyword("BIASAGE", 1.5)
        l1.set_keyword("OSCANSUB", 1)
        l1.set_keyword("BIASFILE", "/m/bias_L1.fits")
        fn = str(tmp_path / "kpf_L1_20240101T000000.fits")
        l1.to_fits(fn)
        back = KPF1.from_fits(fn)
        assert back.headers["QUALITY_CONTROL"]["RNGREEN1"] == 4.2
        assert back.headers["QUALITY_CONTROL"].comments["RNGREEN1"] == (
            "Read noise GREEN amp 1 [e-]"
        )
        assert back.headers["QUALITY_CONTROL"]["BIASAGE"] == 1.5
        assert back.headers["RECEIPT"]["OSCANSUB"] == 1
        assert back.headers["RECEIPT"]["BIASFILE"] == "/m/bias_L1.fits"

    def test_propagation_l0_to_l1_to_l2(self):
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "Bias"
        l0.headers["PRIMARY"]["MJD-OBS"] = 60310.0
        l0.standardize_header_format()
        # QCL0 routes L0 QC flags to QUALITY_CONTROL.
        l0.set_keyword("NOTJUNK", 1)
        l1 = l0.to_kpf1()
        assert l1.headers["QUALITY_CONTROL"]["NOTJUNK"] == 1
        l1.set_keyword("OSCANSUB", 1)
        l1.set_keyword("RNGREEN1", 4.0)
        l2 = l1.to_kpf2()
        assert l2.headers["QUALITY_CONTROL"]["NOTJUNK"] == 1
        assert l2.headers["QUALITY_CONTROL"]["RNGREEN1"] == 4.0
        assert l2.headers["RECEIPT"]["OSCANSUB"] == 1


class TestExtDescript:
    """``EXT_DESCRIPT`` names the model's own extension set, at every level.

    ``_rebuild_ext_descript`` runs off the live extension set, so the table is a
    restatement of the manifest the model was built from -- never the science
    set a master used to inherit.
    """

    @staticmethod
    def _models():
        return [
            ("KPF2", KPF2()),
            ("KPF4", KPF4()),
            ("KPFMasterL1", KPFMasterL1()),
            ("KPFMasterL2-wls", KPFMasterL2(kind="wls")),
            ("KPFMasterL2-flat", KPFMasterL2(kind="flat")),
        ]

    def test_the_table_names_exactly_the_models_extensions(self):
        for label, model in self._models():
            if "EXT_DESCRIPT" not in model.extensions:
                # ML1-extensions.csv declares no such row.
                assert label == "KPFMasterL1", label
                continue
            table = model.data["EXT_DESCRIPT"]
            assert table["Name"].tolist() == list(model.extensions), label

    def test_each_description_comes_from_the_manifest(self):
        for label, model in self._models():
            if "EXT_DESCRIPT" not in model.extensions:
                continue
            # Straight from the CSV, so the oracle is not the lookup under test.
            manifest = kpf_table(f"{model._data_model}-extensions")
            declared = dict(zip(manifest["Name"], manifest["Description"], strict=True))
            table = model.data["EXT_DESCRIPT"]
            for name, description in zip(
                table["Name"], table["Description"], strict=True
            ):
                assert description == declared[name], (label, name)

    def test_it_survives_a_round_trip(self, tmp_path):
        l2 = KPF2()
        before = l2.data["EXT_DESCRIPT"]["Name"].tolist()
        out = str(tmp_path / "kpf_SL2_20240101T000000.fits")
        l2.to_fits(out)
        back = KPF2.from_fits(out)
        assert [str(x) for x in back.data["EXT_DESCRIPT"]["Name"]] == before


class TestBareModelDefaults:
    """A standalone model carries real provenance values, not 'UNKNOWN'.

    rvdata's header map defaults these to the literal string; KPF overrides them
    in the registry, so they are already right before standardize_header_format runs.
    """

    @pytest.mark.parametrize("keyword", ("EPRVTAG", "VOCLASS", "DRPTAG"))
    @pytest.mark.parametrize("factory", (KPF2, KPF4), ids=("KPF2", "KPF4"))
    def test_provenance_defaults_are_not_the_unknown_sentinel(self, factory, keyword):
        assert factory().headers["PRIMARY"].get(keyword) != "UNKNOWN", keyword

    def test_numorder_is_the_kpf_value(self):
        # rvdata's header_map defaults it to 65; KPF reads 35 green + 32 red.
        assert KPF2().headers["PRIMARY"]["NUMORDER"] == 67
        assert DETECTOR["numorder"] == 67


_CFG = importlib.resources.files("kpfpipe.data_models.config")
_REPO_ROOT = Path(__file__).resolve().parents[2]

_KEYWORD_COLUMNS = [
    "Keyword",
    "Description",
    "Units",
    "DataType",
    "ExampleValue",
    "PopulatedBy",
]
_MANIFEST_COLUMNS = ["HDU", "Name", "DataType", "BitDepth", "Required", "Description"]

_DATA_MODELS = ("L0", "L1", "L2", "L4", "ML1", "ML2-flat", "ML2-wls")

# KPF-owned PRIMARY keywords with no EPRV header-map row: the per-CCD RVs the
# standard has no equivalent for.
_KPF_ONLY_PRIMARY = {"RVGREEN", "RVRED", "ERVGREEN", "ERVRED"}


def _keyword_files():
    return sorted(
        (p for p in _CFG.iterdir() if p.name.endswith("-keywords.csv")),
        key=lambda p: p.name,
    )


class TestPinnedRelease:
    """Every EPRV compliance check reads the installed rvdata, so the pin the
    exception lists were written against is asserted here, once."""

    def test_the_installed_release_matches_the_pin(self):
        pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
        pins = [
            d
            for d in pyproject["project"]["dependencies"]
            if d.startswith("rv-data-standard")
        ]
        assert pins == ["rv-data-standard==0.4.0"]
        assert importlib.metadata.version("rv-data-standard") == "0.4.0"


class TestConfigTables:
    """The config tables' own shape. The filename-driven layout only works if no
    file quietly grows or loses a column, or lands under an unknown prefix."""

    def test_keyword_files_have_the_keyword_schema(self):
        for path in _keyword_files():
            assert list(pd.read_csv(path).columns) == _KEYWORD_COLUMNS, path.name

    def test_every_data_model_has_a_manifest_with_the_manifest_schema(self):
        for data_model in _DATA_MODELS:
            columns = list(pd.read_csv(_CFG / f"{data_model}-extensions.csv").columns)
            assert columns == _MANIFEST_COLUMNS, data_model

    def test_keyword_filenames_use_a_known_data_model(self):
        for path in _keyword_files():
            data_model, _, extension = path.name[: -len("-keywords.csv")].rpartition(
                "-"
            )
            assert data_model in _DATA_MODELS, path.name
            assert extension, path.name

    def test_no_structural_card_is_registered(self):
        # rvdata declares XTENSION/EXTNAME as per-extension keywords; astropy
        # writes them from the HDU structure, so they must not be carried over.
        registry = KPF1.keyword_registry
        assert not [k for k in registry.registered if registry.is_structural(k)]

    def test_the_config_csvs_are_packaged(self):
        with open(_REPO_ROOT / "pyproject.toml", "rb") as fh:
            package_data = tomllib.load(fh)["tool"]["setuptools"]["package-data"]
        assert "*.csv" in package_data["kpfpipe.data_models.config"]


class TestHeaderMapIsTheEprvPrimarySet:
    """``EPRV-header-map.csv`` and the PRIMARY keyword CSVs are two authorities
    over one set, so they are asserted equal rather than left to drift."""

    @staticmethod
    def _map():
        return pd.read_csv(_CFG / "EPRV-header-map.csv")

    def test_keys_are_unique(self):
        keys = self._map()["EPRV_KEY"].astype(str).str.strip()
        assert not list(keys[keys.duplicated()])

    def test_keys_are_exactly_the_registered_eprv_primary_set(self):
        mapped = set(self._map()["EPRV_KEY"].astype(str).str.strip())
        # Scoped to the science chain: the masters PRIMARY keywords are registered
        # but outside EPRV scope, so unmapped by construction.
        registered = {
            member
            for level in ("L0", "L1", "L2", "L4")
            for keyword in pd.read_csv(_CFG / f"{level}-PRIMARY-keywords.csv")[
                "Keyword"
            ]
            for member in expand(keyword)
        }
        assert mapped <= registered
        assert registered - mapped == _KPF_ONLY_PRIMARY

    def test_kpf_ext_is_primary_or_quality_control(self):
        for value in self._map()["KPF_EXT"].dropna().astype(str).str.strip():
            assert value in ("", "PRIMARY", "QUALITY_CONTROL"), value


class TestRouting:
    """The refactor must not silently rehome a keyword."""

    def test_routing_prefers_primary(self):
        # PRIMARY-preference is what keeps RVMETHOD and the per-CCD RV keywords
        # routable while they also live on RV1..5.
        routing = KPF1.keyword_registry.routing
        for keyword in ("RVMETHOD", *_KPF_ONLY_PRIMARY):
            assert routing[keyword] == "PRIMARY"

    def test_the_per_ccd_rv_set_is_registered_on_every_rv_extension(self):
        registry = KPF1.keyword_registry
        for i in range(1, DETECTOR["numtrace"] + 1):
            assert _KPF_ONLY_PRIMARY <= registry.allowed[f"RV{i}"]

    def test_multi_home_keywords_stay_unrouted(self):
        routing = KPF1.keyword_registry.routing
        for keyword in ("CTYPE1", "CTYPE2", "VELSTART", "VELWIDTH", "SKYRMVD"):
            assert keyword not in routing
