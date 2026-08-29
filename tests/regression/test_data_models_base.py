"""Tests for shared KPFDataModel behaviour (data_models/base.py).

KPF1 is the vehicle for the contract every model inherits: extension headers are
``fits.Header`` from construction onward, and a commented PRIMARY card survives
``to_fits`` -> ``from_fits`` (rvdata >=0.4.0 copies the ``fits.Header`` directly).
KPF2/KPF4 round-trip guards live in test_data_models_l{2,4}.py.
"""

import pytest
from astropy.io import fits

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.modules.standardize_data_format import StandardizeDataFormat

from ._registry import (
    expected_comment,
    expected_routing,
    read_kpf_header_registry,
)


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
        routing = KPF1.keyword_registry.routing
        table = read_kpf_header_registry()
        homes = table.groupby("Keyword")["Extension"].apply(set)
        for keyword, extensions in homes.items():
            if len(extensions) > 1 and "PRIMARY" not in extensions:
                assert keyword not in routing, f"multi-home {keyword} must not route"
            else:
                assert keyword in routing, f"{keyword} missing from routing table"

    def test_comment_is_description_and_units(self):
        reg = KPF1.keyword_registry
        for _, row in read_kpf_header_registry().iterrows():
            got = reg.comment_for(row["Keyword"], row["Extension"])
            assert got == expected_comment(row["Description"], row["Units"])

    def test_datatype_matches_the_csv(self):
        reg = KPF1.keyword_registry
        for _, row in read_kpf_header_registry().iterrows():
            got = reg.datatype_for(row["Keyword"], row["Extension"])
            assert got == row["DataType"]

    def test_qc_flag_keyword_sets_are_scoped_by_level(self):
        # Relocated from test_checkpoints.py: this is registry scoping, not
        # Checkpoint behaviour, and it reaches the registry through the model
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

    def test_unions_every_profile(self):
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
        # L0 and L1 share the 154-card science skeleton; L2 adds the extraction
        # SNR cards and L4 the RV summary.
        reg = KPF1.keyword_registry
        l0, l1, l2, l4 = (reg.primary_seed(p) for p in ("L0", "L1", "L2", "L4"))
        assert set(l0) == set(l1)
        assert set(l0) < set(l2) < set(l4)
        assert set(l2) - set(l0) == {"EXTRACT"} | {f"EXSNR{i}" for i in range(1, 6)} | {
            f"EXSNRW{i}" for i in range(1, 6)
        }
        assert set(l4) - set(l2) == {
            "BJDTDB",
            "RV",
            "RVERR",
            "BERV",
            "RVMETHOD",
            "SYSVEL",
            "SYSACC",
        }

    def test_primary_seed_is_typed_and_commented(self):
        seed = KPF1.keyword_registry.primary_seed("L0")
        assert seed["NUMTRACE"] == (5, "Number of object-indexed keyword families")
        assert seed["OBSALT"][0] == 4145.0  # injected from kpfpipe.OBSERVATORY
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

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="unknown keyword profile"):
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
        l0.headers["PRIMARY"]["MJD-OBS"] = 60310.0
        StandardizeDataFormat(l0).perform()
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
