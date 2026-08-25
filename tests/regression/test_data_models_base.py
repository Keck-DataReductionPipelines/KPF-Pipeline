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

from ._registry import read_kpf_header_registry


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
        l4 = KPF4()
        l4.set_keyword("GRNRVS1", 1.2345)  # GREEN SCI1 -> RV2
        l4.set_keyword("RVGREEN", 6.789)  # GREEN SCI-combined -> PRIMARY
        assert l4.headers["RV2"]["GRNRVS1"] == 1.2345
        assert l4.headers["PRIMARY"]["RVGREEN"] == 6.789

    def test_targeted_ext_writes_eprv_per_extension_card(self):
        # EPRV per-extension cards (VELSTART on every CCF#, RVMETHOD on every RV#)
        # have no single routed home, so ext= picks one, alias-resolved.
        l4 = KPF4()
        l4.set_keyword("VELSTART", -100.0, ext="SCI2_CCF")  # SCI2_CCF -> CCF3
        l4.set_keyword("RVMETHOD", "CCF", ext="SCI2_RV")  # SCI2_RV -> RV3
        assert l4.headers["CCF3"]["VELSTART"] == -100.0
        assert l4.headers["CCF3"].comments["VELSTART"] == "Velocity Grid Start"
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
    """The routing table the model exposes agrees with the registry CSV column.

    Oracle: ``read_kpf_header_registry`` reads config/L*-headers.csv directly,
    independent of the code under test.
    """

    def test_routing_matches_extension_column(self):
        routing = KPF1.keyword_registry.routing
        mismatches = []
        for _, row in read_kpf_header_registry().iterrows():
            want = str(row["Extension"]).strip()
            if want.endswith("*"):
                continue
            kw = str(row["Keyword"]).strip()
            got = routing.get(kw, (None,))[0]
            if got != want:
                mismatches.append((kw, want, got))
        assert not mismatches, f"routing != registry Extension column: {mismatches}"

    def test_single_home_routable_per_extension_not(self):
        # Per-extension keywords (Extension "CCF*") are ext=-only, so they are
        # excluded from routing; every single-home keyword is in it.
        routing = KPF1.keyword_registry.routing
        for _, row in read_kpf_header_registry().iterrows():
            kw = str(row["Keyword"]).strip()
            if str(row["Extension"]).strip().endswith("*"):
                assert kw not in routing, f"per-extension {kw} must not be routed"
            else:
                assert kw in routing, f"{kw} missing from routing table"

    def test_comment_is_registry_description(self):
        routing = KPF1.keyword_registry.routing
        for _, row in read_kpf_header_registry().iterrows():
            if str(row["Extension"]).strip().endswith("*"):
                continue
            kw = str(row["Keyword"]).strip()
            assert routing[kw][1] == str(row["Description"]).strip()

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
    """The unified registry table and its derived validation lookups."""

    def test_columns(self):
        assert list(KPF1.keyword_registry.table.columns) == [
            "Keyword",
            "Description",
            "Extension",
            "DataType",
            "PopulatedBy",
            "Required",
            "Level",
            "Default",
            "Units",
        ]

    def test_unions_kpf_and_eprv(self):
        keys = KPF1.keyword_registry.registered
        assert "RNGREEN1" in keys and "RV" in keys

    def test_non_registry_headermap_keys_absent(self):
        # PARANG/PARANG2 are header_map STANDARD names that aren't EPRV keywords;
        # they must NOT be in the registry (so _map_header drops them).
        assert "PARANG" not in KPF1.keyword_registry.registered
        assert "PARANG2" not in KPF1.keyword_registry.registered

    def test_primary_allowed_not_level_gated(self):
        # An EPRV-L4 keyword (RV) is allowed on PRIMARY regardless of product level.
        assert "RV" in KPF1.keyword_registry.allowed["PRIMARY"]

    def test_required_keyed_by_minimal_level(self):
        # required maps keyword -> the minimal Level it is Required at; the EPRV
        # L2 PRIMARY set is tagged Level 1 because KPF requires it from L1.
        primary_required = KPF1.keyword_registry.required["PRIMARY"]
        assert primary_required.get("RV") == 4  # L4-only required
        assert primary_required.get("INSTRUME") == 1  # required from L1

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
        # Build-time invariant: structural cards (astropy-authored) are never
        # registered. rvdata redundantly declares XTENSION/EXTNAME as per-extension
        # keywords; the _build_registry sanitizer drops them.
        reg = KPF1.keyword_registry
        assert not [k for k in reg.registered if reg.is_structural(k)]
        assert "XTENSION" not in reg.registered
        assert "EXTNAME" not in reg.registered

    def test_default_units_populated_for_eprv_blank_for_kpf(self):
        # Default and Units carry the EPRV CSV values for EPRV rows, "" for KPF.
        table = KPF1.keyword_registry.table.set_index("Keyword")
        assert table.loc["INSTRUME", "Default"] == "UNKNOWN"  # EPRV row
        assert table.loc["RNGREEN1", "Default"] == ""  # KPF row
        assert table.loc["RNGREEN1", "Units"] == ""

    def test_eprv_primary_seed_is_typed_required_set(self):
        # The seed is the EPRV Required PRIMARY set (Level <= 1), pre-typed as
        # (value, comment) tuples ready to drop into a header.
        reg = KPF1.keyword_registry
        required = {k for k, lvl in reg.required["PRIMARY"].items() if lvl <= 1}
        assert set(reg.eprv_primary_seed) == required
        value, comment = reg.eprv_primary_seed["ISSOLAR"]
        assert value is False and comment  # Boolean parsed, comment present

    def test_eprv_primary_datatypes_cover_emitted_keywords(self):
        # Datatypes use rvdata vocab so parse_value_to_datatype works, and cover
        # the keywords _map_header emits.
        dt = KPF1.keyword_registry.eprv_primary_datatypes
        assert dt["INSTRUME"] == "String" and dt["NUMTRACE"] == "UInt"
        assert "RNGREEN1" not in dt  # KPF keyword, not EPRV PRIMARY

    def test_header_map_sanitized_on_load(self):
        # header_map holds only static native->EPRV mappings: unregistered targets
        # (PARANG) and non-native keywords are dropped, so _map_header needs no
        # filter or per-keyword correction.
        std = set(KPF1.keyword_registry.header_map["STANDARD"].astype(str).str.strip())
        assert not ({"PARANG", "PARANG2"} & std)
        assert not ({"NUMORDER", "DATALVL", "DRPTAG", "JD_UTC"} & std)

    def test_default_overrides_sanitize_the_table(self):
        # NUMORDER/DRPTAG are corrected once, on the table Default, so both the
        # table and the seed derived from it carry the corrected values.
        import importlib.metadata

        reg = KPF1.keyword_registry
        version = importlib.metadata.version("kpfpipe")
        table = reg.table.set_index("Keyword")
        # Table Defaults are strings (like every other Default); the seed types them.
        assert table.loc["NUMORDER", "Default"] == "67"
        assert table.loc["DRPTAG", "Default"] == version
        assert reg.eprv_primary_seed["NUMORDER"][0] == 67
        assert reg.eprv_primary_seed["DRPTAG"][0] == version
        # DATALVL is not overridden -- KPF1.__init__ sets it from the data level.
        assert table.loc["DATALVL", "Default"] == "UNKNOWN"
        assert reg.eprv_primary_seed["DATALVL"][0] == "UNKNOWN"

    def test_kpf_row_may_not_claim_the_eprv_sentinel(self):
        # 'EPRV' is the discriminator every derived lookup keys on, so a KPF CSV
        # row claiming it would silently misclassify itself as EPRV-sourced --
        # a KPF keyword routed to PRIMARY, or a corrupted L1 seed.
        import pandas as pd

        from kpfpipe.data_models.keyword_registry import KeywordRegistry

        df = pd.DataFrame(
            [{"Keyword": "FOO", "Description": "d", "PopulatedBy": "EPRV"}]
        )
        with pytest.raises(ValueError, match="reserved as the EPRV-row discriminator"):
            KeywordRegistry._parse_kpf_keyword_config(df, "fake.csv", lambda r: "L0")


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
