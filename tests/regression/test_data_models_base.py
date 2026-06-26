"""
Tests for shared KPFDataModel behaviour (data_models/base.py).

KPF stores every extension header as an ``astropy.io.fits.Header``, so reads use
``header.get(key)`` / ``header[key]`` and writes use ``header[key] = (value,
comment)`` natively. These tests pin the contract every model inherits from the
shared base: (1) headers are ``fits.Header`` from construction onward, and (2) a
commented PRIMARY card survives ``to_fits`` -> ``from_fits`` with its comment
intact -- the regression guard for the lossless-PRIMARY serialization in
``KPFDataModel._create_hdul`` / ``_restore_primary_comments``.

KPF1 is the representative vehicle for the inherited base path (L0/L1 use
``KPFDataModel._create_hdul`` directly). KPF2/KPF4 override ``_create_hdul`` via
RV2/RV4, so their round-trip guards live in test_data_models_l{2,4}.py.

The WMKO->EPRV conversion (``KPF0.wmko_to_eprv`` / ``build_instrument_header``) is
exercised end-to-end by the to_kpf1/to_kpf2 tests in test_data_models_l{1,2,4}.py.
PRIMARY-header validation no longer lives on the data models (it moved to the QC
runner, ``quality_control/qc_booleans/base.py``).
"""

import warnings

import pytest
from astropy.io import fits

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.level4 import KPF4

from ._registry import read_kpf_header_registry


class TestHeaderStorage:
    """Every extension header is a fits.Header, with native read/write semantics."""

    def test_fresh_headers_are_fits_headers(self):
        l1 = KPF1()
        assert isinstance(l1.headers["PRIMARY"], fits.Header)

    def test_tuple_write_sets_value_and_comment(self):
        # The documented write path: header[key] = (value, comment).
        l1 = KPF1()
        l1.headers["PRIMARY"]["BIASAGE"] = (-0.5, "[day] bias age")
        assert l1.headers["PRIMARY"]["BIASAGE"] == -0.5
        assert l1.headers["PRIMARY"].comments["BIASAGE"] == "[day] bias age"

    def test_get_returns_scalar_and_honors_default(self):
        l1 = KPF1()
        l1.headers["PRIMARY"]["DATE-OBS"] = ("2024-01-13T10:26:56", "obs start")
        assert l1.headers["PRIMARY"].get("DATE-OBS") == "2024-01-13T10:26:56"
        assert l1.headers["PRIMARY"].get("NOPE", "fallback") == "fallback"
        assert l1.headers["PRIMARY"].get("NOPE") is None


class TestPrimaryCommentRoundTrip:
    """A commented PRIMARY write survives to_fits -> from_fits with its comment.

    rvdata's base _create_hdul serializes PRIMARY by iterating ``.items()``, which
    drops a fits.Header's comments; KPFDataModel._create_hdul rebuilds the PRIMARY
    HDU (via _restore_primary_comments) so the comments survive. This test would
    fail without that override.
    """

    def test_primary_comment_round_trips(self, tmp_path):
        l1 = KPF1()
        l1.headers["PRIMARY"]["DATE-OBS"] = "2024-01-13T10:26:56"
        l1.headers["PRIMARY"]["BIASAGE"] = (-0.5, "[day] bias age")

        fn = str(tmp_path / "header_roundtrip_l1.fits")
        l1.to_fits(fn)

        prim = KPF1.from_fits(fn).headers["PRIMARY"]
        assert prim.get("BIASAGE") == -0.5
        assert prim.comments["BIASAGE"] == "[day] bias age"


class TestUndefinedRoundTrip:
    """A None value (non-finite RV) becomes a FITS UNDEFINED card, not a crash."""

    def test_none_value_round_trips_as_undefined(self, tmp_path):
        l1 = KPF1()
        l1.headers["PRIMARY"]["DATE-OBS"] = "2024-01-13T10:26:56"
        # radial_velocity writes None for a non-finite fit; astropy stores it as
        # an UNDEFINED card. The value must round-trip as a blank/undefined card
        # (read back as None), never as a finite number.
        l1.headers["PRIMARY"]["CCFRV"] = None

        fn = str(tmp_path / "header_undefined_l1.fits")
        l1.to_fits(fn)

        prim = KPF1.from_fits(fn).headers["PRIMARY"]
        assert "CCFRV" in prim
        val = prim["CCFRV"]
        assert val is None or isinstance(val, fits.card.Undefined)


class TestSetKeyword:
    """KPFDataModel.set_keyword routes to the registry extension with its comment."""

    def test_routes_to_quality_control(self):
        l1 = KPF1()
        l1.set_keyword("RNGREEN1", 3.5)
        assert l1.headers["QUALITY_CONTROL"]["RNGREEN1"] == 3.5
        # comment comes from the registry Description, not the caller.
        assert l1.headers["QUALITY_CONTROL"].comments["RNGREEN1"] == (
            "Read noise GREEN amp 1 [e-]"
        )
        # it does NOT leak onto PRIMARY.
        assert "RNGREEN1" not in l1.headers["PRIMARY"]

    def test_routes_to_receipt(self):
        l1 = KPF1()
        l1.set_keyword("BIASFILE", "/path/to/master_bias_L1.fits")
        assert l1.headers["RECEIPT"]["BIASFILE"] == "/path/to/master_bias_L1.fits"
        assert "BIASFILE" not in l1.headers["PRIMARY"]

    def test_routes_to_barycorr_and_bjd_extensions(self):
        l2 = KPF2()
        l2.set_keyword("CCD1BKMS", -12.3)
        l2.set_keyword("CCD1BJD", 2460000.5)
        assert l2.headers["BARYCORR_KMS"]["CCD1BKMS"] == -12.3
        assert l2.headers["BJD_TDB"]["CCD1BJD"] == 2460000.5

    def test_routes_l4_orderlet_rv_to_rv_table(self):
        l4 = KPF4()
        l4.set_keyword("CCD1RV1", 1.2345)  # GREEN SCI1 -> RV2
        l4.set_keyword("CCD1RV", 6.789)  # combined -> PRIMARY
        assert l4.headers["RV2"]["CCD1RV1"] == 1.2345
        assert l4.headers["PRIMARY"]["CCD1RV"] == 6.789

    def test_unregistered_keyword_raises_keyerror(self):
        l1 = KPF1()
        with pytest.raises(KeyError, match="not registered"):
            l1.set_keyword("BOGUSKEY", 1)

    def test_missing_extension_raises_valueerror(self):
        # KPF1 has no BARYCORR_KMS extension, so routing CCD1BKMS there fails loud.
        l1 = KPF1()
        with pytest.raises(ValueError, match="does not exist"):
            l1.set_keyword("CCD1BKMS", 1.0)


class TestRegistryConformance:
    """The routing table the model exposes agrees with the registry CSV column.

    Oracle: ``read_kpf_header_registry`` reads config/L*-headers.csv directly
    (tests/regression/_registry.py), independent of the code under test.
    """

    def test_routing_matches_extension_column(self):
        routing = KPF1.KEYWORD_ROUTING
        mismatches = []
        for _, row in read_kpf_header_registry().iterrows():
            kw = str(row["Keyword"]).strip()
            want = str(row["Extension"]).strip()
            got = routing.get(kw, (None,))[0]
            if got != want:
                mismatches.append((kw, want, got))
        assert not mismatches, f"routing != registry Extension column: {mismatches}"

    def test_every_registry_keyword_is_routable(self):
        routing = KPF1.KEYWORD_ROUTING
        for _, row in read_kpf_header_registry().iterrows():
            kw = str(row["Keyword"]).strip()
            assert kw in routing, f"{kw} missing from routing table"

    def test_comment_is_registry_description(self):
        routing = KPF1.KEYWORD_ROUTING
        for _, row in read_kpf_header_registry().iterrows():
            kw = str(row["Keyword"]).strip()
            assert routing[kw][1] == str(row["Description"]).strip()


class TestKeywordRegistry:
    """The unified KEYWORD_REGISTRY table and its derived validation lookups."""

    def test_columns(self):
        assert list(KPF1.KEYWORD_REGISTRY.columns) == [
            "Keyword",
            "Description",
            "Extension",
            "DataType",
            "Populated by",
            "Required",
            "Level",
        ]

    def test_unions_kpf_and_eprv(self):
        # KPF-registered (RNGREEN1) and EPRV (RV) keywords both present.
        keys = KPF1.REGISTERED_KEYWORDS
        assert "RNGREEN1" in keys and "RV" in keys

    def test_non_registry_headermap_targets_absent(self):
        # PARANG/PARANG2 are header_map STANDARD names that aren't EPRV keywords;
        # they must NOT be in the registry (so wmko_to_eprv drops them).
        assert "PARANG" not in KPF1.REGISTERED_KEYWORDS
        assert "PARANG2" not in KPF1.REGISTERED_KEYWORDS

    def test_primary_allowed_not_level_gated(self):
        # An EPRV-L4 keyword (RV) is allowed on PRIMARY regardless of product level.
        assert "RV" in KPF1.EXT_ALLOWED["PRIMARY"]

    def test_required_keyed_by_minimal_level(self):
        # EXT_REQUIRED maps keyword -> the minimal Level it is Required at.
        primary_required = KPF1.EXT_REQUIRED["PRIMARY"]
        assert primary_required.get("RV") == 4  # L4-only required
        assert primary_required.get("INSTRUME") == 2  # L2 required


class TestQualityControlPropagation:
    """QUALITY_CONTROL + RECEIPT header cards survive to_fits and L0->L1->L2.

    (The KPF2 QUALITY_CONTROL round-trip lives in test_data_models_l2.py, since
    KPF2 serializes through the RV2 read path.)
    """

    def test_l1_quality_control_receipt_roundtrip(self, tmp_path):
        l1 = KPF1()
        l1.set_keyword("RNGREEN1", 4.2)
        l1.set_keyword("BIASAGE", 1.5)
        l1.set_keyword("OSCANSUB", 1)
        l1.set_keyword("BIASFILE", "/m/bias_L1.fits")
        fn = str(tmp_path / "kpf_L1_20240101T000000.fits")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        # seed an L0 QC flag (QCL0 routes these to QUALITY_CONTROL).
        l0.set_keyword("NOTJUNK", 1)
        l1 = l0.to_kpf1()
        assert l1.headers["QUALITY_CONTROL"]["NOTJUNK"] == 1
        # add an L1 RECEIPT card; both QUALITY_CONTROL and RECEIPT must reach L2.
        l1.set_keyword("OSCANSUB", 1)
        l1.set_keyword("RNGREEN1", 4.0)
        l2 = l1.to_kpf2()
        assert l2.headers["QUALITY_CONTROL"]["NOTJUNK"] == 1
        assert l2.headers["QUALITY_CONTROL"]["RNGREEN1"] == 4.0
        assert l2.headers["RECEIPT"]["OSCANSUB"] == 1
