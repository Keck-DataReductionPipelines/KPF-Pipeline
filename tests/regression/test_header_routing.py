"""
Tests for registry-driven header routing and the QC header validator.

Covers the keyword-routing layer added when header keywords were distributed
across extension headers (QUALITY_CONTROL, RECEIPT, the barycentric extensions,
the L4 RV tables) per the ``Extension`` column of
``config/L{0,1,2,4}-headers.csv``:

  - ``KPFDataModel.set_keyword`` routing, comments, and fail-loud behaviour
  - registry conformance (routing table matches the ``Extension`` column)
  - QUALITY_CONTROL + RECEIPT round-trip and L0->L1->L2 propagation
  - ``QC._validate_headers`` (raise on unexpected, warn on missing required)

All synthetic in-memory data — no real KPF files required.
"""

import warnings

import pytest

from kpfpipe.data_models.base import (
    _KEYWORD_ROUTING,
    _KPFPIPE_KW,
)
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.quality_control.qc_booleans.level2 import QCL2

# ---------------------------------------------------------------------------
# set_keyword routing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Registry conformance: routing table matches the Extension column
# ---------------------------------------------------------------------------


class TestRegistryConformance:
    """The live routing table agrees with the registry CSV Extension column."""

    def test_routing_matches_extension_column(self):
        mismatches = []
        for _, row in _KPFPIPE_KW.iterrows():
            kw = str(row["Keyword"]).strip()
            want = str(row["Extension"]).strip()
            got = _KEYWORD_ROUTING.get(kw, (None,))[0]
            if got != want:
                mismatches.append((kw, want, got))
        assert not mismatches, f"routing != registry Extension column: {mismatches}"

    def test_every_registry_keyword_is_routable(self):
        for _, row in _KPFPIPE_KW.iterrows():
            kw = str(row["Keyword"]).strip()
            assert kw in _KEYWORD_ROUTING, f"{kw} missing from routing table"

    def test_comment_is_registry_description(self):
        for _, row in _KPFPIPE_KW.iterrows():
            kw = str(row["Keyword"]).strip()
            assert _KEYWORD_ROUTING[kw][1] == str(row["Description"]).strip()


# ---------------------------------------------------------------------------
# Round-trip and propagation of QUALITY_CONTROL + RECEIPT headers
# ---------------------------------------------------------------------------


class TestRoundTripAndPropagation:
    """QUALITY_CONTROL + RECEIPT header cards survive to_fits and L0->L1->L2."""

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

    def test_l2_quality_control_roundtrip(self, tmp_path):
        l2 = KPF2()
        l2.set_keyword("NANSCI1", 7)
        l2.set_keyword("CCD1BKMS", -3.21)
        fn = str(tmp_path / "kpf_SL2_20240101T000000.fits")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            l2.to_fits(fn)
            back = KPF2.from_fits(fn)
        assert back.headers["QUALITY_CONTROL"]["NANSCI1"] == 7
        assert back.headers["BARYCORR_KMS"]["CCD1BKMS"] == -3.21

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


# ---------------------------------------------------------------------------
# QC header validator
# ---------------------------------------------------------------------------


class TestValidateHeaders:
    """QC._validate_headers raises on an unexpected card, warns on missing required."""

    def test_clean_product_passes(self):
        # A fresh KPF2 has RV2-seeded EPRV PRIMARY defaults (all required present)
        # and empty governed extensions -> no unexpected card, no missing required.
        l2 = KPF2()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail this
            QCL2(l2)._validate_headers()

    def test_unexpected_keyword_on_governed_extension_raises(self):
        l2 = KPF2()
        l2.headers["QUALITY_CONTROL"]["BOGUSKEY"] = (1, "not registered")
        with pytest.raises(ValueError, match="unregistered keyword 'BOGUSKEY'"):
            QCL2(l2)._validate_headers()

    def test_native_wmko_leak_on_primary_raises(self):
        l2 = KPF2()
        # GAIAID is a raw WMKO native (kept in INSTRUMENT_HEADER, never on PRIMARY).
        l2.headers["PRIMARY"]["GAIAID"] = (12345, "leaked native")
        with pytest.raises(ValueError, match="native WMKO keyword 'GAIAID'"):
            QCL2(l2)._validate_headers()

    def test_missing_required_primary_keyword_warns(self):
        l2 = KPF2()
        del l2.headers["PRIMARY"]["INSTRUME"]  # a Required EPRV PRIMARY keyword
        with pytest.warns(UserWarning, match="missing required keyword"):
            QCL2(l2)._validate_headers()

    def test_registered_keyword_on_its_extension_passes(self):
        l2 = KPF2()
        l2.set_keyword("NANSCI1", 3)  # registered -> QUALITY_CONTROL
        # should not raise (warnings about other missing required keys are fine).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            QCL2(l2)._validate_headers()
