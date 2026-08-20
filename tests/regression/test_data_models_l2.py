"""Tests for the KPF2 (extracted spectra / L2) data model, the L1->L2 transform,
the AliasedOrderedDict extension-alias machinery, and the KPFMasterL2 calibration
product. Synthetic FITS fixtures only -- no real KPF data needed.
"""

import logging
from collections import OrderedDict

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from rvdata.core.models.level2 import RV2

from kpfpipe import DETECTOR
from kpfpipe.data_models.aliased_dict import AliasedOrderedDict
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.masters import KPFMasterL2
from kpfpipe.data_models.masters.base import KPFMasterModel

from ._catalog import SOURCES, catalog_record_table
from ._dtype_policy import FLUX, WAVE, assert_dtype, assert_roundtrip_dtype

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
NORDER = NORDER_GREEN + NORDER_RED

# synthetic_l1_file fixture lives in tests/conftest.py

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_masters_l2_file(tmp_path_factory):
    """Minimal synthetic Masters L2 FITS file; module-scoped and read-only."""
    rng = np.random.default_rng(20240113)
    fn = str(
        tmp_path_factory.mktemp("ml2") / "KP.20240113.23249.10_master_thar_L2.fits"
    )

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-01-13T10:26:56"
    primary.header["DATALVL"] = "ML2"
    primary.header["MASTYPE"] = "thar"  # from_fits infers kind="wls" from this

    n_pix = 64
    # WAVE is born-64 (EPRV / dtype policy); float32 would trip rvdata's
    # MinBitDepth upcast-and-warn on read.
    wave = rng.random((NORDER_GREEN + NORDER_RED, n_pix)).astype(np.float64)
    trace3_wave = fits.ImageHDU(data=wave)
    trace3_wave.name = "TRACE3_WAVE"

    hdul = fits.HDUList([primary, trace3_wave])
    hdul.writeto(fn, overwrite=True)
    hdul.close()

    return fn


@pytest.fixture
def converted_l1(synthetic_l0_file):
    """An L1 from KPF0.to_kpf1: EPRV-standard PRIMARY plus a populated
    INSTRUMENT_HEADER, the input to_kpf2 expects in production."""
    return KPF0.from_fits(synthetic_l0_file).to_kpf1()


class TestKPF2QualityControlRoundTrip:
    """The KPF-custom QUALITY_CONTROL extension survives KPF2's RV2 read path.

    register_rvdata_extension teaches rvdata's definition-driven L2 reader about
    QUALITY_CONTROL; without it, from_fits raises KeyError on that HDU.
    """

    def test_quality_control_and_barycorr_roundtrip(self, tmp_path):
        l2 = KPF2()
        l2.set_keyword("NANSCI1", 7)
        l2.set_keyword("CCD1BKMS", -3.21)
        fn = str(tmp_path / "kpf_SL2_20240101T000000.fits")
        l2.to_fits(fn)
        back = KPF2.from_fits(fn)
        assert back.headers["QUALITY_CONTROL"]["NANSCI1"] == 7
        assert back.headers["BARYCORR_KMS"]["CCD1BKMS"] == -3.21

    def test_from_fits_recovers_obs_id_from_origid(self, tmp_path):
        # An L2's timestamp-based filename does not embed the obs_id, so from_fits
        # recovers it from RECEIPT's ORIGID; that keeps generate_standard_filename
        # working on the from_fits construction path.
        l2 = KPF2()
        l2.set_keyword("ORIGID", "KP.20240101.00000.00")  # 0 s of day = 00:00:00
        fn = str(tmp_path / "kpf_SL2_20240101T000000.fits")
        l2.to_fits(fn)
        back = KPF2.from_fits(fn)
        assert back.obs_id == "KP.20240101.00000.00"
        assert back.generate_standard_filename() == "kpf_SL2_20240101T000000.fits"


class TestCatalogRecordPassthrough:
    """CATALOG_RECORD rides L1 -> L2 and survives KPF2's RV2 read path.

    Same register_rvdata_extension mechanism as QUALITY_CONTROL above.
    """

    @staticmethod
    def _l1_with_catalog(rv=-16.6):
        # KPF1 creates CATALOG_RECORD only on the L0 pass-through or a read (it is
        # Required=False in L1-extensions.csv), so a bare L1 makes it explicitly.
        l1 = KPF1()
        l1.create_extension("CATALOG_RECORD", "BinTableHDU")
        l1.set_data("CATALOG_RECORD", catalog_record_table(rv=rv))
        l1.headers["CATALOG_RECORD"]["GAIACR"] = (1, "Catalog record present")
        return l1

    def test_rows_and_flags_reach_l2(self):
        l2 = self._l1_with_catalog().to_kpf2()
        assert [str(s) for s in l2.data["CATALOG_RECORD"]["source"]] == list(SOURCES)
        assert l2.headers["CATALOG_RECORD"]["GAIACR"] == 1

    def test_catalog_record_roundtrip(self, tmp_path):
        # The missing rv reads back NaN, not masked -- L2 reads through rvdata's
        # RV2._read, so only the from_fits chokepoint can normalize it.
        l2 = self._l1_with_catalog(rv=None).to_kpf2()
        fn = str(tmp_path / "kpf_SL2_20240101T000000.fits")
        l2.to_fits(fn)
        back = KPF2.from_fits(fn)
        assert [str(s) for s in back.data["CATALOG_RECORD"]["source"]] == list(SOURCES)
        assert back.headers["CATALOG_RECORD"]["GAIACR"] == 1
        rv = back.data["CATALOG_RECORD"][0]["rv"]
        assert rv is not np.ma.masked and np.isnan(rv)

    def test_empty_catalog_record_roundtrips(self, tmp_path):
        # An L2 that never saw AstroQuery carries an empty table.
        fn = str(tmp_path / "kpf_SL2_20240101T000001.fits")
        KPF2().to_fits(fn)
        back = KPF2.from_fits(fn)
        assert "CATALOG_RECORD" in back.extensions
        assert len(back.data["CATALOG_RECORD"]) == 0


class TestToKPF2:
    def test_to_kpf2_creates_kpf2(self, converted_l1):
        kpf2 = converted_l1.to_kpf2()
        assert kpf2.level == 2
        assert isinstance(kpf2, KPF2)

    def test_to_kpf2_passes_through_eprv_primary(self, converted_l1):
        # The keyword conversion happened in to_kpf1; to_kpf2 only forwards it.
        kpf2 = converted_l1.to_kpf2()
        prim = kpf2.headers["PRIMARY"]
        assert prim.get("EXPTIME") == 300.0  # from ELAPSED, set in to_kpf1
        assert prim.get("OBSTYPE") == "Object"  # from IMTYPE
        assert prim.get("OBSERVER") == "Smith"  # from GROBSERV
        # Raw natives never reach the EPRV PRIMARY.
        assert "ELAPSED" not in prim
        assert "IMTYPE" not in prim

    def test_to_kpf2_copies_same_name_keywords(self, converted_l1):
        kpf2 = converted_l1.to_kpf2()
        assert kpf2.headers["PRIMARY"].get("INSTRUME") == "KPF"
        assert kpf2.headers["PRIMARY"].get("DATE-OBS") == "2024-01-13T10:26:56"

    def test_to_kpf2_sets_defaults(self, converted_l1):
        kpf2 = converted_l1.to_kpf2()
        assert kpf2.headers["PRIMARY"].get("DATALVL") == "L2"
        origin = kpf2.headers["PRIMARY"].get("ORIGIN")
        assert origin is not None

    def test_to_kpf2_carries_instrument_header(self, converted_l1):
        # INSTRUMENT_HEADER holds the raw natives, forwarded unchanged from L1.
        kpf2 = converted_l1.to_kpf2()
        assert kpf2.headers["INSTRUMENT_HEADER"]["INSTRUME"] == "KPF"
        assert kpf2.headers["INSTRUMENT_HEADER"]["ELAPSED"] == 300.0

    def test_to_kpf2_maps_passthrough_extensions(self, tmp_path):
        fn = str(tmp_path / "l1_with_extras.fits")
        primary = fits.PrimaryHDU()
        primary.header["INSTRUME"] = "KPF"
        primary.header["DATE-OBS"] = "2024-01-13T10:26:56"
        green = fits.ImageHDU(data=np.zeros((8, 8), dtype=np.float32))
        green.name = "GREEN_CCD"
        green_var = fits.ImageHDU(data=np.zeros((8, 8), dtype=np.float32))
        green_var.name = "GREEN_VAR"
        red = fits.ImageHDU(data=np.zeros((8, 8), dtype=np.float32))
        red.name = "RED_CCD"
        red_var = fits.ImageHDU(data=np.zeros((8, 8), dtype=np.float32))
        red_var.name = "RED_VAR"
        ca_hk = fits.ImageHDU(data=np.ones((4, 4), dtype=np.float32))
        ca_hk.name = "CA_HK"
        telemetry = Table({"keyword": ["T1"], "average": [20.0]})
        tel_hdu = fits.BinTableHDU(data=telemetry)
        tel_hdu.name = "TELEMETRY"
        hdul = fits.HDUList([primary, green, green_var, red, red_var, ca_hk, tel_hdu])
        hdul.writeto(fn, overwrite=True)
        hdul.close()

        l1 = KPF1.from_fits(fn)
        kpf2 = l1.to_kpf2()
        assert "TELEMETRY" in kpf2.extensions
        assert "ANCILLARY_SPECTRUM" in kpf2.extensions

    def test_to_kpf2_leaves_traces_empty(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        kpf2 = l1.to_kpf2()
        assert "TRACE1_FLUX" in kpf2.extensions
        assert len(kpf2.data["TRACE1_FLUX"]) == 0

    def test_to_kpf2_carries_receipt(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        kpf2 = l1.to_kpf2()
        assert "to_kpf2" in kpf2.receipt["FUNCTION"].values

    def test_to_kpf2_receipt_updates_drpstatus(self, synthetic_l1_file):
        # KPF2 subclasses RV2, not KPFDataModel, so it carries its own copy of the
        # DRPSTATU receipt override.
        kpf2 = KPF1.from_fits(synthetic_l1_file).to_kpf2()
        kpf2.receipt_add_entry("barycentric_correction", "", "PASS")
        assert (
            kpf2.headers["RECEIPT"].get("DRPSTATU")
            == "Barycentric Correction module complete"
        )

    def test_to_kpf2_propagates_origid(self, tmp_path):
        # ORIGID is stamped at L0 and rides RECEIPT through to_kpf2; it is not
        # rewritten at L2, and stays off PRIMARY.
        fn = str(tmp_path / "KP.20240113.23249.10_L1.fits")
        primary = fits.PrimaryHDU()
        primary.header["INSTRUME"] = "KPF"
        primary.header["DATE-OBS"] = "2024-01-13T10:26:56"
        green = fits.ImageHDU(data=np.zeros((8, 8), dtype=np.float32))
        green.name = "GREEN_CCD"
        green_var = fits.ImageHDU(data=np.zeros((8, 8), dtype=np.float32))
        green_var.name = "GREEN_VAR"
        red = fits.ImageHDU(data=np.zeros((8, 8), dtype=np.float32))
        red.name = "RED_CCD"
        red_var = fits.ImageHDU(data=np.zeros((8, 8), dtype=np.float32))
        red_var.name = "RED_VAR"
        hdul = fits.HDUList([primary, green, green_var, red, red_var])
        hdul.writeto(fn, overwrite=True)
        hdul.close()

        l1 = KPF1.from_fits(fn)
        # Mimic what KPF0.from_fits + to_kpf1 would have placed on the L1 RECEIPT.
        l1.set_keyword("ORIGID", "KP.20240113.23249.10")
        kpf2 = l1.to_kpf2()
        assert kpf2.headers["RECEIPT"].get("ORIGID") == "KP.20240113.23249.10"
        assert "ORIGID" not in kpf2.headers["PRIMARY"]


class TestAliasedOrderedDict:
    def test_basic_alias(self):
        d = AliasedOrderedDict()
        d["CANONICAL"] = 42
        d.register_alias("ALIAS", "CANONICAL")
        assert d["ALIAS"] == 42
        assert d["CANONICAL"] == 42

    def test_contains_with_alias(self):
        d = AliasedOrderedDict()
        d["CANONICAL"] = "data"
        d.register_alias("ALIAS", "CANONICAL")
        assert "ALIAS" in d
        assert "CANONICAL" in d
        assert "MISSING" not in d

    def test_set_via_alias(self):
        d = AliasedOrderedDict()
        d["CANONICAL"] = "old"
        d.register_alias("ALIAS", "CANONICAL")
        d["ALIAS"] = "new"
        assert d["CANONICAL"] == "new"

    def test_get_with_default(self):
        d = AliasedOrderedDict()
        d["CANONICAL"] = 99
        d.register_alias("ALIAS", "CANONICAL")
        assert d.get("ALIAS") == 99
        assert d.get("MISSING", "default") == "default"

    def test_aliases_for(self):
        d = AliasedOrderedDict()
        d["CANONICAL"] = 1
        d.register_alias("A1", "CANONICAL")
        d.register_alias("A2", "CANONICAL")
        aliases = d.aliases_for("CANONICAL")
        assert aliases == {"A1", "A2"}

    def test_from_ordered_dict(self):
        od = OrderedDict([("A", 1), ("B", 2)])
        aliased = AliasedOrderedDict.from_ordered_dict(od)
        assert aliased["A"] == 1
        assert aliased["B"] == 2
        aliased.register_alias("C", "A")
        assert aliased["C"] == 1

    def test_identity_via_alias(self):
        d = AliasedOrderedDict()
        arr = np.zeros((4, 4))
        d["CANONICAL"] = arr
        d.register_alias("ALIAS", "CANONICAL")
        assert d["ALIAS"] is d["CANONICAL"]


class TestKPF2Aliases:
    def test_kpf2_inherits_rv2(self):
        kpf2 = KPF2()
        assert isinstance(kpf2, RV2)
        assert kpf2.level == 2

    def test_extension_alias_resolves(self):
        kpf2 = KPF2()
        assert "CA_HK" in kpf2.extensions
        assert kpf2.data["CA_HK"] is kpf2.data["ANCILLARY_SPECTRUM"]

    def test_expmeter_alias(self):
        kpf2 = KPF2()
        assert "EXPMETER_SCI" in kpf2.extensions
        assert kpf2.data["EXPMETER_SCI"] is kpf2.data["EXPMETER"]

    def test_set_data_via_alias(self):
        kpf2 = KPF2()
        test_data = np.random.default_rng(42).random((10, 100))
        kpf2.set_data("SCI2_FLUX", test_data)
        np.testing.assert_array_equal(kpf2.data["TRACE3_FLUX"], test_data)

    def test_all_trace_aliases_registered(self):
        kpf2 = KPF2()
        for fiber, trace in [
            ("SKY", 1),
            ("SCI1", 2),
            ("SCI2", 3),
            ("SCI3", 4),
            ("CAL", 5),
        ]:
            for suffix in ["FLUX", "WAVE", "VAR", "BLAZE"]:
                alias = f"{fiber}_{suffix}"
                canonical = f"TRACE{trace}_{suffix}"
                assert alias in kpf2.extensions, f"{alias} not found"
                assert kpf2.data[alias] is kpf2.data[canonical]

    def test_chip_prefix_access(self):
        # A GREEN_/RED_ prefix slices the concatenated (GREEN then RED) trace.
        kpf2 = KPF2()
        n_pix = 100
        rng = np.random.default_rng(42)
        trace_data = rng.random((NORDER_GREEN + NORDER_RED, n_pix))
        kpf2.set_data("TRACE3_FLUX", trace_data)

        green = kpf2.data["GREEN_SCI2_FLUX"]
        red = kpf2.data["RED_SCI2_FLUX"]
        assert green.shape == (NORDER_GREEN, n_pix)
        assert red.shape == (NORDER_RED, n_pix)
        np.testing.assert_array_equal(green, trace_data[:NORDER_GREEN])
        np.testing.assert_array_equal(red, trace_data[NORDER_GREEN:])

    def test_chip_prefix_contains(self):
        kpf2 = KPF2()
        assert "GREEN_SCI2_FLUX" in kpf2.data
        assert "RED_CAL_WAVE" in kpf2.data
        assert "GREEN_NONEXISTENT" not in kpf2.data

    def test_chip_prefix_all_fibers(self):
        kpf2 = KPF2()
        for fiber in ["CAL", "SCI1", "SCI2", "SCI3", "SKY"]:
            for suffix in ["FLUX", "WAVE", "VAR", "BLAZE"]:
                assert f"GREEN_{fiber}_{suffix}" in kpf2.data
                assert f"RED_{fiber}_{suffix}" in kpf2.data

    def test_chip_prefix_write_populates_slices(self):
        kpf2 = KPF2()
        n_pix = 100
        green_data = np.ones((NORDER_GREEN, n_pix), dtype=np.float32)
        red_data = np.full((NORDER_RED, n_pix), 2.0, dtype=np.float32)

        kpf2.set_data("GREEN_SCI2_FLUX", green_data)
        kpf2.set_data("RED_SCI2_FLUX", red_data)

        full = kpf2.data["SCI2_FLUX"]
        assert full.shape == (NORDER_GREEN + NORDER_RED, n_pix)
        np.testing.assert_array_equal(full[:NORDER_GREEN], green_data)
        np.testing.assert_array_equal(full[NORDER_GREEN:], red_data)

    def test_chip_prefix_write_allocates_on_first_write(self):
        # A GREEN-only write must still allocate the full GREEN+RED trace.
        kpf2 = KPF2()
        n_pix = 100
        assert len(kpf2.data["TRACE3_FLUX"]) == 0

        green_data = np.zeros((NORDER_GREEN, n_pix), dtype=np.float32)
        kpf2.set_data("GREEN_SCI2_FLUX", green_data)

        assert kpf2.data["TRACE3_FLUX"].shape == (NORDER_GREEN + NORDER_RED, n_pix)

    def test_chip_prefix_write_via_set_data(self):
        kpf2 = KPF2()
        n_pix = 50
        green_data = np.arange(NORDER_GREEN * n_pix, dtype=np.float32).reshape(
            NORDER_GREEN, n_pix
        )

        kpf2.set_data("GREEN_SCI2_FLUX", green_data)
        np.testing.assert_array_equal(kpf2.data["GREEN_SCI2_FLUX"], green_data)


class TestDtypeProvenance:
    """Storage preserves dtype (no coercion) and the FITS round-trip preserves
    precision (float32 -> BITPIX -32, float64 -> -64)."""

    def _populated(self):
        kpf2 = KPF2()
        kpf2.headers["PRIMARY"]["DATE-OBS"] = "2024-01-13T10:26:56"
        kpf2.set_data("TRACE3_FLUX", np.ones((NORDER, 8), dtype=np.float32))
        kpf2.set_data("TRACE3_WAVE", np.ones((NORDER, 8), dtype=np.float64))
        return kpf2

    def test_set_data_preserves_dtype(self):
        kpf2 = self._populated()
        assert_dtype(kpf2.data["TRACE3_FLUX"], FLUX, "TRACE3_FLUX in-mem")
        assert_dtype(kpf2.data["TRACE3_WAVE"], WAVE, "TRACE3_WAVE in-mem")

    def test_roundtrip_preserves_precision(self, tmp_path):
        kpf2 = self._populated()
        name = "kpf_SL2_20240101T000000.fits"
        assert_roundtrip_dtype(KPF2, kpf2, "TRACE3_FLUX", FLUX, tmp_path, name=name)
        assert_roundtrip_dtype(KPF2, kpf2, "TRACE3_WAVE", WAVE, tmp_path, name=name)

    def test_chip_prefix_wave_write_enforces_min_bit_depth(self, caplog):
        # A chip-prefix write bypasses rvdata's base set_data, so check it still
        # enforces the born-64 WAVE policy (upcast plus MinBitDepth warning).
        kpf2 = KPF2()
        with caplog.at_level(logging.WARNING):
            kpf2.set_data(
                "GREEN_SCI2_WAVE", np.ones((NORDER_GREEN, 8), dtype=np.float32)
            )
        assert "MinBitDepth=64" in caplog.text
        assert_dtype(kpf2.data["TRACE3_WAVE"], WAVE, "underlying TRACE3_WAVE")

    def test_chip_prefix_flux_write_keeps_float32(self, caplog):
        # MinBitDepth enforcement is WAVE/QUALITY-only, so float32 FLUX is kept
        # as-is: no upcast, no warning.
        kpf2 = KPF2()
        with caplog.at_level(logging.WARNING):
            kpf2.set_data(
                "GREEN_SCI2_FLUX", np.ones((NORDER_GREEN, 8), dtype=np.float32)
            )
        assert "MinBitDepth" not in caplog.text
        assert_dtype(kpf2.data["TRACE3_FLUX"], FLUX, "underlying TRACE3_FLUX")


class TestKPFMasterL2:
    def test_wls_extensions_created(self):
        m = KPFMasterL2(kind="wls")
        for ext in ["PRIMARY", "RECEIPT", "QUALITY_CONTROL", "INPUT_FILES"]:
            assert ext in m.extensions
        for n in (1, 3, 5):
            assert f"TRACE{n}_WAVE" in m.extensions
            for suffix in ("FLUX", "VAR", "BLAZE"):
                assert f"TRACE{n}_{suffix}" not in m.extensions

    def test_flat_extensions_created(self):
        m = KPFMasterL2(kind="flat")
        for ext in ["PRIMARY", "RECEIPT", "QUALITY_CONTROL", "INPUT_FILES"]:
            assert ext in m.extensions
        for n in (1, 3, 5):
            for suffix in ("FLUX", "VAR", "BLAZE"):
                assert f"TRACE{n}_{suffix}" in m.extensions
            assert f"TRACE{n}_WAVE" not in m.extensions

    def test_kind_required_and_validated(self):
        with pytest.raises(TypeError):
            KPFMasterL2()  # kind is required, no default
        with pytest.raises(ValueError):
            KPFMasterL2(kind="bogus")

    def test_aliases_work(self):
        m = KPFMasterL2(kind="wls")
        assert m.extensions._resolve("SCI2_WAVE") == "TRACE3_WAVE"
        assert m.extensions._resolve("CAL_WAVE") == "TRACE5_WAVE"
        assert m.extensions._resolve("SKY_WAVE") == "TRACE1_WAVE"

    def test_chip_prefix_access(self):
        m = KPFMasterL2(kind="wls")
        n_pix = 32
        trace_data = (
            np.random.default_rng(42)
            .random((NORDER_GREEN + NORDER_RED, n_pix))
            .astype(np.float32)
        )
        m.data["TRACE3_WAVE"] = trace_data

        green = m.data["GREEN_SCI2_WAVE"]
        red = m.data["RED_SCI2_WAVE"]
        assert green.shape == (NORDER_GREEN, n_pix)
        assert red.shape == (NORDER_RED, n_pix)
        np.testing.assert_array_equal(green, trace_data[:NORDER_GREEN])
        np.testing.assert_array_equal(red, trace_data[NORDER_GREEN:])

    def test_inherits_from_kpf2(self):
        m = KPFMasterL2(kind="wls")
        assert isinstance(m, KPF2)

    def test_inherits_from_kpf_master_model(self):
        m = KPFMasterL2(kind="wls")
        assert isinstance(m, KPFMasterModel)

    def test_class_attributes(self):
        assert KPFMasterL2._DATALVL == "ML2"

    def test_from_fits(self, synthetic_masters_l2_file):
        m = KPFMasterL2.from_fits(synthetic_masters_l2_file)
        assert "TRACE3_WAVE" in m.extensions
        assert m.data["TRACE3_WAVE"].shape == (NORDER_GREEN + NORDER_RED, 64)

    def test_from_fits_adds_receipt_entry(self, synthetic_masters_l2_file):
        m = KPFMasterL2.from_fits(synthetic_masters_l2_file)
        assert "from_fits" in m.receipt["FUNCTION"].values

    def test_round_trip(self, synthetic_masters_l2_file, tmp_path):
        m = KPFMasterL2.from_fits(synthetic_masters_l2_file)
        original = m.data["TRACE3_WAVE"].copy()

        out_fn = str(tmp_path / "KP.20240113.23249.10_master_thar_L2.fits")
        m.to_fits(out_fn)

        m2 = KPFMasterL2.from_fits(out_fn)
        np.testing.assert_array_almost_equal(
            m2.data["TRACE3_WAVE"], original, decimal=4
        )

    def test_datalvl_header_in_fits(self, synthetic_masters_l2_file, tmp_path):
        m = KPFMasterL2.from_fits(synthetic_masters_l2_file)
        out_fn = str(tmp_path / "KP.20240113.23249.10_master_thar_L2.fits")
        m.to_fits(out_fn)
        with fits.open(out_fn) as hdul:
            assert hdul["PRIMARY"].header["DATALVL"] == "ML2"

    def test_no_warning_on_known_extensions(self, caplog, synthetic_masters_l2_file):
        with caplog.at_level(logging.WARNING):
            KPFMasterL2.from_fits(synthetic_masters_l2_file)
        assert "Non-standard extension" not in caplog.text

    def test_set_input_files(self):
        m = KPFMasterL2(kind="wls")
        files = ["/data/a.fits", "/data/b.fits", "/data/c.fits"]
        m.set_input_files(files, "thar")
        assert m.data["INPUT_FILES"]["FILENAME"].tolist() == files
        assert m.headers["PRIMARY"]["MASTYPE"] == "thar"

    def test_input_files_roundtrip(self, tmp_path):
        m = KPFMasterL2(kind="wls")
        files = ["/data/a.fits", "/data/b.fits", "/data/c.fits"]
        m.set_input_files(files, "thar")
        m.headers["PRIMARY"]["DATE-OBS"] = "2024-01-13T10:26:56"

        out_fn = str(tmp_path / "KP.20240113.23249.10_master_thar_L2.fits")
        m.to_fits(out_fn)

        m2 = KPFMasterL2.from_fits(out_fn)
        assert "INPUT_FILES" in m2.extensions
        assert m2.data["INPUT_FILES"]["FILENAME"].tolist() == files

    def test_warns_on_unknown_extension(self, caplog, tmp_path):
        fn = str(tmp_path / "unknown_ext_ml2.fits")
        primary = fits.PrimaryHDU()
        primary.header["DATE-OBS"] = "2024-01-13T00:00:00"
        primary.header["MASTYPE"] = "thar"  # so from_fits can infer kind="wls"
        weird = fits.ImageHDU(data=np.zeros((4, 4)))
        weird.name = "WEIRD_EXTENSION"
        hdul = fits.HDUList([primary, weird])
        hdul.writeto(fn, overwrite=True)
        hdul.close()

        with caplog.at_level(logging.WARNING):
            KPFMasterL2.from_fits(fn)
        assert "Non-standard extension" in caplog.text


class TestKPF2HeaderStorage:
    """KPF2-specific header storage and the KPF2._create_hdul serialization path.

    KPF2 stores headers as fits.Header like every KPF model, but overrides
    _create_hdul via RV2 -- a distinct path from the inherited base one covered in
    test_data_models_base.py.
    """

    def test_fresh_l2_headers_are_fits_headers(self):
        kpf2 = KPF2()
        assert isinstance(kpf2.headers["PRIMARY"], fits.Header)
        assert isinstance(kpf2.headers["TRACE1_FLUX"], fits.Header)

    def test_l2_primary_comment_round_trips(self, tmp_path):
        kpf2 = KPF2()
        # Guards the KPF2._create_hdul comment-preservation override. The card must
        # be non-EPRV: rvdata rewrites its own L2-defined keywords from the
        # definition, dropping any KPF comment on them.
        kpf2.headers["PRIMARY"]["HDRCMNT"] = ("kept", "comment must survive to_fits")

        fn = str(tmp_path / "kpf_SL2_20240113T102656.fits")
        kpf2.to_fits(fn)

        prim = KPF2.from_fits(fn).headers["PRIMARY"]
        assert prim.get("HDRCMNT") == "kept"
        assert prim.comments["HDRCMNT"] == "comment must survive to_fits"
