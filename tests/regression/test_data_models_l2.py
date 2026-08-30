"""Tests for the KPF2 (extracted spectra / L2) data model, the L1->L2 transform,
the AliasedOrderedDict extension-alias machinery, and the KPFMasterL2 calibration
product. Synthetic FITS fixtures only -- no real KPF data needed.
"""

import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.table import Table

from kpfpipe import DETECTOR
from kpfpipe.data_models.aliased_dict import AliasedOrderedDict
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.masters import KPFMasterL2
from kpfpipe.data_models.masters.base import KPFMasterModel
from kpfpipe.modules.standardize_data_format import StandardizeDataFormat

from ._catalog import SOURCES, catalog_record_table
from ._dtype_policy import FLUX, WAVE, assert_dtype, assert_roundtrip_dtype
from ._eprv import expand, kpf_table, rvdata_table

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
NORDER = DETECTOR["numorder"]

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
    # BitDepth rejection on read.
    wave = rng.random((DETECTOR["numorder"], n_pix)).astype(np.float64)
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
    l0 = KPF0.from_fits(synthetic_l0_file)
    StandardizeDataFormat(l0).perform()
    return l0.to_kpf1()


class TestKPF2QualityControlRoundTrip:
    """The KPF-custom QUALITY_CONTROL extension round-trips through KPF2.

    It is a plain L2-extensions.csv row, so the manifest builds it and the
    manifest-driven read accepts it back.
    """

    def test_quality_control_and_barycorr_roundtrip(self, tmp_path):
        l2 = KPF2()
        l2.set_keyword("NANSCI1", 7)
        l2.set_keyword("BVGREEN", -3.21)
        fn = str(tmp_path / "kpf_SL2_20240101T000000.fits")
        l2.to_fits(fn)
        back = KPF2.from_fits(fn)
        assert back.headers["QUALITY_CONTROL"]["NANSCI1"] == 7
        assert back.headers["BARYCORR_KMS"]["BVGREEN"] == -3.21

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
    """CATALOG_RECORD rides L1 -> L2 and reads back.

    Another declared L2-extensions.csv row, like QUALITY_CONTROL above.
    """

    @staticmethod
    def _l1_with_catalog(rv=10.0):
        # Every L1-extensions.csv row is created, so CATALOG_RECORD is present
        # and empty on a bare L1; only its rows need supplying.
        l1 = KPF1()
        l1.set_data("CATALOG_RECORD", catalog_record_table(rv=rv))
        return l1

    def test_rows_reach_l2(self):
        l2 = self._l1_with_catalog().to_kpf2()
        assert [str(s) for s in l2.data["CATALOG_RECORD"]["source"]] == list(SOURCES)

    def test_catalog_record_roundtrip(self, tmp_path):
        # The missing rv reads back NaN, not masked -- only the from_fits
        # chokepoint can normalize it.
        l2 = self._l1_with_catalog(rv=None).to_kpf2()
        fn = str(tmp_path / "kpf_SL2_20240101T000000.fits")
        l2.to_fits(fn)
        back = KPF2.from_fits(fn)
        assert [str(s) for s in back.data["CATALOG_RECORD"]["source"]] == list(SOURCES)
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
        # The keyword conversion happened in StandardizeDataFormat; to_kpf1 and
        # to_kpf2 only forward it.
        kpf2 = converted_l1.to_kpf2()
        prim = kpf2.headers["PRIMARY"]
        assert prim.get("EXPTIME") == 300.0  # from ELAPSED
        assert prim.get("OBSTYPE") == "Object"  # from IMTYPE
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
        assert origin == "Keck"

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
        # The DRPSTATU receipt override lives on KPFDataModel; assert KPF2 reaches
        # it, since to_kpf2 is the only producer that exercises it at L2.
        kpf2 = KPF1.from_fits(synthetic_l1_file).to_kpf2()
        kpf2.receipt_add_entry("barycentric_correction", "", "PASS")
        assert (
            kpf2.headers["RECEIPT"].get("DRPSTATU")
            == "Barycentric Correction module complete"
        )

    def test_receipt_and_drpstatus_survive_roundtrip(self, tmp_path):
        # L2 carries RECEIPT columns and a DRPSTATU card that L0/L1 do not, so the
        # L0/L1 round-trip twins cover none of this.
        kpf2 = KPF2()
        kpf2.headers["PRIMARY"]["DATE-OBS"] = "2024-01-01T00:00:00"
        kpf2.receipt_add_entry("spectral_extraction", "", "PASS")
        fn = str(tmp_path / "kpf_SL2_20240101T000000.fits")
        kpf2.to_fits(fn)

        back = KPF2.from_fits(fn)
        assert "spectral_extraction" in back.receipt["FUNCTION"].values
        assert (
            back.headers["RECEIPT"].get("DRPSTATU")
            == "Spectral Extraction module complete"
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

    def test_delete_via_alias(self):
        d = AliasedOrderedDict()
        d["CANONICAL"] = 1
        d.register_alias("ALIAS", "CANONICAL")
        del d["ALIAS"]
        assert "CANONICAL" not in d

    def test_identity_via_alias(self):
        d = AliasedOrderedDict()
        arr = np.zeros((4, 4))
        d["CANONICAL"] = arr
        d.register_alias("ALIAS", "CANONICAL")
        assert d["ALIAS"] is d["CANONICAL"]


class TestKPF2Aliases:
    def test_kpf2_declares_its_level(self):
        # The level is the manifest key, so KPF2 resolving to 2 is what makes
        # _data_model, _seed_primary and _read read the L2 tables.
        assert KPF2().level == 2

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
        test_data = np.random.default_rng(42).random((10, 100), dtype=np.float32)
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
        trace_data = rng.random((DETECTOR["numorder"], n_pix), dtype=np.float32)
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
        assert full.shape == (DETECTOR["numorder"], n_pix)
        np.testing.assert_array_equal(full[:NORDER_GREEN], green_data)
        np.testing.assert_array_equal(full[NORDER_GREEN:], red_data)

    def test_chip_prefix_write_allocates_on_first_write(self):
        # A GREEN-only write must still allocate the full GREEN+RED trace.
        kpf2 = KPF2()
        n_pix = 100
        assert len(kpf2.data["TRACE3_FLUX"]) == 0

        green_data = np.zeros((NORDER_GREEN, n_pix), dtype=np.float32)
        kpf2.set_data("GREEN_SCI2_FLUX", green_data)

        assert kpf2.data["TRACE3_FLUX"].shape == (DETECTOR["numorder"], n_pix)

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

    def test_set_data_rejects_the_wrong_bit_depth(self):
        # The manifest declares an exact width, so a float32 WAVE is a producer
        # bug and is refused rather than quietly widened.
        kpf2 = KPF2()
        with pytest.raises(TypeError, match="TRACE1_WAVE: manifest declares 64-bit"):
            kpf2.set_data("TRACE1_WAVE", np.ones((NORDER, 8), dtype=np.float32))

    def test_chip_prefix_write_rejects_the_wrong_bit_depth(self):
        # A chip-prefix write bypasses rvdata's base set_data, so check the KPF
        # path enforces the born-64 WAVE policy too.
        kpf2 = KPF2()
        with pytest.raises(
            TypeError, match="GREEN_SCI2_WAVE: manifest declares 64-bit"
        ):
            kpf2.set_data(
                "GREEN_SCI2_WAVE", np.ones((NORDER_GREEN, 8), dtype=np.float32)
            )

    def test_chip_prefix_flux_write_keeps_float32(self):
        kpf2 = KPF2()
        kpf2.set_data("GREEN_SCI2_FLUX", np.ones((NORDER_GREEN, 8), dtype=np.float32))
        assert_dtype(kpf2.data["TRACE3_FLUX"], FLUX, "underlying TRACE3_FLUX")

    def test_empty_arrays_are_exempt(self):
        # Every extension is born np.array([]) -- 64-bit -- so a 32-bit slot must
        # accept an empty array or no product could be constructed or re-read.
        KPF2().set_data("TRACE3_FLUX", np.array([]))


class TestKPFMasterL2:
    def test_wls_extensions_created(self):
        m = KPFMasterL2(kind="wls")
        for ext in ["PRIMARY", "RECEIPT", "QUALITY_CONTROL", "INPUT_FILES"]:
            assert ext in m.extensions
        for n in (1, 3, 5):
            assert f"TRACE{n}_WAVE" in m.extensions
            for suffix in ("FLUX", "VAR", "BLAZE"):
                assert f"TRACE{n}_{suffix}" not in m.extensions
        # ML2-wls-extensions.csv is the whole inventory: no science L2 row it
        # omits survives, and both coefficient extensions are built.
        assert len(m.extensions) == 14
        assert "GREEN_WLS_COEFFS" in m.extensions
        assert "RED_WLS_COEFFS" in m.extensions

    def test_flat_extensions_created(self):
        m = KPFMasterL2(kind="flat")
        for ext in ["PRIMARY", "RECEIPT", "QUALITY_CONTROL", "INPUT_FILES"]:
            assert ext in m.extensions
        for n in (1, 3, 5):
            for suffix in ("FLUX", "VAR", "BLAZE"):
                assert f"TRACE{n}_{suffix}" in m.extensions
            assert f"TRACE{n}_WAVE" not in m.extensions
        assert len(m.extensions) == 22

    @pytest.mark.parametrize("kind", ("wls", "flat"))
    def test_ext_descript_is_the_master_own_extension_set(self, kind):
        # The master builds its own manifest, so EXT_DESCRIPT (written last)
        # names exactly what it carries -- not the science L2 set.
        m = KPFMasterL2(kind=kind)
        assert m.data["EXT_DESCRIPT"]["Name"].tolist() == list(m.extensions)

    @pytest.mark.parametrize("kind", ("wls", "flat"))
    def test_primary_is_seeded_from_the_master_data_model(self, kind):
        m = KPFMasterL2(kind=kind)
        seed = m.keyword_registry.primary_seed(f"ML2-{kind}")
        assert set(seed) <= set(m.headers["PRIMARY"])
        assert m.headers["PRIMARY"]["DATALVL"] == "ML2"
        # The EPRV science skeleton is not inherited.
        science = set(m.keyword_registry.primary_seed("L2"))
        assert not (science & set(m.headers["PRIMARY"])) - {"DATALVL"}

    @pytest.mark.parametrize("kind", ("flat", "wls"))
    def test_the_master_builds_its_whole_manifest(self, kind):
        master = KPFMasterL2(kind=kind)
        assert set(master.extensions) == set(
            kpf_table(f"ML2-{kind}-extensions")["Name"]
        )
        # A master is not a translation of a native instrument product, so it
        # carries no verbatim instrument header.
        assert "INSTRUMENT_HEADER" not in master.extensions

    @pytest.mark.parametrize("data_model", ("ML2-flat", "ML2-wls"))
    def test_shared_rows_agree_with_the_science_manifest(self, data_model):
        # The duplication between a master manifest and its science level's is
        # intentional -- each stays a complete spec of one product -- so a shared
        # row must not disagree, or a master would ship a TRACE1_WAVE unlike
        # L2's.
        master = kpf_table(f"{data_model}-extensions").set_index("Name")
        science = kpf_table("L2-extensions").set_index("Name")
        shared = set(master.index) & set(science.index)
        assert shared
        for name in shared:
            assert master.loc[name, "DataType"] == science.loc[name, "DataType"], name
            ours, theirs = master.loc[name, "BitDepth"], science.loc[name, "BitDepth"]
            assert (pd.isna(ours) and pd.isna(theirs)) or ours == theirs, name

    def test_bit_depth_from_the_master_manifest(self):
        manifest = KPFMasterL2.extension_manifest
        assert manifest.bit_depth("ML2-wls", "TRACE1_WAVE") == 64
        assert manifest.bit_depth("ML2-wls", "GREEN_WLS_COEFFS") == 64
        assert manifest.bit_depth("ML2-flat", "TRACE1_FLUX") == 32
        assert manifest.bit_depth("ML2-wls", "RECEIPT") is None

    def test_reads_through_the_one_base_reader(self):
        # KPFMasterL2 declares no read/_read of its own: KPFDataModel.read is
        # what detaches rvdata's level-keyed dispatch, once, for every level.
        assert "read" not in vars(KPFMasterL2)
        assert "_read" not in vars(KPFMasterL2)

    def test_kind_required_and_validated(self):
        with pytest.raises(TypeError):
            KPFMasterL2()  # kind is required, no default
        with pytest.raises(ValueError, match="kind must be one of"):
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
            .random((DETECTOR["numorder"], n_pix))
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

    def test_from_fits(self, synthetic_masters_l2_file):
        m = KPFMasterL2.from_fits(synthetic_masters_l2_file)
        assert "TRACE3_WAVE" in m.extensions
        assert m.data["TRACE3_WAVE"].shape == (DETECTOR["numorder"], 64)

    def test_from_fits_adds_receipt_entry(self, synthetic_masters_l2_file):
        m = KPFMasterL2.from_fits(synthetic_masters_l2_file)
        assert "from_fits" in m.receipt["FUNCTION"].values

    def test_from_fits_rejects_unreadable_input(self, tmp_path):
        with pytest.raises(OSError, match="does not exist"):
            KPFMasterL2.from_fits(str(tmp_path / "absent.fits"))
        not_fits = tmp_path / "master.txt"
        not_fits.write_text("not a FITS file")
        with pytest.raises(OSError, match="must be FITS files"):
            KPFMasterL2.from_fits(str(not_fits))

    def test_from_fits_requires_a_known_mastype(self, tmp_path):
        # MASTYPE picks the extension manifest, so an unmappable one must not
        # fall back to a default kind -- a flat read under the WLS schema would
        # land extracted spectra in TRACE*_WAVE.
        fn = str(tmp_path / "KP.20240113.23249.10_master_bogus_L2.fits")
        primary = fits.PrimaryHDU()
        primary.header["MASTYPE"] = "bogus"
        fits.HDUList([primary]).writeto(fn, overwrite=True)
        with pytest.raises(ValueError, match="cannot infer KPFMasterL2 kind"):
            KPFMasterL2.from_fits(fn)

    def test_round_trip(self, synthetic_masters_l2_file, tmp_path):
        m = KPFMasterL2.from_fits(synthetic_masters_l2_file)
        original = m.data["TRACE3_WAVE"].copy()

        out_fn = str(tmp_path / "KP.20240113.23249.10_master_thar_L2.fits")
        m.to_fits(out_fn)

        m2 = KPFMasterL2.from_fits(out_fn)
        np.testing.assert_array_equal(m2.data["TRACE3_WAVE"], original)

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

    def test_seeded_but_unpopulated_cards_read_back_as_none(self, tmp_path):
        m = KPFMasterL2(kind="wls")
        m.headers["PRIMARY"]["DATE-OBS"] = "2024-04-05T01:00:37"
        m.set_input_files(["KP.20240405.63499.95.fits"], "thar")
        out = str(tmp_path / "KP.20240405.63499.95_master_thar_L2.fits")
        m.to_fits(out)

        back = KPFMasterL2.from_fits(out)
        assert set(back.extensions) == set(kpf_table("ML2-wls-extensions")["Name"])
        for key in ("ROUGHWLS", "LINELIST", "POLYDEGX"):
            assert back.headers["PRIMARY"][key] is None, key

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
    """KPF2-specific header storage and its serialization path.

    KPF2 stores headers as fits.Header like every KPF model; these cover the L2
    extension set, which the base twins in test_data_models_base.py do not.
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


class TestRvdataReadersAreDetached:
    """rvdata's L2 reader never runs, and the assertion is not redundant.

    ``RVDataModel.read`` dispatches to ``RV2._read`` through a hardcoded class
    reference keyed on ``self.level``, not through the MRO, so KPF2 no longer
    subclassing RV2 is not by itself what stops it -- ``KPFDataModel.read`` is.
    """

    def test_rv2_read_never_fires(self, tmp_path, monkeypatch):
        from rvdata.core.models.level2 import RV2

        fired = []
        monkeypatch.setattr(
            RV2, "_read", lambda self, hdul: fired.append("RV2"), raising=True
        )
        fn = str(tmp_path / "kpf_SL2_20240101T000000.fits")
        KPF2().to_fits(fn)
        KPF2.from_fits(fn)
        assert fired == []


# rvdata rows KPF deliberately does not build. These are the EPRV-optional set
# rvdata's own name-guessing readers tolerated; KPF's manifest-driven read
# declares what it accepts, so an undeclared extension is rejected, not inferred.
_UNBUILT = {
    "IMAGE": "EPRV-optional; KPF ships no whole-detector image at L2",
    "TRACE1_DRIFT": "EPRV-optional; KPF has no drift product",
    "TRACE1_QUALITY": "EPRV-optional; KPF carries QC in QUALITY_CONTROL",
    "TRACE1_SKYMODEL": "EPRV-optional; KPF has no sky model",
    "TRACE1_TELLURIC": "EPRV-optional; KPF does no telluric correction",
    "CUSTOM1_TRACE1_FLUX": "EPRV CUSTOM slot; unused by KPF",
    "CUSTOM1_TRACE1_VAR": "EPRV CUSTOM slot; unused by KPF",
    "CUSTOM1_TRACE1_WAVE": "EPRV CUSTOM slot; unused by KPF",
}

# Extensions whose HDU type deliberately differs from the standard's.
_DATATYPE_DEVIATIONS = {
    "ANCILLARY_SPECTRUM": (
        "EPRV says ImageHDU; KPF ships Ca H&K as a BinTableHDU placeholder while "
        "extraction is WIP, and existing products encode it that way"
    ),
}

# EPRV keywords KPF does not register.
_UNREGISTERED = {
    "PVN_#": "variable-length parametric WLS family; KPF writes no parametric WLS",
}

_PER_EXTENSION_TABLES = [
    ("L2-TRACE_FLUX-keywords", "TRACE1_FLUX"),
    ("L2-TRACE_VAR-keywords", "TRACE1_VAR"),
    ("L2-TRACE_BLAZE-keywords", "TRACE1_BLAZE"),
    ("L2-TRACE_WAVE-keywords", "TRACE1_WAVE"),
    ("L2-BJD_TDB-keywords", "BJD_TDB"),
    ("L2-BARYCORR_KMS-keywords", "BARYCORR_KMS"),
    ("L2-BARYCORR_Z-keywords", "BARYCORR_Z"),
]


class TestEPRVCompliance:
    """L2 against the installed rvdata tables.

    Existence and shape only: every EPRV keyword is registered, every required
    extension is declared and built. Where a keyword is *written* is the data
    model's business -- ``set_keyword`` routes it off the registry -- not this
    file's, so nothing here re-checks a write site. The three dicts above are the
    complete list of divergences; anything else must match.
    """

    def test_every_eprv_primary_keyword_is_registered(self):
        want = {
            member
            for keyword in rvdata_table("L2-PRIMARY-keywords")["Keyword"]
            for member in expand(keyword)
        }
        assert not sorted(want - KPF2.keyword_registry.allowed["PRIMARY"])

    @pytest.mark.parametrize(("table", "extension"), _PER_EXTENSION_TABLES)
    def test_every_eprv_per_extension_keyword_is_registered(self, table, extension):
        registry = KPF2.keyword_registry
        missing = [
            keyword
            for keyword in map(str.strip, rvdata_table(table)["Keyword"])
            if not registry.is_structural(keyword)
            and keyword not in _UNREGISTERED
            and keyword not in registry.allowed[extension]
        ]
        assert not missing

    def test_every_required_extension_is_built(self):
        rvdata = rvdata_table("L2-extensions")
        assert set(rvdata[rvdata["Required"]]["Name"]) <= set(KPF2().extensions)

    def test_the_model_builds_its_whole_manifest(self):
        model = KPF2()
        assert set(model.extensions) == set(kpf_table("L2-extensions")["Name"])

    def test_undeclared_rvdata_extensions_are_listed(self):
        undeclared = set(rvdata_table("L2-extensions")["Name"]) - set(
            kpf_table("L2-extensions")["Name"]
        )
        assert undeclared == set(_UNBUILT)

    def test_shared_extensions_agree_on_hdu_type(self):
        rvdata = rvdata_table("L2-extensions")
        kpf = kpf_table("L2-extensions")
        theirs = dict(zip(rvdata["Name"], rvdata["DataType"], strict=True))
        ours = dict(zip(kpf["Name"], kpf["DataType"], strict=True))
        differing = {n for n in set(theirs) & set(ours) if theirs[n] != ours[n]}
        assert differing == set(_DATATYPE_DEVIATIONS)

    def test_bit_depth_meets_the_eprv_floor(self):
        # rvdata's column is a floor and KPF's is exact, so this is >=, not ==:
        # KPF declares the wider policy, filling in the float32 rows rvdata
        # leaves blank.
        rvdata = rvdata_table("L2-extensions")
        ours = kpf_table("L2-extensions").set_index("Name")["BitDepth"]
        shared = rvdata[rvdata["MinBitDepth"].notna() & ~rvdata["Name"].isin(_UNBUILT)]
        assert not shared.empty
        for _, row in shared.iterrows():
            assert ours[row["Name"]] >= row["MinBitDepth"], row["Name"]
