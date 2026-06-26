"""
Tests for the KPF2 (extracted spectra / L2) data model, the L1->L2 transform,
the AliasedOrderedDict extension-alias machinery, and the KPFMasterL2
calibration product.

Uses synthetic FITS fixtures — no real KPF data needed.
"""

import numpy as np
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

from ._dtype_policy import FLUX, WAVE, assert_dtype, assert_roundtrip_dtype

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
NORDER = NORDER_GREEN + NORDER_RED

# synthetic_l1_file fixture lives in tests/conftest.py

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_masters_l2_file(tmp_path):
    """Create a minimal synthetic Masters L2 FITS file."""
    rng = np.random.default_rng(20240113)
    fn = str(tmp_path / "KP.20240113.23249.10_master_thar_L2.fits")

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-01-13T10:26:56"
    primary.header["DATALVL"] = "ML2"

    n_pix = 64
    wave = rng.random((NORDER_GREEN + NORDER_RED, n_pix)).astype(np.float32)
    trace3_wave = fits.ImageHDU(data=wave)
    trace3_wave.name = "TRACE3_WAVE"

    hdul = fits.HDUList([primary, trace3_wave])
    hdul.writeto(fn, overwrite=True)
    hdul.close()

    return fn


@pytest.fixture
def converted_l1(synthetic_l0_file):
    """An L1 produced via KPF0.to_kpf1 — already EPRV-standard PRIMARY plus a
    populated INSTRUMENT_HEADER (the input to_kpf2 expects in production)."""
    return KPF0.from_fits(synthetic_l0_file).to_kpf1()


class TestToKPF2:
    def test_to_kpf2_creates_kpf2(self, converted_l1):
        kpf2 = converted_l1.to_kpf2()
        assert kpf2.level == 2
        assert isinstance(kpf2, KPF2)

    def test_to_kpf2_passes_through_eprv_primary(self, converted_l1):
        """Conversion happened in to_kpf1; to_kpf2 forwards the EPRV PRIMARY."""
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
        """INSTRUMENT_HEADER (raw natives) is forwarded unchanged from L1."""
        kpf2 = converted_l1.to_kpf2()
        assert kpf2.headers["INSTRUMENT_HEADER"]["INSTRUME"] == "KPF"
        assert kpf2.headers["INSTRUMENT_HEADER"]["ELAPSED"] == 300.0

    def test_to_kpf2_maps_passthrough_extensions(self, tmp_path):
        """Build an L1 with TELEMETRY and CA_HK, verify they map to KPF2 extensions."""
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
        assert "to_kpf2" in kpf2.receipt["Module_Name"].values

    def test_to_kpf2_receipt_updates_drpstatus(self, synthetic_l1_file):
        """The DRPSTATU receipt override is active on KPF2 too (it subclasses
        RV2, not KPFDataModel, so it carries its own override)."""
        kpf2 = KPF1.from_fits(synthetic_l1_file).to_kpf2()
        kpf2.receipt_add_entry("barycentric_correction", "PASS")
        assert (
            kpf2.headers["PRIMARY"].get("DRPSTATU")
            == "Barycentric Correction module complete"
        )

    def test_to_kpf2_sets_origid(self, tmp_path):
        """Verify obs_id is stored as ORIGID in KPF2 PRIMARY."""
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
        assert l1.obs_id == "KP.20240113.23249.10"
        kpf2 = l1.to_kpf2()
        origid = kpf2.headers["PRIMARY"].get("ORIGID")
        assert origid == "KP.20240113.23249.10"


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
        from collections import OrderedDict

        od = OrderedDict([("A", 1), ("B", 2)])
        aliased = AliasedOrderedDict.from_ordered_dict(od)
        assert aliased["A"] == 1
        assert aliased["B"] == 2
        aliased.register_alias("C", "A")
        assert aliased["C"] == 1

    def test_identity_via_alias(self):
        """Alias access returns the exact same object (not a copy)."""
        d = AliasedOrderedDict()
        arr = np.zeros((4, 4))
        d["CANONICAL"] = arr
        d.register_alias("ALIAS", "CANONICAL")
        assert d["ALIAS"] is d["CANONICAL"]


class TestKPF2Aliases:
    def test_kpf2_inherits_rv2(self):
        from rvdata.core.models.level2 import RV2

        kpf2 = KPF2()
        assert isinstance(kpf2, RV2)
        assert kpf2.level == 2

    def test_fiber_alias_resolves(self):
        kpf2 = KPF2()
        # SCI2_FLUX should resolve to TRACE3_FLUX
        assert kpf2.data["SCI2_FLUX"] is kpf2.data["TRACE3_FLUX"]
        assert kpf2.data["CAL_FLUX"] is kpf2.data["TRACE1_FLUX"]
        assert kpf2.data["SKY_WAVE"] is kpf2.data["TRACE5_WAVE"]

    def test_extension_alias_resolves(self):
        kpf2 = KPF2()
        # CA_HK should resolve to ANCILLARY_SPECTRUM
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
        # Check all 5 fibers x 4 suffixes = 20 aliases
        for fiber, trace in [
            ("CAL", 1),
            ("SCI1", 2),
            ("SCI2", 3),
            ("SCI3", 4),
            ("SKY", 5),
        ]:
            for suffix in ["FLUX", "WAVE", "VAR", "BLAZE"]:
                alias = f"{fiber}_{suffix}"
                canonical = f"TRACE{trace}_{suffix}"
                assert alias in kpf2.extensions, f"{alias} not found"
                assert kpf2.data[alias] is kpf2.data[canonical]

    def test_chip_prefix_access(self):
        """Test GREEN_/RED_ prefix returns correct slices of concatenated trace."""
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
        """GREEN_SCI2_FLUX should be 'in' the data dict."""
        kpf2 = KPF2()
        assert "GREEN_SCI2_FLUX" in kpf2.data
        assert "RED_CAL_WAVE" in kpf2.data
        assert "GREEN_NONEXISTENT" not in kpf2.data

    def test_chip_prefix_all_fibers(self):
        """All chip+fiber+suffix combinations should work."""
        kpf2 = KPF2()
        for fiber in ["CAL", "SCI1", "SCI2", "SCI3", "SKY"]:
            for suffix in ["FLUX", "WAVE", "VAR", "BLAZE"]:
                assert f"GREEN_{fiber}_{suffix}" in kpf2.data
                assert f"RED_{fiber}_{suffix}" in kpf2.data

    def test_chip_prefix_write_populates_slices(self):
        """Writing via chip-prefix should fill the correct slice of the trace."""
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
        """Writing GREEN first should allocate the full (67, ncol) trace."""
        kpf2 = KPF2()
        n_pix = 100
        assert len(kpf2.data["TRACE3_FLUX"]) == 0

        green_data = np.zeros((NORDER_GREEN, n_pix), dtype=np.float32)
        kpf2.set_data("GREEN_SCI2_FLUX", green_data)

        assert kpf2.data["TRACE3_FLUX"].shape == (NORDER_GREEN + NORDER_RED, n_pix)

    def test_chip_prefix_write_via_set_data(self):
        """set_data() should route chip-prefix keys through __setitem__."""
        kpf2 = KPF2()
        n_pix = 50
        green_data = np.arange(NORDER_GREEN * n_pix, dtype=np.float32).reshape(
            NORDER_GREEN, n_pix
        )

        kpf2.set_data("GREEN_SCI2_FLUX", green_data)
        np.testing.assert_array_equal(kpf2.data["GREEN_SCI2_FLUX"], green_data)


class TestDtypeProvenance:
    """The data-model storage layer preserves dtype (no coercion); FITS
    round-trip preserves precision (float32 -> BITPIX -32, float64 -> -64)."""

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
        assert_roundtrip_dtype(KPF2, kpf2, "TRACE3_FLUX", FLUX, tmp_path)
        assert_roundtrip_dtype(KPF2, kpf2, "TRACE3_WAVE", WAVE, tmp_path)


class TestKPFMasterL2:
    def test_required_extensions_created(self):
        m = KPFMasterL2()
        for ext in [
            "PRIMARY",
            "RECEIPT",
            "TRACE1_FLUX",
            "TRACE1_WAVE",
            "TRACE1_VAR",
            "TRACE1_BLAZE",
            "TRACE3_FLUX",
            "TRACE3_WAVE",
            "TRACE3_VAR",
            "TRACE3_BLAZE",
            "TRACE5_FLUX",
            "TRACE5_WAVE",
            "TRACE5_VAR",
            "TRACE5_BLAZE",
        ]:
            assert ext in m.extensions

    def test_aliases_work(self):
        m = KPFMasterL2()
        assert m.extensions._resolve("SCI2_WAVE") == "TRACE3_WAVE"
        assert m.extensions._resolve("CAL_WAVE") == "TRACE1_WAVE"
        assert m.extensions._resolve("SKY_WAVE") == "TRACE5_WAVE"

    def test_chip_prefix_access(self):
        m = KPFMasterL2()
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
        m = KPFMasterL2()
        assert isinstance(m, KPF2)

    def test_inherits_from_kpf_master_model(self):
        m = KPFMasterL2()
        assert isinstance(m, KPFMasterModel)

    def test_class_attributes(self):
        assert KPFMasterL2._DATALVL == "ML2"

    def test_from_fits(self, synthetic_masters_l2_file):
        m = KPFMasterL2.from_fits(synthetic_masters_l2_file)
        assert "TRACE3_WAVE" in m.extensions
        assert m.data["TRACE3_WAVE"].shape == (NORDER_GREEN + NORDER_RED, 64)

    def test_from_fits_adds_receipt_entry(self, synthetic_masters_l2_file):
        m = KPFMasterL2.from_fits(synthetic_masters_l2_file)
        assert "from_fits" in m.receipt["Module_Name"].values

    def test_round_trip(self, synthetic_masters_l2_file, tmp_path):
        m = KPFMasterL2.from_fits(synthetic_masters_l2_file)
        original = m.data["TRACE3_WAVE"].copy()

        out_fn = str(tmp_path / "roundtrip_ml2.fits")
        m.to_fits(out_fn)

        m2 = KPFMasterL2.from_fits(out_fn)
        np.testing.assert_array_almost_equal(m2.data["TRACE3_WAVE"], original)

    def test_datalvl_header_in_fits(self, synthetic_masters_l2_file, tmp_path):
        m = KPFMasterL2.from_fits(synthetic_masters_l2_file)
        out_fn = str(tmp_path / "datalvl_ml2.fits")
        m.to_fits(out_fn)
        with fits.open(out_fn) as hdul:
            assert hdul["PRIMARY"].header["DATALVL"] == "ML2"

    def test_no_warning_on_known_extensions(self, synthetic_masters_l2_file):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            KPFMasterL2.from_fits(synthetic_masters_l2_file)

    def test_set_input_files(self):
        m = KPFMasterL2()
        files = ["/data/a.fits", "/data/b.fits", "/data/c.fits"]
        m.set_input_files(files, "thar")
        assert m.data["INPUT_FILES"]["FILENAME"].tolist() == files
        assert m.headers["PRIMARY"]["MASTYPE"] == "thar"

    def test_input_files_roundtrip(self, tmp_path):
        m = KPFMasterL2()
        files = ["/data/a.fits", "/data/b.fits", "/data/c.fits"]
        m.set_input_files(files, "thar")
        m.headers["PRIMARY"]["DATE-OBS"] = "2024-01-13T10:26:56"

        out_fn = str(tmp_path / "ml2_input_files.fits")
        m.to_fits(out_fn)

        m2 = KPFMasterL2.from_fits(out_fn)
        assert "INPUT_FILES" in m2.extensions
        assert m2.data["INPUT_FILES"]["FILENAME"].tolist() == files

    def test_warns_on_unknown_extension(self, tmp_path):
        fn = str(tmp_path / "unknown_ext_ml2.fits")
        primary = fits.PrimaryHDU()
        primary.header["DATE-OBS"] = "2024-01-13T00:00:00"
        weird = fits.ImageHDU(data=np.zeros((4, 4)))
        weird.name = "WEIRD_EXTENSION"
        hdul = fits.HDUList([primary, weird])
        hdul.writeto(fn, overwrite=True)
        hdul.close()

        with pytest.warns(UserWarning, match="Non-standard extension"):
            KPFMasterL2.from_fits(fn)


class TestKPF2HeaderStorage:
    """KPF2-specific header storage and the KPF2._create_hdul serialization path.

    KPF2 stores headers as fits.Header (like all KPF models) but overrides
    _create_hdul via RV2, a distinct code path from the inherited base one tested
    in test_data_models_base.py.
    """

    def test_fresh_l2_headers_are_fits_headers(self):
        kpf2 = KPF2()
        assert isinstance(kpf2.headers["PRIMARY"], fits.Header)
        # A non-PRIMARY extension header too (created via create_extension).
        assert isinstance(kpf2.headers["TRACE1_FLUX"], fits.Header)

    def test_l2_primary_comment_round_trips(self, tmp_path):
        # Guards the KPF2._create_hdul override (not the inherited base path):
        # a commented PRIMARY card must survive to_fits -> from_fits.
        kpf2 = KPF2()
        kpf2.headers["PRIMARY"]["CCFRV"] = (1.2345, "[km/s] Combined radial velocity")

        fn = str(tmp_path / "kpf_SL2_20240113T102656.fits")
        kpf2.to_fits(fn)

        prim = KPF2.from_fits(fn).headers["PRIMARY"]
        assert prim.get("CCFRV") == 1.2345
        assert prim.comments["CCFRV"] == "[km/s] Combined radial velocity"
