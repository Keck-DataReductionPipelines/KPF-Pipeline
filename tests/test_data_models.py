"""
Tests for KPF data models (KPF0, KPF1, KPF2, KPF4).

Uses synthetic FITS fixtures — no real KPF data needed.
"""

import os
import tempfile

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from kpfpipe import DETECTOR
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.data_models.aliased_dict import AliasedOrderedDict

NORDER_GREEN = DETECTOR['norder']['GREEN']
NORDER_RED = DETECTOR['norder']['RED']


@pytest.fixture
def synthetic_l0_file(tmp_path):
    """Create a minimal synthetic L0 FITS file."""
    fn = str(tmp_path / "KP.20240113.23249.10.fits")

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-01-13T10:26:56"
    primary.header["EXPTIME"] = 300.0
    primary.header["OBJECT"] = "HD_10700"
    primary.header["IMTYPE"] = "Object"

    green_amp1 = fits.ImageHDU(data=np.random.random((32, 32)).astype(np.float32))
    green_amp1.name = "GREEN_AMP1"

    green_amp2 = fits.ImageHDU(data=np.random.random((32, 32)).astype(np.float32))
    green_amp2.name = "GREEN_AMP2"

    red_amp1 = fits.ImageHDU(data=np.random.random((32, 32)).astype(np.float32))
    red_amp1.name = "RED_AMP1"

    ca_hk = fits.ImageHDU(data=np.random.random((16, 16)).astype(np.float32))
    ca_hk.name = "CA_HK"

    telemetry = Table({"keyword": ["TEMP1", "TEMP2"], "average": [20.0, 21.0]})
    telemetry_hdu = fits.BinTableHDU(data=telemetry)
    telemetry_hdu.name = "TELEMETRY"

    hdul = fits.HDUList([primary, green_amp1, green_amp2, red_amp1, ca_hk, telemetry_hdu])
    hdul.writeto(fn, overwrite=True)
    hdul.close()

    return fn


@pytest.fixture
def synthetic_l0_minimal(tmp_path):
    """Create an L0 file with only PRIMARY (no optional extensions)."""
    fn = str(tmp_path / "KP.20240113.00001.00.fits")

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-01-13T00:00:01"

    hdul = fits.HDUList([primary])
    hdul.writeto(fn, overwrite=True)
    hdul.close()

    return fn


@pytest.fixture
def synthetic_l1_file(tmp_path):
    """Create a minimal synthetic L1 FITS file."""
    fn = str(tmp_path / "kpf_L1_20240113T102656.fits")

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-01-13T10:26:56"
    primary.header["EXPTIME"] = 300.0
    primary.header["DATALVL"] = "L1"

    green_ccd = fits.ImageHDU(data=np.random.random((32, 32)).astype(np.float32))
    green_ccd.name = "GREEN_CCD"

    green_var = fits.ImageHDU(data=np.random.random((32, 32)).astype(np.float32))
    green_var.name = "GREEN_VAR"

    red_ccd = fits.ImageHDU(data=np.random.random((32, 32)).astype(np.float32))
    red_ccd.name = "RED_CCD"

    red_var = fits.ImageHDU(data=np.random.random((32, 32)).astype(np.float32))
    red_var.name = "RED_VAR"

    hdul = fits.HDUList([primary, green_ccd, green_var, red_ccd, red_var])
    hdul.writeto(fn, overwrite=True)
    hdul.close()

    return fn


class TestKPF0:
    def test_from_fits(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        assert l0.level == 0
        assert l0.obs_id == "KP.20240113.23249.10"
        assert "GREEN_AMP1" in l0.extensions
        assert "GREEN_AMP2" in l0.extensions
        assert "RED_AMP1" in l0.extensions
        assert "CA_HK" in l0.extensions
        assert "TELEMETRY" in l0.extensions
        assert l0.data["GREEN_AMP1"].shape == (32, 32)
        assert l0.headers["PRIMARY"]["INSTRUME"] == "KPF"

    def test_from_fits_minimal(self, synthetic_l0_minimal):
        l0 = KPF0.from_fits(synthetic_l0_minimal)
        assert l0.level == 0
        assert l0.obs_id == "KP.20240113.00001.00"
        assert "PRIMARY" in l0.extensions
        assert len(l0.extensions) == 1

    def test_round_trip(self, synthetic_l0_file, tmp_path):
        l0 = KPF0.from_fits(synthetic_l0_file)
        original_green = l0.data["GREEN_AMP1"].copy()

        out_fn = str(tmp_path / "roundtrip_l0.fits")
        l0.to_fits(out_fn)

        l0_reread = KPF0.from_fits(out_fn)
        np.testing.assert_array_almost_equal(
            l0_reread.data["GREEN_AMP1"], original_green
        )
        assert l0_reread.headers["PRIMARY"]["INSTRUME"] == "KPF"

    def test_receipt_tracking(self, synthetic_l0_file, tmp_path):
        l0 = KPF0.from_fits(synthetic_l0_file)
        assert len(l0.receipt) >= 1
        assert "from_fits" in l0.receipt["Module_Name"].values

        out_fn = str(tmp_path / "receipt_test.fits")
        l0.to_fits(out_fn)
        assert "to_fits" in l0.receipt["Module_Name"].values

    def test_generate_filename(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        assert l0.generate_standard_filename() == "KP.20240113.23249.10.fits"

    def test_file_not_found(self):
        with pytest.raises(IOError):
            KPF0.from_fits("/nonexistent/path.fits")

    def test_non_fits_file(self, tmp_path):
        fn = str(tmp_path / "not_a_fits.txt")
        with open(fn, "w") as f:
            f.write("hello")
        with pytest.raises(IOError):
            KPF0.from_fits(fn)


class TestKPF1:
    def test_from_fits(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        assert l1.level == 1
        assert "GREEN_CCD" in l1.extensions
        assert "GREEN_VAR" in l1.extensions
        assert "RED_CCD" in l1.extensions
        assert "RED_VAR" in l1.extensions
        assert l1.data["GREEN_CCD"].shape == (32, 32)

    def test_required_extensions_created(self):
        l1 = KPF1()
        assert "PRIMARY" in l1.extensions
        assert "GREEN_CCD" in l1.extensions
        assert "GREEN_VAR" in l1.extensions
        assert "RED_CCD" in l1.extensions
        assert "RED_VAR" in l1.extensions
        assert "RECEIPT" in l1.extensions

    def test_round_trip(self, synthetic_l1_file, tmp_path):
        l1 = KPF1.from_fits(synthetic_l1_file)
        original_green = l1.data["GREEN_CCD"].copy()

        out_fn = str(tmp_path / "roundtrip_l1.fits")
        l1.to_fits(out_fn)

        l1_reread = KPF1.from_fits(out_fn)
        np.testing.assert_array_almost_equal(
            l1_reread.data["GREEN_CCD"], original_green
        )

    def test_receipt_tracking(self, synthetic_l1_file, tmp_path):
        l1 = KPF1.from_fits(synthetic_l1_file)
        assert len(l1.receipt) >= 1

        out_fn = str(tmp_path / "receipt_l1.fits")
        l1.to_fits(out_fn)
        assert "to_fits" in l1.receipt["Module_Name"].values

    def test_generate_filename(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        fn = l1.generate_standard_filename()
        assert fn.startswith("kpf_L1_")
        assert fn.endswith(".fits")

    def test_datalvl_header(self, synthetic_l1_file, tmp_path):
        l1 = KPF1.from_fits(synthetic_l1_file)
        out_fn = str(tmp_path / "datalvl_test.fits")
        l1.to_fits(out_fn)

        with fits.open(out_fn) as hdul:
            assert hdul["PRIMARY"].header["DATALVL"] == "L1"

    def test_warns_on_unknown_extension(self, tmp_path):
        fn = str(tmp_path / "unknown_ext.fits")
        primary = fits.PrimaryHDU()
        primary.header["DATE-OBS"] = "2024-01-13T00:00:00"
        weird = fits.ImageHDU(data=np.zeros((4, 4)))
        weird.name = "WEIRD_EXTENSION"
        hdul = fits.HDUList([primary, weird])
        hdul.writeto(fn, overwrite=True)
        hdul.close()

        with pytest.warns(UserWarning, match="Non-standard extension"):
            KPF1.from_fits(fn)


class TestToL1:
    def test_to_l1_creates_kpf1(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert l1.level == 1
        assert isinstance(l1, KPF1)

    def test_to_l1_copies_primary_header(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert l1.headers["PRIMARY"]["INSTRUME"] == "KPF"
        assert l1.headers["PRIMARY"]["DATE-OBS"] == "2024-01-13T10:26:56"
        assert l1.headers["PRIMARY"]["OBJECT"] == "HD_10700"

    def test_to_l1_copies_passthrough_extensions(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        # CA_HK and TELEMETRY were in the synthetic file
        assert "CA_HK" in l1.extensions
        assert "TELEMETRY" in l1.extensions
        np.testing.assert_array_equal(l1.data["CA_HK"], l0.data["CA_HK"])

    def test_to_l1_skips_missing_extensions(self, synthetic_l0_minimal):
        l0 = KPF0.from_fits(synthetic_l0_minimal)
        l1 = l0.to_kpf1()
        assert "CA_HK" not in l1.extensions
        assert "TELEMETRY" not in l1.extensions

    def test_to_l1_leaves_ccd_empty(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert "GREEN_CCD" in l1.extensions
        assert "RED_CCD" in l1.extensions
        # Extensions exist but data is empty (not populated yet)
        assert len(l1.data["GREEN_CCD"]) == 0
        assert len(l1.data["RED_CCD"]) == 0

    def test_to_l1_carries_receipt(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert len(l1.receipt) >= 2  # from_fits + to_l1
        assert "to_l1" in l1.receipt["Module_Name"].values

    def test_to_l1_copies_obs_id(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert l1.obs_id == "KP.20240113.23249.10"

    def test_to_l1_drops_amp_extensions(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert "GREEN_AMP1" not in l1.extensions
        assert "GREEN_AMP2" not in l1.extensions
        assert "RED_AMP1" not in l1.extensions


class TestToKPF2:
    def test_to_kpf2_creates_kpf2(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        kpf2 = l1.to_kpf2()
        assert kpf2.level == 2
        assert isinstance(kpf2, KPF2)

    def test_to_kpf2_maps_header_keywords(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        l1.headers["PRIMARY"]["ELAPSED"] = 300.0
        l1.headers["PRIMARY"]["IMTYPE"] = "Object"
        l1.headers["PRIMARY"]["GROBSERV"] = "Smith"
        kpf2 = l1.to_kpf2()
        assert kpf2.headers["PRIMARY"]["EXPTIME"] == 300.0
        assert kpf2.headers["PRIMARY"]["OBSTYPE"] == "Object"
        assert kpf2.headers["PRIMARY"]["OBSERVER"] == "Smith"

    def test_to_kpf2_copies_same_name_keywords(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        kpf2 = l1.to_kpf2()
        assert kpf2.headers["PRIMARY"]["INSTRUME"] == "KPF"
        assert kpf2.headers["PRIMARY"]["DATE-OBS"] == "2024-01-13T10:26:56"

    def test_to_kpf2_sets_defaults(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        kpf2 = l1.to_kpf2()
        datalvl = kpf2.headers["PRIMARY"]["DATALVL"]
        assert (datalvl[0] if isinstance(datalvl, tuple) else datalvl) == "L2"
        origin = kpf2.headers["PRIMARY"].get("ORIGIN")
        assert origin is not None

    def test_to_kpf2_sets_instrument_header(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        kpf2 = l1.to_kpf2()
        assert "INSTRUME" in kpf2.headers["INSTRUMENT_HEADER"]
        assert kpf2.headers["INSTRUMENT_HEADER"]["INSTRUME"] == "KPF"

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
        origid = kpf2.headers["PRIMARY"]["ORIGID"]
        assert (origid[0] if isinstance(origid, tuple) else origid) == "KP.20240113.23249.10"


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
        test_data = np.random.random((10, 100)).astype(np.float64)
        kpf2.set_data("SCI2_FLUX", test_data)
        np.testing.assert_array_equal(kpf2.data["TRACE3_FLUX"], test_data)

    def test_all_trace_aliases_registered(self):
        kpf2 = KPF2()
        # Check all 5 fibers x 4 suffixes = 20 aliases
        for fiber, trace in [("CAL", 1), ("SCI1", 2), ("SCI2", 3), ("SCI3", 4), ("SKY", 5)]:
            for suffix in ["FLUX", "WAVE", "VAR", "BLAZE"]:
                alias = f"{fiber}_{suffix}"
                canonical = f"TRACE{trace}_{suffix}"
                assert alias in kpf2.extensions, f"{alias} not found"
                assert kpf2.data[alias] is kpf2.data[canonical]

    def test_chip_prefix_access(self):
        """Test GREEN_/RED_ prefix returns correct slices of concatenated trace."""
        kpf2 = KPF2()
        n_pix = 100
        trace_data = np.random.random((NORDER_GREEN + NORDER_RED, n_pix))
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
        green_data = np.arange(NORDER_GREEN * n_pix, dtype=np.float32).reshape(NORDER_GREEN, n_pix)

        kpf2.set_data("GREEN_SCI2_FLUX", green_data)
        np.testing.assert_array_equal(kpf2.data["GREEN_SCI2_FLUX"], green_data)


class TestToKPF4:
    def test_to_kpf4_creates_kpf4(self):
        kpf2 = KPF2()
        kpf4 = kpf2.to_kpf4()
        assert isinstance(kpf4, KPF4)
        assert kpf4.level == 4

    def test_to_kpf4_forwards_primary_header(self):
        kpf2 = KPF2()
        kpf2.headers["PRIMARY"]["INSTRUME"] = "KPF"
        kpf2.headers["PRIMARY"]["OBJECT"] = "HD_10700"
        kpf4 = kpf2.to_kpf4()
        assert kpf4.headers["PRIMARY"]["INSTRUME"] == "KPF"
        assert kpf4.headers["PRIMARY"]["OBJECT"] == "HD_10700"

    def test_to_kpf4_sets_datalvl(self):
        kpf2 = KPF2()
        kpf4 = kpf2.to_kpf4()
        datalvl = kpf4.headers["PRIMARY"]["DATALVL"]
        assert (datalvl[0] if isinstance(datalvl, tuple) else datalvl) == "L4"

    def test_to_kpf4_carries_receipt(self):
        kpf2 = KPF2()
        kpf4 = kpf2.to_kpf4()
        assert "to_kpf4" in kpf4.receipt["Module_Name"].values

    def test_to_kpf4_leaves_rv_empty(self):
        kpf2 = KPF2()
        kpf4 = kpf2.to_kpf4()
        assert "RV1" in kpf4.extensions
        assert len(kpf4.data["RV1"]) == 0


class TestKPF4:
    def test_kpf4_inherits_rv4(self):
        from rvdata.core.models.level4 import RV4
        kpf4 = KPF4()
        assert isinstance(kpf4, RV4)
        assert kpf4.level == 4

    def test_rv_alias(self):
        kpf4 = KPF4()
        assert "RV" in kpf4.extensions
        assert kpf4.data["RV"] is kpf4.data["RV1"]


class TestImports:
    def test_data_models_import(self):
        from kpfpipe.data_models import KPF0, KPF1, KPF2, KPF4
        assert KPF0 is not None
        assert KPF1 is not None
        assert KPF2 is not None
        assert KPF4 is not None

    def test_rvdata_import(self):
        from rvdata.core.models.level2 import RV2
        from rvdata.core.models.level4 import RV4
        assert RV2 is not None
        assert RV4 is not None
