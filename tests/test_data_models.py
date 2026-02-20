"""
Tests for KPF data models (KPF0 and KPF1).

Uses synthetic FITS fixtures — no real KPF data needed.
"""

import os
import tempfile

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1


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
        l1 = l0.to_l1()
        assert l1.level == 1
        assert isinstance(l1, KPF1)

    def test_to_l1_copies_primary_header(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_l1()
        assert l1.headers["PRIMARY"]["INSTRUME"] == "KPF"
        assert l1.headers["PRIMARY"]["DATE-OBS"] == "2024-01-13T10:26:56"
        assert l1.headers["PRIMARY"]["OBJECT"] == "HD_10700"

    def test_to_l1_copies_passthrough_extensions(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_l1()
        # CA_HK and TELEMETRY were in the synthetic file
        assert "CA_HK" in l1.extensions
        assert "TELEMETRY" in l1.extensions
        np.testing.assert_array_equal(l1.data["CA_HK"], l0.data["CA_HK"])

    def test_to_l1_skips_missing_extensions(self, synthetic_l0_minimal):
        l0 = KPF0.from_fits(synthetic_l0_minimal)
        l1 = l0.to_l1()
        assert "CA_HK" not in l1.extensions
        assert "TELEMETRY" not in l1.extensions

    def test_to_l1_leaves_ccd_empty(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_l1()
        assert "GREEN_CCD" in l1.extensions
        assert "RED_CCD" in l1.extensions
        # Extensions exist but data is empty (not populated yet)
        assert len(l1.data["GREEN_CCD"]) == 0
        assert len(l1.data["RED_CCD"]) == 0

    def test_to_l1_carries_receipt(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_l1()
        assert len(l1.receipt) >= 2  # from_fits + to_l1
        assert "to_l1" in l1.receipt["Module_Name"].values

    def test_to_l1_copies_obs_id(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_l1()
        assert l1.obs_id == "KP.20240113.23249.10"

    def test_to_l1_drops_amp_extensions(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_l1()
        assert "GREEN_AMP1" not in l1.extensions
        assert "GREEN_AMP2" not in l1.extensions
        assert "RED_AMP1" not in l1.extensions


class TestToRV2:
    def test_to_rv2_creates_rv2(self, synthetic_l1_file):
        from rvdata.core.models.level2 import RV2
        l1 = KPF1.from_fits(synthetic_l1_file)
        rv2 = l1.to_rv2()
        assert rv2.level == 2
        assert isinstance(rv2, RV2)

    def test_to_rv2_maps_header_keywords(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        # Set a KPF-native keyword that maps to an EPRV standard keyword
        l1.headers["PRIMARY"]["ELAPSED"] = 300.0
        l1.headers["PRIMARY"]["IMTYPE"] = "Object"
        l1.headers["PRIMARY"]["GROBSERV"] = "Smith"
        rv2 = l1.to_rv2()
        assert rv2.headers["PRIMARY"]["EXPTIME"] == 300.0
        assert rv2.headers["PRIMARY"]["OBSTYPE"] == "Object"
        assert rv2.headers["PRIMARY"]["OBSERVER"] == "Smith"

    def test_to_rv2_copies_same_name_keywords(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        rv2 = l1.to_rv2()
        # Keywords with same name in KPF and EPRV standard
        assert rv2.headers["PRIMARY"]["INSTRUME"] == "KPF"
        assert rv2.headers["PRIMARY"]["DATE-OBS"] == "2024-01-13T10:26:56"

    def test_to_rv2_sets_defaults(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        rv2 = l1.to_rv2()
        # DATALVL is set as (value, comment) tuple
        datalvl = rv2.headers["PRIMARY"]["DATALVL"]
        assert (datalvl[0] if isinstance(datalvl, tuple) else datalvl) == "L2"
        # ORIGIN comes from header_map.csv defaults
        origin = rv2.headers["PRIMARY"].get("ORIGIN")
        assert origin is not None

    def test_to_rv2_sets_instrument_header(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        rv2 = l1.to_rv2()
        # Full L1 PRIMARY should be stored in INSTRUMENT_HEADER
        assert "INSTRUME" in rv2.headers["INSTRUMENT_HEADER"]
        assert rv2.headers["INSTRUMENT_HEADER"]["INSTRUME"] == "KPF"

    def test_to_rv2_maps_passthrough_extensions(self, tmp_path):
        """Build an L1 with TELEMETRY and CA_HK, verify they map to RV2 extensions."""
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
        rv2 = l1.to_rv2()
        assert "TELEMETRY" in rv2.extensions
        assert "ANCILLARY_SPECTRUM" in rv2.extensions

    def test_to_rv2_leaves_traces_empty(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        rv2 = l1.to_rv2()
        assert "TRACE1_FLUX" in rv2.extensions
        # RV2 init creates empty arrays for trace extensions
        assert len(rv2.data["TRACE1_FLUX"]) == 0

    def test_to_rv2_carries_receipt(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        rv2 = l1.to_rv2()
        assert "to_rv2" in rv2.receipt["Module_Name"].values

    def test_to_rv2_sets_origid(self, tmp_path):
        """Verify obs_id is stored as ORIGID in RV2 PRIMARY."""
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
        # obs_id is set by from_fits since filename starts with KP pattern;
        # in production this would come from to_l1()
        assert l1.obs_id == "KP.20240113.23249.10"
        rv2 = l1.to_rv2()
        # ORIGID is stored as (value, comment) tuple
        origid = rv2.headers["PRIMARY"]["ORIGID"]
        assert (origid[0] if isinstance(origid, tuple) else origid) == "KP.20240113.23249.10"


class TestImports:
    def test_data_models_import(self):
        from kpfpipe.data_models import KPF0, KPF1
        assert KPF0 is not None
        assert KPF1 is not None

    def test_rvdata_import(self):
        from rvdata.core.models.level2 import RV2
        from rvdata.core.models.level4 import RV4
        assert RV2 is not None
        assert RV4 is not None
