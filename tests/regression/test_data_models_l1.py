"""
Tests for the KPF1 (assembled FFI / L1) data model, the L0->L1 transform,
and the KPFMasterL1 calibration product.

Uses synthetic FITS fixtures — no real KPF data needed.
"""

import importlib.metadata

import numpy as np
import pytest
from astropy.io import fits

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.data_models.masters.base import KPFMasterModel

# synthetic_l0_file, synthetic_l0_minimal, synthetic_l1_file fixtures live in
# tests/conftest.py

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_masters_l1_file(tmp_path):
    """Create a minimal synthetic Masters L1 FITS file."""
    rng = np.random.default_rng(20240113)
    fn = str(tmp_path / "KP.20240113.23249.10_master_bias_L1.fits")

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-01-13T10:26:56"
    primary.header["DATALVL"] = "ML1"

    def img(name):
        return fits.ImageHDU(data=rng.random((32, 32)).astype(np.float32), name=name)

    def mask(name):
        return fits.ImageHDU(data=np.ones((32, 32), dtype=np.uint8), name=name)

    hdul = fits.HDUList(
        [
            primary,
            img("GREEN_IMG"),
            img("GREEN_SNR"),
            mask("GREEN_MASK"),
            img("RED_IMG"),
            img("RED_SNR"),
            mask("RED_MASK"),
        ]
    )
    hdul.writeto(fn, overwrite=True)
    hdul.close()

    return fn


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

    def test_receipt_survives_roundtrip(self, synthetic_l1_file, tmp_path):
        """The processing history must be written to the FITS RECEIPT extension,
        not just held in memory. rvdata's _create_hdul serializes
        self.data["RECEIPT"], so KPFDataModel._create_hdul syncs self.receipt
        into it before writing; without that the receipt is silently lost."""
        l1 = KPF1.from_fits(synthetic_l1_file)
        l1.receipt_add_entry("image_assembly", "PASS")
        out_fn = str(tmp_path / "roundtrip_l1.fits")
        l1.to_fits(out_fn)

        modules = KPF1.from_fits(out_fn).receipt["Module_Name"].values
        assert "image_assembly" in modules
        assert "to_fits" in modules

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

    def test_to_l1_converts_native_to_eprv(self, synthetic_l0_file):
        """to_kpf1 renames WMKO natives to their EPRV PRIMARY counterparts."""
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        prim = l1.headers["PRIMARY"]
        assert prim.get("OBSTYPE") == "Object"  # IMTYPE -> OBSTYPE
        assert prim.get("EXPTIME") == 300.0  # ELAPSED -> EXPTIME
        assert prim.get("OBSERVER") == "Smith"  # GROBSERV -> OBSERVER
        # Raw native names must not remain on the EPRV PRIMARY.
        assert "IMTYPE" not in prim
        assert "ELAPSED" not in prim
        assert "GROBSERV" not in prim

    def test_to_l1_preserves_raw_header_in_instrument_header(self, synthetic_l0_file):
        """INSTRUMENT_HEADER is a verbatim copy of the raw L0 PRIMARY."""
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert "INSTRUMENT_HEADER" in l1.extensions
        inst = l1.headers["INSTRUMENT_HEADER"]
        assert inst["IMTYPE"] == "Object"
        assert inst["ELAPSED"] == 300.0
        assert inst["GROBSERV"] == "Smith"
        assert inst["INSTRUME"] == "KPF"

    def test_to_l1_fixes_value_bugs(self, synthetic_l0_file):
        """NUMORDER, JD_UTC, and the DRP version keywords are corrected/stamped."""
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        prim = l1.headers["PRIMARY"]
        assert prim.get("NUMORDER") == 67  # 35 green + 32 red, not 65
        # JD_UTC is the full Julian Date of DATE-OBS (not a raw MJD).
        assert prim.get("JD_UTC") == pytest.approx(2460322.93537, abs=1e-3)
        version = importlib.metadata.version("kpfpipe")
        assert prim.get("DRPTAG") == version
        assert prim.get("DRPVERNO") == version

    def test_to_l1_stamps_native_program_ids(self, synthetic_l0_file):
        """PROGID/KOAID carry from the native L0 PRIMARY onto the L1 EPRV PRIMARY."""
        l0 = KPF0.from_fits(synthetic_l0_file)
        l0.headers["PRIMARY"]["PROGID"] = "U999"
        l0.headers["PRIMARY"]["KOAID"] = "KP.20201122.34567.89"
        prim = l0.to_kpf1().headers["PRIMARY"]
        assert prim.get("PROGID") == "U999"
        assert prim.get("KOAID") == "KP.20201122.34567.89"

    def test_to_l1_defaults_program_ids_to_unknown(self, synthetic_l0_file):
        """Absent PROGID/KOAID default to UNKNOWN (the card is always written)."""
        l0 = KPF0.from_fits(synthetic_l0_file)
        for key in ("PROGID", "KOAID"):
            if key in l0.headers["PRIMARY"]:
                del l0.headers["PRIMARY"][key]
        prim = l0.to_kpf1().headers["PRIMARY"]
        assert prim.get("PROGID") == "UNKNOWN"
        assert prim.get("KOAID") == "UNKNOWN"

    def test_to_l1_sets_drpstatus_default(self, synthetic_l0_file):
        """to_kpf1 seeds DRPSTATU; its own to_l1 receipt is denylisted, so the
        default survives until the first real module runs."""
        prim = KPF0.from_fits(synthetic_l0_file).to_kpf1().headers["PRIMARY"]
        assert prim.get("DRPSTATU") == "File ingested into KPF-DRP"

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


class TestDrpStatus:
    """DRPSTATU advances to '<Module Name> module complete' via the
    receipt_add_entry override; data-model conversion/IO receipts are denylisted
    so it names the last real science/masters stage (DRP-RUN-20)."""

    def test_module_receipt_updates_status(self, synthetic_l0_file):
        l1 = KPF0.from_fits(synthetic_l0_file).to_kpf1()
        l1.receipt_add_entry("image_assembly", "PASS")
        status = l1.headers["PRIMARY"].get("DRPSTATU")
        assert status == "Image Assembly module complete"

    def test_master_receipt_updates_status(self, synthetic_l0_file):
        l1 = KPF0.from_fits(synthetic_l0_file).to_kpf1()
        l1.receipt_add_entry("master_bias", "PASS")
        status = l1.headers["PRIMARY"].get("DRPSTATU")
        assert status == "Master Bias module complete"

    def test_internal_receipts_do_not_change_status(self, synthetic_l0_file):
        l1 = KPF0.from_fits(synthetic_l0_file).to_kpf1()
        l1.receipt_add_entry("radial_velocity", "PASS")
        for internal in ("to_kpf2", "to_kpf4", "to_fits", "from_fits"):
            l1.receipt_add_entry(internal, "PASS")
        status = l1.headers["PRIMARY"].get("DRPSTATU")
        assert status == "Radial Velocity module complete"


class TestKPFMasterL1:
    def test_required_extensions_created(self):
        m = KPFMasterL1()
        for ext in [
            "PRIMARY",
            "GREEN_IMG",
            "GREEN_SNR",
            "GREEN_MASK",
            "RED_IMG",
            "RED_SNR",
            "RED_MASK",
            "RECEIPT",
        ]:
            assert ext in m.extensions

    def test_no_science_extensions(self):
        m = KPFMasterL1()
        for ext in ["GREEN_CCD", "GREEN_VAR", "RED_CCD", "RED_VAR", "CA_HK"]:
            assert ext not in m.extensions

    def test_inherits_from_kpf1(self):
        m = KPFMasterL1()
        assert isinstance(m, KPF1)

    def test_inherits_from_kpf_master_model(self):
        m = KPFMasterL1()
        assert isinstance(m, KPFMasterModel)

    def test_class_attributes(self):
        assert KPFMasterL1._DATALVL == "ML1"

    def test_from_fits(self, synthetic_masters_l1_file):
        m = KPFMasterL1.from_fits(synthetic_masters_l1_file)
        assert "GREEN_IMG" in m.extensions
        assert "GREEN_SNR" in m.extensions
        assert "GREEN_MASK" in m.extensions
        assert "RED_IMG" in m.extensions
        assert m.data["GREEN_IMG"].shape == (32, 32)

    def test_round_trip(self, synthetic_masters_l1_file, tmp_path):
        m = KPFMasterL1.from_fits(synthetic_masters_l1_file)
        original = m.data["GREEN_IMG"].copy()

        out_fn = str(tmp_path / "roundtrip_ml1.fits")
        m.to_fits(out_fn)

        m2 = KPFMasterL1.from_fits(out_fn)
        np.testing.assert_array_almost_equal(m2.data["GREEN_IMG"], original)

    def test_datalvl_header_in_fits(self, synthetic_masters_l1_file, tmp_path):
        m = KPFMasterL1.from_fits(synthetic_masters_l1_file)
        out_fn = str(tmp_path / "datalvl_ml1.fits")
        m.to_fits(out_fn)
        with fits.open(out_fn) as hdul:
            assert hdul["PRIMARY"].header["DATALVL"] == "ML1"

    def test_generate_filename(self):
        m = KPFMasterL1()
        m.set_input_files(
            ["KP.20240113.23249.10.fits", "KP.20240113.23310.00.fits"], "bias"
        )
        assert (
            m.generate_standard_filename() == "KP.20240113.23249.10_master_bias_L1.fits"
        )

    def test_generate_filename_requires_inputs(self):
        # A master can never produce a non-compliant name: with no recorded
        # inputs / type, generate_standard_filename raises rather than falling
        # back to a KOAID-less name.
        with pytest.raises(ValueError):
            KPFMasterL1().generate_standard_filename()

    def test_no_warning_on_known_extensions(self, synthetic_masters_l1_file):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            KPFMasterL1.from_fits(synthetic_masters_l1_file)

    def test_set_input_files(self):
        m = KPFMasterL1()
        files = ["/data/a.fits", "/data/b.fits", "/data/c.fits"]
        m.set_input_files(files, "bias")
        assert m.data["INPUT_FILES"]["FILENAME"].tolist() == files
        assert m.headers["PRIMARY"]["MASTYPE"] == "bias"
