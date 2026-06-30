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
        assert "to_fits" in l1.receipt["FUNCTION"].values

    def test_receipt_survives_roundtrip(self, synthetic_l1_file, tmp_path):
        """The processing history must be written to the FITS RECEIPT extension,
        not just held in memory. rvdata's _create_hdul serializes
        self.data["RECEIPT"], so KPFDataModel._create_hdul syncs self.receipt
        into it before writing; without that the receipt is silently lost."""
        l1 = KPF1.from_fits(synthetic_l1_file)
        l1.receipt_add_entry("image_assembly", "", "PASS")
        out_fn = str(tmp_path / "roundtrip_l1.fits")
        l1.to_fits(out_fn)

        modules = KPF1.from_fits(out_fn).receipt["FUNCTION"].values
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


class TestL1PrimarySeed:
    """KPF1.__init__ seeds the EPRV Required PRIMARY skeleton from the registry,
    mirroring rvdata's RV2.__init__ (which KPF2/KPF4 inherit but KPF1 cannot, as
    L1 is not an EPRV level). This makes KWRDPRL1 a meaningful presence check.
    """

    @staticmethod
    def _required_l1_primary():
        # The EPRV L2 PRIMARY set is tagged Level 1 (KPF requires it from L1).
        reg = KPF1.keyword_registry
        return {k for k, lvl in reg.required["PRIMARY"].items() if lvl <= 1}

    def test_fresh_kpf1_carries_eprv_skeleton(self):
        l1 = KPF1()
        present = set(l1.headers["PRIMARY"])
        assert self._required_l1_primary() <= present

    def test_seed_is_typed_with_comments(self):
        l1 = KPF1()
        prim = l1.headers["PRIMARY"]
        # Boolean/UInt EPRV datatypes parse to real Python types, not strings.
        assert prim["ISSOLAR"] is False
        # The comment comes from the EPRV description (registry), not the caller.
        assert prim.comments["INSTRUME"] == "Instrument name"

    def test_datalvl_corrected_to_l1(self):
        # The seed defaults DATALVL (EPRV Required) to "UNKNOWN"; __init__ fixes it.
        assert KPF1().headers["PRIMARY"]["DATALVL"] == "L1"

    def test_seed_matches_registry_lookup(self):
        # The 40 seeded keys are exactly the registry's eprv_primary_seed.
        assert (
            set(KPF1.keyword_registry.eprv_primary_seed) == self._required_l1_primary()
        )

    def test_converted_l1_has_all_required(self, synthetic_l0_file):
        # The original goal: a converted L1 carries every EPRV-Required PRIMARY
        # keyword, so QCL1's KWRDPRL1 presence check is meaningful (and passes).
        l1 = KPF0.from_fits(synthetic_l0_file).to_kpf1()
        assert self._required_l1_primary() <= set(l1.headers["PRIMARY"])

    def test_native_overlay_is_typed(self, synthetic_l0_file):
        # _map_header coerces native/header_map-default values to their EPRV
        # DataType (was raw strings) and preserves the seeded comment.
        prim = KPF0.from_fits(synthetic_l0_file).to_kpf1().headers["PRIMARY"]
        assert prim["NUMTRACE"] == 5 and isinstance(prim["NUMTRACE"], int)
        assert isinstance(prim["OBSALT"], float)
        assert prim["ISSOLAR"] is False
        assert prim.comments["NUMTRACE"]  # comment survived the typed overlay

    def test_master_l1_not_seeded(self):
        # KPFMasterL1 bypasses KPF1.__init__ (its __init__ chains straight to
        # KPFDataModel), so masters stay out of EPRV scope -- no EPRV science
        # skeleton. Masters carry their own minimal PRIMARY: only DATALVL ("ML1").
        prim = set(KPFMasterL1().headers["PRIMARY"])
        assert prim == {"DATALVL"}
        assert not (self._required_l1_primary() & prim) - {"DATALVL"}


class TestToKpf1:
    def test_to_kpf1_creates_kpf1(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert l1.level == 1
        assert isinstance(l1, KPF1)

    def test_to_kpf1_copies_primary_header(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert l1.headers["PRIMARY"]["INSTRUME"] == "KPF"
        assert l1.headers["PRIMARY"]["DATE-OBS"] == "2024-01-13T10:26:56"
        assert l1.headers["PRIMARY"]["OBJECT"] == "HD_10700"

    def test_to_kpf1_converts_native_to_eprv(self, synthetic_l0_file):
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

    def test_to_kpf1_preserves_raw_header_in_instrument_header(self, synthetic_l0_file):
        """INSTRUMENT_HEADER is a pure verbatim copy of the raw L0 PRIMARY."""
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert "INSTRUMENT_HEADER" in l1.extensions
        inst = l1.headers["INSTRUMENT_HEADER"]
        assert inst["IMTYPE"] == "Object"
        assert inst["ELAPSED"] == 300.0
        assert inst["GROBSERV"] == "Smith"
        assert inst["INSTRUME"] == "KPF"
        # Pipeline-stamped DRP provenance lives on RECEIPT, never on the raw
        # PRIMARY snapshot, so INSTRUMENT_HEADER stays pure instrument metadata.
        assert "DRPVERNO" not in inst
        assert "DRPSTATU" not in inst

    def test_to_kpf1_filters_non_registry_headermap_keys(self, tmp_path):
        """_map_header emits only registered keywords; header_map's non-standard
        STANDARD keys (e.g. PARANG <- PARANTEL) are dropped, not leaked onto the
        EPRV PRIMARY. The raw value survives verbatim in INSTRUMENT_HEADER."""
        import warnings

        fn = str(tmp_path / "KP.20240113.00009.00.fits")
        p = fits.PrimaryHDU()
        p.header["INSTRUME"] = "KPF"
        p.header["PARANTEL"] = 108.03  # header_map maps PARANTEL -> non-standard PARANG
        fits.HDUList([p]).writeto(fn)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # PROGID/KOAID absent -> UNKNOWN
            l1 = KPF0.from_fits(fn).to_kpf1()
        assert "PARANG" not in l1.headers["PRIMARY"]
        assert l1.headers["INSTRUMENT_HEADER"]["PARANTEL"] == 108.03

    def test_to_kpf1_fixes_value_bugs(self, synthetic_l0_file):
        """NUMORDER, JD_UTC, and the DRP version keywords are corrected/stamped."""
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        prim = l1.headers["PRIMARY"]
        assert prim.get("NUMORDER") == 67  # 35 green + 32 red, not 65
        # JD_UTC is the full Julian Date of DATE-OBS (not a raw MJD).
        assert prim.get("JD_UTC") == pytest.approx(2460322.93537, abs=1e-3)
        version = importlib.metadata.version("kpfpipe")
        assert prim.get("DRPTAG") == version  # EPRV version keyword stays on PRIMARY
        # DRPVERNO (WMKO DRP-RUN-11) now lives on RECEIPT, not PRIMARY.
        assert prim.get("DRPVERNO") is None
        assert l1.headers["RECEIPT"].get("DRPVERNO") == version

    def test_map_header_is_pure_tabular_except_jd_utc(self, synthetic_l0_file):
        """The above values are correct on the L1 PRIMARY, but _map_header itself
        no longer special-cases NUMORDER/DRPTAG/DATALVL (those ride the seed /
        model level); JD_UTC is the one transform it still performs."""
        out = KPF0.from_fits(synthetic_l0_file)._map_header()
        assert "NUMORDER" not in out  # seeded (registry _DEFAULT_OVERRIDES)
        assert "DRPTAG" not in out  # seeded (registry _DEFAULT_OVERRIDES)
        assert "DATALVL" not in out  # set by KPF1.__init__ (model level)
        assert "JD_UTC" in out  # the one per-frame transform kept in _map_header

    def test_to_kpf1_forwards_program_ids(self, synthetic_l0_file):
        """Native PROGID/KOAID stamped to the L0 RECEIPT at read carry onto the L1
        RECEIPT via the RECEIPT-header forward (no longer onto PRIMARY)."""
        l1 = KPF0.from_fits(synthetic_l0_file).to_kpf1()
        receipt = l1.headers["RECEIPT"]
        assert receipt.get("PROGID") == "K123"
        assert receipt.get("KOAID") == "KP.20240113.23249.10"
        assert "PROGID" not in l1.headers["PRIMARY"]

    def test_to_kpf1_forwards_drpstatus(self, synthetic_l0_file):
        """DRPSTATU stamped at read carries onto the L1 RECEIPT; the to_kpf1 receipt
        is denylisted, so the ingest default survives until the first real module."""
        receipt = KPF0.from_fits(synthetic_l0_file).to_kpf1().headers["RECEIPT"]
        assert receipt.get("DRPSTATU") == "File ingested into KPF-DRP"

    def test_to_kpf1_copies_passthrough_extensions(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        # CA_HK and TELEMETRY were in the synthetic file
        assert "CA_HK" in l1.extensions
        assert "TELEMETRY" in l1.extensions
        np.testing.assert_array_equal(l1.data["CA_HK"], l0.data["CA_HK"])

    def test_to_kpf1_skips_missing_extensions(self, synthetic_l0_minimal):
        l0 = KPF0.from_fits(synthetic_l0_minimal)
        l1 = l0.to_kpf1()
        assert "CA_HK" not in l1.extensions
        assert "TELEMETRY" not in l1.extensions

    def test_to_kpf1_leaves_ccd_empty(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert "GREEN_CCD" in l1.extensions
        assert "RED_CCD" in l1.extensions
        # Extensions exist but data is empty (not populated yet)
        assert len(l1.data["GREEN_CCD"]) == 0
        assert len(l1.data["RED_CCD"]) == 0

    def test_to_kpf1_carries_receipt(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert len(l1.receipt) >= 2  # from_fits + to_kpf1
        assert "to_kpf1" in l1.receipt["FUNCTION"].values

    def test_to_kpf1_copies_obs_id(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert l1.obs_id == "KP.20240113.23249.10"

    def test_to_kpf1_drops_amp_extensions(self, synthetic_l0_file):
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
        l1.receipt_add_entry("image_assembly", "", "PASS")
        status = l1.headers["RECEIPT"].get("DRPSTATU")
        assert status == "Image Assembly module complete"

    def test_master_receipt_updates_status(self, synthetic_l0_file):
        l1 = KPF0.from_fits(synthetic_l0_file).to_kpf1()
        l1.receipt_add_entry("master_bias", "", "PASS")
        status = l1.headers["RECEIPT"].get("DRPSTATU")
        assert status == "Master Bias module complete"

    def test_internal_receipts_do_not_change_status(self, synthetic_l0_file):
        l1 = KPF0.from_fits(synthetic_l0_file).to_kpf1()
        l1.receipt_add_entry("radial_velocity", "", "PASS")
        for internal in ("to_kpf2", "to_kpf4", "to_fits", "from_fits"):
            l1.receipt_add_entry(internal, "", "PASS")
        status = l1.headers["RECEIPT"].get("DRPSTATU")
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
