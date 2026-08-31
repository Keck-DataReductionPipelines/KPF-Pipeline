"""Tests for the KPF1 (assembled FFI / L1) data model, the L0->L1 transform, and
the KPFMasterL1 calibration product. Synthetic FITS fixtures throughout -- no real
KPF data needed.
"""

import logging

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.data_models.masters.base import KPFMasterModel
from kpfpipe.utils.astro import compute_redshift

from ._catalog import SOURCES, catalog_record_table
from ._eprv import kpf_table

# synthetic_l0_file, synthetic_l0_minimal, synthetic_l1_file fixtures live in
# tests/conftest.py

# Fixtures


@pytest.fixture(scope="module")
def synthetic_masters_l1_file(tmp_path_factory):
    """Minimal synthetic Masters L1 FITS file (module-scoped: consumers only read
    it via from_fits)."""
    rng = np.random.default_rng(20240113)
    fn = str(
        tmp_path_factory.mktemp("ml1") / "KP.20240113.23249.10_master_bias_L1.fits"
    )

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

        out_fn = str(tmp_path / "kpf_L1_20240113T102656.fits")
        l1.to_fits(out_fn)

        l1_reread = KPF1.from_fits(out_fn)
        np.testing.assert_array_equal(l1_reread.data["GREEN_CCD"], original_green)

    def test_receipt_tracking(self, synthetic_l1_file, tmp_path):
        l1 = KPF1.from_fits(synthetic_l1_file)
        assert len(l1.receipt) >= 1

        out_fn = str(tmp_path / "kpf_L1_20240113T102656.fits")
        l1.to_fits(out_fn)
        assert "to_fits" in l1.receipt["FUNCTION"].values

    def test_receipt_survives_roundtrip(self, synthetic_l1_file, tmp_path):
        # rvdata's _create_hdul serializes self.data["RECEIPT"], so
        # KPFDataModel._create_hdul syncs self.receipt into it before writing;
        # without that the in-memory history is silently lost on write.
        l1 = KPF1.from_fits(synthetic_l1_file)
        l1.receipt_add_entry("image_assembly", "", "PASS")
        out_fn = str(tmp_path / "kpf_L1_20240113T102656.fits")
        l1.to_fits(out_fn)

        modules = KPF1.from_fits(out_fn).receipt["FUNCTION"].values
        assert "image_assembly" in modules
        assert "to_fits" in modules

    def test_generate_filename(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        l1.obs_id = "KP.20240113.37616.10"  # 37616 s of day = 10:26:56 UT
        assert l1.generate_standard_filename() == "kpf_L1_20240113T102656.fits"

    def test_datalvl_header(self, synthetic_l1_file, tmp_path):
        l1 = KPF1.from_fits(synthetic_l1_file)
        out_fn = str(tmp_path / "kpf_L1_20240113T102656.fits")
        l1.to_fits(out_fn)

        with fits.open(out_fn) as hdul:
            assert hdul["PRIMARY"].header["DATALVL"] == "L1"

    def test_raises_on_unknown_extension(self, tmp_path):
        fn = str(tmp_path / "unknown_ext.fits")
        primary = fits.PrimaryHDU()
        primary.header["DATE-OBS"] = "2024-01-13T00:00:00"
        weird = fits.ImageHDU(data=np.zeros((4, 4)))
        weird.name = "WEIRD_EXTENSION"
        hdul = fits.HDUList([primary, weird])
        hdul.writeto(fn, overwrite=True)
        hdul.close()

        with pytest.raises(
            ValueError, match="Non-standard extension 'WEIRD_EXTENSION'"
        ):
            KPF1.from_fits(fn)

    def test_to_fits_warns_on_nonconforming_name_but_writes(
        self, caplog, synthetic_l1_file, tmp_path
    ):
        # The filename advisory is warn-only, as at L0: the write still proceeds.
        l1 = KPF1.from_fits(synthetic_l1_file)
        out = str(tmp_path / "not_kpf_convention.fits")
        with caplog.at_level(logging.WARNING):
            out_path = l1.to_fits(out)
        assert "does not follow the KPF L1 naming" in caplog.text
        assert out_path == out


class TestToKpf1:
    """L0 -> L1 is a pure forward once standardize_header_format has run.

    The native -> EPRV conversion itself is tested in test_data_models_l0.py;
    here the subject is what to_kpf1 carries across, and the fail-loud gate on
    an L0 that skipped standardization.
    """

    @staticmethod
    def _standardized(fn):
        l0 = KPF0.from_fits(fn)
        l0.standardize_header_format()
        return l0

    def test_raw_l0_is_rejected(self, synthetic_l0_file):
        # The fail-loud gate: forwarding a raw WMKO PRIMARY onto an EPRV L1
        # PRIMARY would be silent corruption, so a mis-ordered call raises.
        l0 = KPF0.from_fits(synthetic_l0_file)
        with pytest.raises(ValueError, match="has not been standardized"):
            l0.to_kpf1()

    def test_to_kpf1_creates_kpf1(self, synthetic_l0_file):
        l1 = self._standardized(synthetic_l0_file).to_kpf1()
        assert l1.level == 1
        assert isinstance(l1, KPF1)

    def test_to_kpf1_forwards_the_eprv_primary(self, synthetic_l0_file):
        l1 = self._standardized(synthetic_l0_file).to_kpf1()
        assert l1.headers["PRIMARY"]["INSTRUME"] == "KPF"
        assert l1.headers["PRIMARY"]["DATE-OBS"] == "2024-01-13T10:26:56"
        assert l1.headers["PRIMARY"]["OBJECT"] == "10700"
        assert l1.headers["PRIMARY"]["DATALVL"] == "L1"

    # Canonical CATALOG_RECORD row (EPRV C*# format) overlaid onto the SCI cards.
    _KPF_DRP = {
        "object": "Gaia DR3 12345",
        "radec_src": "gaia",
        "plx_src": "gaia",
        "rv_src": "gaia",
        "ra": "12:00:00.0000",
        "dec": "+40:00:00.000",
        "pmra": 0.5,
        "pmdec": -0.3,
        "parallax": 50.0,
        "rv": 10.0,
        "frame": "icrs",
        "epoch": 2016.0,
        "equinox": 2000.0,
        "color": 1.23,
        "color_name": "Gaia BP-RP",
    }

    @staticmethod
    def _l0_with_catalog(record):
        # A standardized science KPF0 carrying a canonical 'kpf-drp'
        # CATALOG_RECORD row and AstroQuery's PRIMARY overlay, no network.
        # Deferred, not for a cycle: astro_query pulls in astroquery, and this
        # module would otherwise pay that import at collection in every worker
        # for the sake of one test. Mirrors tests/conftest.py's _catalog_record_hdu.
        from kpfpipe.modules.astro_query import AstroQuery

        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "Object"
        l0.headers["PRIMARY"]["MJD-OBS"] = 60310.0
        l0.standardize_header_format()
        astro_query = AstroQuery(l0)
        astro_query._write_catalog_record("kpf-drp", record)
        for keyword, value in astro_query._catalog_primary_cards().items():
            l0.set_keyword(keyword, value)
        return l0

    def test_catalog_overlay_populates_sci_cards(self):
        # Direct copy of the canonical row onto every SCI fiber (2,3,4), no conversion.
        p = self._l0_with_catalog(dict(self._KPF_DRP)).to_kpf1().headers["PRIMARY"]
        for i in (2, 3, 4):
            assert p[f"CID{i}"] == "Gaia DR3 12345"
            assert p[f"CSRC{i}"] == "gaia"
            assert p[f"CRA{i}"] == "12:00:00.0000"
            assert p[f"CDEC{i}"] == "+40:00:00.000"
            assert p[f"CPMR{i}"] == 0.5
            assert p[f"CPMD{i}"] == -0.3
            assert p[f"CPLX{i}"] == 50.0
            assert p[f"CRV{i}"] == 10.0
            assert p[f"CZ{i}"] == pytest.approx(compute_redshift(10.0 * u.km / u.s))
            assert p[f"CEPCH{i}"] == 2016.0
            assert p[f"CEQNX{i}"] == 2000.0
            assert p[f"CCLR{i}"] == 1.23
            assert p[f"CCLRN{i}"] == "Gaia BP-RP"

    def test_catalog_overlay_skips_missing_optional(self):
        # parallax/rv absent from the canonical row -> those cards keep the blank
        # the seed stamped; the coherent position block is still written.
        record = {
            **self._KPF_DRP,
            "parallax": None,
            "rv": None,
            "color": None,
            "color_name": None,
        }
        p = self._l0_with_catalog(record).to_kpf1().headers["PRIMARY"]
        assert not p.get("CPLX2")
        assert not p.get("CRV2")
        assert not p.get("CZ2")  # z derives from rv, so it is blank when rv is absent
        assert not p.get("CCLR2")
        assert not p.get("CCLRN2")
        assert p["CRA2"] == "12:00:00.0000"

    def test_frame_without_catalog_leaves_sci_cards_present_and_blank(self):
        # No AstroQuery overlay at all: the seed has still stamped every member of
        # every C*# family, so the cards are present and empty rather than absent.
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "Bias"
        l0.headers["PRIMARY"]["MJD-OBS"] = 60310.0
        l0.standardize_header_format()
        p = l0.to_kpf1().headers["PRIMARY"]
        for kw in ("CRA2", "CID2", "CSRC2", "CPMR2"):
            assert kw in p
            assert not p.get(kw)

    def test_catalog_overlay_warns_on_mixed_sources(self, caplog):
        # rv from a different catalog than the position -> WARNING at PRIMARY commit.
        record = {**self._KPF_DRP, "rv_src": "simbad"}
        with caplog.at_level(logging.WARNING):
            self._l0_with_catalog(record)
        assert "mixed sources" in caplog.text

    def test_catalog_overlay_single_source_no_warning(self, caplog):
        # All provenance labels agree (gaia) -> no mixed-source warning.
        with caplog.at_level(logging.WARNING):
            self._l0_with_catalog(dict(self._KPF_DRP))
        assert "mixed sources" not in caplog.text

    def test_to_kpf1_forwards_the_instrument_header(self, synthetic_l0_file):
        # INSTRUMENT_HEADER is a verbatim copy of the raw L0 PRIMARY, written by
        # standardize_header_format and carried across as a pass-through extension.
        l1 = self._standardized(synthetic_l0_file).to_kpf1()
        assert "INSTRUMENT_HEADER" in l1.extensions
        inst = l1.headers["INSTRUMENT_HEADER"]
        assert inst["IMTYPE"] == "Object"
        assert inst["ELAPSED"] == 300.0
        assert inst["GROBSERV"] == "Smith"
        assert inst["INSTRUME"] == "KPF"
        # The snapshot is taken before the DRP provenance is stamped, so
        # INSTRUMENT_HEADER stays pure instrument metadata.
        assert "DRPVERNO" not in inst
        assert "DRPSTATU" not in inst

    def test_to_kpf1_forwards_program_ids(self, synthetic_l0_file):
        # PROGID/KOAID are stamped onto the L0 PRIMARY when it is standardized
        # and forwarded to the L1 PRIMARY.
        l1 = self._standardized(synthetic_l0_file).to_kpf1()
        prim = l1.headers["PRIMARY"]
        assert prim.get("PROGID") == "K123"
        assert prim.get("KOAID") == "KP.20240113.23249.10.fits"

    def test_to_kpf1_forwards_drpstatus(self, synthetic_l0_file):
        # standardize_header_format is not an internal receipt, so it advances
        # DRPSTATU; to_kpf1 is denylisted and leaves it alone.
        l0 = self._standardized(synthetic_l0_file)
        prim = l0.to_kpf1().headers["PRIMARY"]
        assert prim.get("DRPSTATU") == "Standardize Header Format module complete"

    def test_to_kpf1_copies_passthrough_extensions(self, synthetic_l0_file):
        l0 = self._standardized(synthetic_l0_file)
        l1 = l0.to_kpf1()
        # CA_HK and TELEMETRY were in the synthetic file
        assert "CA_HK" in l1.extensions
        assert "TELEMETRY" in l1.extensions
        np.testing.assert_array_equal(l1.data["CA_HK"], l0.data["CA_HK"])

    def test_to_kpf1_leaves_absent_extensions_empty(self, synthetic_l0_minimal):
        # Both manifests create every row, so a pass-through the file never
        # supplied is present and empty rather than absent.
        l0 = self._standardized(synthetic_l0_minimal)
        l1 = l0.to_kpf1()
        assert len(l1.data["CA_HK"]) == 0
        assert len(l1.data["TELEMETRY"]) == 0

    def test_to_kpf1_leaves_ccd_empty(self, synthetic_l0_file):
        l1 = self._standardized(synthetic_l0_file).to_kpf1()
        assert "GREEN_CCD" in l1.extensions
        assert "RED_CCD" in l1.extensions
        # Extensions exist but ImageAssembly has not populated them yet.
        assert len(l1.data["GREEN_CCD"]) == 0
        assert len(l1.data["RED_CCD"]) == 0

    def test_to_kpf1_carries_receipt(self, synthetic_l0_file):
        l1 = self._standardized(synthetic_l0_file).to_kpf1()
        assert len(l1.receipt) >= 3  # from_fits + standardize_header_format + to_kpf1
        assert "to_kpf1" in l1.receipt["FUNCTION"].values
        assert "standardize_header_format" in l1.receipt["FUNCTION"].values

    def test_to_kpf1_copies_obs_id(self, synthetic_l0_file):
        l1 = self._standardized(synthetic_l0_file).to_kpf1()
        assert l1.obs_id == "KP.20240113.23249.10"

    def test_to_kpf1_drops_amp_extensions(self, synthetic_l0_file):
        l1 = self._standardized(synthetic_l0_file).to_kpf1()
        assert "GREEN_AMP1" not in l1.extensions
        assert "GREEN_AMP2" not in l1.extensions
        assert "RED_AMP1" not in l1.extensions


class TestCatalogRecordPassthrough:
    """AstroQuery's CATALOG_RECORD rows ride L0 -> L1 unchanged.

    The L1 -> L2 and L2 -> L4 hops have the same class in test_data_models_l{2,4}.py.
    """

    @staticmethod
    def _l0_with_catalog_table(rv=10.0):
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "Object"
        l0.headers["PRIMARY"]["MJD-OBS"] = 60310.0
        l0.set_data("CATALOG_RECORD", catalog_record_table(rv=rv))
        l0.standardize_header_format()
        return l0

    def test_rows_reach_l1(self):
        # Beyond the C*# overlay, L1 keeps every source row -- not just the merged
        # one -- so the astrometry stays auditable.
        l1 = self._l0_with_catalog_table().to_kpf1()
        assert [str(s) for s in l1.data["CATALOG_RECORD"]["source"]] == list(SOURCES)

    def test_catalog_record_roundtrip(self, tmp_path):
        # CATALOG_RECORD is registered in L1-extensions.csv, so it reads back at
        # all (an unlisted extension raises), and a missing rv comes back NaN
        # rather than masked.
        fn = str(tmp_path / "kpf_L1_20240405T000000.fits")
        self._l0_with_catalog_table(rv=None).to_kpf1().to_fits(fn)
        back = KPF1.from_fits(fn)
        assert [str(s) for s in back.data["CATALOG_RECORD"]["source"]] == list(SOURCES)
        rv = back.data["CATALOG_RECORD"][0]["rv"]
        assert rv is not np.ma.masked and np.isnan(rv)


class TestDrpStatus:
    """DRPSTATU advances to '<Module Name> module complete' via the
    receipt_add_entry override; data-model conversion/IO receipts are denylisted
    so it names the last real science/masters stage."""

    def test_module_receipt_updates_status(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l0.standardize_header_format()
        l1 = l0.to_kpf1()
        l1.receipt_add_entry("image_assembly", "", "PASS")
        status = l1.headers["PRIMARY"].get("DRPSTATU")
        assert status == "Image Assembly module complete"

    def test_master_receipt_updates_status(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l0.standardize_header_format()
        l1 = l0.to_kpf1()
        l1.receipt_add_entry("master_bias", "", "PASS")
        status = l1.headers["PRIMARY"].get("DRPSTATU")
        assert status == "Master Bias module complete"

    def test_internal_receipts_do_not_change_status(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l0.standardize_header_format()
        l1 = l0.to_kpf1()
        l1.receipt_add_entry("radial_velocity", "", "PASS")
        for internal in ("to_kpf2", "to_kpf4", "to_fits", "from_fits"):
            l1.receipt_add_entry(internal, "", "PASS")
        status = l1.headers["PRIMARY"].get("DRPSTATU")
        assert status == "Radial Velocity module complete"


class TestKPFMasterL1:
    def test_the_master_builds_its_whole_manifest(self):
        master = KPFMasterL1()
        assert set(master.extensions) == set(kpf_table("ML1-extensions")["Name"])
        # A master is not a translation of a native instrument product, so it
        # carries no verbatim instrument header.
        assert "INSTRUMENT_HEADER" not in master.extensions

    @pytest.mark.parametrize("data_model", ("ML1",))
    def test_shared_rows_agree_with_the_science_manifest(self, data_model):
        # The duplication between a master manifest and its science level's is
        # intentional -- each stays a complete spec of one product -- so a shared
        # row must not disagree, or a master would ship a GREEN_IMG unlike
        # L1's.
        master = kpf_table(f"{data_model}-extensions").set_index("Name")
        science = kpf_table("L1-extensions").set_index("Name")
        shared = set(master.index) & set(science.index)
        assert shared
        for name in shared:
            assert master.loc[name, "DataType"] == science.loc[name, "DataType"], name
            ours, theirs = master.loc[name, "BitDepth"], science.loc[name, "BitDepth"]
            assert (pd.isna(ours) and pd.isna(theirs)) or ours == theirs, name

    def test_manifest_extensions_created(self):
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
        # Every manifest row is built, DRP_CONFIG (Required=False) included.
        assert len(m.extensions) == 11
        assert "DRP_CONFIG" in m.extensions

    def test_primary_is_seeded_from_the_master_data_model(self):
        m = KPFMasterL1()
        prim = set(m.headers["PRIMARY"])
        assert set(m.keyword_registry.primary_seed("ML1")) <= prim
        assert m.headers["PRIMARY"]["DATALVL"] == "ML1"
        # Masters are outside EPRV scope, so the science skeleton must not
        # reach a built master, DATALVL aside.
        science = set(m.keyword_registry.primary_seed("L1"))
        assert not (science & prim) - {"DATALVL"}

    def test_bit_depth_from_the_master_manifest(self):
        m = KPFMasterL1()
        assert m.extension_manifest.bit_depth("ML1", "GREEN_IMG") == 32
        assert m.extension_manifest.bit_depth("ML1", "GREEN_MASK") == 8
        assert m.extension_manifest.bit_depth("ML1", "RECEIPT") is None

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

        out_fn = str(tmp_path / "KP.20240113.23249.10_master_bias_L1.fits")
        m.to_fits(out_fn)

        m2 = KPFMasterL1.from_fits(out_fn)
        np.testing.assert_array_equal(m2.data["GREEN_IMG"], original)

    def test_datalvl_header_in_fits(self, synthetic_masters_l1_file, tmp_path):
        m = KPFMasterL1.from_fits(synthetic_masters_l1_file)
        out_fn = str(tmp_path / "KP.20240113.23249.10_master_bias_L1.fits")
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

    def test_generate_filename_requires_mastype(self):
        # A master can never produce a non-compliant name: with no recorded
        # inputs / type, generate_standard_filename raises rather than falling
        # back to a KOAID-less name. MASTYPE is checked first.
        with pytest.raises(ValueError, match="MASTYPE"):
            KPFMasterL1().generate_standard_filename()

    def test_generate_filename_requires_inputs(self):
        # The second guard, reachable only once MASTYPE is set -- what a
        # partially-failed stack leaves behind.
        m = KPFMasterL1()
        m.set_keyword("MASTYPE", "bias")
        with pytest.raises(ValueError, match="INPUT_FILES"):
            m.generate_standard_filename()

    def test_set_input_files(self):
        m = KPFMasterL1()
        files = ["/data/a.fits", "/data/b.fits", "/data/c.fits"]
        m.set_input_files(files, "bias")
        assert m.data["INPUT_FILES"]["FILENAME"].tolist() == files
        assert m.headers["PRIMARY"]["MASTYPE"] == "bias"


class TestEPRVCompliance:
    """L1 against the EPRV standard.

    rvdata publishes no L1 tables either -- the assembled FFI is a KPF stage
    between the raw readout and the EPRV L2 -- so, as at L0, the oracle is KPF's
    own registry. Unlike L0, an L1 seeds its PRIMARY at construction, so the
    cards are asserted on a bare model.
    """

    def test_the_primary_carries_the_whole_seed(self):
        registry = KPF1.keyword_registry
        assert set(registry.primary_seed("L1")) <= set(KPF1().headers["PRIMARY"])

    def test_the_seed_is_registered_on_primary(self):
        registry = KPF1.keyword_registry
        assert set(registry.primary_seed("L1")) <= registry.allowed["PRIMARY"]

    def test_the_model_builds_its_whole_manifest(self):
        model = KPF1()
        assert set(model.extensions) == set(kpf_table("L1-extensions")["Name"])
