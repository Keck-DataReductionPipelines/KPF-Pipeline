"""Tests for the KPF0 (raw CCD / L0) data model, on synthetic FITS fixtures."""

import importlib.metadata
import logging
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from kpfpipe.data_models.level0 import KPF0

from ._catalog import catalog_record_table
from ._data_models import standardized_l0, write_minimal_l0
from ._dtype_policy import assert_not_float64
from ._eprv import kpf_table

# synthetic_l0_file and synthetic_l0_minimal fixtures live in tests/conftest.py


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
        # The manifest is a complete, literal statement of the level's shape, so
        # every row exists -- empty where the file supplied nothing.
        assert "QUALITY_CONTROL" in l0.extensions
        assert "RECEIPT" in l0.extensions
        assert "CATALOG_RECORD" in l0.extensions
        assert "INSTRUMENT_HEADER" in l0.extensions
        assert len(l0.extensions) == len(kpf_table("L0-extensions"))

    def test_round_trip(self, synthetic_l0_file, tmp_path):
        l0 = KPF0.from_fits(synthetic_l0_file)
        original_green = l0.data["GREEN_AMP1"].copy()

        out_fn = str(tmp_path / "KP.20240113.23249.10.fits")
        l0.to_fits(out_fn)

        l0_reread = KPF0.from_fits(out_fn)
        np.testing.assert_array_equal(l0_reread.data["GREEN_AMP1"], original_green)
        assert l0_reread.headers["PRIMARY"]["INSTRUME"] == "KPF"

    def test_receipt_tracking(self, synthetic_l0_file, tmp_path):
        l0 = KPF0.from_fits(synthetic_l0_file)
        assert len(l0.receipt) >= 1
        assert "from_fits" in l0.receipt["FUNCTION"].values

        out_fn = str(tmp_path / "KP.20240113.00002.00.fits")
        l0.to_fits(out_fn)
        assert "to_fits" in l0.receipt["FUNCTION"].values

    def test_receipt_survives_roundtrip(self, synthetic_l0_file, tmp_path):
        # The history must reach the FITS RECEIPT extension, not just live in
        # memory; _create_hdul syncs it and creates the extension, which L0's
        # default extension set lacks.
        l0 = KPF0.from_fits(synthetic_l0_file)
        l0.receipt_add_entry("image_assembly", "", "PASS")
        out_fn = str(tmp_path / "KP.20240113.23249.10.fits")
        l0.to_fits(out_fn)

        modules = KPF0.from_fits(out_fn).receipt["FUNCTION"].values
        assert "image_assembly" in modules
        assert "to_fits" in modules

    def test_generate_filename(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        assert l0.generate_standard_filename() == "KP.20240113.23249.10.fits"

    def test_file_not_found(self):
        with pytest.raises(IOError, match="does not exist"):
            KPF0.from_fits("/nonexistent/path.fits")

    def test_non_fits_file(self, tmp_path):
        fn = str(tmp_path / "not_a_fits.txt")
        with open(fn, "w") as f:
            f.write("hello")
        with pytest.raises(IOError, match="must be FITS files"):
            KPF0.from_fits(fn)


class TestKPF0ErrorPaths:
    """Malformed-input, provenance, and write-path guards."""

    def _minimal_l0(self, tmp_path, extra_hdus=()):
        return write_minimal_l0(
            tmp_path / "KP.20240113.00002.00.fits",
            primary_cards={"DATE-OBS": "2024-01-13T00:00:02"},
            extra_hdus=extra_hdus,
        )

    def test_raises_on_unknown_extension(self, tmp_path):
        weird = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="MYSTERY")
        fn = self._minimal_l0(tmp_path, extra_hdus=[weird])
        with pytest.raises(ValueError, match="Non-standard extension 'MYSTERY'"):
            KPF0.from_fits(fn)

    def test_parses_receipt_with_entries(self, tmp_path):
        receipt = Table({"FUNCTION": ["init"], "STATUS": ["PASS"]})
        rec_hdu = fits.BinTableHDU(data=receipt, name="RECEIPT")
        fn = self._minimal_l0(tmp_path, extra_hdus=[rec_hdu])
        l0 = KPF0.from_fits(fn)
        assert "init" in l0.receipt["FUNCTION"].values
        # The receipt is reindexed to include the standard provenance columns.
        assert "COMMIT_HASH" in l0.receipt.columns

    def test_parses_empty_receipt(self, tmp_path):
        receipt = Table(names=["FUNCTION"], dtype=["U10"])  # zero rows
        rec_hdu = fits.BinTableHDU(data=receipt, name="RECEIPT")
        fn = self._minimal_l0(tmp_path, extra_hdus=[rec_hdu])
        l0 = KPF0.from_fits(fn)
        # An empty receipt is seeded with the standard columns.
        assert "CODE_RELEASE" in l0.receipt.columns
        assert "from_fits" in l0.receipt["FUNCTION"].values

    def test_generate_filename_without_obs_id_raises(self):
        with pytest.raises(ValueError, match="valid observation ID"):
            KPF0().generate_standard_filename()

    def test_to_fits_rejects_non_fits_name(self, synthetic_l0_file, tmp_path):
        l0 = KPF0.from_fits(synthetic_l0_file)
        with pytest.raises(NameError, match="must end with .fits"):
            l0.to_fits(str(tmp_path / "output.txt"))

    def test_to_fits_creates_parent_dirs(self, synthetic_l0_file, tmp_path):
        l0 = KPF0.from_fits(synthetic_l0_file)
        out = str(tmp_path / "nested" / "sub" / "KP.20240113.00002.00.fits")
        l0.to_fits(out)
        assert os.path.isfile(out)

    def test_from_fits_raises_on_unparseable_filename(self, tmp_path):
        # Fail loud rather than silently read with obs_id=None.
        fn = str(tmp_path / "not_a_kpf_name.fits")
        primary = fits.PrimaryHDU()
        primary.header["INSTRUME"] = "KPF"
        fits.HDUList([primary]).writeto(fn, overwrite=True)
        with pytest.raises(ValueError, match="No obs_id found"):
            KPF0.from_fits(fn)

    def test_to_fits_warns_on_nonconforming_name_but_writes(
        self, caplog, synthetic_l0_file, tmp_path
    ):
        l0 = KPF0.from_fits(synthetic_l0_file)
        out = str(tmp_path / "not_kpf_convention.fits")
        with caplog.at_level(logging.WARNING):
            l0.to_fits(out)
        assert "does not follow the KPF L0 naming" in caplog.text
        assert os.path.isfile(out)


class TestKPF0Provenance:
    """standardize_header_format stamps the DRP provenance cards onto PRIMARY,
    their registry home. The INSTRUMENT_HEADER snapshot is taken first and stays
    raw; to_kpf1 forwards the PRIMARY header downstream."""

    def test_a_raw_read_carries_no_provenance(self, synthetic_l0_file):
        # The stamp is part of the conversion, so an unstandardized L0 reflects
        # the file on disk: no DRP card anywhere on it.
        l0 = KPF0.from_fits(synthetic_l0_file)
        for keyword in ("DRPVERNO", "DRPSTATU", "ORIGID", "KOAID", "PROGID"):
            assert keyword not in l0.headers["PRIMARY"]
            assert keyword not in l0.headers["RECEIPT"]

    def test_standardizing_stamps_version_and_status(self, synthetic_l0_file):
        prim = standardized_l0(synthetic_l0_file).headers["PRIMARY"]
        assert prim.get("DRPVERNO") == importlib.metadata.version("kpfpipe")
        assert prim.get("DRPSTATU") == "Standardize Header Format module complete"

    def test_standardizing_maps_native_program_ids(self, synthetic_l0_file):
        # The native OFNAME/PROGNAME cards map to KOAID/PROGID on PRIMARY.
        prim = standardized_l0(synthetic_l0_file).headers["PRIMARY"]
        assert prim.get("PROGID") == "K123"
        assert prim.get("KOAID") == "KP.20240113.23249.10.fits"

    def test_standardizing_stamps_origid_from_obs_id(self, synthetic_l0_file):
        l0 = standardized_l0(synthetic_l0_file)
        assert l0.obs_id == "KP.20240113.23249.10"
        assert l0.headers["PRIMARY"].get("ORIGID") == "KP.20240113.23249.10"

    def test_progid_defaults_to_unknown_and_warns(self, caplog, synthetic_l0_minimal):
        with caplog.at_level(logging.WARNING):
            l0 = standardized_l0(synthetic_l0_minimal)
        assert "PROGNAME absent" in caplog.text
        prim = l0.headers["PRIMARY"]
        assert prim.get("PROGID") == "UNKNOWN"
        assert prim.get("KOAID") == "KP.20240113.00001.00.fits"

    def test_raises_when_ofname_absent(self, tmp_path):
        # Without OFNAME there is no KOAID (the archive obs_id), so fail loud
        # rather than stamp a placeholder.
        fn = write_minimal_l0(
            tmp_path / "KP.20240113.00003.00.fits", primary_cards={"OFNAME": None}
        )
        with pytest.raises(ValueError, match="OFNAME absent"):
            KPF0.from_fits(fn, standardize=True)


class TestKPF0CatalogRecord:
    """KPF0.read() creates the CATALOG_RECORD extension but never populates it --
    AstroQuery is its sole writer."""

    def test_read_leaves_catalog_record_empty(self, tmp_path):
        # TARG* is present and the extension still comes back empty.
        fn = str(tmp_path / "KP.20240405.00001.00.fits")
        primary = fits.PrimaryHDU()
        primary.header["INSTRUME"] = "KPF"
        primary.header["OFNAME"] = "KP.20240405.00001.00.fits"
        primary.header["PROGNAME"] = "K123"
        primary.header["OBJECT"] = "testtarget"
        primary.header["IMTYPE"] = "Object"
        primary.header["TARGRA"] = "12:00:00.00"
        primary.header["TARGDEC"] = "+40:00:00.0"
        fits.HDUList([primary]).writeto(fn, overwrite=True)

        l0 = KPF0.from_fits(fn)
        assert "CATALOG_RECORD" in l0.extensions
        assert len(l0.data["CATALOG_RECORD"]) == 0


class TestCatalogRecordMissingValues:
    """A missing CATALOG_RECORD value survives a FITS round-trip as NaN.

    Astropy's FITS reader returns a NaN float cell masked, and a masked cell is not
    NaN, so a missing-value check would read it as present; ``KPFDataModel.from_fits``
    fills those cells back. L0 is the vehicle because it is where AstroQuery writes.
    """

    @staticmethod
    def _l0_written_and_read(tmp_path, rv):
        fn = str(tmp_path / "KP.20240405.00002.00.fits")
        l0 = KPF0()
        l0.headers["PRIMARY"]["INSTRUME"] = "KPF"
        l0.headers["PRIMARY"]["OFNAME"] = "KP.20240405.00002.00.fits"
        l0.headers["PRIMARY"]["PROGNAME"] = "K123"
        l0.headers["PRIMARY"]["IMTYPE"] = "Object"
        l0.headers["PRIMARY"]["MJD-OBS"] = 60405.0
        l0.set_data("CATALOG_RECORD", catalog_record_table(rv=rv))
        l0.to_fits(fn)
        return KPF0.from_fits(fn)

    def test_missing_value_reads_back_as_nan_not_masked(self, tmp_path):
        row = self._l0_written_and_read(tmp_path, rv=None).data["CATALOG_RECORD"][0]
        assert row["rv"] is not np.ma.masked
        assert np.isnan(row["rv"])

    def test_present_value_reads_back_unchanged(self, tmp_path):
        row = self._l0_written_and_read(tmp_path, rv=10.0).data["CATALOG_RECORD"][0]
        assert row["rv"] == pytest.approx(10.0)

    def test_missing_value_leaves_catalog_card_blank(self, tmp_path):
        # The regression the normalization exists for: a masked cell defeats the
        # 'skip missing' branch in AstroQuery._catalog_primary_cards, so the C*#
        # card is written as 'nan' and the L1 write then raises.
        from kpfpipe.modules.astro_query import AstroQuery

        l0 = self._l0_written_and_read(tmp_path, rv=None)
        l0.standardize_header_format()
        cards = AstroQuery(l0)._catalog_primary_cards()
        assert "CRV2" not in cards  # skipped, so the seeded blank stands
        for keyword, value in cards.items():
            l0.set_keyword(keyword, value)

        l1 = l0.to_kpf1()
        assert not l1.headers["PRIMARY"].get("CRV2")
        assert l1.headers["PRIMARY"]["CRA2"] == "12:00:00.0000"
        l1.to_fits(str(tmp_path / "kpf_L1_20240405T000000.fits"))


class TestDtypeProvenance:
    """L0 is the raw product: whatever the detector wrote is what we keep.

    There is no single L0 dtype to assert -- amp data arrives as the instrument
    recorded it -- so the policy here is one-directional: reading a frame must
    never upcast it, because every downstream array inherits that width.
    """

    def test_amp_data_not_upcast_on_read(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        for ext in ("GREEN_AMP1", "GREEN_AMP2", "RED_AMP1", "CA_HK"):
            assert_not_float64(l0.data[ext], ext)

    def test_amp_data_not_upcast_on_round_trip(self, synthetic_l0_file, tmp_path):
        l0 = KPF0.from_fits(synthetic_l0_file)
        out_fn = str(tmp_path / "KP.20240113.23249.10.fits")
        l0.to_fits(out_fn)
        reread = KPF0.from_fits(out_fn)
        for ext in ("GREEN_AMP1", "RED_AMP1"):
            assert_not_float64(reread.data[ext], f"{ext} after round-trip")


class TestEPRVCompliance:
    """L0 against the EPRV standard.

    rvdata publishes no L0 tables -- L0 is the raw WMKO readout, not an EPRV
    product -- so the oracle is KPF's own ``EPRV-header-map.csv``. A bare KPF0 is
    deliberately unseeded, because ``_read`` replaces PRIMARY wholesale;
    standardize_header_format is what puts the EPRV skeleton on it.
    """

    def test_standardization_stamps_every_mapped_keyword(self, tmp_path):
        fn = tmp_path / "KP.20240101.00001.00.fits"
        write_minimal_l0(fn)
        l0 = standardized_l0(fn)
        registry = KPF0.keyword_registry
        assert set(registry.primary_seed("L0")) <= set(l0.headers["PRIMARY"])

    def test_the_seed_is_registered_on_primary(self):
        registry = KPF0.keyword_registry
        assert set(registry.primary_seed("L0")) <= registry.allowed["PRIMARY"]

    def test_the_model_builds_its_whole_manifest(self):
        model = KPF0()
        assert set(model.extensions) == set(kpf_table("L0-extensions")["Name"])
