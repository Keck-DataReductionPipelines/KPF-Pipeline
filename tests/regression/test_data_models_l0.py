"""
Tests for the KPF0 (raw CCD / L0) data model.

Uses synthetic FITS fixtures — no real KPF data needed.
"""

import importlib.metadata
import logging
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from kpfpipe.data_models.level0 import KPF0

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
        # Real KPF0 objects always carry QUALITY_CONTROL, RECEIPT, and CATALOG_RECORD
        # extensions (RECEIPT is the registry home of the DRP-RUN provenance cards,
        # stamped at read; CATALOG_RECORD holds AstroQuery's astrometry).
        assert "QUALITY_CONTROL" in l0.extensions
        assert "RECEIPT" in l0.extensions
        assert "CATALOG_RECORD" in l0.extensions
        assert len(l0.extensions) == 4

    def test_round_trip(self, synthetic_l0_file, tmp_path):
        l0 = KPF0.from_fits(synthetic_l0_file)
        original_green = l0.data["GREEN_AMP1"].copy()

        out_fn = str(tmp_path / "KP.20240113.23249.10.fits")
        l0.to_fits(out_fn)

        l0_reread = KPF0.from_fits(out_fn)
        np.testing.assert_array_almost_equal(
            l0_reread.data["GREEN_AMP1"], original_green
        )
        assert l0_reread.headers["PRIMARY"]["INSTRUME"] == "KPF"

    def test_receipt_tracking(self, synthetic_l0_file, tmp_path):
        l0 = KPF0.from_fits(synthetic_l0_file)
        assert len(l0.receipt) >= 1
        assert "from_fits" in l0.receipt["FUNCTION"].values

        out_fn = str(tmp_path / "receipt_test.fits")
        l0.to_fits(out_fn)
        assert "to_fits" in l0.receipt["FUNCTION"].values

    def test_receipt_survives_roundtrip(self, synthetic_l0_file, tmp_path):
        """The processing history must reach the FITS RECEIPT extension, not just
        live in memory; KPFDataModel._create_hdul syncs it (and creates the
        extension, which L0's default extension set lacks)."""
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
        with pytest.raises(IOError):
            KPF0.from_fits("/nonexistent/path.fits")

    def test_non_fits_file(self, tmp_path):
        fn = str(tmp_path / "not_a_fits.txt")
        with open(fn, "w") as f:
            f.write("hello")
        with pytest.raises(IOError):
            KPF0.from_fits(fn)


class TestKPF0ErrorPaths:
    """Malformed-input, provenance, and write-path guards."""

    def _minimal_l0(self, tmp_path, extra_hdus=()):
        fn = str(tmp_path / "KP.20240113.00002.00.fits")
        primary = fits.PrimaryHDU()
        primary.header["INSTRUME"] = "KPF"
        primary.header["DATE-OBS"] = "2024-01-13T00:00:02"
        primary.header["OFNAME"] = "KP.20240113.00002.00.fits"
        hdul = fits.HDUList([primary, *extra_hdus])
        hdul.writeto(fn, overwrite=True)
        hdul.close()
        return fn

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
        # An empty receipt is seeded with the standard columns; from_fits then
        # appends its own entry.
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
        out = str(tmp_path / "nested" / "sub" / "out.fits")
        l0.to_fits(out)
        assert os.path.isfile(out)

    def test_from_fits_raises_on_unparseable_filename(self, tmp_path):
        """A filename carrying no obs_id fails loud on read (get_obs_id) rather
        than silently reading with obs_id=None."""
        fn = str(tmp_path / "not_a_kpf_name.fits")
        primary = fits.PrimaryHDU()
        primary.header["INSTRUME"] = "KPF"
        fits.HDUList([primary]).writeto(fn, overwrite=True)
        with pytest.raises(ValueError, match="No obs_id found"):
            KPF0.from_fits(fn)

    def test_to_fits_warns_on_nonconforming_name_but_writes(
        self, caplog, synthetic_l0_file, tmp_path
    ):
        """to_fits runs the warn-only filename advisory: a non-conforming output
        name warns but the write still proceeds."""
        l0 = KPF0.from_fits(synthetic_l0_file)
        out = str(tmp_path / "not_kpf_convention.fits")
        with caplog.at_level(logging.WARNING):
            l0.to_fits(out)
        assert "does not follow the KPF L0 naming" in caplog.text
        assert os.path.isfile(out)


class TestKPF0Provenance:
    """from_fits stamps the WMKO DRP-RUN provenance cards onto the L0 RECEIPT
    (their registry home; config/L0-headers.csv PopulatedBy = KPF0.from_fits).
    PRIMARY (and its INSTRUMENT_HEADER snapshot) is left raw; to_kpf1 forwards the
    RECEIPT header downstream."""

    def test_from_fits_stamps_version_and_status(self, synthetic_l0_file):
        receipt = KPF0.from_fits(synthetic_l0_file).headers["RECEIPT"]
        assert receipt.get("DRPVERNO") == importlib.metadata.version("kpfpipe")
        assert receipt.get("DRPSTATU") == "File ingested into KPF-DRP"

    def test_from_fits_maps_native_program_ids(self, synthetic_l0_file):
        """The native OFNAME/PROGNAME cards map to the KOAID/PROGID RECEIPT cards."""
        receipt = KPF0.from_fits(synthetic_l0_file).headers["RECEIPT"]
        assert receipt.get("PROGID") == "K123"
        assert receipt.get("KOAID") == "KP.20240113.23249.10.fits"

    def test_from_fits_stamps_origid_from_obs_id(self, synthetic_l0_file):
        """ORIGID is inferred from the resolved obs_id and stamped onto RECEIPT."""
        l0 = KPF0.from_fits(synthetic_l0_file)
        assert l0.obs_id == "KP.20240113.23249.10"
        assert l0.headers["RECEIPT"].get("ORIGID") == "KP.20240113.23249.10"
        assert "ORIGID" not in l0.headers["PRIMARY"]

    def test_from_fits_defaults_progid_to_unknown_and_warns(
        self, caplog, synthetic_l0_minimal
    ):
        """A file lacking PROGNAME defaults PROGID to UNKNOWN and warns; KOAID is
        still mapped from the present OFNAME."""
        with caplog.at_level(logging.WARNING):
            l0 = KPF0.from_fits(synthetic_l0_minimal)
        assert "PROGNAME absent" in caplog.text
        receipt = l0.headers["RECEIPT"]
        assert receipt.get("PROGID") == "UNKNOWN"
        assert receipt.get("KOAID") == "KP.20240113.00001.00.fits"

    def test_from_fits_raises_when_ofname_absent(self, tmp_path):
        """A file lacking OFNAME cannot set KOAID (the archive obs_id) and must
        fail loud rather than stamp a placeholder."""
        fn = str(tmp_path / "KP.20240113.00003.00.fits")
        primary = fits.PrimaryHDU()
        primary.header["INSTRUME"] = "KPF"
        primary.header["DATE-OBS"] = "2024-01-13T00:00:03"
        fits.HDUList([primary]).writeto(fn, overwrite=True)
        with pytest.raises(ValueError, match="OFNAME absent"):
            KPF0.from_fits(fn)
