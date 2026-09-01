"""Tests for the KPF0 (raw CCD / L0) data model, on synthetic FITS fixtures.

``KPF0.standardize_headers`` is the single WMKO-native -> EPRV conversion
site and runs at every raw-L0 load, so the shape of the EPRV PRIMARY is pinned
here: what the registry seeds, what the header map fills, what stays blank, and
that the whole thing is idempotent and fail-loud.
"""

import importlib.metadata
import logging
import os
import re

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from kpfpipe import DETECTOR
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.utils.astro import KECK_LOCATION

from ._catalog import catalog_record_table
from ._data_models import standardized_l0, write_minimal_l0
from ._dtype_policy import assert_not_float64
from ._eprv import kpf_table

# synthetic_l0_file and synthetic_l0_minimal fixtures live in tests/conftest.py

# The families that must carry a card for every trace, on every frame.
_TRACE_FAMILIES = (
    "TRACE",
    "CLSRC",
    "CSRC",
    "CID",
    "CRA",
    "CDEC",
    "CEQNX",
    "CEPCH",
    "CPLX",
    "CPMR",
    "CPMD",
    "CRV",
    "CZ",
    "CCLR",
    "CCLRN",
)


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
    """standardize_headers stamps the DRP provenance cards onto PRIMARY,
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
        assert prim.get("DRPSTATU") == "Standardize Headers module complete"

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

    def test_program_and_observer_fall_back_to_their_copies(self, tmp_path):
        fn = write_minimal_l0(
            tmp_path / "KP.20240113.00004.00.fits",
            primary_cards={
                "PROGNAME": None,
                "GRPROGNA": "K123",
                "RDPROGNA": "K123",
                "GROBSERV": "Isaacson",  # RDOBSERV absent: one copy is enough
            },
        )
        prim = standardized_l0(fn).headers["PRIMARY"]
        assert prim["PROGID"] == "K123"
        assert prim["PROGRAM"] == "K123"
        assert prim["OBSERVER"] == "Isaacson"

    def test_disagreeing_copies_default_to_unknown_and_warn(self, caplog, tmp_path):
        fn = write_minimal_l0(
            tmp_path / "KP.20240113.00005.00.fits",
            primary_cards={"PROGNAME": None, "GRPROGNA": "K123", "RDPROGNA": "K456"},
        )
        with caplog.at_level(logging.WARNING):
            prim = standardized_l0(fn).headers["PRIMARY"]
        assert prim["PROGRAM"] == "UNKNOWN"
        assert "GRPROGNA/RDPROGNA disagree" in caplog.text

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
        l0.standardize_headers()
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
    product -- so the oracle is KPF's own registry. A bare KPF0 is deliberately
    unseeded, because ``_read`` replaces PRIMARY wholesale;
    standardize_headers is what puts the EPRV skeleton on it.
    """

    def test_standardization_stamps_the_whole_seed(self, tmp_path):
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


class TestPrimarySeed:
    """The registry stamps the whole PRIMARY skeleton, typed and commented.

    Nothing is filtered on REQUIRED: that column is a compliance label, so every
    registered PRIMARY keyword gets a card, blank where nothing supplies a value.
    """

    def test_seed_covers_every_registered_primary_keyword(self):
        l0 = KPF0()
        l0._seed_primary()
        expected = set(l0.keyword_registry.primary_seed("L0"))
        assert expected <= set(l0.headers["PRIMARY"])

    def test_seed_is_typed_with_comments(self):
        prim = KPF1().headers["PRIMARY"]
        # UInt parses to a real Python int, not a string.
        assert prim["NUMTRACE"] == 5 and isinstance(prim["NUMTRACE"], int)
        # The comment comes from the registry Description, not the caller.
        assert prim.comments["INSTRUME"] == "Instrument name"
        # Description [Units]; a unit-less row carries the description alone.
        assert prim.comments["EXPTIME"] == "Exposure time [s]"

    def test_datalvl_is_the_model_level(self):
        assert KPF1().headers["PRIMARY"]["DATALVL"] == "L1"


class TestStandardizedPrimary:
    """What a standardized L0 PRIMARY holds."""

    def test_natives_are_converted_to_eprv_names(self, synthetic_l0_file):
        prim = standardized_l0(synthetic_l0_file).headers["PRIMARY"]
        assert prim["OBSTYPE"] == "Object"  # IMTYPE -> OBSTYPE
        assert prim["EXPTIME"] == 300.0  # ELAPSED -> EXPTIME
        assert prim["INSTRUME"] == "KPF"
        assert prim["OBJECT"] == "10700"
        # Raw native names must not remain on the EPRV PRIMARY.
        for native in ("IMTYPE", "ELAPSED", "GROBSERV", "OFNAME", "PROGNAME"):
            assert native not in prim

    def test_the_fill_is_typed(self, synthetic_l0_file):
        prim = standardized_l0(synthetic_l0_file).headers["PRIMARY"]
        assert prim["NUMTRACE"] == 5 and isinstance(prim["NUMTRACE"], int)
        assert isinstance(prim["OBSALT"], float)
        assert prim.comments["NUMTRACE"]  # comment survived the typed overlay

    def test_observatory_cards_come_from_config(self, synthetic_l0_file):
        # Geodetic and geocentric views of the one KECK_LOCATION, to the mm.
        prim = standardized_l0(synthetic_l0_file).headers["PRIMARY"]
        assert prim["GEOSYS"] == KECK_LOCATION.ellipsoid
        assert prim["OBSLON"] == pytest.approx(KECK_LOCATION.lon.deg)
        assert prim["OBSLAT"] == pytest.approx(KECK_LOCATION.lat.deg)
        assert prim["OBSALT"] == pytest.approx(KECK_LOCATION.height.to_value("m"))
        for axis in ("X", "Y", "Z"):
            assert prim[f"OBSGEO-{axis}"] == pytest.approx(
                getattr(KECK_LOCATION, axis.lower()).to_value("m"), abs=1e-3
            )

    def test_value_bugs_are_fixed(self, synthetic_l0_file):
        l0 = standardized_l0(synthetic_l0_file)
        prim = l0.headers["PRIMARY"]
        assert prim["NUMORDER"] == 67  # 35 green + 32 red, not 65
        # JD_UTC is the full Julian Date of DATE-OBS (not a raw MJD).
        assert prim["JD_UTC"] == pytest.approx(2460322.93537, abs=1e-3)
        version = importlib.metadata.version("kpfpipe")
        assert prim["DRPTAG"] == version  # EPRV version keyword stays on PRIMARY
        # DRPVERNO is the WMKO counterpart of DRPTAG; both sit on PRIMARY.
        assert prim["DRPVERNO"] == version

    def test_compliance_tags_from_the_rvdata_pin(self, synthetic_l0_file):
        prim = standardized_l0(synthetic_l0_file).headers["PRIMARY"]
        version = importlib.metadata.version("rv-data-standard")
        assert prim["EPRVTAG"] == f"v{version}"
        assert re.fullmatch(r"EPRVSTANDARD\d{4}\.\d{2}", prim["VOCLASS"])

    def test_jd_utc_is_the_only_non_tabular_value(self, synthetic_l0_file):
        # Everything else is a straight application of header-map.csv; the
        # MJD -> JD epoch transform is the one thing the table cannot express.
        l0 = KPF0.from_fits(synthetic_l0_file)
        native = l0.as_fits_header(l0.headers["PRIMARY"])
        l0.standardize_headers()
        assert l0.headers["PRIMARY"]["JD_UTC"] != native["MJD-OBS"]
        assert l0.headers["PRIMARY"]["JD_UTC"] == pytest.approx(
            float(native["MJD-OBS"]) + 2400000.5
        )

    def test_unmapped_natives_do_not_reach_primary(self, tmp_path):
        # A native the map does not name stays in INSTRUMENT_HEADER: the fill
        # loop writes only mapped EPRV_KEYs, so nothing leaks past it.
        fn = str(tmp_path / "KP.20240113.00009.00.fits")
        primary = fits.PrimaryHDU()
        primary.header["INSTRUME"] = "KPF"
        primary.header["IMTYPE"] = "Object"
        primary.header["OFNAME"] = "KP.20240113.00009.00.fits"
        primary.header["PROGNAME"] = "K123"
        primary.header["MJD-OBS"] = 60310.0
        primary.header["PARANTEL"] = 108.03
        fits.HDUList([primary]).writeto(fn)

        l0 = standardized_l0(fn)
        assert "PARANTEL" not in l0.headers["PRIMARY"]
        assert "PARANG" not in l0.headers["PRIMARY"]
        assert l0.headers["INSTRUMENT_HEADER"]["PARANTEL"] == 108.03

    def test_instrument_header_is_the_verbatim_native(self, synthetic_l0_file):
        l0 = standardized_l0(synthetic_l0_file)
        native = l0.headers["INSTRUMENT_HEADER"]
        assert native["IMTYPE"] == "Object"
        assert native["ELAPSED"] == 300.0
        assert native["GROBSERV"] == "Smith"
        # DRP provenance lives on RECEIPT, never on the raw PRIMARY snapshot.
        assert "DRPVERNO" not in native
        assert "DRPSTATU" not in native

    def test_instrument_era_is_stamped(self):
        l0 = self._dated_l0(60310.0)  # 2024-01-01, era 1.0
        l0.standardize_headers()
        assert l0.headers["PRIMARY"]["INSTERA"] == "1.0"

    def test_undated_frame_is_rejected(self):
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "Bias"
        with pytest.raises(ValueError, match="No KPF instrument era covers NaT"):
            l0.standardize_headers()

    def test_frame_between_eras_is_rejected(self):
        l0 = self._dated_l0(60355.0)  # 2024-02-15, eras 1.5 -> 2.0 gap
        with pytest.raises(ValueError, match="No KPF instrument era covers"):
            l0.standardize_headers()

    @staticmethod
    def _dated_l0(mjd):
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "Bias"
        l0.headers["PRIMARY"]["MJD-OBS"] = mjd
        return l0


class TestObservingMode:
    """OBSMODE and ISSOLAR from the mapped OBSTYPE, CLSRC# from the TRACE# cards.

    KPF has one optical configuration, so OBSMODE restates OBSTYPE and ISSOLAR
    as sci/cal/solar rather than naming a configuration of its own.
    """

    @staticmethod
    def _standardize(imtype, **native):
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = imtype
        l0.headers["PRIMARY"]["MJD-OBS"] = 60310.0
        for key, value in native.items():
            l0.headers["PRIMARY"][key] = value
        return l0.standardize_headers().headers["PRIMARY"]

    def test_object_frame_is_sci(self, synthetic_l0_file):
        prim = standardized_l0(synthetic_l0_file).headers["PRIMARY"]
        assert prim["OBSTYPE"] == "Object"
        assert prim["OBSMODE"] == "sci"
        assert prim["ISSOLAR"] is False

    @pytest.mark.parametrize(
        "imtype", ("Bias", "Dark", "Flatlamp", "Arclamp", "Etalon")
    )
    def test_calibration_frames_are_cal(self, imtype):
        prim = self._standardize(imtype)
        assert prim["OBSMODE"] == "cal"
        assert prim["ISSOLAR"] is False

    @pytest.mark.parametrize("key", ("OBJECT", "TARGNAME"))
    def test_socal_frame_is_solar(self, key):
        prim = self._standardize("Object", **{key: "SoCal"})
        assert prim["OBSMODE"] == "solar"
        assert prim["ISSOLAR"] is True

    def test_socal_match_is_case_insensitive(self):
        assert self._standardize("Object", OBJECT="socal")["ISSOLAR"] is True

    def test_unrecognized_imtype_is_rejected(self):
        with pytest.raises(ValueError, match="IMTYPE 'Sky'"):
            self._standardize("Sky")

    def test_clsrc_normalizes_each_trace(self):
        prim = self._standardize(
            "Object",
            **{"SKY-OBJ": "Sky", "SCI-OBJ": "Target", "CAL-OBJ": "Th_gold"},
        )
        assert prim["CLSRC1"] == "Sky"
        assert [prim[f"CLSRC{trace}"] for trace in (2, 3, 4)] == ["Target"] * 3
        assert prim["CLSRC5"] == "ThAr"

    @pytest.mark.parametrize(
        "cal_obj, source",
        [
            ("Th_gold", "ThAr"),
            ("Th_daily", "ThAr"),
            ("LFCFiber", "LFC"),
            ("EtalonFiber", "Etalon"),
            ("BrdbandFiber", "BrdbandFiber"),
        ],
    )
    def test_clsrc_vocabulary(self, cal_obj, source):
        prim = self._standardize("Arclamp", **{"CAL-OBJ": cal_obj})
        assert prim["CLSRC5"] == source

    def test_clsrc_is_blank_when_the_trace_names_no_source(self):
        assert self._standardize("Bias")["CLSRC5"] is None


class TestFiveTraceShape:
    """Verification step 5 (L0/L1): every ``#`` family carries a card for every
    trace, on every frame, regardless of IMTYPE or which fibers were illuminated.

    Presence, not value: what must not vary between frames is the *set of card
    names*, so an absent card can never be mistaken for "no value".
    """

    @staticmethod
    def _expected():
        return {
            f"{base}{i}"
            for base in _TRACE_FAMILIES
            for i in range(1, DETECTOR["numtrace"] + 1)
        }

    @pytest.mark.parametrize(
        "imtype", ["Object", "Bias", "Dark", "Flatlamp", "Arclamp"]
    )
    def test_every_family_member_is_present_at_l0(self, imtype):
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = imtype
        l0.headers["PRIMARY"]["MJD-OBS"] = 60310.0
        l0.standardize_headers()
        assert self._expected() <= set(l0.headers["PRIMARY"])

    @pytest.mark.parametrize("imtype", ["Object", "Bias", "Arclamp"])
    def test_every_family_member_survives_to_l1(self, imtype):
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = imtype
        l0.headers["PRIMARY"]["MJD-OBS"] = 60310.0
        l0.standardize_headers()
        assert self._expected() <= set(l0.to_kpf1().headers["PRIMARY"])

    def test_the_card_set_does_not_vary_with_imtype(self):
        def cards(imtype):
            l0 = KPF0()
            l0.headers["PRIMARY"]["IMTYPE"] = imtype
            l0.headers["PRIMARY"]["MJD-OBS"] = 60310.0
            l0.standardize_headers()
            return set(l0.headers["PRIMARY"])

        assert cards("Object") == cards("Bias") == cards("Arclamp")


class TestIdempotencyAndGate:
    """Verification step 7: idempotent, fail-loud, and round-trip stable."""

    def test_standardized_is_false_on_a_fresh_object(self):
        # A fresh KPF0 has a zero-column receipt, which is why the property needs
        # its `not r.empty` guard rather than indexing FUNCTION directly.
        assert KPF0().standardized is False

    def test_standardized_is_false_on_a_raw_read(self, synthetic_l0_file):
        assert KPF0.from_fits(synthetic_l0_file).standardized is False

    def test_standardized_is_true_after_the_module(self, synthetic_l0_file):
        assert standardized_l0(synthetic_l0_file).standardized is True

    def test_standardized_survives_a_round_trip(self, synthetic_l0_file, tmp_path):
        l0 = standardized_l0(synthetic_l0_file)
        out = str(tmp_path / "KP.20240113.23249.10.fits")
        l0.to_fits(out)
        back = KPF0.from_fits(out)
        assert back.standardized is True
        assert back.headers["PRIMARY"]["INSTRUME"] == "KPF"
        assert back.headers["INSTRUMENT_HEADER"]["IMTYPE"] == "Object"

    def test_running_twice_is_a_no_op(self, synthetic_l0_file):
        l0 = standardized_l0(synthetic_l0_file)
        before = dict(l0.headers["PRIMARY"])
        native_before = dict(l0.headers["INSTRUMENT_HEADER"])
        receipts_before = list(l0.receipt["FUNCTION"])

        l0.standardize_headers()

        assert dict(l0.headers["PRIMARY"]) == before
        assert dict(l0.headers["INSTRUMENT_HEADER"]) == native_before
        assert list(l0.receipt["FUNCTION"]) == receipts_before

    def test_to_kpf1_rejects_an_unstandardized_l0(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        with pytest.raises(ValueError, match="call standardize_headers"):
            l0.to_kpf1()

    def test_a_receipt_row_is_written(self, synthetic_l0_file):
        l0 = standardized_l0(synthetic_l0_file)
        assert "standardize_headers" in l0.receipt["FUNCTION"].values
        # Not an internal receipt, so it advances DRPSTATU like any module.
        assert (
            l0.headers["PRIMARY"]["DRPSTATU"] == "Standardize Headers module complete"
        )


class TestKPF0TcsPointing:
    """The pointing cards KPF0 derives from the TCS cards the header map fills."""

    # The TCS cards of the 2024-04-05 science frame, and the PARANG that frame
    # carries at mid-exposure -- an independent value to check the formula against.
    _POINTING = {
        "EL": 49.66,
        "DEC": "+40:25:50.0",
        "HA": "+02:44:48.20",
        "ELAPSED": 75.022,
    }
    _NATIVE_PARANG = 108.08

    def _pointed(self, tmp_path):
        fn = write_minimal_l0(
            tmp_path / "KP.20240113.00006.00.fits", primary_cards=self._POINTING
        )
        return standardized_l0(fn).headers["PRIMARY"]

    def test_zenith_angle_complements_the_elevation(self, tmp_path):
        assert self._pointed(tmp_path)["TZA1"] == 40.34

    def test_parallactic_angle_matches_the_native_parang(self, tmp_path):
        prim = self._pointed(tmp_path)
        # Neglecting refraction, so a tenth of a degree at this airmass.
        mid = (prim["PARST1"] + prim["PAREND1"]) / 2
        assert mid == pytest.approx(self._NATIVE_PARANG, abs=0.2)
        # Past the meridian with dec > latitude: the angle falls back from 180.
        assert prim["PARST1"] > prim["PAREND1"]

    def test_an_unpointed_frame_leaves_the_cards_blank(self, synthetic_l0_minimal):
        prim = standardized_l0(synthetic_l0_minimal).headers["PRIMARY"]
        for keyword in ("TZA1", "PARST1", "PAREND1"):
            assert prim[keyword] is None


class TestBlankCards:
    """Cards with no source stay present and blank rather than absent."""

    def test_unsourced_cards_read_back_as_none(self, synthetic_l0_file, tmp_path):
        l0 = standardized_l0(synthetic_l0_file)
        # FILENAME is blank until to_fits stamps it, so it is not a blank card.
        blank = [
            k
            for k in l0.headers["PRIMARY"]
            if l0.headers["PRIMARY"][k] is None and k != "FILENAME"
        ]
        assert blank  # the frame does not supply everything

        out = str(tmp_path / "KP.20240113.23249.10.fits")
        l0.to_fits(out)
        prim = KPF0.from_fits(out).headers["PRIMARY"]
        for keyword in blank:
            assert keyword in prim
            assert not prim[keyword]

    def test_every_card_carries_its_registry_comment(self, synthetic_l0_file):
        l0 = standardized_l0(synthetic_l0_file)
        registry = l0.keyword_registry
        prim = l0.headers["PRIMARY"]
        for keyword in prim:
            if registry.is_structural(keyword):
                continue
            assert prim.comments[keyword] == registry.comment_for(keyword, "PRIMARY")
