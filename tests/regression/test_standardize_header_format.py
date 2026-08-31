"""Tests for the WMKO-native -> EPRV-standard L0 conversion.

``KPF0.standardize_header_format`` is the single conversion site and runs at
every raw-L0 load, so this module is where the shape of the EPRV PRIMARY is
pinned: what the registry seeds, what the header map fills, what stays blank, and
that the whole thing is idempotent and fail-loud.
"""

import importlib.metadata
import re

import pytest
from astropy.io import fits

from kpfpipe import DETECTOR
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.utils.astro import KECK_LOCATION

from ._data_models import standardized_l0

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

    def test_fresh_kpf1_carries_the_l1_skeleton(self):
        l1 = KPF1()
        assert set(l1.keyword_registry.primary_seed("L1")) <= set(l1.headers["PRIMARY"])

    def test_seed_is_typed_with_comments(self):
        prim = KPF1().headers["PRIMARY"]
        # UInt parses to a real Python int, not a string.
        assert prim["NUMTRACE"] == 5 and isinstance(prim["NUMTRACE"], int)
        # The comment comes from the registry Description, not the caller.
        assert prim.comments["INSTRUME"] == "Instrument name"
        # Description [Units], with the EPRV "N/A" placeholder suppressed.
        assert prim.comments["EXPTIME"] == "Exposure time [s]"

    def test_datalvl_is_the_model_level(self):
        assert KPF1().headers["PRIMARY"]["DATALVL"] == "L1"

    def test_master_l1_is_seeded_from_its_own_data_model(self):
        # Masters are outside EPRV scope: they seed ML1-PRIMARY-keywords.csv,
        # never the science header-map skeleton.
        master = KPFMasterL1()
        prim = set(master.headers["PRIMARY"])
        assert set(master.keyword_registry.primary_seed("ML1")) <= prim
        science = set(master.keyword_registry.primary_seed("L1"))
        assert not (science & prim) - {"DATALVL"}


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
        # Everything else is a straight application of EPRV-header-map.csv; the
        # MJD -> JD epoch transform is the one thing the table cannot express.
        l0 = KPF0.from_fits(synthetic_l0_file)
        native = l0.as_fits_header(l0.headers["PRIMARY"])
        l0.standardize_header_format()
        assert l0.headers["PRIMARY"]["JD_UTC"] != native["MJD-OBS"]
        assert l0.headers["PRIMARY"]["JD_UTC"] == pytest.approx(
            float(native["MJD-OBS"]) + 2400000.5
        )

    def test_unmapped_natives_do_not_reach_primary(self, tmp_path):
        # A native the map does not name stays in INSTRUMENT_HEADER; the map is
        # the definition of the EPRV PRIMARY set, so nothing leaks past it.
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
        l0.standardize_header_format()
        assert l0.headers["PRIMARY"]["INSTERA"] == "1.0"

    def test_undated_frame_is_rejected(self):
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "Bias"
        with pytest.raises(ValueError, match="Cannot infer the instrument era"):
            l0.standardize_header_format()

    def test_frame_between_eras_is_rejected(self):
        l0 = self._dated_l0(60355.0)  # 2024-02-15, eras 1.5 -> 2.0 gap
        with pytest.raises(ValueError, match="No KPF instrument era covers"):
            l0.standardize_header_format()

    @staticmethod
    def _dated_l0(mjd):
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "Bias"
        l0.headers["PRIMARY"]["MJD-OBS"] = mjd
        return l0


class TestObservingMode:
    """OBSMODE and ISSOLAR, both derived from the mapped OBSTYPE.

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
        return l0.standardize_header_format().headers["PRIMARY"]

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
        l0.standardize_header_format()
        assert self._expected() <= set(l0.headers["PRIMARY"])

    @pytest.mark.parametrize("imtype", ["Object", "Bias", "Arclamp"])
    def test_every_family_member_survives_to_l1(self, imtype):
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = imtype
        l0.headers["PRIMARY"]["MJD-OBS"] = 60310.0
        l0.standardize_header_format()
        assert self._expected() <= set(l0.to_kpf1().headers["PRIMARY"])

    def test_the_card_set_does_not_vary_with_imtype(self):
        def cards(imtype):
            l0 = KPF0()
            l0.headers["PRIMARY"]["IMTYPE"] = imtype
            l0.headers["PRIMARY"]["MJD-OBS"] = 60310.0
            l0.standardize_header_format()
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

        l0.standardize_header_format()

        assert dict(l0.headers["PRIMARY"]) == before
        assert dict(l0.headers["INSTRUMENT_HEADER"]) == native_before
        assert list(l0.receipt["FUNCTION"]) == receipts_before

    def test_to_kpf1_rejects_an_unstandardized_l0(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        with pytest.raises(ValueError, match="call standardize_header_format"):
            l0.to_kpf1()

    def test_a_receipt_row_is_written(self, synthetic_l0_file):
        l0 = standardized_l0(synthetic_l0_file)
        assert "standardize_header_format" in l0.receipt["FUNCTION"].values
        # Not an internal receipt, so it advances DRPSTATU like any module.
        assert (
            l0.headers["PRIMARY"]["DRPSTATU"]
            == "Standardize Header Format module complete"
        )


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
