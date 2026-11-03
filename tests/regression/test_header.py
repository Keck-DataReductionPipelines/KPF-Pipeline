"""HeaderParser: frame identity, trace illumination, readout and warnings from a
native L0 PRIMARY header."""

import logging
import re
from pathlib import Path

import pytest
from astropy.io import fits

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.utils import header as header_module
from kpfpipe.utils.header import HeaderError, HeaderParser
from kpfpipe.utils.kpf import kpf_target_name

TESTDATA_L0 = Path(__file__).parent.parent / "testdata" / "L0" / "20240405"

ALL = ("open", "open", "open")
SCI = ("open", "closed", "closed")
SKY = ("closed", "open", "closed")
CAL = ("closed", "closed", "open")
HK = ("open", "closed", "open")
SHUT = ("closed", "closed", "closed")
CCDS = ("YES", "YES", "YES")


def _hdr(imtype, octagon="Home", shutters=None, cameras=CCDS, **cards):
    """A native PRIMARY as a dict: the identity cards plus regular-readout timing
    for both CCDs. ``shutters`` defaults to all shut on a bias/dark, else all
    open. ``cards`` adds or overrides any card (dashes as underscores)."""
    if shutters is None:
        shutters = SHUT if imtype in ("Bias", "Dark") else ALL
    header = {
        "OFNAME": "KP.20240405.00000.00.fits",
        "IMTYPE": imtype,
        "OCTAGON": octagon,
        "OBJECT": "target",
        "TARGNAME": "target" if imtype == "Object" else "unknown",
        "ELAPSED": 30.0,
        "EXPTIME": 29.0,
        "GRDATE": "2024-04-05T00:01:00",
        "GRDATE-E": "2024-04-05T00:00:12",
        "RDDATE": "2024-04-05T00:01:00",
        "RDDATE-E": "2024-04-05T00:00:12",
        "GRACFFLN": "regular-read-green.acf",
        "RDACFFLN": "regular-read-red.acf",
        "EXPMETER": "NO",
    }
    header.update(zip(("SCISEL", "SKYSEL", "SIMCALSH"), shutters, strict=True))
    header.update(zip(("GREEN", "RED", "CA_HK"), cameras, strict=True))
    header.update({k.replace("_", "-"): v for k, v in cards.items()})
    return header


class TestFrameIdentity:
    """OBSTYPE/OBSMODE/OBJECT/EXPTIME from IMTYPE, OCTAGON and shutters."""

    @pytest.mark.parametrize(
        "imtype, octagon",
        [
            ("Flatlamp", "BrdbandFiber"),
            ("Arclamp", "LFCFiber"),
            ("Arclamp", "EtalonFiber"),
            ("Arclamp", "Th_daily"),
            ("Arclamp", "U_daily"),
        ],
    )
    def test_lamp_frame_with_shutters_closed_raises(self, imtype, octagon):
        # No lamp light reaches the CCD; the wide flat (all closed) is exempt.
        with pytest.raises(HeaderError, match="every fiber shutter closed"):
            HeaderParser(_hdr(imtype, octagon, shutters=SHUT))

    @pytest.mark.parametrize(
        "header, obstype, obsmode, obj",
        [
            (_hdr("Bias", shutters=SHUT), "Bias", "Calibration", "Bias"),
            (_hdr("Dark", shutters=SHUT), "Dark", "Calibration", "Dark"),
            (
                _hdr("Object", "EtalonFiber", shutters=SCI),
                "Target",
                "Science",
                "target",
            ),
            (
                _hdr("Object", OBJECT="SoCal", TARGNAME="Sun"),
                "Target",
                "Science",
                "SoCal",
            ),
            (_hdr("Flatlamp", shutters=SHUT), "Flat", "Calibration", "WideFlat"),
            (_hdr("Flatlamp", "BrdbandFiber"), "Flat", "Calibration", "Flat"),
            (
                _hdr("Arclamp", "Th_daily", shutters=SCI),
                "Arclamp",
                "Calibration",
                "ThAr_daily",
            ),
            (_hdr("Arclamp", "Th_gold"), "Arclamp", "Calibration", "ThAr_gold"),
            (_hdr("Arclamp", "U_daily"), "Arclamp", "Calibration", "UNe_daily"),
            (
                _hdr("Arclamp", "U_gold", shutters=CAL),
                "Arclamp",
                "Calibration",
                "UNe_gold",
            ),
            (_hdr("Arclamp", "EtalonFiber"), "Etalon", "Calibration", "Etalon"),
            (_hdr("Arclamp", "LFCFiber"), "LFC", "Calibration", "LFC"),
            (
                _hdr("Arclamp", "Th_daily", shutters=HK, cameras=("NO", "NO", "YES")),
                "Arclamp",
                "Calibration",
                "ThAr_daily",
            ),
            (_hdr("Arclamp", "BrdbandFiber"), "Unknown", "Unknown", "Unknown"),
            (_hdr("Sky"), "Unknown", "Unknown", "Unknown"),
            # A bias/dark must record every shutter closed: open, absent or garbled
            # is unclassifiable.
            (_hdr("Bias", shutters=CAL), "Unknown", "Unknown", "Unknown"),
            (_hdr("Dark", shutters=SCI), "Unknown", "Unknown", "Unknown"),
            (_hdr("Bias", SKYSEL=None), "Unknown", "Unknown", "Unknown"),
            (_hdr("Dark", SIMCALSH="moving"), "Unknown", "Unknown", "Unknown"),
        ],
    )
    def test_frame_identity(self, header, obstype, obsmode, obj):
        keywords = HeaderParser(header).DRP_KEYWORDS
        assert keywords["OBSTYPE"] == obstype
        assert keywords["OBSMODE"] == obsmode
        assert keywords["OBJECT"] == obj
        assert keywords["ISSOLAR"] is (obj == "SoCal")

    def test_exptime_is_elapsed(self):
        # EPRV EXPTIME is the native ELAPSED, not the requested native EXPTIME.
        assert HeaderParser(_hdr("Bias")).DRP_KEYWORDS["EXPTIME"] == 30.0

    @pytest.mark.parametrize(
        "header, obj",
        [
            # Target: the canonical target, when OBJECT and TARGNAME agree.
            (_hdr("Object", OBJECT=" 10700 ", TARGNAME="10700"), "HD 10700"),
            (_hdr("Object", OBJECT="WASP-85", TARGNAME="WASP-85"), "WASP-85"),
            # A blank card takes the other.
            (_hdr("Object", OBJECT="", TARGNAME="K2-157"), "K2-157"),
            (_hdr("Object", OBJECT="KEPLER-65", TARGNAME=""), "Kepler-65"),
            (_hdr("Object", OBJECT="", TARGNAME="Sun"), "Unknown"),
            # Solar: exactly OBJECT SoCal with TARGNAME Sun; one alone is Unknown.
            (_hdr("Object", OBJECT="SoCal", TARGNAME="target"), "Unknown"),
            (_hdr("Object", OBJECT="x", TARGNAME="Sun"), "Unknown"),
            (_hdr("Object", OBJECT="Sun", TARGNAME="Sun"), "Unknown"),
            # Calibration: IMTYPE/OCTAGON alone; the native OBJECT only warns.
            (_hdr("Bias", shutters=SHUT, OBJECT="autocal-bias", TARGNAME=""), "Bias"),
            (_hdr("Arclamp", "Th_gold", OBJECT="autocal-thar-all-morn"), "ThAr_gold"),
            (_hdr("Arclamp", "LFCFiber", OBJECT="LFCStability"), "LFC"),
            (_hdr("Bias", shutters=SHUT, OBJECT=""), "Bias"),
            (_hdr("Sky", OBJECT="autocal-bias"), "Unknown"),
        ],
    )
    def test_object(self, header, obj):
        assert HeaderParser(header).DRP_KEYWORDS["OBJECT"] == obj

    @pytest.mark.parametrize(
        "header, reason",
        [
            # A science frame needs the science shutter open.
            (_hdr("Object", shutters=SKY), "SCISEL='closed'"),
            (_hdr("Object", SCISEL=None), "SCISEL=None"),
            # A calibration word in OBJECT contradicts IMTYPE Object.
            (_hdr("Object", OBJECT="autocal-bias", TARGNAME="T004201"), "calibration"),
            (_hdr("Object", OBJECT="manualcal-fast-bias", TARGNAME=""), "calibration"),
            (_hdr("Object", OBJECT="acf_change_bias_flush", TARGNAME="x"), "label"),
            # Solar only as exactly the (SoCal, Sun) pair.
            (_hdr("Object", OBJECT="SoCalSimulEtalonOD0", TARGNAME="Sun"), "SoCal/Sun"),
            (_hdr("Object", OBJECT="SoCal", TARGNAME="Sun1"), "SoCal/Sun"),
        ],
    )
    def test_object_frame_contradicted(self, header, reason):
        """An IMTYPE Object frame the rest of the header contradicts is Unknown,
        with one fatal warning naming why."""
        parsed = HeaderParser(header)
        keywords = parsed.DRP_KEYWORDS
        assert (keywords["OBSTYPE"], keywords["OBSMODE"], keywords["OBJECT"]) == (
            "Unknown",
            "Unknown",
            "Unknown",
        )
        (fatal,) = [w for w in parsed.warnings if "cannot classify frame" in w]
        assert "IMTYPE 'Object' but" in fatal and reason in fatal

    @pytest.mark.parametrize(
        "obj, targname, expected",
        [
            ("10700", "HD 10700", "HD 10700"),
            ("Callisto Transit Test", "Callisto Transit Test", "Callisto"),
            ("SoCal", "Sun", "SoCal"),
            ("test", "test", "test"),
        ],
    )
    def test_object_frame_not_contradicted(self, obj, targname, expected):
        keywords = HeaderParser(
            _hdr("Object", OBJECT=obj, TARGNAME=targname)
        ).DRP_KEYWORDS
        assert (keywords["OBSMODE"], keywords["OBJECT"]) == ("Science", expected)

    @pytest.mark.parametrize(
        "obj, targname, expected",
        [
            # Known aliases reconcile to one canonical form.
            ("HD57555", "HD 57555", "HD 57555"),
            ("57555", "HD 57555", "HD 57555"),
            ("hd 57555", "57555", "HD 57555"),
            ("HIP9618", "HIP 9618", "HIP 9618"),
            ("TOI 233", "TOI-233", "TOI-233"),
            ("T001824", "TOI-1824", "TOI-1824"),
            ("Kepler 86", "Kepler-86", "Kepler-86"),
            ("KOI-270", "KOI 270", "KOI-270"),
            ("K2 157", "K2-157", "K2-157"),
            (
                "Gaia 2077618037775709568",
                "2077618037775709568",
                "Gaia 2077618037775709568",
            ),
            (
                "Gaia DR3 2077618037775709568",
                "Gaia DR3 2077618037775709568",
                "Gaia DR3 2077618037775709568",
            ),
            ("TIC436314778", "TIC 436314778", "TIC 436314778"),
            ("KIC4736208", "KIC 4736208", "KIC 4736208"),
            ("WASP127", "WASP-127", "WASP-127"),
            # A bare system name reconciles with one of its components.
            ("WASP-173", "WASP-173B", "WASP-173 B"),
            ("WASP-173B", "WASP-173", "WASP-173 B"),
            ("WASP-173A", "WASP-173B", "Unknown"),
            ("115404A", "HD 115404 A", "HD 115404 A"),
            ("hd21019A", "21019", "HD 21019 A"),
            ("hd21019a", "21019", "HD 21019 A"),
            ("WASP-173a", "WASP-173", "WASP-173 A"),
            ("Kepler-444a", "Kepler-444 A", "Kepler-444 A"),
            ("TOI-1338B", "TOI-1338", "TOI-1338 B"),
            # A lowercase planet letter (b-z) is dropped for the host star.
            ("TOI-969b", "TOI-969", "TOI-969"),
            # Purely numeric catalogs take no letter.
            ("TIC 436314778b", "TIC 436314778", "Unknown"),
            ("KIC4736208A", "KIC 4736208", "Unknown"),
            (
                "Gaia DR3 2077618037775709568 A",
                "Gaia DR3 2077618037775709568",
                "Unknown",
            ),
            ("HD 189733b", "HD 189733", "HD 189733"),
            ("WASP-12b", "WASP-12", "WASP-12"),
            ("Kepler-86 b", "Kepler-86", "Kepler-86"),
            ("21019 A", "HD 21019B", "Unknown"),
            # A bare catalog prefix reconciles with a full id in that catalog.
            ("TOI", "TOI-969", "TOI-969"),
            ("TIC", "TIC 436314778", "TIC 436314778"),
            ("HIP 9618", "HIP", "HIP 9618"),
            # Unreconcilable: other catalogs, other ids, unknown lengths.
            ("HD", "TOI-969", "Unknown"),
            ("HD 10700", "HD 10701", "Unknown"),
            ("343600706696532", "Gaia 3436007066965322240", "Unknown"),
            ("343600706696532", "343600706696532", "Unknown"),
            ("Gaia 2077618037775709568", "Gaia DR3 2077618037775709568", "Unknown"),
        ],
    )
    def test_target_aliases(self, obj, targname, expected):
        # An Unknown Target needs a GAIAID, else it raises (TestUnknownTarget).
        gaiaid = f"Gaia DR3 {_G19}" if expected == "Unknown" else None
        header = _hdr("Object", OBJECT=obj, TARGNAME=targname, GAIAID=gaiaid)
        assert HeaderParser(header).DRP_KEYWORDS["OBJECT"] == expected

    @pytest.mark.parametrize(
        "obj, targname, expected",
        [
            # Both cards name one body as a whole word, any case.
            ("URANUS", "uranus", "Uranus"),
            ("Ganymede Umbra", "Ganymede", "Ganymede"),
            ("Where It Was Europa 7:10", "Where It Was Europa 7:10", "Europa"),
            ("Io", "Io Eclipse", "Io"),
            ("Dione In Sunlight Test", "Dione", "Dione"),
            ("Jupiter", "Jupiter Calibration", "Jupiter"),
            # Two bodies on a card, or cards that don't agree.
            ("Tethys & Enceladus", "Tethys & Enceladus", "Unknown"),
            ("Europa Transiting Jupiter", "Europa", "Unknown"),
            ("Io", "Where It Was 06:40", "Unknown"),
            ("Ganymed Penumbra", "Ganymede Penumbra", "Unknown"),
            # No body: the existing rules.
            ("Where It Was 08:00", "Where It Was 08:00", "Where It Was 08:00"),
            ("Titania", "Titania", "Titania"),
        ],
    )
    def test_solar_system_bodies(self, obj, targname, expected):
        gaiaid = f"Gaia DR3 {_G19}" if expected == "Unknown" else None
        header = _hdr("Object", OBJECT=obj, TARGNAME=targname, GAIAID=gaiaid)
        assert HeaderParser(header).DRP_KEYWORDS["OBJECT"] == expected

    def test_solar_system_blank_card_warned(self):
        parser = HeaderParser(_hdr("Object", OBJECT="Ganymede Penumbra", TARGNAME=""))
        assert parser.DRP_KEYWORDS["OBJECT"] == "Ganymede"
        assert any("blank card" in w for w in parser.warnings)

    def test_reconciliation_logged_at_info(self, caplog):
        with caplog.at_level(logging.INFO, logger="kpfpipe.utils.header"):
            HeaderParser(_hdr("Object", OBJECT="HD73344", TARGNAME="HD 73344"))
        assert "OBJECT 'HD73344' / TARGNAME 'HD 73344' resolved to 'HD 73344'" in (
            caplog.text
        )

    def test_canonical_pair_not_logged(self, caplog):
        with caplog.at_level(logging.INFO, logger="kpfpipe.utils.header"):
            HeaderParser(_hdr("Object", OBJECT="HD 73344", TARGNAME="HD 73344"))
        assert "resolved to" not in caplog.text

    def test_bare_number_in_no_catalog_warned(self):
        header = _hdr(
            "Object",
            OBJECT="343600706696532",
            TARGNAME="343600706696532",
            GAIAID=f"Gaia DR3 {_G19}",
        )
        parser = HeaderParser(header)
        assert any("bare number in no known catalog" in w for w in parser.warnings)

    def test_planet_letter_dropped_and_warned(self):
        parser = HeaderParser(_hdr("Object", OBJECT="HD 189733b", TARGNAME="HD 189733"))
        assert parser.DRP_KEYWORDS["OBJECT"] == "HD 189733"
        assert any("names planet 'b'" in w for w in parser.warnings)

    @pytest.mark.parametrize(
        "obj, targname",
        [
            ("HD 189733b", "HD 189733B"),
            ("WASP-173B", "WASP-173b"),
            ("HD 189733b", "HD 189733 A"),
        ],
    )
    def test_planet_and_component_letter_raises(self, obj, targname):
        with pytest.raises(HeaderError, match="planet letter with a component"):
            HeaderParser(_hdr("Object", OBJECT=obj, TARGNAME=targname))

    def test_component_reconciliation_warned(self):
        parser = HeaderParser(_hdr("Object", OBJECT="WASP-173", TARGNAME="WASP-173B"))
        assert any("component letter" in w for w in parser.warnings)
        exact = HeaderParser(_hdr("Object", OBJECT="WASP-173B", TARGNAME="WASP-173B"))
        assert not any("component letter" in w for w in exact.warnings)


_G19 = "2452378776434477184"  # tau Cet, Gaia DR3
_G18 = "806000816970521728"


class TestGaiaId:
    """GAIAID: parsed, validated and canonicalized, and reconciled with OBJECT."""

    @staticmethod
    def _parse(gaiaid, obj="HD 10700", targname=None):
        return HeaderParser(
            _hdr("Object", OBJECT=obj, TARGNAME=targname or obj, GAIAID=gaiaid)
        )

    @staticmethod
    def _gaia_warnings(parser):
        return [w for w in parser.warnings if "GAIAID" in w]

    @pytest.mark.parametrize(
        "native, expected",
        [
            (f"DR3 {_G19}", f"Gaia DR3 {_G19}"),  # the usual native form
            (f"DR3_{_G19}", f"Gaia DR3 {_G19}"),  # failure mode 2
            (f"DR3_{_G18}", f"Gaia DR3 {_G18}"),
            (f"Gaia_DR3_{_G19}", f"Gaia DR3 {_G19}"),
            (f"DR3{_G19}", f"Gaia DR3 {_G19}"),
            (f"Gaia DR3 {_G19}", f"Gaia DR3 {_G19}"),
            (f"dr2 {_G19}", f"Gaia DR2 {_G19}"),
            (f"EDR3_{_G19}", f"Gaia DR3 {_G19}"),  # EDR3 shares DR3 source_ids
            (_G19, f"Gaia {_G19}"),  # bare: AstroQuery resolves the release
            (f"Gaia {_G19}", f"Gaia {_G19}"),
            ("DR3 4295806720", "Gaia DR3 4295806720"),  # low-HEALPix source_id
            ("Gaia 4295806720", "Gaia 4295806720"),
        ],
    )
    def test_canonical(self, native, expected):
        parser = self._parse(native)
        assert parser.DRP_KEYWORDS["GAIAID"] == expected
        assert not self._gaia_warnings(parser)

    @pytest.mark.parametrize(
        "native",
        [
            "0",
            "2",
            "123456789",
            _G19[:14],
            _G19 + "0",
            "DR3",
            f"DR1 {_G19}",
            f"DR4 {_G19}",
            f"Gaia DR3 {_G19[:9]}",
            "4295806720",  # a bare short number is not taken for Gaia
            f"{_G19}b",  # Gaia ids take no trailing letter
            f"Gaia DR3 {_G19} A",
        ],
    )
    def test_malformed_raises(self, native):
        with pytest.raises(HeaderError, match="is not a Gaia DR2/DR3 source_id"):
            self._parse(native)

    @pytest.mark.parametrize("obj", ["Gaia 12 b", "Gaia DR1 123456789012345678"])
    def test_gaia_like_object_not_copied(self, obj):
        # Only an OBJECT kpf_target_name names a Gaia id lends it to GAIAID.
        assert self._parse(None, obj=obj).DRP_KEYWORDS["GAIAID"] is None

    @pytest.mark.parametrize("native", [None, "", "None", "none"])
    def test_absent_is_blank_and_silent(self, native):
        parser = self._parse(native)
        assert parser.DRP_KEYWORDS["GAIAID"] is None
        assert not self._gaia_warnings(parser)

    def test_blank_off_a_science_target(self):
        cal = HeaderParser(_hdr("Bias", GAIAID=f"DR3 {_G19}"))
        assert cal.DRP_KEYWORDS["GAIAID"] is None
        # Malformed is irrelevant where no GAIAID is needed.
        assert HeaderParser(_hdr("Bias", GAIAID="2")).DRP_KEYWORDS["GAIAID"] is None
        solar = self._parse(f"DR3 {_G19}", obj="SoCal", targname="Sun")
        assert solar.DRP_KEYWORDS["GAIAID"] is None

    @pytest.mark.parametrize("native", [None, "", "None"])
    def test_solar_system_absent_is_silent(self, native):
        parser = self._parse(native, obj="Uranus")
        assert parser.DRP_KEYWORDS["GAIAID"] is None
        assert not self._gaia_warnings(parser)

    @pytest.mark.parametrize("native", [f"DR3 {_G19}", "2"])
    def test_solar_system_id_is_blanked_and_warned(self, native):
        # Well-formed or not: a solar-system target needs no GAIAID, so never raises.
        parser = self._parse(native, obj="Io", targname="Io Eclipse")
        assert parser.DRP_KEYWORDS["GAIAID"] is None
        assert any("solar-system target 'Io'" in w for w in self._gaia_warnings(parser))

    @pytest.mark.parametrize(
        "obj, native, expected",
        [
            # The release on either card is lent to the other.
            (_G19, f"DR3 {_G19}", f"Gaia DR3 {_G19}"),
            (f"Gaia DR3 {_G19}", _G19, f"Gaia DR3 {_G19}"),
            # A blank GAIAID takes the OBJECT's.
            (f"Gaia DR3 {_G19}", None, f"Gaia DR3 {_G19}"),
        ],
    )
    def test_reconciled_with_gaia_object(self, obj, native, expected):
        parser = self._parse(native, obj=obj)
        assert parser.DRP_KEYWORDS["GAIAID"] == expected
        assert parser.DRP_KEYWORDS["OBJECT"] == expected
        assert self._gaia_warnings(parser)

    def test_agreeing_gaia_object_is_silent(self):
        parser = self._parse(f"DR3 {_G19}", obj=f"Gaia DR3 {_G19}")
        assert parser.DRP_KEYWORDS["GAIAID"] == f"Gaia DR3 {_G19}"
        assert not self._gaia_warnings(parser)

    def test_short_bare_id_message_names_the_fix(self):
        with pytest.raises(HeaderError, match="shorter bare number is refused"):
            self._parse("4295806720")

    @pytest.mark.parametrize(
        "obj, native, reason",
        [
            (_G19, f"DR3 {_G18}0", f"source_id {_G19} vs {_G18}0"),
            (f"Gaia DR2 {_G19}", f"DR3 {_G19}", "releases DR2 and DR3"),
        ],
    )
    def test_disagreement_with_gaia_object_raises(self, obj, native, reason):
        with pytest.raises(HeaderError, match="name different Gaia sources") as err:
            self._parse(native, obj=obj)
        assert reason in str(err.value)


def _traces(keywords, prefix):
    return [keywords.get(f"{prefix}{i}") for i in range(1, 6)]


class TestIllumination:
    """Per-fiber sources, trace labels and SIMULCAL."""

    @pytest.mark.parametrize(
        "header, srcs",
        [
            (_hdr("Arclamp", "Th_daily", shutters=SCI), ["None", "ThAr_daily", "None"]),
            (_hdr("Flatlamp", "BrdbandFiber", shutters=CAL), ["None", "None", "Flat"]),
            (_hdr("Object", "EtalonFiber"), ["Sky", "Target", "Etalon"]),
            (
                _hdr(
                    "Object",
                    "EtalonFiber",
                    OBJECT="SoCal",
                    TARGNAME="Sun",
                    SOLSCISH="open",
                ),
                ["Sky", "SoCal", "Etalon"],
            ),
            # SoCal reaches SCI through its own shutter, SOLSCISH.
            (
                _hdr(
                    "Object",
                    "EtalonFiber",
                    OBJECT="SoCal",
                    TARGNAME="Sun",
                    SOLSCISH="closed",
                ),
                ["Sky", "None", "Etalon"],
            ),
            (
                _hdr("Object", "EtalonFiber", OBJECT="SoCal", TARGNAME="Sun"),
                ["Sky", "Unknown", "Etalon"],
            ),
            (
                # The octagon parks on the etalon, but the SimulCal shutter is shut.
                _hdr("Object", "EtalonFiber", shutters=("open", "open", "closed")),
                ["Sky", "Target", "None"],
            ),
            (_hdr("Bias", shutters=SHUT), ["None"] * 3),
            (_hdr("Flatlamp", shutters=ALL), ["None"] * 3),
            (_hdr("Sky"), ["Unknown"] * 3),
            # A shutter the source depends on that is absent or garbled.
            (
                _hdr("Arclamp", "Th_daily", SKYSEL=None),
                ["Unknown"] + ["ThAr_daily"] * 2,
            ),
            (
                _hdr("Object", "EtalonFiber", SIMCALSH="moving"),
                ["Sky", "Target", "Unknown"],
            ),
            # An open CAL fiber on a Target frame whose OCTAGON names no lamp.
            (_hdr("Object", "Home"), ["Sky", "Target", "Unknown"]),
            (_hdr("Object", OCTAGON=None), ["Sky", "Target", "Unknown"]),
        ],
    )
    def test_illumination(self, header, srcs):
        keywords = HeaderParser(header).DRP_KEYWORDS
        assert [keywords[k] for k in ("SKYSRC", "SCISRC", "CALSRC")] == srcs
        assert _traces(keywords, "TRACE") == ["SKY", "SCI", "SCI", "SCI", "CAL"]
        # CLSRC# names a calibration lamp only, never Sky/Target/SoCal.
        sky, sci, cal = ("None" if s in ("Sky", "Target", "SoCal") else s for s in srcs)
        assert _traces(keywords, "CLSRC") == [sky, sci, sci, sci, cal]
        # No catalog object until AstroQuery runs.
        assert _traces(keywords, "CSRC") == _traces(keywords, "CID") == ["None"] * 5

    @pytest.mark.parametrize(
        "header, simulcal",
        [
            (_hdr("Object", "EtalonFiber"), "On"),
            (
                _hdr(
                    "Object",
                    "EtalonFiber",
                    OBJECT="SoCal",
                    TARGNAME="Sun",
                    SOLSCISH="open",
                ),
                "On",
            ),
            # Only the etalon is a simultaneous calibration source.
            (_hdr("Object", "LFCFiber"), "Unknown"),
            (_hdr("Object", "EtalonFiber", shutters=("open", "open", "closed")), "Off"),
            (_hdr("Object", "EtalonFiber", SIMCALSH=None), "Unknown"),
            (_hdr("Object", "Home"), "Unknown"),
            (_hdr("Arclamp", "EtalonFiber"), "N/A"),
            (_hdr("Bias", shutters=SHUT), "N/A"),
            (_hdr("Sky"), "Unknown"),
        ],
    )
    def test_simulcal(self, header, simulcal):
        assert HeaderParser(header).DRP_KEYWORDS["SIMULCAL"] == simulcal


class TestCamerasAndReadout:
    """Camera flags, read mode/time, and nonsensical values."""

    def test_cameras(self):
        keywords = HeaderParser(
            _hdr("Bias", cameras=("YES", "no", " YES "))
        ).DRP_KEYWORDS
        assert [keywords[k] for k in ("GREEN", "RED", "CA_HK", "EXPMETER")] == [
            True,
            False,
            True,
            False,
        ]

    def test_unrecognized_camera_card_warned_and_omitted(self):
        parsed = HeaderParser(_hdr("Bias", cameras=("YES", "YES", "maybe")))
        assert "CA_HK" not in parsed.DRP_KEYWORDS
        assert any("CA_HK='maybe' is not YES/NO" in w for w in parsed.warnings)

    def test_readout_from_acf(self):
        keywords = HeaderParser(_hdr("Bias")).DRP_KEYWORDS
        assert keywords["READMODE"] == "regular"
        assert keywords["READTIMG"] == keywords["READTIMR"] == 48.0

    def test_readmode_fallback_logs_debug_not_warning(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="kpfpipe.utils.header"):
            parser = HeaderParser(_hdr("Bias", GRACFFLN="", RDACFFLN=""))
        assert parser.DRP_KEYWORDS["READMODE"] == "regular"
        assert not any("READMODE" in w for w in parser.warnings)
        assert "READMODE inferred from read time" in caplog.text

    @pytest.mark.parametrize(
        "cards, match",
        [
            # GRDATE-E (shutter close) after GRDATE (file write) is impossible.
            ({"GRDATE_E": "2024-04-05T00:02:00"}, "negative GREEN read time"),
            ({"RDDATE": "garbled"}, "RDDATE='garbled' or RDDATE-E=.* is not an ISO"),
            ({"GRDATE": 5.0}, "is not an ISO timestamp"),
            ({"ELAPSED": "N/A"}, "ELAPSED='N/A' is not a number"),
        ],
    )
    def test_nonsensical_value_raises_header_error(self, cards, match):
        with pytest.raises(HeaderError, match=match):
            HeaderParser(_hdr("Bias", **cards))


def _consistent(imtype, octagon, shutters, sci, sky, cal, **cards):
    return _hdr(
        imtype,
        octagon,
        shutters,
        **{"SCI_OBJ": sci, "SKY_OBJ": sky, "CAL_OBJ": cal},
        **cards,
    )


class TestHeaderFixMessages:
    """A target-identity defect (OBJECT, TARGNAME, GAIAID) says the raw L0 header
    must be fixed, in one greppable phrase for batch triage."""

    @pytest.mark.parametrize(
        "header",
        [
            _hdr("Object", OBJECT="10700", TARGNAME="10700", GAIAID="NoGaiaName"),
            _hdr(
                "Object",
                OBJECT=f"Gaia DR3 {_G19}",
                TARGNAME=f"Gaia DR3 {_G19}",
                GAIAID=f"Gaia DR2 {_G19}",
            ),
            _hdr("Object", OBJECT="WASP-173b", TARGNAME="WASP-173B"),
            _hdr("Object", OBJECT="10700", TARGNAME="HD 10701"),
        ],
    )
    def test_identity_error_names_the_fix(self, header):
        with pytest.raises(HeaderError, match="L0 header must be fixed: "):
            HeaderParser(header)


class TestUnknownTarget:
    """A Target whose OBJECT is Unknown: one warning naming the reason, the native
    cards and the GAIAID that still locates the star; with GAIAID blank, a
    HeaderError carrying the same diagnosis."""

    _CASES = [
        # Legitimate aliases of one star, which kpf_target_name doesn't cross-identify.
        ("201092", "61 Cyg B", "canonicalize to different names ('HD 201092'"),
        ("", "", "OBJECT and TARGNAME are both blank"),
        ("343600706696532", "343600706696532", "bare number in no known catalog"),
        ("Tethys & Enceladus", "Tethys", "more than one solar-system body"),
        ("Io", "Where It Was 06:40", "name solar-system body ['Io']"),
    ]

    @pytest.mark.parametrize("obj, targname, reason", _CASES)
    def test_gaiaid_locates_it(self, obj, targname, reason):
        parser = HeaderParser(
            _hdr("Object", OBJECT=obj, TARGNAME=targname, GAIAID=f"DR3 {_G19}")
        )
        assert parser.DRP_KEYWORDS["OBJECT"] == "Unknown"
        assert parser.DRP_KEYWORDS["GAIAID"] == f"Gaia DR3 {_G19}"
        (warning,) = [w for w in parser.warnings if "OBJECT is Unknown" in w]
        assert reason in warning
        assert f"[native OBJECT={obj!r}, TARGNAME={targname!r}]" in warning
        assert f"astrometry from GAIAID 'Gaia DR3 {_G19}'" in warning

    @pytest.mark.parametrize("obj, targname, reason", _CASES)
    def test_blank_gaiaid_raises(self, obj, targname, reason):
        with pytest.raises(HeaderError, match="GAIAID is blank") as err:
            HeaderParser(_hdr("Object", OBJECT=obj, TARGNAME=targname))
        assert reason in str(err.value)


class TestWarnings:
    """Each unreliable-metadata case is recorded and logged."""

    def test_consistent_frame_has_no_warnings(self):
        header = _consistent(
            "Arclamp",
            "Th_daily",
            SCI,
            "Th_daily",
            "None",
            "None",
            OBJECT="autocal-thar-sci",
        )
        assert HeaderParser(header).warnings == []

    @pytest.mark.parametrize(
        "header, expected",
        [
            (_hdr("Sky"), "cannot classify frame"),
            (
                _hdr("Dark", shutters=CAL),
                "cannot classify frame (IMTYPE='Dark', OCTAGON='Home', "
                "SCISEL='closed', SKYSEL='closed', SIMCALSH='open', "
                "OBJECT='target', TARGNAME='unknown')",
            ),
            (_hdr("Bias", shutters=SHUT, OBJECT="  "), "OBJECT '' is no autocal label"),
            (
                _hdr("Arclamp", "LFCFiber", OBJECT="LFCStability"),
                "OBJECT 'LFCStability' is no autocal label",
            ),
            (
                _hdr(
                    "Object",
                    OBJECT="10700",
                    TARGNAME="HD 10701",
                    GAIAID="Gaia DR3 2452378776434477184",
                ),
                "OBJECT is Unknown: the cards canonicalize to different names "
                "('HD 10700' vs 'HD 10701')",
            ),
            (
                _hdr("Object", OBJECT="K2-157", TARGNAME=""),
                "TARGNAME blank; OBJECT is 'K2-157'",
            ),
            (
                _hdr("Object", OBJECT="", TARGNAME="K2-157"),
                "OBJECT blank; OBJECT is TARGNAME 'K2-157'",
            ),
            (
                _hdr(
                    "Arclamp",
                    "LFCFiber",
                    OBJECT="autocal-lfc-all",
                    TARGNAME="AFP Failsafe",
                ),
                "TARGNAME 'AFP Failsafe' names a target on a LFC frame",
            ),
            (
                _hdr(
                    "Arclamp", "Th_daily", shutters=SCI, OBJECT="autocal-thar-all-eve"
                ),
                "disagrees with derived autocal-thar-sci",
            ),
            (
                # A daily-lamp OBJECT on a gold-lamp frame: the lamps must not conflate.
                _hdr("Arclamp", "Th_gold", OBJECT="autocal-thar-all-morn"),
                "disagrees with derived autocal-thar_gold-all",
            ),
            (
                # SCI+SKY lit: no autocal label names that fiber set.
                _hdr(
                    "Arclamp",
                    "Th_daily",
                    shutters=("open", "open", "closed"),
                    OBJECT="autocal-thar-hk",
                ),
                "OBJECT 'autocal-thar-hk' disagrees with the shutters "
                "(SCISEL='open', SKYSEL='open', SIMCALSH='closed'), which no "
                "autocal label names",
            ),
            (
                # The shutter-blind native pair: SKY-OBJ names the lamp on a shut fiber.
                _consistent("Arclamp", "Th_daily", SCI, "Th_daily", "Th_daily", "None"),
                "SKY-OBJ='Th_daily' disagrees with derived SKYSRC='None'",
            ),
            (
                # CAL-OBJ carrying an octagon position index (20240405).
                _consistent("Arclamp", "Th_daily", ALL, "Th_daily", "Th_daily", "10"),
                "CAL-OBJ='10' disagrees with derived CALSRC='ThAr_daily'",
            ),
            (
                _hdr("Arclamp", "Th_daily", SIMCALSH="moving"),
                "SIMCALSH='moving' is not open/closed; CALSRC is Unknown",
            ),
            (_hdr("Bias", GRDATE=None), "GREEN=YES but GRDATE/GRDATE-E absent"),
            (_hdr("Bias", ELAPSED=None), "ELAPSED absent"),
        ],
    )
    def test_warnings(self, header, expected, caplog):
        with caplog.at_level(logging.WARNING, logger="kpfpipe.utils.header"):
            parser = HeaderParser(header)
        assert any(expected in w for w in parser.warnings)
        assert expected in caplog.text

    @pytest.mark.parametrize(
        "header, obj",
        [
            (_hdr("Arclamp", "Th_daily", OBJECT="autocal-thar-all"), "ThAr_daily"),
            (_hdr("Flatlamp", shutters=SHUT, OBJECT="autocal-wideflat-f5"), "WideFlat"),
            (_hdr("Bias", shutters=SHUT, OBJECT="autocal-bias"), "Bias"),
            (
                _hdr(
                    "Arclamp",
                    "Th_daily",
                    HK,
                    ("NO", "NO", "YES"),
                    OBJECT="autocal-thar-hk",
                ),
                "ThAr_daily",
            ),
        ],
    )
    def test_object_labels_that_agree_do_not_warn(self, header, obj):
        parser = HeaderParser(header)
        assert not any("OBJECT" in w or "TARGNAME" in w for w in parser.warnings)
        assert parser.DRP_KEYWORDS["OBJECT"] == obj

    def test_hk_frame_does_not_warn_missing_ccd_timing(self):
        header = _hdr("Arclamp", "Th_daily", shutters=HK, cameras=("NO", "NO", "YES"))
        for card in ("GRDATE", "GRDATE-E", "RDDATE", "RDDATE-E"):
            del header[card]
        parser = HeaderParser(header)
        assert not any("DATE" in w for w in parser.warnings)
        assert "READMODE" in parser.DRP_KEYWORDS  # still named by the ACF filenames


def _write_l0(tmp_path, header):
    path = tmp_path / "KP.20240405.00000.00.fits"
    hdul = fits.HDUList([fits.PrimaryHDU(header=fits.Header(header)), fits.ImageHDU()])
    hdul.writeto(path)
    return path


class TestSources:
    """A file path, a fits.Header and a plain mapping parse identically."""

    def test_sources_parse_identically(self, tmp_path):
        native = _hdr("Arclamp", "Th_daily", shutters=SCI)
        path = _write_l0(tmp_path, native)
        loaded = fits.getheader(path, ext=0)
        expected = HeaderParser(native).DRP_KEYWORDS
        for source in (path, str(path), loaded, dict(loaded)):
            assert HeaderParser(source).DRP_KEYWORDS == expected

    def test_path_reads_only_primary(self, tmp_path, monkeypatch):
        path = _write_l0(tmp_path, _hdr("Bias"))
        calls = []
        real = header_module.fits.getheader

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(header_module.fits, "getheader", spy)
        HeaderParser(path)
        assert calls == [{"ext": 0}]

    def test_header_is_not_modified(self):
        native = _hdr("Arclamp", "Th_daily", shutters=SCI)
        before = dict(native)
        parser = HeaderParser(native)
        assert parser.WMKO_HEADER is native
        assert native == before


class TestRealData:
    """Every truth frame standardizes, carrying HeaderParser's keywords."""

    @pytest.mark.slow
    @pytest.mark.requires_testdata
    @pytest.mark.parametrize(
        "path",
        sorted(TESTDATA_L0.glob("*.fits")) or [None],
        ids=lambda p: getattr(p, "name", "none"),
    )
    def test_kpf0_stamps_every_keyword(self, path):
        l0 = KPF0.from_fits(str(path), standardize=True)
        keywords = HeaderParser(l0.headers["INSTRUMENT_HEADER"]).DRP_KEYWORDS
        assert keywords["OBSMODE"] != "Unknown"
        for key, value in keywords.items():
            ext = "QUALITY_CONTROL" if key.startswith("READTIM") else "PRIMARY"
            assert l0.headers[ext].get(key) == value, key


class TestKpfTargetName:
    """kpf_target_name: the one apparatus for target identifiers."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("57555", ("HD", "HD 57555", None)),
            ("hd21019a", ("HD", "HD 21019 A", None)),
            ("HD 189733b", ("HD", "HD 189733", "b")),
            (
                "Gaia_EDR3_2077618037775709568",
                ("Gaia", "Gaia DR3 2077618037775709568", None),
            ),
            ("tau Cet", (None, "tau Cet", None)),
        ],
    )
    def test_canonical(self, name, expected):
        assert kpf_target_name(name) == expected

    def test_name_matching_several_rows_raises(self, monkeypatch):
        # Row order never picks a winner: overlapping patterns are a csv bug.
        rows = [
            ("HD", re.compile(r"(\d+)"), "HD {}"),
            ("HIP", re.compile(r"(\d+)"), "HIP {}"),
        ]
        monkeypatch.setattr("kpfpipe.utils.kpf._STAR_CATALOG_ALIASES", rows)
        with pytest.raises(ValueError, match="matches several"):
            kpf_target_name("10700")
