"""HeaderParser: frame identity, trace illumination, readout and warnings from a
native L0 PRIMARY header."""

import logging
from pathlib import Path

import pytest
from astropy.io import fits

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.utils import header as header_module
from kpfpipe.utils.header import HeaderError, HeaderParser

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


@pytest.mark.parametrize(
    "header, obstype, obsmode, obj",
    [
        (_hdr("Bias", shutters=SHUT), "Bias", "Calibration", "Bias"),
        (_hdr("Dark", shutters=SHUT), "Dark", "Calibration", "Dark"),
        (_hdr("Object", "EtalonFiber", shutters=SKY), "Target", "Science", "target"),
        (_hdr("Object", OBJECT="SoCal", TARGNAME="Sun"), "Target", "Science", "SoCal"),
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
        (_hdr("Arclamp", "U_gold", shutters=CAL), "Arclamp", "Calibration", "UNe_gold"),
        (_hdr("Arclamp", "EtalonFiber"), "Etalon", "Calibration", "Etalon"),
        (_hdr("Arclamp", "LFCFiber", shutters=SHUT), "LFC", "Calibration", "LFC"),
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
def test_frame_identity(header, obstype, obsmode, obj):
    keywords = HeaderParser(header).DRP_KEYWORDS
    assert keywords["OBSTYPE"] == obstype
    assert keywords["OBSMODE"] == obsmode
    assert keywords["OBJECT"] == obj
    assert keywords["ISSOLAR"] is (obj == "SoCal")


def test_exptime_is_elapsed():
    # EPRV EXPTIME is the native ELAPSED, not the requested native EXPTIME.
    assert HeaderParser(_hdr("Bias")).DRP_KEYWORDS["EXPTIME"] == 30.0


@pytest.mark.parametrize(
    "header, obj",
    [
        # Target: the target, when native OBJECT and TARGNAME agree.
        (_hdr("Object", OBJECT=" 10700 ", TARGNAME="10700"), "10700"),
        (_hdr("Object", OBJECT="10700", TARGNAME="HD 10700"), "Unknown"),
        (_hdr("Object", OBJECT="", TARGNAME=""), "Unknown"),
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
def test_object(header, obj):
    assert HeaderParser(header).DRP_KEYWORDS["OBJECT"] == obj


def _traces(keywords, prefix):
    return [keywords.get(f"{prefix}{i}") for i in range(1, 6)]


@pytest.mark.parametrize(
    "header, srcs",
    [
        (_hdr("Arclamp", "Th_daily", shutters=SCI), ["None", "ThAr_daily", "None"]),
        (_hdr("Flatlamp", "BrdbandFiber", shutters=CAL), ["None", "None", "Flat"]),
        (_hdr("Object", "EtalonFiber"), ["Sky", "Target", "Etalon"]),
        (
            _hdr(
                "Object", "EtalonFiber", OBJECT="SoCal", TARGNAME="Sun", SOLSCISH="open"
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
        (_hdr("Arclamp", "LFCFiber", shutters=SHUT), ["None"] * 3),
        (_hdr("Sky"), ["Unknown"] * 3),
        # A shutter the source depends on that is absent or garbled.
        (_hdr("Arclamp", "Th_daily", SKYSEL=None), ["Unknown"] + ["ThAr_daily"] * 2),
        (
            _hdr("Object", "EtalonFiber", SIMCALSH="moving"),
            ["Sky", "Target", "Unknown"],
        ),
        # An open CAL fiber on a Target frame whose OCTAGON names no lamp.
        (_hdr("Object", "Home"), ["Sky", "Target", "Unknown"]),
        (_hdr("Object", OCTAGON=None), ["Sky", "Target", "Unknown"]),
    ],
)
def test_illumination(header, srcs):
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
                "Object", "EtalonFiber", OBJECT="SoCal", TARGNAME="Sun", SOLSCISH="open"
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
def test_simulcal(header, simulcal):
    assert HeaderParser(header).DRP_KEYWORDS["SIMULCAL"] == simulcal


def test_cameras():
    keywords = HeaderParser(_hdr("Bias", cameras=("YES", "no", " YES "))).DRP_KEYWORDS
    assert [keywords[k] for k in ("GREEN", "RED", "CA_HK", "EXPMETER")] == [
        True,
        False,
        True,
        False,
    ]


def test_unrecognized_camera_card_warned_and_omitted():
    parsed = HeaderParser(_hdr("Bias", cameras=("YES", "YES", "maybe")))
    assert "CA_HK" not in parsed.DRP_KEYWORDS
    assert any("CA_HK='maybe' is not YES/NO" in w for w in parsed.warnings)


def test_readout_from_acf():
    keywords = HeaderParser(_hdr("Bias")).DRP_KEYWORDS
    assert keywords["READMODE"] == "regular"
    assert keywords["READTIMG"] == keywords["READTIMR"] == 48.0


def test_readmode_fallback_logs_debug_not_warning(caplog):
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
def test_nonsensical_value_raises_header_error(cards, match):
    with pytest.raises(HeaderError, match=match):
        HeaderParser(_hdr("Bias", **cards))


# Warnings: each unreliable-metadata case is recorded and logged.


def _consistent(imtype, octagon, shutters, sci, sky, cal, **cards):
    return _hdr(
        imtype,
        octagon,
        shutters,
        **{"SCI_OBJ": sci, "SKY_OBJ": sky, "CAL_OBJ": cal},
        **cards,
    )


def test_consistent_frame_has_no_warnings():
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
            "SCISEL='closed', SKYSEL='closed', SIMCALSH='open')",
        ),
        (_hdr("Bias", shutters=SHUT, OBJECT="  "), "OBJECT '' is no autocal label"),
        (
            _hdr("Arclamp", "LFCFiber", OBJECT="LFCStability"),
            "OBJECT 'LFCStability' is no autocal label",
        ),
        (
            _hdr("Object", OBJECT="10700", TARGNAME="HD 10700"),
            "OBJECT '10700' disagrees with TARGNAME 'HD 10700'",
        ),
        (
            _hdr(
                "Arclamp", "LFCFiber", OBJECT="autocal-lfc-all", TARGNAME="AFP Failsafe"
            ),
            "TARGNAME 'AFP Failsafe' names a target on a LFC frame",
        ),
        (
            _hdr("Arclamp", "Th_daily", shutters=SCI, OBJECT="autocal-thar-all-eve"),
            "disagrees with derived autocal-thar-sci",
        ),
        (
            # A daily-lamp OBJECT on a gold-lamp frame: the lamps must not conflate.
            _hdr("Arclamp", "Th_gold", OBJECT="autocal-thar-all-morn"),
            "disagrees with derived autocal-thar_gold-all",
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
def test_warnings(header, expected, caplog):
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
                "Arclamp", "Th_daily", HK, ("NO", "NO", "YES"), OBJECT="autocal-thar-hk"
            ),
            "ThAr_daily",
        ),
    ],
)
def test_object_labels_that_agree_do_not_warn(header, obj):
    parser = HeaderParser(header)
    assert not any("OBJECT" in w or "TARGNAME" in w for w in parser.warnings)
    assert parser.DRP_KEYWORDS["OBJECT"] == obj


def test_hk_frame_does_not_warn_missing_ccd_timing():
    header = _hdr("Arclamp", "Th_daily", shutters=HK, cameras=("NO", "NO", "YES"))
    for card in ("GRDATE", "GRDATE-E", "RDDATE", "RDDATE-E"):
        del header[card]
    parser = HeaderParser(header)
    assert not any("DATE" in w for w in parser.warnings)
    assert "READMODE" in parser.DRP_KEYWORDS  # still named by the ACF filenames


# Sources: a file path, a fits.Header and a plain mapping parse identically.


def _write_l0(tmp_path, header):
    path = tmp_path / "KP.20240405.00000.00.fits"
    hdul = fits.HDUList([fits.PrimaryHDU(header=fits.Header(header)), fits.ImageHDU()])
    hdul.writeto(path)
    return path


def test_sources_parse_identically(tmp_path):
    native = _hdr("Arclamp", "Th_daily", shutters=SCI)
    path = _write_l0(tmp_path, native)
    loaded = fits.getheader(path, ext=0)
    expected = HeaderParser(native).DRP_KEYWORDS
    for source in (path, str(path), loaded, dict(loaded)):
        assert HeaderParser(source).DRP_KEYWORDS == expected


def test_path_reads_only_primary(tmp_path, monkeypatch):
    path = _write_l0(tmp_path, _hdr("Bias"))
    calls = []
    real = header_module.fits.getheader

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(header_module.fits, "getheader", spy)
    HeaderParser(path)
    assert calls == [{"ext": 0}]


def test_header_is_not_modified():
    native = _hdr("Arclamp", "Th_daily", shutters=SCI)
    before = dict(native)
    parser = HeaderParser(native)
    assert parser.WMKO_HEADER is native
    assert native == before


# Every truth frame standardizes, carrying HeaderParser's keywords.


@pytest.mark.slow
@pytest.mark.requires_testdata
@pytest.mark.parametrize(
    "path",
    sorted(TESTDATA_L0.glob("*.fits")) or [None],
    ids=lambda p: getattr(p, "name", "none"),
)
def test_kpf0_stamps_every_keyword(path):
    l0 = KPF0.from_fits(str(path), standardize=True)
    keywords = HeaderParser(l0.headers["INSTRUMENT_HEADER"]).DRP_KEYWORDS
    assert keywords["OBSMODE"] != "Unknown"
    for key, value in keywords.items():
        ext = "QUALITY_CONTROL" if key.startswith("READTIM") else "PRIMARY"
        assert l0.headers[ext].get(key) == value, key
