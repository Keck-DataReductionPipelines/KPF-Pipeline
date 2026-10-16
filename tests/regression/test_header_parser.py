"""HeaderParser: frame identity, trace illumination, readout and warnings from a
native L0 PRIMARY header."""

import logging
from pathlib import Path

import pytest
from astropy.io import fits

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.utils import header as header_module
from kpfpipe.utils.header import HeaderParser

TESTDATA_L0 = Path(__file__).parent.parent / "testdata" / "L0" / "20240405"

ALL = ("open", "open", "open")
SCI = ("open", "closed", "closed")
SKY = ("closed", "open", "closed")
CAL = ("closed", "closed", "open")
HK = ("open", "closed", "open")
SHUT = ("closed", "closed", "closed")
CCDS = ("YES", "YES", "YES")


def _hdr(imtype, octagon="Home", shutters=ALL, cameras=CCDS, **cards):
    """A native PRIMARY as a dict: the identity cards plus regular-readout timing
    for both CCDs. ``cards`` adds or overrides any card (dashes as underscores)."""
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
    }
    header.update(zip(("SCISEL", "SKYSEL", "SIMCALSH"), shutters, strict=True))
    header.update(zip(("GREEN", "RED", "CA_HK"), cameras, strict=True))
    header.update({k.replace("_", "-"): v for k, v in cards.items()})
    return header


@pytest.mark.parametrize(
    "header, obsmode, obstype",
    [
        (_hdr("Bias", shutters=SHUT), "bias", "Bias"),
        (_hdr("Dark", shutters=SHUT), "dark", "Dark"),
        (_hdr("Object", "EtalonFiber", shutters=SKY), "star", "Object"),
        (_hdr("Object", OBJECT="SoCal"), "socal", "Object"),
        (_hdr("Object", OBJECT="x", TARGNAME=" SUN "), "socal", "Object"),
        (_hdr("Flatlamp", shutters=SHUT), "wideflat", "Flat"),
        (_hdr("Flatlamp", "BrdbandFiber"), "flat-all", "Flat"),
        (_hdr("Arclamp", "Th_daily", shutters=SCI), "thar-sci", "ThAr"),
        (_hdr("Arclamp", "Th_daily", shutters=SKY), "thar-sky", "ThAr"),
        (_hdr("Arclamp", "Th_daily", shutters=CAL), "thar-cal", "ThAr"),
        (_hdr("Arclamp", "Th_gold"), "thar_gold-all", "ThAr"),
        (_hdr("Arclamp", "U_daily"), "une-all", "UNe"),
        (_hdr("Arclamp", "U_gold", shutters=CAL), "une_gold-cal", "UNe"),
        (_hdr("Arclamp", "EtalonFiber"), "etalon-all", "Etalon"),
        (_hdr("Arclamp", "LFCFiber", cameras=("NO", "YES", "NO")), "lfc-all", "LFC"),
        (
            _hdr("Arclamp", "Th_daily", shutters=HK, cameras=("NO", "NO", "YES")),
            "thar-hk",
            "ThAr",
        ),
        (_hdr("Arclamp", "LFCFiber", shutters=SHUT), "unknown", None),
        (_hdr("Arclamp", "BrdbandFiber"), "unknown", None),
        (_hdr("Sky"), "unknown", None),
    ],
)
def test_frame_identity(header, obsmode, obstype):
    keywords = HeaderParser(header).DRP_KEYWORDS
    assert keywords["OBSMODE"] == obsmode
    assert keywords["OBSTYPE"] == obstype
    assert keywords["ISSOLAR"] is (obsmode == "socal")


def test_exptime_is_elapsed():
    # EPRV EXPTIME is the native ELAPSED, not the requested native EXPTIME.
    assert HeaderParser(_hdr("Bias")).DRP_KEYWORDS["EXPTIME"] == 30.0


@pytest.mark.parametrize(
    "header, obj",
    [
        # Science: the target, when native OBJECT and TARGNAME agree.
        (_hdr("Object", OBJECT=" 10700 ", TARGNAME="10700"), "10700"),
        (_hdr("Object", OBJECT="10700", TARGNAME="HD 10700"), "unknown"),
        (_hdr("Object", OBJECT="", TARGNAME=""), "unknown"),
        # Calibration: the OBSTYPE, when OBJECT is an agreeing autocal label.
        (_hdr("Bias", shutters=SHUT, OBJECT="autocal-bias", TARGNAME=""), "Bias"),
        (_hdr("Arclamp", "Th_daily", OBJECT="autocal-thar-all-morn"), "ThAr"),
        (_hdr("Arclamp", "Th_gold", OBJECT="autocal-thar-all-morn"), "unknown"),
        (_hdr("Arclamp", "LFCFiber", OBJECT="LFCStability"), "unknown"),
        (_hdr("Bias", shutters=SHUT, OBJECT=""), "unknown"),
        (
            _hdr(
                "Arclamp", "LFCFiber", OBJECT="autocal-lfc-all", TARGNAME="AFP Failsafe"
            ),
            "unknown",
        ),
        (_hdr("Sky", OBJECT="autocal-bias"), "unknown"),
    ],
)
def test_object(header, obj):
    assert HeaderParser(header).DRP_KEYWORDS["OBJECT"] == obj


def _traces(keywords, prefix):
    return [keywords.get(f"{prefix}{i}") for i in range(1, 6)]


@pytest.mark.parametrize(
    "header, traces, clsrc",
    [
        (
            _hdr("Arclamp", "Th_daily", shutters=SCI),
            ["None", "Th_daily", "Th_daily", "Th_daily", "None"],
            ["None", "ThAr", "ThAr", "ThAr", "None"],
        ),
        (
            _hdr("Flatlamp", "BrdbandFiber", shutters=CAL),
            ["None", "None", "None", "None", "BrdbandFiber"],
            ["None", "None", "None", "None", "Flat"],
        ),
        (
            _hdr("Object", "EtalonFiber"),
            ["Sky", "Target", "Target", "Target", "EtalonFiber"],
            ["Sky", "Target", "Target", "Target", "Etalon"],
        ),
        (
            # The octagon parks on the etalon, but the SimulCal shutter is shut.
            _hdr("Object", "EtalonFiber", shutters=("open", "open", "closed")),
            ["Sky", "Target", "Target", "Target", "None"],
            ["Sky", "Target", "Target", "Target", "None"],
        ),
        (_hdr("Bias", shutters=SHUT), ["None"] * 5, ["None"] * 5),
        (_hdr("Flatlamp", shutters=SHUT), ["None"] * 5, ["None"] * 5),
        (_hdr("Arclamp", "LFCFiber", shutters=SHUT), [None] * 5, [None] * 5),
    ],
)
def test_illumination(header, traces, clsrc):
    keywords = HeaderParser(header).DRP_KEYWORDS
    assert _traces(keywords, "TRACE") == traces
    assert _traces(keywords, "CLSRC") == clsrc


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
        (_hdr("Arclamp", "LFCFiber", shutters=SHUT), "cannot classify frame"),
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
            "TARGNAME 'AFP Failsafe' names a target on a lfc-all frame",
        ),
        (
            _hdr("Arclamp", "Th_daily", shutters=SCI, OBJECT="autocal-thar-all-eve"),
            "disagrees with OBSMODE 'thar-sci'",
        ),
        (
            # A daily-lamp OBJECT on a gold-lamp frame: the lamps must not conflate.
            _hdr("Arclamp", "Th_gold", OBJECT="autocal-thar-all-morn"),
            "disagrees with OBSMODE 'thar_gold-all'",
        ),
        (
            # The shutter-blind native pair: SKY-OBJ names the lamp on a shut fiber.
            _consistent("Arclamp", "Th_daily", SCI, "Th_daily", "Th_daily", "None"),
            "SKY-OBJ='Th_daily' disagrees with derived TRACE1='None'",
        ),
        (
            # CAL-OBJ carrying an octagon position index (20240405).
            _consistent("Arclamp", "Th_daily", ALL, "Th_daily", "Th_daily", "10"),
            "CAL-OBJ='10' disagrees with derived TRACE5='Th_daily'",
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
    "obj, octagon, shutters",
    [
        ("autocal-thar-all", "Th_daily", ALL),  # 2023's bare ThAr name
        ("autocal-wideflat-f5", "Home", SHUT),
        ("autocal-bias", "Home", SHUT),
    ],
)
def test_object_labels_that_agree_do_not_warn(obj, octagon, shutters):
    imtype = {"Th_daily": "Arclamp"}.get(octagon, "Flatlamp")
    if obj == "autocal-bias":
        imtype = "Bias"
    parser = HeaderParser(
        _hdr(imtype, octagon, shutters, OBJECT=obj, TARGNAME="unknown")
    )
    assert not any("OBJECT" in w or "TARGNAME" in w for w in parser.warnings)
    assert parser.DRP_KEYWORDS["OBJECT"] == parser.DRP_KEYWORDS["OBSTYPE"]


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
    assert keywords["OBSMODE"] != "unknown"
    for key, value in keywords.items():
        ext = "QUALITY_CONTROL" if key.startswith("READTIM") else "PRIMARY"
        assert l0.headers[ext].get(key) == value, key
