"""HeaderParser: read and sanitize a native (WMKO) L0 PRIMARY header."""

import logging
import re
from datetime import datetime
from pathlib import Path

from astropy.io import fits

logger = logging.getLogger(__name__)

# OCTAGON position -> the source it feeds the fibers, in OBJECT vocabulary.
_SOURCES = {
    "th_daily": "ThAr_daily",
    "th_gold": "ThAr_gold",
    "u_daily": "UNe_daily",
    "u_gold": "UNe_gold",
    "lfcfiber": "LFC",
    "etalonfiber": "Etalon",
    "brdbandfiber": "Flat",
}

# (SCISEL, SKYSEL, SIMCALSH) at exposure midpoint -> autocal fiber suffix.
_SHUTTER_SUFFIXES = {
    ("open", "open", "open"): "all",
    ("open", "closed", "closed"): "sci",
    ("closed", "open", "closed"): "sky",
    ("closed", "closed", "open"): "cal",
}

# TRACE# -> (EPRV fiber type, its shutter card, the native *-OBJ card naming its
# source, the per-fiber source card).
_TRACES = {
    1: ("SKY", "SKYSEL", "SKY-OBJ", "SKYSRC"),
    2: ("SCI", "SCISEL", "SCI-OBJ", "SCISRC"),
    3: ("SCI", "SCISEL", "SCI-OBJ", None),
    4: ("SCI", "SCISEL", "SCI-OBJ", None),
    5: ("CAL", "SIMCALSH", "CAL-OBJ", "CALSRC"),
}


class HeaderError(ValueError):
    """A native header that was read but carries a nonsensical value."""


def _norm(value):
    """A header value as a stripped, lowercased string (None -> '')."""
    return "" if value is None else str(value).strip().lower()


class HeaderParser:
    """Read and sanitize a native L0 PRIMARY header.

    The single source of truth for what an L0 frame is: its frame type, the
    illumination of each trace, which cameras read out, and its readout. Native
    cards are usually but not always reliable, so each derived value is inferred
    from the cards that record what the instrument actually did (IMTYPE, OCTAGON,
    the shutter and camera cards), and every fallback or disagreement with the
    other cards is warned.

    Parameters
    ----------
    source : str, pathlib.Path, astropy.io.fits.Header or Mapping
        An L0 FITS file, of which only the PRIMARY header is read (no
        extensions, so a night's scan stays fast), or a pre-loaded native
        header, which is read but never modified.

    Attributes
    ----------
    WMKO_HEADER : astropy.io.fits.Header or Mapping
        The native PRIMARY header.
    DRP_KEYWORDS : dict
        Sanitized and derived values under their EPRV PRIMARY names: OBSTYPE,
        OBSMODE, OBJECT, ISSOLAR, EXPTIME, GREEN, RED, CA_HK, EXPMETER, TRACE1-5,
        CLSRC1-5, CSRC1-5, CID1-5, SKYSRC, SCISRC, CALSRC, SIMULCAL, READMODE, READTIMG,
        READTIMR. A value that cannot be determined
        is omitted.
    warnings : list of str
        Every unreliable-metadata warning raised while parsing, also logged.

    Raises
    ------
    OSError
        If ``source`` is a file that cannot be read.
    HeaderError
        If the header was read but a value is nonsensical (a corrupt card, an
        unparseable ELAPSED or readout timestamp, or a negative read time).
    """

    def __init__(self, source):
        if isinstance(source, (str, Path)):
            self.WMKO_HEADER = fits.getheader(source, ext=0)
            self._label = Path(source).name
        else:
            self.WMKO_HEADER = source
            self._label = source.get("OFNAME") or "<header>"
        self.DRP_KEYWORDS = {}
        self.warnings = []

        # astropy reads a card's value lazily, so a corrupt one raises on lookup.
        try:
            self._cameras()
            self._frame_identity()
            self._illumination()
            self._readout()
        except fits.verify.VerifyError as e:
            raise HeaderError(f"{self._label}: {e}") from None

    def _warn(self, message):
        """Record one unreliable-metadata warning and log it with the file label."""
        self.warnings.append(message)
        logger.warning("%s: %s", self._label, message)

    def _cameras(self):
        """Stamp GREEN, RED, CA_HK, EXPMETER: whether each camera read out, from
        its native YES/NO card. An absent or unrecognized card is warned and
        omitted."""
        for card in ("GREEN", "RED", "CA_HK", "EXPMETER"):
            value = _norm(self.WMKO_HEADER.get(card))
            if value in ("yes", "no"):
                self.DRP_KEYWORDS[card] = value == "yes"
            else:
                self._warn(f"{card}={self.WMKO_HEADER.get(card)!r} is not YES/NO")

    def _classify(self):
        """(OBSTYPE, calibration OBJECT) from IMTYPE and OCTAGON, and on a bias
        or dark the shutters; see ``_frame_identity``. The OBJECT is None on a
        Target or Unknown frame."""
        native = self.WMKO_HEADER
        imtype = _norm(native.get("IMTYPE"))
        source = _SOURCES.get(_norm(native.get("OCTAGON")))

        if imtype in ("bias", "dark"):
            # No fiber may be lit on a bias or dark: every shutter must record closed.
            if any(
                _norm(native.get(k)) != "closed"
                for k in ("SCISEL", "SKYSEL", "SIMCALSH")
            ):
                return "Unknown", None
            return imtype.capitalize(), imtype.capitalize()
        if imtype == "object":
            return "Target", None
        if imtype == "flatlamp" and _norm(native.get("OCTAGON")) == "home":
            return "Flat", "WideFlat"
        if imtype == "flatlamp" and source == "Flat":
            return "Flat", "Flat"
        if imtype == "arclamp" and source in ("LFC", "Etalon"):
            return source, source
        if imtype == "arclamp" and source and source.startswith(("ThAr", "UNe")):
            return "Arclamp", source
        return "Unknown", None

    def _frame_identity(self):
        """Stamp OBSTYPE, OBSMODE, OBJECT, ISSOLAR and EXPTIME.

        IMTYPE gives OBSTYPE (Bias, Dark, Flat, Arclamp, LFC, Etalon, Target)
        and, on a lamp frame, OCTAGON the lamp; anything else, including a
        bias or dark with any shutter not recorded closed, is ``Unknown``.
        OBSMODE is ``Science`` for a Target, ``Calibration`` for any other
        known OBSTYPE, else ``Unknown``. OBJECT, set once OBSTYPE is known, is
        sanitized by ``_object``. EXPTIME is the EPRV exposure time: the native
        ELAPSED, not the requested native EXPTIME.
        """
        native = self.WMKO_HEADER
        obstype, cal_object = self._classify()
        self.DRP_KEYWORDS["OBSTYPE"] = obstype
        if obstype == "Unknown":
            cards = ", ".join(
                f"{k}={native.get(k)!r}"
                for k in ("IMTYPE", "OCTAGON", "SCISEL", "SKYSEL", "SIMCALSH")
            )
            self._warn(f"cannot classify frame ({cards})")
            self.DRP_KEYWORDS["OBSMODE"] = "Unknown"
        else:
            self.DRP_KEYWORDS["OBSMODE"] = (
                "Science" if obstype == "Target" else "Calibration"
            )

        self.DRP_KEYWORDS["OBJECT"] = self._object(cal_object)
        self.DRP_KEYWORDS["ISSOLAR"] = self.DRP_KEYWORDS["OBJECT"] == "SoCal"

        elapsed = native.get("ELAPSED")
        if elapsed in (None, ""):
            self._warn("ELAPSED absent; EXPTIME not set")
        else:
            try:
                self.DRP_KEYWORDS["EXPTIME"] = float(elapsed)
            except ValueError:
                raise HeaderError(
                    f"{self._label}: ELAPSED={elapsed!r} is not a number"
                ) from None

    def _object(self, cal_object):
        """The sanitized OBJECT: the target on a Target frame, else `cal_object`.

        WMKO uses OBJECT as free-form metadata; the DRP uses it to name what
        was observed. A solar Target frame (native OBJECT ``SoCal`` and TARGNAME
        ``Sun``) is ``SoCal``; any other Target frame's OBJECT is its target
        identifier (e.g. ``10700``): the native OBJECT, trusted only when it and
        TARGNAME agree, or TARGNAME when the native OBJECT is blank (warned).
        A Target frame failing its check (including only one
        of the two cards naming the Sun) is ``Unknown``. A calibration frame's OBJECT is
        `cal_object`, derived from IMTYPE/OCTAGON alone (e.g. ``ThAr_daily``,
        ``WideFlat``); a native OBJECT that is no ``autocal-*`` label agreeing
        with it and the lit fibers, or a TARGNAME naming a target, is only
        warned. An Unknown frame's OBJECT is ``Unknown``.
        """
        native = self.WMKO_HEADER
        obj = str(native.get("OBJECT") or "").strip()
        targname = str(native.get("TARGNAME") or "").strip()
        obstype = self.DRP_KEYWORDS["OBSTYPE"]
        if obstype == "Unknown":
            return "Unknown"

        if obstype == "Target":
            if (obj, targname) == ("SoCal", "Sun"):
                return "SoCal"
            solar = {obj.lower(), targname.lower()} & {"socal", "sun"}
            if obj and obj == targname and not solar:
                return obj
            if not obj and targname and not solar:
                self._warn(f"OBJECT blank; OBJECT is TARGNAME {targname!r}")
                return targname
            self._warn(
                f"OBJECT {obj!r} disagrees with TARGNAME {targname!r}; "
                "OBJECT is Unknown"
            )
            return "Unknown"

        family = cal_object.lower().replace("_daily", "")
        if family in ("bias", "dark", "wideflat"):
            expected = family
        elif tuple(self.DRP_KEYWORDS.get(k) for k in ("GREEN", "RED", "CA_HK")) == (
            False,
            False,
            True,
        ):
            expected = f"{family}-hk"
        else:
            shutters = tuple(
                _norm(native.get(k)) for k in ("SCISEL", "SKYSEL", "SIMCALSH")
            )
            expected = f"{family}-{_SHUTTER_SUFFIXES.get(shutters)}"
        match = re.match(r"autocal-([a-z_]+)(?:-([a-z]+))?", obj.lower())
        if not match:
            self._warn(f"OBJECT {obj!r} is no autocal label")
        else:
            lamp, fibers = match.groups()
            label = (
                lamp
                if lamp in ("bias", "dark", "wideflat")
                else (f"{lamp}-{fibers or 'all'}")
            )
            if label != expected:
                self._warn(f"OBJECT {obj!r} disagrees with derived autocal-{expected}")
        if targname.lower() not in ("", "unknown"):
            self._warn(f"TARGNAME {targname!r} names a target on a {cal_object} frame")
        return cal_object

    def _illumination(self):
        """Stamp SKYSRC, SCISRC, CALSRC (the source lighting each fiber), and the
        EPRV TRACE1-5 (fiber types), CLSRC1-5 (calibration lamp or ``None``) and
        CSRC1-5/CID1-5 (``None``; AstroQuery fills the SCI traces of a Target).

        Each trace's source follows from the frame and its fiber's midpoint
        shutter, not from the native SCI-OBJ/SKY-OBJ/CAL-OBJ, which can ignore
        the shutters or carry placeholders; those only cross-check the result.
        On a lamp frame an open fiber sees the OCTAGON source; on a Target
        frame SCI sees the Target, SKY the Sky and an open CAL the OCTAGON
        source (``Unknown``, warned, if OCTAGON names no lamp). On a solar
        frame SCI sees ``SoCal`` only when the SoCal feed's own shutter
        (SOLSCISH) is open; bias, dark and wide flat light no trace
        (``None``). A shutter it needs that is absent or neither open nor
        closed makes that source ``Unknown`` (warned), as is every source of
        an Unknown frame.

        SIMULCAL is the convenience summary of CALSRC: ``On`` when a Target
        frame's CAL fiber sees the etalon (the only simultaneous calibration
        source), ``Off`` when it is dark, ``Unknown`` when CALSRC is or names
        another source (warned); ``N/A`` on a calibration frame and ``Unknown``
        on an Unknown one.
        """
        native = self.WMKO_HEADER
        obj = self.DRP_KEYWORDS["OBJECT"]
        obstype = self.DRP_KEYWORDS["OBSTYPE"]

        octagon = str(native.get("OCTAGON") or "").strip()
        for trace, (fiber, shutter, obj_card, src_card) in _TRACES.items():
            self.DRP_KEYWORDS[f"TRACE{trace}"] = fiber
            solar_sci = fiber == "SCI" and obj == "SoCal"
            if solar_sci:
                shutter = "SOLSCISH"
            state = _norm(native.get(shutter))
            # The source in native SCI-OBJ/SKY-OBJ/CAL-OBJ terms, for the cross-check.
            lit = {"open": "Target" if solar_sci else octagon, "closed": "None"}.get(
                state, "Unknown"
            )
            if obstype == "Unknown":
                source = "Unknown"
            elif obj in ("Bias", "Dark", "WideFlat"):
                source = "None"
            elif obstype == "Target" and not solar_sci:
                source = {"SCI": "Target", "SKY": "Sky"}.get(fiber, lit)
            else:
                source = lit

            src = _SOURCES.get(source.lower(), source)
            if source == "Target" and obj == "SoCal":
                src = "SoCal"
            # An open CAL fiber on a Target frame sees an OCTAGON naming no lamp.
            if state == "open" and source == octagon and _norm(octagon) not in _SOURCES:
                self._warn(
                    f"OCTAGON={native.get('OCTAGON')!r} names no calibration "
                    f"source; {src_card} is Unknown"
                )
                src = "Unknown"
            if src_card:
                self.DRP_KEYWORDS[src_card] = src
            # CLSRC# names a calibration lamp only; CSRC#/CID# await AstroQuery.
            lamp = src in _SOURCES.values() or src == "Unknown"
            self.DRP_KEYWORDS[f"CLSRC{trace}"] = src if lamp else "None"
            self.DRP_KEYWORDS[f"CSRC{trace}"] = "None"
            self.DRP_KEYWORDS[f"CID{trace}"] = "None"

            # Trace 2 stands for 3-4, which share SCI-OBJ/SCISEL: warn once per card.
            if src_card and source == "Unknown" and obstype != "Unknown":
                self._warn(
                    f"{shutter}={native.get(shutter)!r} is not open/closed; "
                    f"{src_card} is Unknown"
                )
            if (
                src_card
                and source != "Unknown"
                and _norm(native.get(obj_card)) != source.lower()
            ):
                self._warn(
                    f"{obj_card}={native.get(obj_card)!r} disagrees with derived "
                    f"{src_card}={src!r} ({shutter}={native.get(shutter)!r})"
                )

        mode = self.DRP_KEYWORDS["OBSMODE"]
        if mode == "Science":
            calsrc = self.DRP_KEYWORDS["CALSRC"]
            if calsrc not in ("Etalon", "None", "Unknown"):
                self._warn(
                    f"CALSRC={calsrc!r} is no simultaneous calibration source; "
                    "SIMULCAL is Unknown"
                )
            self.DRP_KEYWORDS["SIMULCAL"] = {"Etalon": "On", "None": "Off"}.get(
                calsrc, "Unknown"
            )
        else:
            self.DRP_KEYWORDS["SIMULCAL"] = "N/A" if mode == "Calibration" else mode

    def _readout(self):
        """Stamp READTIMG/READTIMR and READMODE.

        READTIM{G,R} is the shutter-close to file-write duration. The ACF
        waveform filenames name the read mode outright; failing that, the read
        time separates ~12 s fast readout from ~48 s regular (logged at debug,
        since most frames need it). A camera that read out but lacks its timing
        cards is warned; a negative read time (file written before the shutter
        closed) or an unparseable timestamp raises HeaderError.
        """
        native = self.WMKO_HEADER
        read_time = {}
        for chip, prefix, key in (
            ("GREEN", "GR", "READTIMG"),
            ("RED", "RD", "READTIMR"),
        ):
            start, end = native.get(f"{prefix}DATE"), native.get(f"{prefix}DATE-E")
            if start and end:
                try:
                    read_time[chip] = (
                        datetime.fromisoformat(start) - datetime.fromisoformat(end)
                    ).total_seconds()
                except (TypeError, ValueError):
                    raise HeaderError(
                        f"{self._label}: {prefix}DATE={start!r} or "
                        f"{prefix}DATE-E={end!r} is not an ISO timestamp"
                    ) from None
                if read_time[chip] < 0:
                    raise HeaderError(
                        f"{self._label}: {prefix}DATE={start!r} precedes "
                        f"{prefix}DATE-E={end!r}; negative {chip} read time"
                    )
                self.DRP_KEYWORDS[key] = round(read_time[chip], 3)
            elif self.DRP_KEYWORDS.get(chip):
                self._warn(f"{chip}=YES but {prefix}DATE/{prefix}DATE-E absent")

        acf = f"{native.get('GRACFFLN', '')} {native.get('RDACFFLN', '')}"
        if "fast" in acf:
            self.DRP_KEYWORDS["READMODE"] = "fast"
        elif "regular" in acf:
            self.DRP_KEYWORDS["READMODE"] = "regular"
        elif read_time:
            fast = min(read_time.values()) < 20
            self.DRP_KEYWORDS["READMODE"] = "fast" if fast else "regular"
            logger.debug(
                "%s: ACF filenames name no read mode; READMODE inferred from read time",
                self._label,
            )
