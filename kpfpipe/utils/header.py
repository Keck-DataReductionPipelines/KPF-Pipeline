"""HeaderParser: read and sanitize a native (WMKO) L0 PRIMARY header."""

import logging
import re
from datetime import datetime
from pathlib import Path

from astropy.io import fits

logger = logging.getLogger(__name__)

# Arclamp OCTAGON position -> OBSMODE lamp family.
_LAMP_FAMILIES = {
    "th_daily": "thar",
    "th_gold": "thar_gold",
    "u_daily": "une",
    "u_gold": "une_gold",
    "lfcfiber": "lfc",
    "etalonfiber": "etalon",
}

# (SCISEL, SKYSEL, SIMCALSH) at exposure midpoint -> OBSMODE fiber suffix.
_SHUTTER_SUFFIXES = {
    ("open", "open", "open"): "all",
    ("open", "closed", "closed"): "sci",
    ("closed", "open", "closed"): "sky",
    ("closed", "closed", "open"): "cal",
}

# OBSMODE family -> EPRV OBSTYPE.
_OBSTYPES = {
    "bias": "Bias",
    "dark": "Dark",
    "flat": "Flat",
    "wideflat": "Flat",
    "thar": "ThAr",
    "thar_gold": "ThAr",
    "une": "UNe",
    "une_gold": "UNe",
    "lfc": "LFC",
    "etalon": "Etalon",
    "star": "Object",
    "socal": "Object",
}

# KPF fiber-source name (TRACE#) -> EPRV illumination source (CLSRC#); names not
# listed (Target, Sky, None) pass through.
_EPRV_SOURCES = {
    "th_daily": "ThAr",
    "th_gold": "ThAr",
    "u_daily": "UNe",
    "u_gold": "UNe",
    "lfcfiber": "LFC",
    "etalonfiber": "Etalon",
    "brdbandfiber": "Flat",
}

# TRACE# -> (fiber, its shutter card, the native *-OBJ card naming its source).
_TRACES = {
    1: ("SKY", "SKYSEL", "SKY-OBJ"),
    2: ("SCI", "SCISEL", "SCI-OBJ"),
    3: ("SCI", "SCISEL", "SCI-OBJ"),
    4: ("SCI", "SCISEL", "SCI-OBJ"),
    5: ("CAL", "SIMCALSH", "CAL-OBJ"),
}


def _norm(value):
    """A header value as a stripped, lowercased string (None -> '')."""
    return "" if value is None else str(value).strip().lower()


class HeaderParser:
    """Read and sanitize a native L0 PRIMARY header.

    The single source of truth for what an L0 frame is: its frame type, the
    illumination of each trace, and its readout. Native cards are usually but
    not always reliable, so each derived value is inferred from the cards that
    record what the instrument actually did (IMTYPE, OCTAGON, the shutter and
    camera cards), and every fallback or disagreement with the other cards is
    warned.

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
        Sanitized and derived values under their EPRV PRIMARY names: OBSMODE,
        OBSTYPE, ISSOLAR, OBJECT, EXPTIME, TRACE1-5, CLSRC1-5, READMODE,
        READTIMG, READTIMR. A value that cannot be determined is omitted.
    warnings : list of str
        Every unreliable-metadata warning raised while parsing, also logged.
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

        self._frame_identity()
        self._illumination()
        self._readout()

    def _warn(self, message):
        """Record and log one unreliable-metadata warning."""
        message = f"{self._label}: {message}"
        self.warnings.append(message)
        logger.warning(message)

    def _obsmode(self):
        """The sanitized frame label; see ``_frame_identity``."""
        native = self.WMKO_HEADER
        imtype = _norm(native.get("IMTYPE"))
        octagon = _norm(native.get("OCTAGON"))

        if imtype in ("bias", "dark"):
            return imtype
        if imtype == "object":
            names = {_norm(native.get(k)) for k in ("OBJECT", "TARGNAME")}
            return "socal" if names & {"socal", "sun"} else "star"
        if imtype == "flatlamp" and octagon == "home":
            return "wideflat"

        if imtype == "flatlamp" and octagon == "brdbandfiber":
            family = "flat"
        elif imtype == "arclamp" and octagon in _LAMP_FAMILIES:
            family = _LAMP_FAMILIES[octagon]
        else:
            return "unknown"

        cameras = tuple(_norm(native.get(k)) for k in ("GREEN", "RED", "CA_HK"))
        if cameras == ("no", "no", "yes"):
            return f"{family}-hk"
        shutters = tuple(_norm(native.get(k)) for k in ("SCISEL", "SKYSEL", "SIMCALSH"))
        suffix = _SHUTTER_SUFFIXES.get(shutters)
        return f"{family}-{suffix}" if suffix else "unknown"

    def _frame_identity(self):
        """Stamp OBSMODE, OBSTYPE, ISSOLAR, OBJECT and EXPTIME.

        OBSMODE is what OBJECT would say if it were reliable: ``bias``,
        ``dark``, ``wideflat``, ``star``, ``socal``, ``{lamp}-{fibers}`` (e.g.
        ``thar-all``, ``flat-sci``, ``thar_gold-hk``) or ``unknown``. IMTYPE
        gives the frame type and, on an arclamp, OCTAGON the lamp; the
        midpoint shutters give the lit fibers, and a frame that read out only
        the Ca HK camera is ``-hk``. OBJECT is sanitized by ``_object``.
        EXPTIME is the EPRV exposure time: the native ELAPSED, not the
        requested native EXPTIME.
        """
        native = self.WMKO_HEADER
        obsmode = self._obsmode()
        family = obsmode.split("-")[0]
        self.DRP_KEYWORDS["OBSMODE"] = obsmode
        self.DRP_KEYWORDS["OBSTYPE"] = _OBSTYPES.get(family)
        self.DRP_KEYWORDS["ISSOLAR"] = obsmode == "socal"
        if obsmode == "unknown":
            cards = ", ".join(
                f"{k}={native.get(k)!r}"
                for k in ("IMTYPE", "OCTAGON", "SCISEL", "SKYSEL", "SIMCALSH")
                + ("GREEN", "RED", "CA_HK")
            )
            self._warn(f"cannot classify frame ({cards})")

        self.DRP_KEYWORDS["OBJECT"] = self._object(obsmode)

        elapsed = native.get("ELAPSED")
        if elapsed in (None, ""):
            self._warn("ELAPSED absent; EXPTIME not set")
        else:
            self.DRP_KEYWORDS["EXPTIME"] = float(elapsed)

    def _object(self, obsmode):
        """The sanitized OBJECT: the target on a science frame, else OBSTYPE.

        WMKO uses OBJECT as free-form metadata; the DRP uses it to name what
        was observed. A science frame's OBJECT is its target identifier (e.g.
        ``10700``), trusted only when the native OBJECT and TARGNAME agree. A
        calibration frame's OBJECT is its OBSTYPE (e.g. ``ThAr``), trusted only
        when the native OBJECT is an ``autocal-*`` label agreeing with OBSMODE
        and TARGNAME names no target. Anything else -- an engineering string
        such as ``LFCStability``, a blank, a mismatch -- is ``unknown``.
        """
        native = self.WMKO_HEADER
        obj = str(native.get("OBJECT") or "").strip()
        targname = str(native.get("TARGNAME") or "").strip()
        if obsmode == "unknown":
            return "unknown"

        if self.DRP_KEYWORDS["OBSTYPE"] == "Object":
            if obj and obj == targname:
                return obj
            self._warn(
                f"OBJECT {obj!r} disagrees with TARGNAME {targname!r}; "
                "OBJECT is unknown"
            )
            return "unknown"

        match = re.match(r"autocal-([a-z]+)(?:-([a-z]+))?", obj.lower())
        if not match:
            self._warn(f"OBJECT {obj!r} is no autocal label; OBJECT is unknown")
            return "unknown"
        lamp, fibers = match.groups()
        expected = (
            lamp
            if lamp in ("bias", "dark", "wideflat")
            else f"{lamp}-{fibers or 'all'}"
        )
        if expected != obsmode:
            self._warn(
                f"OBJECT {obj!r} disagrees with OBSMODE {obsmode!r}; OBJECT is unknown"
            )
            return "unknown"
        if targname.lower() not in ("", "unknown"):
            self._warn(
                f"TARGNAME {targname!r} names a target on a {obsmode} frame; "
                "OBJECT is unknown"
            )
            return "unknown"
        return self.DRP_KEYWORDS["OBSTYPE"]

    def _illumination(self):
        """Stamp TRACE1-5 (KPF source names) and CLSRC1-5 (EPRV names).

        Each trace's source follows from OBSMODE and its fiber's midpoint
        shutter, not from the native SCI-OBJ/SKY-OBJ/CAL-OBJ, which can ignore
        the shutters or carry placeholders; those only cross-check the result.
        On a lamp frame an open fiber sees the OCTAGON source; on a science
        frame SCI sees the Target, SKY the Sky and an open CAL the OCTAGON
        source; bias, dark and wideflat light no trace. An unknown frame gets
        neither card.
        """
        native = self.WMKO_HEADER
        obsmode = self.DRP_KEYWORDS["OBSMODE"]
        if obsmode == "unknown":
            return

        octagon = str(native.get("OCTAGON") or "").strip()
        for trace, (fiber, shutter, obj_card) in _TRACES.items():
            is_open = _norm(native.get(shutter)) == "open"
            if obsmode in ("bias", "dark", "wideflat"):
                source = "None"
            elif obsmode in ("star", "socal"):
                source = {"SCI": "Target", "SKY": "Sky"}.get(
                    fiber, octagon if is_open else "None"
                )
            else:
                source = octagon if is_open else "None"

            self.DRP_KEYWORDS[f"TRACE{trace}"] = source
            self.DRP_KEYWORDS[f"CLSRC{trace}"] = _EPRV_SOURCES.get(
                source.lower(), source
            )

            # Trace 2 stands for 3-4, which share SCI-OBJ: warn once per card.
            if trace in (1, 2, 5) and _norm(native.get(obj_card)) != source.lower():
                self._warn(
                    f"{obj_card}={native.get(obj_card)!r} disagrees with derived "
                    f"TRACE{trace}={source!r} ({shutter}={native.get(shutter)!r})"
                )

    def _readout(self):
        """Stamp READTIMG/READTIMR and READMODE.

        READTIM{G,R} is the shutter-close to file-write duration. The ACF
        waveform filenames name the read mode outright; failing that, the read
        time separates ~12 s fast readout from ~48 s regular (logged at debug,
        since most frames need it). A camera that read out but lacks its timing
        cards is warned.
        """
        native = self.WMKO_HEADER
        read_time = {}
        for chip, prefix, key in (
            ("GREEN", "GR", "READTIMG"),
            ("RED", "RD", "READTIMR"),
        ):
            start, end = native.get(f"{prefix}DATE"), native.get(f"{prefix}DATE-E")
            if start and end:
                read_time[chip] = (
                    datetime.fromisoformat(start) - datetime.fromisoformat(end)
                ).total_seconds()
                self.DRP_KEYWORDS[key] = round(read_time[chip], 3)
            elif _norm(native.get(chip)) == "yes":
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
