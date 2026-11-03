"""HeaderParser: read and sanitize a native (WMKO) L0 PRIMARY header."""

import logging
import re
from datetime import datetime
from pathlib import Path

from astropy.io import fits

from kpfpipe import SOLAR_SYSTEM_OBJECTS
from kpfpipe.utils.kpf import kpf_target_name

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

# The fiber shutter cards, and their midpoint states -> autocal fiber suffix.
_SHUTTERS = ("SCISEL", "SKYSEL", "SIMCALSH")
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
    """A native header that was read but carries a nonsensical value.

    A target-identity error (OBJECT, TARGNAME, GAIAID) is one the raw L0 header
    must fix; its message says so with the greppable phrase ``L0 header must be
    fixed:`` and the action needed. Other errors (e.g. ELAPSED, readout
    timestamps) likely reflect an operations fault, not a card to edit."""


def _lower(value):
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
        OBSMODE, OBJECT, ISSOLAR, GAIAID, EXPTIME, GREEN, RED, CA_HK, EXPMETER,
        TRACE1-5, CLSRC1-5, CSRC1-5, CID1-5, SKYSRC, SCISRC, CALSRC, SIMULCAL,
        READMODE, READTIMG, READTIMR. A value that cannot be determined is
        omitted.
    warnings : list of str
        Every unreliable-metadata warning raised while parsing, also logged.

    Raises
    ------
    OSError
        If ``source`` is a file that cannot be read.
    HeaderError
        If the header was read but a value is nonsensical (a corrupt card, an
        unparseable ELAPSED or readout timestamp, a negative read time, a
        malformed or conflicting GAIAID, an ambiguous planet/component letter, or
        a Target whose OBJECT is ``Unknown`` with GAIAID blank).
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
            value = _lower(self.WMKO_HEADER.get(card))
            if value in ("yes", "no"):
                self.DRP_KEYWORDS[card] = value == "yes"
            else:
                self._warn(f"{card}={self.WMKO_HEADER.get(card)!r} is not YES/NO")

    def _classify(self):
        """(OBSTYPE, calibration OBJECT) from IMTYPE, OCTAGON and the shutters;
        the OBJECT is None on a Target or Unknown frame.

        An IMTYPE ``Object`` frame is a Target only if nothing else in the header
        says otherwise: SCISEL open, no calibration word in OBJECT, and any
        SoCal/Sun name exactly the (``SoCal``, ``Sun``) pair."""
        native = self.WMKO_HEADER
        imtype = _lower(native.get("IMTYPE"))
        octagon = _lower(native.get("OCTAGON"))
        source = _SOURCES.get(octagon)

        def unknown(why=""):
            keys = ("IMTYPE", "OCTAGON", *_SHUTTERS, "OBJECT", "TARGNAME")
            cards = ", ".join(f"{k}={native.get(k)!r}" for k in keys)
            self._warn(f"cannot classify frame{why} ({cards})")
            return "Unknown", None

        if imtype in ("bias", "dark"):
            # No fiber may be lit on a bias or dark: every shutter must record closed.
            if any(_lower(native.get(k)) != "closed" for k in _SHUTTERS):
                return unknown()
            return imtype.capitalize(), imtype.capitalize()
        if imtype == "object":
            obj = str(native.get("OBJECT") or "").strip()
            targname = str(native.get("TARGNAME") or "").strip()
            calibration_words = {
                "autocal",
                "manualcal",
                "bias",
                "dark",
                "flat",
                "thar",
                "une",
                "fast",
                "slow",
            }
            if _lower(native.get("SCISEL")) != "open":
                return unknown(
                    f": IMTYPE 'Object' but SCISEL={native.get('SCISEL')!r}; "
                    "a science frame needs the science shutter open"
                )
            if set(re.split(r"[^a-z0-9]+", obj.lower())) & calibration_words:
                return unknown(
                    f": IMTYPE 'Object' but OBJECT {obj!r} is a calibration label"
                )
            solar = any(s.lower().startswith(("socal", "sun")) for s in (obj, targname))
            if solar and (obj, targname) != ("SoCal", "Sun"):
                return unknown(
                    f": IMTYPE 'Object' but OBJECT {obj!r} / TARGNAME {targname!r} "
                    "is not the SoCal/Sun pair"
                )
            return "Target", None
        if imtype == "flatlamp" and octagon == "home":
            return "Flat", "WideFlat"
        if imtype == "flatlamp" and source == "Flat":
            return "Flat", "Flat"
        if imtype == "arclamp" and source in ("LFC", "Etalon"):
            return source, source
        if imtype == "arclamp" and source and source.startswith(("ThAr", "UNe")):
            return "Arclamp", source
        return unknown()

    def _frame_identity(self):
        """Stamp OBSTYPE, OBSMODE, OBJECT, ISSOLAR, GAIAID and EXPTIME (the
        native ELAPSED, not the requested native EXPTIME).

        WMKO uses OBJECT as free-form metadata; the DRP uses it to name what was
        observed: the target, the calibration, or ``Unknown``.
        """
        obstype, cal_object = self._classify()
        self.DRP_KEYWORDS["OBSTYPE"] = obstype
        self.DRP_KEYWORDS["OBSMODE"] = {
            "Target": "Science",
            "Unknown": "Unknown",
        }.get(obstype, "Calibration")

        if obstype == "Unknown":
            self.DRP_KEYWORDS["OBJECT"] = "Unknown"
        elif obstype == "Target":
            self.DRP_KEYWORDS["OBJECT"] = self._target_object()
        else:
            # Lamp light reaches the CCD only through an open fiber shutter; the
            # wide flat illuminates the CCD directly, so its shutters stay closed.
            closed = all(_lower(self.WMKO_HEADER.get(k)) == "closed" for k in _SHUTTERS)
            if (
                obstype in ("Flat", "LFC", "Etalon", "Arclamp")
                and cal_object != "WideFlat"
                and closed
            ):
                raise HeaderError(
                    f"{self._label}: {cal_object} frame with every fiber shutter "
                    f"closed ({', '.join(_SHUTTERS)}): no lamp light reached the "
                    "CCD, so the frame is no calibration"
                )
            self._check_cal_object(cal_object)
            self.DRP_KEYWORDS["OBJECT"] = cal_object
        self.DRP_KEYWORDS["ISSOLAR"] = self.DRP_KEYWORDS["OBJECT"] == "SoCal"
        self.DRP_KEYWORDS["GAIAID"] = self._gaiaid()

        elapsed = self.WMKO_HEADER.get("ELAPSED")
        if elapsed in (None, ""):
            self._warn("ELAPSED absent; EXPTIME not set")
        else:
            try:
                self.DRP_KEYWORDS["EXPTIME"] = float(elapsed)
            except ValueError:
                raise HeaderError(
                    f"{self._label}: ELAPSED={elapsed!r} is not a number"
                ) from None

    def _native_gaiaid(self):
        """The native GAIAID card, stripped; '' for blank or 'None'."""
        raw = str(self.WMKO_HEADER.get("GAIAID") or "").strip()
        return "" if raw.lower() == "none" else raw

    def _gaiaid(self):
        """The sanitized GAIAID (``Gaia DR3 <id>``, ``Gaia DR2 <id>`` or
        ``Gaia <id>``); None for a blank card, a frame other than a non-solar
        Target, or a solar-system OBJECT.

        A non-blank card that is no Gaia id raises HeaderError. A Gaia-named
        OBJECT must name the same source: a blank GAIAID takes the OBJECT's, and
        whichever carries the release lends it to the other.
        """
        if self.DRP_KEYWORDS["OBSTYPE"] != "Target" or self.DRP_KEYWORDS["ISSOLAR"]:
            return None
        raw = self._native_gaiaid()
        obj = self.DRP_KEYWORDS["OBJECT"]
        if obj in SOLAR_SYSTEM_OBJECTS:
            if raw:
                self._warn(
                    f"GAIAID {raw!r} on solar-system target {obj!r}; GAIAID blank"
                )
            return None
        catalog, gaiaid, _ = kpf_target_name(raw)
        if catalog != "Gaia":
            if raw:
                raise HeaderError(
                    f"{self._label}: GAIAID {raw!r} is not a Gaia DR2/DR3 "
                    "source_id. Accepted: '[Gaia] DR3 <id>', '[Gaia] DR2 <id>' "
                    "(EDR3 = DR3) or 'Gaia <id>' with a 10-19 digit source_id, or a "
                    "bare 17-19 digit id; a shorter bare number is refused, as it "
                    "could be another catalog's (e.g. an HD number). L0 header must "
                    "be fixed: correct GAIAID (a short id needs its release, e.g. "
                    "'Gaia DR3 4295806720') or blank it."
                )
            gaiaid = None

        if kpf_target_name(obj)[0] != "Gaia":
            return gaiaid
        if gaiaid is None:
            self._warn(f"GAIAID blank; GAIAID is OBJECT {obj!r}")
            return obj
        *obj_release, obj_id = obj.split()[1:]
        *gaia_release, gaia_id = gaiaid.split()[1:]
        if obj_id != gaia_id:
            raise HeaderError(
                f"{self._label}: OBJECT {obj!r} and GAIAID {gaiaid!r} name "
                f"different Gaia sources (source_id {obj_id} vs {gaia_id}). L0 "
                "header must be fixed: make OBJECT and GAIAID name the same source."
            )
        if obj_release and gaia_release and obj != gaiaid:
            raise HeaderError(
                f"{self._label}: OBJECT {obj!r} and GAIAID {gaiaid!r} name "
                f"different Gaia sources: one source_id under releases "
                f"{obj_release[0]} and {gaia_release[0]}. DR2 and DR3 source_ids are "
                "separate catalogs (a DR2 id can name another source, or none, in "
                "DR3), so the pair is not taken as one star. L0 header must be "
                "fixed: give OBJECT and GAIAID the same release, or blank one."
            )
        resolved = max(obj, gaiaid, key=len)
        if obj != resolved:
            self._warn(
                f"OBJECT {obj!r} takes the GAIAID release; OBJECT is {resolved!r}"
            )
            self.DRP_KEYWORDS["OBJECT"] = resolved
        if gaiaid != resolved:
            self._warn(
                f"GAIAID {raw!r} takes the OBJECT release; GAIAID is {resolved!r}"
            )
        return resolved

    def _target_object(self):
        """A Target frame's OBJECT, from the native OBJECT and TARGNAME:

        - ``SoCal`` for the solar pair (OBJECT ``SoCal``, TARGNAME ``Sun``);
        - the solar-system body (reference/solar_system_objects.txt) the cards
          name as a whole word (``Io`` with ``Io Eclipse`` -> ``Io``);
        - else the canonical target name (``kpf_target_name``, e.g. ``HD 10700``;
          a planet letter is dropped for the host star). The cards must agree,
          or one must be the other's bare catalog prefix (``TOI`` with
          ``TOI-969``) or lack only its component letter (``WASP-173`` with
          ``WASP-173B`` -> ``WASP-173 B``); a planet letter on the other card
          then raises HeaderError.

        A blank card yields the other; anything else is ``Unknown``
        (``_unknown_target``).
        """
        obj = str(self.WMKO_HEADER.get("OBJECT") or "").strip()
        targname = str(self.WMKO_HEADER.get("TARGNAME") or "").strip()
        if (obj, targname) == ("SoCal", "Sun"):
            return "SoCal"
        # The solar-system bodies each non-blank card names.
        bodies = [
            {
                body
                for body in SOLAR_SYSTEM_OBJECTS
                if re.search(rf"\b{body}\b", s, re.IGNORECASE)
            }
            for s in (obj, targname)
            if s
        ]
        named = sorted(set().union(*bodies))
        if any(len(b) > 1 for b in bodies):
            return self._unknown_target(
                f"the cards name more than one solar-system body {named}"
            )
        obj_catalog, obj_name, obj_planet = kpf_target_name(obj)
        targ_catalog, targ_name, targ_planet = kpf_target_name(targname)
        for card, raw, planet, host in (
            ("OBJECT", obj, obj_planet, obj_name),
            ("TARGNAME", targname, targ_planet, targ_name),
        ):
            if planet:
                self._warn(
                    f"{card} {raw!r} names planet {planet!r}; host star {host!r}"
                )
        shorter, longer = sorted((obj_name, targ_name), key=len)
        if bodies and bodies[0] and all(b == bodies[0] for b in bodies):
            (result,) = bodies[0]
            if not (obj and targname):
                self._warn(
                    f"OBJECT {obj!r} / TARGNAME {targname!r} has a blank card; "
                    f"OBJECT is {result!r}"
                )
        elif obj and targname and obj_name == targ_name:
            result = obj_name
        elif (
            obj_catalog
            and obj_catalog == targ_catalog
            and re.fullmatch(re.escape(shorter) + " ?[A-Z]", longer)
        ):
            if obj_planet or targ_planet:
                raise HeaderError(
                    f"{self._label}: OBJECT {obj!r} / TARGNAME {targname!r} "
                    "pair a planet letter with a component letter; ambiguous. L0 "
                    "header must be fixed: name the same star on both cards."
                )
            result = longer
            self._warn(
                f"OBJECT {obj!r} / TARGNAME {targname!r} differ only by a "
                f"component letter; OBJECT is {result!r}"
            )
        elif targ_catalog and obj.lower() == targ_catalog.lower():
            result = targ_name
        elif obj_catalog and targname.lower() == obj_catalog.lower():
            result = obj_name
        elif not obj and targname:
            self._warn(f"OBJECT blank; OBJECT is TARGNAME {targname!r}")
            result = targ_name
        elif obj and not targname:
            self._warn(f"TARGNAME blank; OBJECT is {obj!r}")
            result = obj_name
        elif not (obj or targname):
            return self._unknown_target("OBJECT and TARGNAME are both blank")
        elif named:
            return self._unknown_target(
                f"the cards disagree and name solar-system body {named} "
                f"(canonical {obj_name!r} vs {targ_name!r})"
            )
        else:
            return self._unknown_target(
                f"the cards canonicalize to different names ({obj_name!r} vs "
                f"{targ_name!r}): often legitimate aliases of one star (e.g. HD "
                "number vs common name; kpf_target_name does not cross-identify "
                "catalogs), else different stars or components, or a typo"
            )
        if result.isdigit():
            return self._unknown_target(
                f"{result!r} is a bare number in no known catalog (e.g. a "
                "truncated Gaia source_id)"
            )
        if result != obj or result != targname:
            logger.info(
                "%s: OBJECT %r / TARGNAME %r resolved to %r",
                self._label,
                obj,
                targname,
                result,
            )
        return result

    def _unknown_target(self, why):
        """Return ``Unknown`` for a Target whose cards name no single target,
        warning ``why`` with the native cards; raise HeaderError if GAIAID is
        blank too.

        ``Unknown`` is common and usually not bad data: the cards often carry
        legitimate aliases of one star (OBJECT ``201092`` / TARGNAME ``61 Cyg B``),
        which ``kpf_target_name`` does not cross-identify. Most such frames carry
        a GAIAID, from which AstroQuery resolves OBJECT
        (``AstroQuery._resolve_identity``); with GAIAID blank nothing locates the
        target, so fail here rather than in AstroQuery.
        """
        native = self.WMKO_HEADER
        context = (
            f"OBJECT is Unknown: {why} [native OBJECT={native.get('OBJECT')!r}, "
            f"TARGNAME={native.get('TARGNAME')!r}]"
        )
        raw = self._native_gaiaid()
        if not raw:
            raise HeaderError(
                f"{self._label}: {context}; GAIAID is blank, so nothing "
                "identifies the target for astrometry. L0 header must be fixed: "
                "correct OBJECT/TARGNAME or set GAIAID."
            )
        catalog, gaiaid, _ = kpf_target_name(raw)
        if catalog == "Gaia":  # else _gaiaid raises on the malformed card
            self._warn(
                f"{context}; AstroQuery resolves OBJECT and astrometry from GAIAID "
                f"{gaiaid!r}"
            )
        return "Unknown"

    def _check_cal_object(self, cal_object):
        """Warn if the native OBJECT is no ``autocal-*`` label agreeing with
        `cal_object` and the lit fibers, or if TARGNAME names a target."""
        native = self.WMKO_HEADER
        obj = str(native.get("OBJECT") or "").strip()
        targname = str(native.get("TARGNAME") or "").strip()
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
            shutters = tuple(_lower(native.get(k)) for k in _SHUTTERS)
            suffix = _SHUTTER_SUFFIXES.get(shutters)
            expected = f"{family}-{suffix}" if suffix else None
        match = re.match(r"autocal-([a-z_]+)(?:-([a-z]+))?", obj.lower())
        if not match:
            self._warn(f"OBJECT {obj!r} is no autocal label")
        else:
            lamp, fibers = match.groups()
            label = (
                lamp
                if lamp in ("bias", "dark", "wideflat")
                else f"{lamp}-{fibers or 'all'}"
            )
            if expected is None:
                states = ", ".join(
                    f"{k}={v!r}" for k, v in zip(_SHUTTERS, shutters, strict=True)
                )
                self._warn(
                    f"OBJECT {obj!r} disagrees with the shutters ({states}), which "
                    "no autocal label names"
                )
            elif label != expected:
                self._warn(f"OBJECT {obj!r} disagrees with derived autocal-{expected}")
        if targname.lower() not in ("", "unknown"):
            self._warn(f"TARGNAME {targname!r} names a target on a {cal_object} frame")

    def _illumination(self):
        """Stamp TRACE1-5 (fiber types), SKYSRC/SCISRC/CALSRC (the source lighting
        each fiber), CLSRC1-5 (calibration lamp or ``None``), CSRC1-5/CID1-5
        (``None``; AstroQuery fills them) and SIMULCAL.

        Each source follows from the frame and its fiber's midpoint shutter
        (SOLSCISH for SCI on a solar frame), not from the native
        SCI-OBJ/SKY-OBJ/CAL-OBJ, which can ignore the shutters or carry
        placeholders; those only cross-check the result.
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
            state = _lower(native.get(shutter))
            # Sources in native SCI-OBJ/SKY-OBJ/CAL-OBJ terms, for the cross-check:
            # what the shutter state admits, then what the fiber sees.
            shutter_source = {
                "open": "Target" if solar_sci else octagon,
                "closed": "None",
            }.get(state, "Unknown")
            if obstype == "Unknown":
                native_source = "Unknown"
            elif obj in ("Bias", "Dark", "WideFlat"):
                native_source = "None"
            elif obstype == "Target" and not solar_sci:
                native_source = {"SCI": "Target", "SKY": "Sky"}.get(
                    fiber, shutter_source
                )
            else:
                native_source = shutter_source

            source = _SOURCES.get(native_source.lower(), native_source)
            if native_source == "Target" and obj == "SoCal":
                source = "SoCal"
            # An open CAL fiber on a Target frame sees an OCTAGON naming no lamp.
            if (
                state == "open"
                and native_source == octagon
                and _lower(octagon) not in _SOURCES
            ):
                self._warn(
                    f"OCTAGON={native.get('OCTAGON')!r} names no calibration "
                    f"source; {src_card} is Unknown"
                )
                source = "Unknown"
            if src_card:
                self.DRP_KEYWORDS[src_card] = source
            # CLSRC# names a calibration lamp only.
            is_lamp = source in _SOURCES.values() or source == "Unknown"
            self.DRP_KEYWORDS[f"CLSRC{trace}"] = source if is_lamp else "None"
            self.DRP_KEYWORDS[f"CSRC{trace}"] = "None"
            self.DRP_KEYWORDS[f"CID{trace}"] = "None"

            # Trace 2 stands for 3-4, which share SCI-OBJ/SCISEL: warn once per card.
            if src_card and native_source == "Unknown" and obstype != "Unknown":
                self._warn(
                    f"{shutter}={native.get(shutter)!r} is not open/closed; "
                    f"{src_card} is Unknown"
                )
            if (
                src_card
                and native_source != "Unknown"
                and _lower(native.get(obj_card)) != native_source.lower()
            ):
                self._warn(
                    f"{obj_card}={native.get(obj_card)!r} disagrees with derived "
                    f"{src_card}={source!r} ({shutter}={native.get(shutter)!r})"
                )

        # The etalon is the only simultaneous calibration source.
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
        """Stamp READTIMG/READTIMR (shutter close to file write) and READMODE.

        The ACF waveform filenames name the read mode outright; failing that, the
        read time separates ~12 s fast readout from ~48 s regular.
        """
        native = self.WMKO_HEADER
        read_time = {}
        for chip, prefix, key in (
            ("GREEN", "GR", "READTIMG"),
            ("RED", "RD", "READTIMR"),
        ):
            # {prefix}DATE is the file write, {prefix}DATE-E the shutter close.
            written = native.get(f"{prefix}DATE")
            closed = native.get(f"{prefix}DATE-E")
            if written and closed:
                try:
                    read_time[chip] = (
                        datetime.fromisoformat(written) - datetime.fromisoformat(closed)
                    ).total_seconds()
                except (TypeError, ValueError):
                    raise HeaderError(
                        f"{self._label}: {prefix}DATE={written!r} or "
                        f"{prefix}DATE-E={closed!r} is not an ISO timestamp"
                    ) from None
                if read_time[chip] < 0:
                    raise HeaderError(
                        f"{self._label}: {prefix}DATE={written!r} precedes "
                        f"{prefix}DATE-E={closed!r}; negative {chip} read time"
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
