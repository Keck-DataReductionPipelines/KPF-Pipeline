"""Target identity: parse, compare and reconcile target names.

Pure, offline functions on strings (header card values, catalog identifiers).
HeaderParser and AstroQuery read the cards and query results, call these, and
write the keywords. A contradiction raises ValueError with the fix the L0 header
needs; the caller adds its file label. Warnings go to the caller's ``warn``.
"""

import csv
import re
from collections import namedtuple

from kpfpipe import REPO_ROOT, SOLAR_SYSTEM_OBJECTS

# Known target-name aliases: (catalog, compiled pattern, canonical format), matched
# against the normalized name. A name must fully (case-insensitively) match at most
# one row, so row order never decides a match; parse_target raises on a name matching
# several. Row order is the catalog preference that orders ALIASES (resolve_identity).
with open(REPO_ROOT / "reference/star_catalog_aliases.csv", newline="") as _f:
    _rows = list(csv.DictReader(_f))
_ALIASES = [
    (row["catalog"], re.compile(row["pattern"], re.IGNORECASE), row["canonical"])
    for row in _rows
]
_CATALOG_RANK = {
    catalog: rank
    for rank, catalog in enumerate(dict.fromkeys(row["catalog"] for row in _rows))
}
# Catalogs of purely numeric ids (Gaia, TIC, ...): they take no component or planet
# letter, and name OBJECT only when no other catalog names the star.
_NUMERIC_CATALOGS = frozenset(r["catalog"] for r in _rows if r["numeric"] == "yes")

# OBJECT for a header naming only a numeric id: its mission's own designation,
# in preference order. Others (Gaia, Unknown) take the most preferred catalog.
_SUCCESSORS = {
    "TIC": ("TOI",),
    "KIC": ("Kepler", "KOI"),
    "EPIC": ("K2",),
}


class TargetName(
    namedtuple(
        "TargetName",
        "catalog name component planet gaia_release gaia_source_id",
        defaults=("", "", "", ""),
    )
):
    """A target name parsed by ``parse_target``; absent fields are ''.

    Fields: ``catalog`` (reference/star_catalog_aliases.csv catalog), ``name``
    (canonical, SIMBAD's spelling; the planet letter dropped), ``component``
    (stellar component letter, kept on ``name``), ``planet`` (planet letter,
    dropped from ``name``), ``gaia_release`` ('DR2' or 'DR3' on a Gaia id) and
    ``gaia_source_id`` (the digits of a Gaia id).
    """

    __slots__ = ()


def normalize_name(raw):
    """Strip a SIMBAD marker ('NAME ', 'V*', '*'), read '_' as a space and
    collapse whitespace: '*  61 Cyg B' -> '61 Cyg B', 'BL_Lac' -> 'BL Lac'."""
    name = " ".join(str(raw).replace("_", " ").split())
    return re.sub(r"^(?:NAME |V\*|\*+) ?", "", name)


def parse_target(raw):
    """Parse a target name against reference/star_catalog_aliases.csv.

    'HD57555' -> TargetName('HD', 'HD 57555'); all-digit fields drop leading zeros
    ('HR 0001' -> 'HR 1'). On a non-numeric catalog id a trailing uppercase letter
    or 'a' is a stellar component ('hd21019a' -> 'HD 21019A'), any other letter a
    planet ('HD 189733b' -> name 'HD 189733', planet 'b'). An uncatalogued name is
    returned normalized, with catalog ''.

    Raises
    ------
    ValueError
        If the name matches several csv rows (the patterns must be disjoint).
    """
    name = normalize_name(raw)
    matches = {
        (
            catalog,
            canonical.format(
                *(str(int(g)) if g.isdigit() else g for g in match.groups())
            ),
        )
        for catalog, pattern, canonical in _ALIASES
        if (match := pattern.fullmatch(name))
    }
    if len(matches) > 1:
        raise ValueError(
            f"target name {name!r} matches several reference/star_catalog_aliases.csv "
            f"rows {sorted(matches)}; the patterns must be disjoint"
        )
    if matches:
        catalog, canonical = matches.pop()
        if catalog != "Gaia":
            return TargetName(catalog, canonical)
        *release, source_id = canonical.split()[1:]
        return TargetName(
            catalog, canonical, gaia_release="".join(release), gaia_source_id=source_id
        )
    lettered = re.fullmatch(r"(.*\d) ?([A-Za-z])", name)
    if lettered:
        host, letter = parse_target(lettered[1]), lettered[2]
        if host.catalog and host.catalog not in _NUMERIC_CATALOGS:
            if letter.isupper() or letter == "a":
                letter = letter.upper()
                return TargetName(host.catalog, host.name + letter, component=letter)
            return TargetName(host.catalog, host.name, planet=letter)
    return TargetName("", name)


def same_target(a, b):
    """Whether two names are the same identifier: equal once parsed, ignoring case
    and a space between a letter and a digit ('V1298Tau' = 'V1298 Tau',
    'Gl 15A' = 'GJ 15 A'); '+', '-' and a space between digits count
    ('BD+03 2870' != 'BD-03 2870')."""
    return _key(a) == _key(b)


def _key(name):
    """The comparison form of a name (``same_target``)."""
    name = parse_target(name).name
    return re.sub(r"(?<=[A-Za-z]) (?=\d)|(?<=\d) (?=[A-Za-z])", "", name).casefold()


def _solar_bodies(card):
    """The solar-system bodies (reference/solar_system_objects.csv) a card names
    as a whole word, in any case ('Io Eclipse' -> {'Io'})."""
    return {
        body
        for body in SOLAR_SYSTEM_OBJECTS
        if re.search(rf"\b{body}\b", card, re.IGNORECASE)
    }


def reconcile_object(obj, targname, gaiaid, warn):
    """A Target frame's OBJECT, from its native OBJECT and TARGNAME cards:

    - ``SoCal`` for the solar pair (OBJECT ``SoCal``, TARGNAME ``Sun``);
    - the solar-system body the cards name (``Io`` with ``Io Eclipse`` -> ``Io``);
    - else the canonical target name (``parse_target``; a planet letter is dropped
      for the host star). The cards must be the same target (``same_target``), or
      one must be the other's bare catalog prefix (``TOI`` with ``TOI-969``) or
      lack only its component letter (``WASP-173`` with ``WASP-173B``).

    A blank card yields the other. Anything else is ``Unknown``, warned with the
    reason; AstroQuery then resolves OBJECT from GAIAID.

    Parameters
    ----------
    obj, targname : str
        The stripped native OBJECT and TARGNAME ('' if blank).
    gaiaid : str
        The stripped native GAIAID ('' if blank).
    warn : callable
        Receives each warning message.

    Raises
    ------
    ValueError
        If the cards pair a planet letter with a component letter, or OBJECT is
        ``Unknown`` with GAIAID blank (nothing locates the target).
    """

    def unknown(why):
        context = f"OBJECT {obj!r} / TARGNAME {targname!r}: {why}; OBJECT is Unknown"
        if not gaiaid:
            raise ValueError(
                f"{context}; GAIAID blank. L0 header must be fixed: correct "
                "OBJECT/TARGNAME or set GAIAID."
            )
        gaia = parse_target(gaiaid)
        if gaia.catalog == "Gaia":  # else reconcile_gaiaid raises on the card
            warn(
                f"{context}; AstroQuery resolves OBJECT and astrometry from GAIAID "
                f"{gaia.name!r}"
            )
        return "Unknown"

    if (obj, targname) == ("SoCal", "Sun"):
        return "SoCal"
    if not (obj or targname):
        return unknown("both cards blank")
    blank = "" if obj and targname else ("TARGNAME" if obj else "OBJECT")

    card_bodies = [_solar_bodies(card) for card in (obj, targname) if card]
    all_bodies = sorted(set().union(*card_bodies))
    if any(len(b) > 1 for b in card_bodies):
        return unknown(f"the cards name more than one solar-system body {all_bodies}")
    if card_bodies[0] and all(b == card_bodies[0] for b in card_bodies):
        (body,) = card_bodies[0]
        if blank:
            warn(f"{blank} blank; OBJECT is {body!r}")
        return body

    obj_id, targ_id = parse_target(obj), parse_target(targname)
    lettered = {}  # one warning per letter, naming every card that carries it
    for card, raw, parsed in (("OBJECT", obj, obj_id), ("TARGNAME", targname, targ_id)):
        if parsed.planet or parsed.component:
            lettered.setdefault(parsed, []).append(f"{card} {raw!r}")
    for parsed, cards in lettered.items():
        _warn_letter(" / ".join(cards), parsed, warn)
    if blank:
        result = (obj_id if obj else targ_id).name
    else:
        result = _matching_name(obj_id, targ_id, obj, targname, warn)
        if result is None and all_bodies:
            return unknown(f"the cards disagree on solar-system body {all_bodies}")
        if result is None:
            return unknown(
                f"the cards name different targets ({obj_id.name!r} vs "
                f"{targ_id.name!r})"
            )
    if result.isdigit():
        return unknown("a bare number in no known catalog")
    if blank:
        warn(f"{blank} blank; OBJECT is {result!r}")
    return result


def _warn_letter(cards, parsed, warn):
    """Warn a trailing letter: it can name a stellar component or a planet
    (``parse_target`` decides by case), and the two are easily confused."""
    ambiguity = "a trailing letter can denote a binary component or a planet"
    if parsed.planet:
        warn(
            f"{cards} names planet {parsed.planet!r}; host star "
            f"{parsed.name!r} ({ambiguity})"
        )
    else:
        warn(f"{cards} names stellar component {parsed.name!r} ({ambiguity})")


def _matching_name(obj_id, targ_id, obj, targname, warn):
    """The name two non-blank cards agree on (``reconcile_object``), or None.

    Raises
    ------
    ValueError
        If the cards differ only by a letter that is a planet on one card and a
        component on the other.
    """
    if same_target(obj_id.name, targ_id.name):
        # same_target ignores case, so an uncatalogued name's trailing letter can be
        # a planet ('b') on one card and a component ('B') on the other.
        obj_letter, targ_letter = (
            re.search(r"\d ?([A-Za-z])$", t.name) for t in (obj_id, targ_id)
        )
        if obj_letter and targ_letter and obj_letter[1] != targ_letter[1]:
            warn(
                f"OBJECT {obj!r} / TARGNAME {targname!r} differ in the case of a "
                "trailing letter, which can denote a planet or a binary component; "
                f"OBJECT is {obj_id.name!r}"
            )
        return obj_id.name
    bare, lettered = sorted((obj_id, targ_id), key=lambda t: bool(t.component))
    if (
        obj_id.catalog
        and obj_id.catalog == targ_id.catalog
        and not bare.component
        and lettered.component
        and lettered.name == bare.name + lettered.component
    ):
        if obj_id.planet or targ_id.planet:
            raise ValueError(
                f"OBJECT {obj!r} / TARGNAME {targname!r} pair a planet letter with "
                "a component letter. L0 header must be fixed: name the same star on "
                "both cards."
            )
        warn(
            f"OBJECT {obj!r} / TARGNAME {targname!r} differ only by a component "
            f"letter; OBJECT is {lettered.name!r}"
        )
        return lettered.name
    if targ_id.catalog and same_target(obj, targ_id.catalog):
        return targ_id.name
    if obj_id.catalog and same_target(targname, obj_id.catalog):
        return obj_id.name
    return None


def reconcile_gaiaid(obj, gaiaid, warn):
    """``(OBJECT, GAIAID)`` for a non-solar Target, GAIAID made canonical.

    GAIAID becomes ``Gaia DR3 <id>``, ``Gaia DR2 <id>`` or ``Gaia <id>`` ('' if
    blank). For a Gaia-named OBJECT, a blank GAIAID takes the OBJECT's, and
    whichever card carries the release lends it to the other.

    Parameters
    ----------
    obj : str
        The sanitized OBJECT (``reconcile_object``).
    gaiaid : str
        The stripped native GAIAID ('' if blank).
    warn : callable
        Receives each warning message.

    Raises
    ------
    ValueError
        If GAIAID is no Gaia id, or names another source or release than OBJECT.
    """
    gaia = parse_target(gaiaid)
    if gaia.catalog != "Gaia" and gaiaid:
        raise ValueError(
            f"GAIAID {gaiaid!r} is not a Gaia source_id ('Gaia DR3 <id>', 'DR2 <id>', "
            "or 17-19 bare digits; a shorter id needs its release). L0 header must "
            "be fixed: correct GAIAID or blank it."
        )
    target = parse_target(obj)
    if target.catalog != "Gaia":
        return obj, gaia.name if gaiaid else ""
    if not gaiaid:
        warn(f"GAIAID blank; GAIAID is OBJECT {obj!r}")
        return obj, obj
    if target.gaia_source_id != gaia.gaia_source_id:
        raise ValueError(
            f"OBJECT {obj!r} and GAIAID {gaia.name!r} name different Gaia sources. "
            "L0 header must be fixed: make OBJECT and GAIAID match."
        )
    if target.gaia_release and gaia.gaia_release and obj != gaia.name:
        raise ValueError(
            f"OBJECT {obj!r} and GAIAID {gaia.name!r} name different Gaia releases. "
            "L0 header must be fixed: give OBJECT and GAIAID the same release, or "
            "blank one."
        )
    resolved = obj if target.gaia_release else gaia.name
    if obj != resolved:
        warn(f"OBJECT {obj!r} takes the GAIAID release; OBJECT is {resolved!r}")
    if gaia.name != resolved:
        warn(f"GAIAID {gaiaid!r} takes the OBJECT release; GAIAID is {resolved!r}")
    return resolved, resolved


def resolve_identity(obj, gaiaid, gaia_designation, simbad_ids, common_name):
    """``(OBJECT, ALIASES)`` from the header identity and the catalog matches.

    - A bare-Gaia OBJECT takes the Gaia designation of its source.
    - SIMBAD's identifiers must include OBJECT (``same_target``), whose spelling
      OBJECT then takes ('V1298Tau' -> 'V1298 Tau'), and must list no Gaia id of
      GAIAID's release other than GAIAID.
    - OBJECT is kept unless ``Unknown`` or a numeric id (``_NUMERIC_CATALOGS``);
      those take the catalogued alias of their mission (``_SUCCESSORS``: TIC ->
      TOI), else the first non-numeric alias, else stay.
    - ALIASES is OBJECT, then the non-numeric and numeric aliases (from OBJECT, the
      Gaia designation and SIMBAD's catalogued ids), each ordered by catalog
      preference (reference/star_catalog_aliases.csv row order) then name, then the
      SIMBAD common name.

    Parameters
    ----------
    obj, gaiaid : str
        The PRIMARY OBJECT and GAIAID ('' if blank).
    gaia_designation : str
        Gaia's designation of the source ('' if Gaia gave none).
    simbad_ids : list of TargetName
        SIMBAD's identifiers of its match, parsed (empty if SIMBAD gave none).
    common_name : str
        SIMBAD's common name ('' if none).

    Raises
    ------
    ValueError
        If SIMBAD's match contradicts the header (the cards name another star).
    """
    target = parse_target(obj)
    designation = parse_target(gaia_designation)
    if (
        target.catalog == "Gaia"
        and not target.gaia_release
        and target.gaia_source_id == designation.gaia_source_id
    ):
        obj = gaia_designation
    if simbad_ids:
        obj = _simbad_spelling(obj, simbad_ids, common_name)
        _check_simbad_gaiaid(obj, gaiaid, simbad_ids, common_name)

    names = [obj] if obj != "Unknown" else []
    names += [gaia_designation] if gaia_designation else []
    names += [i.name for i in simbad_ids if i.catalog]
    catalogs = {n: parse_target(n).catalog for n in names}
    # Sorted so ALIASES is independent of SIMBAD's id order.
    names.sort(key=lambda n: (_CATALOG_RANK.get(catalogs[n], -1), n))
    numeric = [n for n in names if catalogs[n] in _NUMERIC_CATALOGS]
    non_numeric = [n for n in names if n not in numeric]

    chosen = obj
    if obj == "Unknown" or catalogs[obj] in _NUMERIC_CATALOGS:
        successors = _SUCCESSORS.get(catalogs.get(obj), ())
        chosen = next(
            (n for s in successors for n in non_numeric if catalogs[n] == s),
            (non_numeric or numeric or [obj])[0],
        )
    aliases = [chosen, *non_numeric, *numeric] + ([common_name] if common_name else [])
    return chosen, "; ".join(n for n in dict.fromkeys(aliases) if n != "Unknown")


def _simbad_spelling(obj, simbad_ids, common_name):
    """SIMBAD's spelling of OBJECT (``Unknown`` stays).

    Raises
    ------
    ValueError
        If OBJECT is none of SIMBAD's identifiers.
    """
    if obj == "Unknown":
        return obj
    spelling = next((i.name for i in simbad_ids if same_target(obj, i.name)), None)
    if spelling is None:
        raise ValueError(
            f"OBJECT {obj!r} is not an identifier of SIMBAD's match "
            f"({common_name!r}). L0 header must be fixed: correct OBJECT/TARGNAME."
        )
    return spelling


def _check_simbad_gaiaid(obj, gaiaid, simbad_ids, common_name):
    """Check SIMBAD lists GAIAID among its Gaia ids of GAIAID's release (any
    release for a bare GAIAID).

    Raises
    ------
    ValueError
        If SIMBAD lists other Gaia ids of that release but not GAIAID.
    """
    gaia = parse_target(gaiaid)
    listed = sorted(
        i
        for i in simbad_ids
        if i.catalog == "Gaia" and gaia.gaia_release in ("", i.gaia_release)
    )
    if (
        gaia.catalog == "Gaia"
        and listed
        and gaia.gaia_source_id not in {i.gaia_source_id for i in listed}
    ):
        raise ValueError(
            f"GAIAID {gaiaid!r} and OBJECT {obj!r} name different stars: SIMBAD's "
            f"match ({common_name!r}) is {', '.join(i.name for i in listed)}. L0 "
            "header must be fixed: make OBJECT and GAIAID match."
        )
