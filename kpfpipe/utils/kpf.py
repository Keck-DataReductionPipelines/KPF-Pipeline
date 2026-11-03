"""obs_id, datecode, and timestamp parsing, UTC/HST and KPF/EPRV conversions,
per-era reference selection, and target-name canonicalization."""

import csv
import glob
import os
import re
from datetime import datetime, timedelta

from astropy.time import Time

from kpfpipe import INSTRUMENT_ERAS, REPO_ROOT

# Known target-name aliases: (catalog, compiled pattern, canonical format). A name
# must fully (case-insensitively) match at most one row, so row order is irrelevant;
# kpf_target_name raises on a name that matches several.
with open(REPO_ROOT / "reference/star_catalog_aliases.csv", newline="") as _f:
    _STAR_CATALOG_ALIASES = [
        (row["catalog"], re.compile(row["pattern"], re.IGNORECASE), row["canonical"])
        for row in csv.DictReader(_f)
    ]

# The obs_id/KOAID shape, as a string so utils/io.py can embed it in the larger
# filename patterns it builds, and compiled here for matching.
OBS_ID_REGEX = r"KP\.\d{8}\.\d{5}\.\d{2}"
_OBS_ID_PATTERN = re.compile(rf"({OBS_ID_REGEX})")
_DATECODE_PATTERN = re.compile(r"\d{8}")
_KPF_TIMESTAMP_PATTERN = re.compile(r"(\d{8}\.\d{5}\.\d{2})")

_SECONDS_PER_DAY = 86400

# Hawaii Standard Time is UTC-10 (KPF timestamps are UTC)
_HST_UTC_OFFSET_SECONDS = 36000


def is_timestamp(s):
    """Return True if ``s`` is a valid KPF timestamp, e.g. '20240113.23249.10'.

    Checks all three: the 'YYYYMMDD.SSSSS.FF' format, that YYYYMMDD is a real
    calendar date, and that SSSSS (seconds past midnight) is in [0, 86399]; the
    sub-second frame field FF is format-checked only. The module's single source
    of timestamp validity -- every other predicate and raising converter
    validates through it.
    """
    if not isinstance(s, str) or not _KPF_TIMESTAMP_PATTERN.fullmatch(s):
        return False
    date_str, seconds_str, _ = s.split(".")
    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return False
    return int(seconds_str) < _SECONDS_PER_DAY


def is_obs_id(s):
    """Return True if ``s`` is a valid KPF observation ID, e.g.
    'KP.20240113.23249.10'. Both the format and the embedded date/seconds
    are checked.
    """
    if not isinstance(s, str) or not _OBS_ID_PATTERN.fullmatch(s):
        return False
    return is_timestamp(s[3:])


def is_datecode(s):
    """Return True if ``s`` is a valid 8-digit datecode that parses as a real
    calendar date, e.g. '20240405'.
    """
    if not isinstance(s, str) or not _DATECODE_PATTERN.fullmatch(s):
        return False
    try:
        datetime.strptime(s, "%Y%m%d")
    except ValueError:
        return False
    return True


def get_obs_id(fn):
    """Extract the obs_id (e.g. 'KP.20240113.23249.10') from a filename or path;
    raises ``ValueError`` if none is found."""
    if not isinstance(fn, str):
        raise ValueError(f"input must be a string; got {type(fn).__name__}")
    match = _OBS_ID_PATTERN.search(os.path.basename(fn))
    if not match:
        raise ValueError(f"No obs_id found in: {fn}")
    obs_id = match.group(1)
    if not is_timestamp(obs_id[3:]):
        raise ValueError(f"Invalid KPF timestamp in obs_id: {obs_id!r}")
    return obs_id


def get_datecode(s):
    """Extract the datecode (e.g. '20230708') from an obs_id or filename; raises
    ``ValueError`` if no valid obs_id is found."""
    if not isinstance(s, str):
        raise ValueError(f"input must be a string; got {type(s).__name__}")
    match = _OBS_ID_PATTERN.search(s)
    if not match:
        raise ValueError(f"Cannot extract datecode from: {s}")
    obs_id = match.group(1)
    if not is_timestamp(obs_id[3:]):
        raise ValueError(f"Invalid KPF timestamp in {s!r}")
    return obs_id.split(".")[1]


def get_timestamp(s):
    """Extract the KPF timestamp (e.g. '20240113.23249.10') from an obs_id,
    filename, or path; raises ``ValueError`` if none is found."""
    if not isinstance(s, str):
        raise ValueError(f"input must be a string; got {type(s).__name__}")
    match = _KPF_TIMESTAMP_PATTERN.search(os.path.basename(s))
    if not match:
        raise ValueError(f"No KPF timestamp found in: {s}")
    timestamp = match.group(1)
    if not is_timestamp(timestamp):
        raise ValueError(f"Invalid KPF timestamp: {timestamp!r}")
    return timestamp


def utc_to_hst(timestamp):
    """Convert a KPF UTC timestamp ('YYYYMMDD.SSSSS.FF') to HST (Hawaii Standard
    Time, UTC-10), returned in the same KPF format."""
    if not is_timestamp(timestamp):
        raise ValueError(f"Invalid KPF timestamp: {timestamp!r}")
    date_str, seconds_str, frame_str = timestamp.split(".")
    hst_seconds = int(seconds_str) - _HST_UTC_OFFSET_SECONDS
    date = datetime.strptime(date_str, "%Y%m%d")
    if hst_seconds < 0:
        hst_seconds += _SECONDS_PER_DAY
        date -= timedelta(days=1)
    return f"{date.strftime('%Y%m%d')}.{hst_seconds:05d}.{frame_str}"


def hst_to_utc(timestamp):
    """Convert a KPF HST timestamp ('YYYYMMDD.SSSSS.FF') to UTC, in the same KPF
    format. The inverse of ``utc_to_hst``, kept as its symmetric counterpart: the
    pipeline only calls ``utc_to_hst`` (the data tree is UTC-keyed), so this backs
    round-trip tests and any future HST-keyed input."""
    if not is_timestamp(timestamp):
        raise ValueError(f"Invalid KPF timestamp: {timestamp!r}")
    date_str, seconds_str, frame_str = timestamp.split(".")
    utc_seconds = int(seconds_str) + _HST_UTC_OFFSET_SECONDS
    date = datetime.strptime(date_str, "%Y%m%d")
    if utc_seconds >= _SECONDS_PER_DAY:
        utc_seconds -= _SECONDS_PER_DAY
        date += timedelta(days=1)
    return f"{date.strftime('%Y%m%d')}.{utc_seconds:05d}.{frame_str}"


def kpf_timestamp_to_datetime(timestamp):
    """Parse a KPF UTC timestamp ('YYYYMMDD.SSSSS.FF', sub-second field ignored)
    into a naive UTC ``datetime``, e.g. '20240405.40113.57' ->
    ``datetime(2024, 4, 5, 11, 8, 33)``."""
    if not is_timestamp(timestamp):
        raise ValueError(f"Invalid KPF timestamp: {timestamp!r}")
    date_str, seconds_str, _ = timestamp.split(".")
    return datetime.strptime(date_str, "%Y%m%d") + timedelta(seconds=int(seconds_str))


def kpf_timestamp_to_eprv_timestamp(timestamp):
    """Convert a KPF timestamp to EPRV format ('YYYYMMDDTHHMMSS', 1-second
    resolution, sub-second field dropped), e.g. '20240405.40113.57' ->
    '20240405T110833'."""
    if not is_timestamp(timestamp):
        raise ValueError(f"Invalid KPF timestamp: {timestamp!r}")
    date_str, seconds_str, _ = timestamp.split(".")
    total_seconds = int(seconds_str)
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{date_str}T{hh:02d}{mm:02d}{ss:02d}"


def instrument_era(obs_time):
    """The INSTERA (str) of the ``INSTRUMENT_ERAS`` era covering ``obs_time``.

    A time no era covers has no reference calibrations, so this raises rather
    than returning an unattributable era.
    """
    eras = INSTRUMENT_ERAS
    in_era = eras[
        (eras["UT_start_date"] <= obs_time) & (obs_time <= eras["UT_end_date"])
    ]
    if in_era.empty:
        raise ValueError(
            f"No KPF instrument era covers {obs_time}; the eras of "
            f"reference/instrument_eras.csv do not span it"
        )
    return str(in_era.iloc[0]["INSTERA"])


def select_era_reference(pattern, kind, primary, obs_id, eras):
    """The most recent ``kind`` reference measured before a frame within the
    frame's instrument era (``INSTERA``, stamped at ``KPF0.to_kpf1``).

    Parameters
    ----------
    pattern : str
        Glob of the references, its one ``*`` standing for the datecode each
        was measured on (e.g. ``.../order_traces/order_trace_*.csv``).
    kind : str
        Reference name for error messages.
    primary : astropy.io.fits.Header
        The frame's PRIMARY header.
    obs_id : str
        The frame's obs_id, for error messages.
    eras : pandas.DataFrame
        The instrument-era table.

    Returns
    -------
    str
        Path of the selected reference.

    Raises
    ------
    ValueError
        If no era carries the frame's ``INSTERA``.
    FileNotFoundError
        If the era has no reference measured before the frame.
    """
    instera = str(primary["INSTERA"])
    obs_time = Time(primary["JD_UTC"], format="jd", scale="utc").datetime

    match = eras[eras["INSTERA"].astype(str) == instera]
    if match.empty:
        raise ValueError(
            f"reference/instrument_eras.csv carries no instrument era "
            f"{instera!r}, the INSTERA of {obs_id}"
        )
    era = match.iloc[0]

    # The instrument changes whenever it is opened, so only a reference
    # measured earlier in this era describes this frame.
    latest = min(era["UT_end_date"], obs_time)
    prefix, suffix = os.path.basename(pattern).split("*")
    in_era = {}
    for path in glob.glob(pattern):
        datecode = os.path.basename(path)[len(prefix) : -len(suffix)]
        # +2 s places a reference dated on an era's start day inside that era
        measured = datetime.strptime(datecode, "%Y%m%d") + timedelta(seconds=2)
        if era["UT_start_date"] <= measured <= latest:
            in_era[measured] = path
    if not in_era:
        raise FileNotFoundError(
            f"No {kind} measured before {obs_time} within KPF instrument era {instera}"
        )
    return in_era[max(in_era)]


def kpf_target_name(name):
    """``(catalog, name, planet)`` of the known alias `name` matches, else
    ``(None, name, None)``: the one apparatus for target identifiers
    (reference/star_catalog_aliases.csv, whose row order is irrelevant; raises
    ValueError on a name matching several rows, a csv bug).

    The canonical name is what HeaderParser writes to OBJECT and what AstroQuery
    sends to SIMBAD. E.g. 'HD57555', 'HD 57555' and '57555' -> ('HD', 'HD 57555', None);
    'T001824' -> ('TOI', 'TOI-1824', None). A trailing letter on a known id is a
    stellar component if uppercase or 'a' ('hd21019a' -> ('HD', 'HD 21019 A',
    None)), else a planet, dropped and returned ('HD 189733b' -> ('HD',
    'HD 189733', 'b')). Gaia, TIC and KIC ids are purely numeric and take no
    letter.
    """
    matches = {
        (catalog, canonical.format(*(g.upper() for g in match.groups())))
        for catalog, pattern, canonical in _STAR_CATALOG_ALIASES
        if (match := pattern.fullmatch(name))
    }
    if len(matches) > 1:
        raise ValueError(
            f"target name {name!r} matches several reference/star_catalog_aliases.csv "
            f"rows {sorted(matches)}; the patterns must be disjoint"
        )
    if matches:
        return *matches.pop(), None
    lettered = re.fullmatch(r"(.*\d)\s*([A-Za-z])", name)
    if lettered:
        catalog, host, _ = kpf_target_name(lettered[1])
        letter = lettered[2]
        if catalog and catalog not in ("Gaia", "TIC", "KIC"):
            if letter.isupper() or letter == "a":
                return catalog, f"{host} {letter.upper()}", None
            return catalog, host, letter
    return None, name, None
