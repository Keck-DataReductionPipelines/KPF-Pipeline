"""KPF pipeline package root: version, detector and observatory config, and
shared defaults."""

import importlib.metadata
import subprocess
import tomllib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    __version__ = importlib.metadata.version("kpfpipe")
except importlib.metadata.PackageNotFoundError as e:
    raise RuntimeError(
        "kpfpipe package metadata not found -- this can occur if the repository "
        "was cloned but not installed. Install it into the kpfpipe conda env "
        "before importing: `pip install -e KPF-Pipeline/`."
    ) from e

# The commit the pipeline runs from, resolved once beside __version__ and
# stamped onto PRIMARY as DRPHASH. "UNKNOWN" when the installed tree is not a
# git checkout (a released tarball or wheel), which is not an error.
try:
    __githash__ = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
except (subprocess.CalledProcessError, OSError):
    __githash__ = "UNKNOWN"


def load_detector_config():
    path = Path(REPO_ROOT) / "reference/detector.toml"
    config = tomllib.loads(path.read_text())

    # Uppercase 'red'/'green' keys recursively, but keep them in dict
    def traverse(obj):
        if isinstance(obj, dict):
            return {
                (
                    k.upper()
                    if isinstance(k, str) and k.lower() in ("red", "green")
                    else k
                ): traverse(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [traverse(v) for v in obj]
        return obj

    _detector = dict(traverse(config))

    # Derived counts, so consumers read them instead of re-summing: total
    # echelle orders across both CCDs, and the number of fibers (traces) on
    # the slicer.
    _detector["numorder"] = sum(_detector["norder"].values())
    _detector["numtrace"] = len(_detector["fiber_positions"])

    return _detector


def load_observatory_config():
    path = Path(REPO_ROOT) / "reference/observatory.toml"
    return tomllib.loads(path.read_text())


def load_instrument_era_reference():
    """The KPF instrument eras, one row per era, dated bounds parsed.

    INSTERA is left as read (a float, e.g. 2.0); consumers that compare it to a
    header value cast to str, as the keyword carries it.
    """
    path = Path(REPO_ROOT) / "reference/instrument_eras.csv"
    return pd.read_csv(path, parse_dates=["UT_start_date", "UT_end_date"])


CHIPS = ("GREEN", "RED")
FIBERS = ("SKY", "SCI1", "SCI2", "SCI3", "CAL")
SCI_FIBERS = ("SCI1", "SCI2", "SCI3")
DETECTOR = load_detector_config()
OBSERVATORY = load_observatory_config()
INSTRUMENT_ERAS = load_instrument_era_reference()

# By default use both CCDs and all five fibers
DEFAULT_CFG = {
    "chips": CHIPS,
    "fibers": FIBERS,
}
