import importlib.metadata
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Installed package version (WMKO DRP-RUN-11); stamped onto PRIMARY as DRPTAG/
# DRPVERNO. Falls back to "unknown" if the package metadata is unavailable
# (e.g. running from a source tree that was never installed).
try:
    __version__ = importlib.metadata.version("kpfpipe")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

# By default use both CCDs and all five fibers
DEFAULTS = {
    "chips": ["GREEN", "RED"],
    "fibers": ["SKY", "SCI1", "SCI2", "SCI3", "CAL"],
}

# Lazy-load the detector config on first access


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

    return _detector


DETECTOR = load_detector_config()
DEFAULTS.update(DETECTOR)
