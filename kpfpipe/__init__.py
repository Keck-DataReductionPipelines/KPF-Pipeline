from pathlib import Path
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]

# By default use both CCDs and all five fibers
DEFAULTS = { 
    'chips' : ['GREEN', 'RED'], 
    'fibers' : ['SKY','SCI1','SCI2','SCI3','CAL'],
}

# Lazy-load the detector config on first access
_detector = None

def load_detector_config():
    global _detector
    if _detector is None:
        path = Path(REPO_ROOT) / "reference/detector.toml"
        config = tomllib.loads(path.read_text())

        # Uppercase 'red'/'green' keys recursively, but keep them in dict
        def traverse(obj):
            if isinstance(obj, dict):
                return {
                    (k.upper() if isinstance(k, str) and k.lower() in ("red", "green") else k):
                    traverse(v)
                    for k, v in obj.items()
                }
            if isinstance(obj, list):
                return [traverse(v) for v in obj]
            return obj

        _detector = dict(traverse(config))

    return _detector

DETECTOR = load_detector_config()