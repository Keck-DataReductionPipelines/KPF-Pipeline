"""KPF detector constants, loaded from data_models/config/detector.toml."""

from pathlib import Path
import tomllib

_toml_path = Path(__file__).parent / "data_models" / "config" / "detector.toml"
_detector = tomllib.loads(_toml_path.read_text())

NROW = _detector["ccd"]["nrow"]
NCOL = _detector["ccd"]["ncol"]

NORDER_GREEN = _detector["orders"]["green"]
NORDER_RED = _detector["orders"]["red"]
