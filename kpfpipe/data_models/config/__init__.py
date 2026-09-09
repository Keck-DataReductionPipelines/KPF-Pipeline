"""Makes this directory a package for importlib.resources.

``PATH`` is the traversable every config reader opens its tables through -- the
extension manifest, the keyword registry, and the level data models -- so the
location is declared once rather than in each.
"""

import importlib.resources

import pandas as pd

PATH = importlib.resources.files(__name__)

# Trace index -> fiber name, and the 1:1 KPF -> EPRV extension synonyms.
TRACE_MAP = pd.read_csv(PATH / "trace-map.csv")
EXTENSION_ALIASES = pd.read_csv(PATH / "extension-aliases.csv")
