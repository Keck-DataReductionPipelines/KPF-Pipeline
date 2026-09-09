"""Makes this directory a package for importlib.resources.

``PATH`` is the traversable the config tables are opened through.
"""

import importlib.resources

import pandas as pd

PATH = importlib.resources.files(__name__)

# Trace index -> fiber name, and the 1:1 KPF -> EPRV extension synonyms.
TRACE_MAP = pd.read_csv(PATH / "trace-map.csv")
EXTENSION_ALIASES = pd.read_csv(PATH / "extension-aliases.csv")
