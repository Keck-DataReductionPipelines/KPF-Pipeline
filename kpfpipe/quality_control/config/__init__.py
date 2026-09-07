"""Makes this directory a package for importlib.resources.

``PATH`` is the traversable ``applicability`` opens the per-class applicability
tables through, so the location is declared once rather than in each reader.
"""

import importlib.resources

PATH = importlib.resources.files(__name__)
