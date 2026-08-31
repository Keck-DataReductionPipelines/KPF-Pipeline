"""Makes this directory a package for importlib.resources.

``PATH`` is the traversable every config reader opens its tables through -- the
extension manifest, the keyword registry, and the level data models -- so the
location is declared once rather than in each.
"""

import importlib.resources

PATH = importlib.resources.files(__name__)
