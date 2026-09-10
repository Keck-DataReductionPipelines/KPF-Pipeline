"""Makes this directory a package, so ``importlib.resources`` locates the
applicability tables without a path hard-coded in each reader."""

import importlib.resources

PATH = importlib.resources.files(__name__)
