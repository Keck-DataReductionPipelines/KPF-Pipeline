# -*- coding: utf-8 -*-
#
# KPF-DRP vNext documentation build configuration.
#
# The docs build does NOT install the kpfpipe package (that would demand the
# exact "==3.14.3" interpreter pinned in pyproject.toml). Instead it puts the
# source tree on sys.path and lets Sphinx autodoc introspect it. The package's
# real runtime dependencies ARE installed (see docs/requirements.txt) so every
# kpfpipe.* module imports cleanly for introspection — mocking them proved too
# fragile for a package this size (module-level astropy-unit math,
# importlib.metadata version lookups, etc. run at import time). The build Python
# is matched to the project's 3.14 line (see .readthedocs.yaml).

import os
import sys

# Repo root = docs/source/../.. — resolved from this file so it is independent
# of the directory sphinx-build is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# -- Project information ------------------------------------------------------

project = "KPF Data Reduction Pipeline (vNext)"
author = "The KPF Team"
copyright = "2020-2026, The KPF Team"

# Placeholder while the docs are structure-only; wire to kpfpipe.__version__
# once the build installs the package.
version = "vNext"
release = "vNext"

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"
exclude_patterns = []
pygments_style = "sphinx"

# -- autodoc / autosummary ----------------------------------------------------

# Generate the per-module autosummary stub pages at build time (nothing to
# commit; the output dir is gitignored).
autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# -- Options for HTML output --------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
