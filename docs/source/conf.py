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

# Modules omitted from the generated API reference: internal machinery that is
# not part of the user-facing surface. This is a docs-presentation choice, not
# an API-privacy change — these modules are legitimately imported across the
# package and tests, so they are NOT renamed private (no leading underscore).
# The recursive autosummary template (_templates/autosummary/module.rst) reads
# this list via autosummary_context and drops matching submodules. Use exact
# dotted names (e.g. keep kpfpipe.utils.config, drop kpfpipe.data_models.config).
autosummary_context = {
    "skip_modules": [
        "kpfpipe.data_models.config",
        "kpfpipe.data_models.keyword_registry",
        "kpfpipe.data_models.aliased_dict",
    ],
}

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
