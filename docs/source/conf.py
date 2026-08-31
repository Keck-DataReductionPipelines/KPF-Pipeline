# -*- coding: utf-8 -*-
#
# KPF-DRP vNext documentation build configuration.
#
# The build installs kpfpipe (see the post_install job in .readthedocs.yaml,
# which waives the exact "==3.14.3" interpreter pin) and lets Sphinx autodoc
# import and introspect it. The package must genuinely be installed, not merely
# on sys.path: kpfpipe/__init__.py raises unless its distribution metadata is
# present. Its real runtime dependencies are installed too (see
# docs/requirements.txt) — mocking them proved too fragile for a package this
# size (module-level astropy-unit math, importlib.metadata version lookups, etc.
# run at import time).

import os
import sys

import kpfpipe

# Repo root = docs/source/../.. — resolved from this file so it is independent
# of the directory sphinx-build is invoked from. Used to locate the source tree
# and docs/dev/; kpfpipe itself is imported from its install, not from here.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Local Sphinx extensions (source/_ext/), importable by bare module name below.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_ext"))


def _find_subpackages(root):
    """Dotted names of every kpfpipe (sub)package — dirs with an __init__.py.

    The recursive autosummary template collapses a *curated* package (one whose
    __init__ re-exports classes via __all__) down to its inline class docs and
    drops the with-stem submodule pages. True subpackages must survive that
    collapse (e.g. kpfpipe.modules is curated, but kpfpipe.modules.masters is a
    subpackage that keeps its own page), so the template checks membership in
    this set. Walked from the source tree rather than imported, so a package
    that fails to import is still classified correctly.
    """
    pkg_dir = os.path.join(root, "kpfpipe")
    found = set()
    for dirpath, _dirnames, filenames in os.walk(pkg_dir):
        if "__init__.py" in filenames:
            rel = os.path.relpath(dirpath, root)
            found.add(rel.replace(os.sep, "."))
    return found

# -- Project information ------------------------------------------------------

project = "KPF Data Reduction Pipeline (vNext)"
author = "The KPF Team"
copyright = "2020-2026, The KPF Team"

# Taken from the installed package, so the rendered docs always name the
# version they were built from.
version = kpfpipe.__version__
release = kpfpipe.__version__

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    # Local: derives the Data Products tables from kpfpipe/data_models/config/.
    "kpf_tables",
]

templates_path = ["_templates"]
# The governing docs under guidelines/ are Markdown (myst_parser); the rest of
# the site (and the generated API stubs) are reStructuredText.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
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
    # Subpackages that keep their own page even when the parent package is
    # curated (has __all__) and its leaf submodules are collapsed inline.
    "subpackages": _find_subpackages(_repo_root),
}

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# Order documented members by source order rather than alphabetically. For a
# package documented via automodule, "bysource" follows the __all__ order, so a
# base class listed first in __all__ leads the page above its subclasses
# (e.g. Diagnostics above DiagL0..DiagL4, matching the base-first __all__
# convention in the quality_control subpackages). For classes it orders methods
# by definition order instead of alphabetically.
autodoc_member_order = "bysource"

# -- Guidelines bridge --------------------------------------------------------

# The governing docs are stored ONCE, as Markdown, in docs/dev/ (outside the
# Sphinx source tree, so developers and AI assistants find them in a single
# obvious place). Sphinx only builds files under source/, so copy them into
# source/dev/guidelines/ at build time and let myst_parser render them. The
# copies are gitignored (like api/_generated) -- docs/dev/ stays the sole tracked
# copy, so there is no duplication in the repo. source/dev/guidelines/index.rst
# (the toctree) is the only tracked file under source/dev/guidelines/.


def _sync_guidelines(*_):
    """Copy docs/dev/*.md into source/dev/guidelines/ before Sphinx reads.

    Connected to Sphinx's ``config-inited`` event, which passes ``(app, config)``
    positionally; neither is needed here (paths come from ``_repo_root``).
    """
    import shutil

    src_dir = os.path.join(_repo_root, "docs", "dev")
    dst_dir = os.path.join(os.path.dirname(__file__), "dev", "guidelines")
    os.makedirs(dst_dir, exist_ok=True)
    for name in sorted(os.listdir(src_dir)):
        if name.endswith(".md"):
            shutil.copyfile(
                os.path.join(src_dir, name), os.path.join(dst_dir, name)
            )


def setup(app):
    # config-inited fires before source discovery, so the copies exist in time.
    app.connect("config-inited", _sync_guidelines)


# -- Options for HTML output --------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
