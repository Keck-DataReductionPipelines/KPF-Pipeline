# -*- coding: utf-8 -*-
#
# KPF-DRP vNext documentation build configuration.
#
# Structure-only phase: this config imports NO project code (no `import
# kpfpipe`), so the docs build needs only Sphinx + the RTD theme and does not
# require installing the package or its heavy dependencies. When the API pages
# gain autodoc content, add the package install (in .readthedocs.yaml) and the
# autodoc/napoleon extensions here.

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
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"
exclude_patterns = []
pygments_style = "sphinx"

# -- Options for HTML output --------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
