"""Sphinx extension: derive the Data Products reference tables from the pipeline config.

The Data Products pages document extensions and header keywords. Rather than
restate them in prose (which drifts), the tables are rendered from the CSVs in
``kpfpipe.data_models.config`` -- the same single source of truth
``extension_manifest`` and ``keyword_registry`` read at runtime, reached the
same way (``importlib.resources``), so a documented extension or keyword cannot
disagree with the one the code enforces.

Sphinx only reads files under ``source/``, so this extension derives the tables
into ``source/data_products/_generated/`` on ``config-inited`` (the same
build-time-generation pattern as conf.py's guidelines bridge). The generated
CSVs are gitignored; the config CSVs stay the sole tracked copy. Pages consume
them with a plain ``.. csv-table:: :file:`` directive.

One table shape is generated: one table per level, passed through verbatim --
``{L0,L1,L2,L4}-extensions.csv`` (that model's extension manifest) and
``{L0,L1,L2,L4}-QUALITY_CONTROL-keywords.csv`` (the QC keywords that level's QC
pass writes).

The PRIMARY keyword table is *not* generated: ``data_products/PRIMARY-keywords.csv``
is maintained by hand, so the whole header can be presented in reading order
rather than in config-file order. ``tests/regression/test_docs.py`` asserts it
against the config registries, so drift fails the suite instead of the build.

A missing config CSV is left to raise: these tables exist to keep the docs and
the config in step, so config drift must fail the build rather than quietly
render a short table.
"""

import csv
import importlib.resources
import os

# The science data models documented under Data Products, in level order. The
# masters models (ML1, ML2-flat, ML2-wls) are outside Data Products scope.
LEVELS = ("L0", "L1", "L2", "L4")

# Config CSVs rendered one table per level, named by the config's own file
# naming: "{level}-{stem}.csv".
PER_LEVEL_TABLES = ("extensions", "QUALITY_CONTROL-keywords")

# Which config CSVs are rendered is an editorial choice, fixed to match the pages
# in data_products/ (a section per level in data_models.rst and in
# quality_control.rst) rather than discovered from the config -- discovery would
# emit tables no page references.

# Config column name -> table heading, for the columns whose config spelling is
# not what a reader should see. Anything absent here keeps its own name.
DISPLAY_NAMES = {
    "DataType": "Data Type",
    "BitDepth": "Bit Depth",
    "ExampleValue": "Example",
    "PopulatedBy": "Populated By",
}

# Generated tables land here, relative to the Sphinx source dir.
_OUTPUT_DIR = os.path.join("data_products", "_generated")


def _read(path):
    """``(header, rows)`` from a config CSV, both as lists of strings.

    Blank lines are dropped: the config CSVs use them as visual group
    separators, and ``pandas.read_csv`` -- what the runtime registry and
    manifest read them with -- skips them too.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader), [row for row in reader if row]


def _write(path, header, rows):
    """Write a generated table, translating the header to its display names."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(DISPLAY_NAMES.get(name, name) for name in header)
        writer.writerows(rows)


def _generate(output_dir):
    """Derive every Data Products table into ``output_dir``."""
    config = importlib.resources.files("kpfpipe.data_models.config")
    os.makedirs(output_dir, exist_ok=True)

    for stem in PER_LEVEL_TABLES:
        for level in LEVELS:
            name = f"{level}-{stem}.csv"
            _write(os.path.join(output_dir, name), *_read(config / name))


def _on_config_inited(app, _config):
    """``config-inited`` handler: generate before Sphinx discovers source files."""
    _generate(os.path.join(app.srcdir, _OUTPUT_DIR))


def setup(app):
    app.connect("config-inited", _on_config_inited)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
