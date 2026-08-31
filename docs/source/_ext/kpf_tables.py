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

Two table shapes are generated:

* ``{L0,L1,L2,L4}-extensions.csv`` -- one data model's extension manifest,
  passed through verbatim.
* ``{PRIMARY,QUALITY_CONTROL}-keywords.csv`` -- that extension's keywords
  across all four science levels, concatenated in level order with a ``Level``
  column marking where each is introduced. The levels register disjoint keyword
  sets, so the concatenation drops nothing.

A missing config CSV is left to raise: these tables exist to keep the docs and
the config in step, so config drift must fail the build rather than quietly
render a short table.
"""

import csv
import importlib.resources
import os

# The science data models documented under Data Products, in level order, and
# the extensions whose keywords get a page of their own. Both are editorial
# choices, fixed to match the pages in data_products/ (one section per level in
# data_models.rst, one page per keyword extension) rather than discovered from
# the config -- discovery would emit tables no page references. The masters
# models (ML1, ML2-flat, ML2-wls) are outside Data Products scope.
LEVELS = ("L0", "L1", "L2", "L4")
KEYWORD_EXTENSIONS = ("PRIMARY", "QUALITY_CONTROL")

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

    for level in LEVELS:
        name = f"{level}-extensions.csv"
        _write(os.path.join(output_dir, name), *_read(config / name))

    for extension in KEYWORD_EXTENSIONS:
        header, rows = None, []
        for level in LEVELS:
            level_header, level_rows = _read(
                config / f"{level}-{extension}-keywords.csv"
            )
            # Level goes second: the keyword is the identifier the reader scans
            # for, the level an attribute of it.
            if header is None:
                header = [level_header[0], "Level", *level_header[1:]]
            rows.extend([row[0], level, *row[1:]] for row in level_rows)
        _write(os.path.join(output_dir, f"{extension}-keywords.csv"), header, rows)


def _on_config_inited(app, _config):
    """``config-inited`` handler: generate before Sphinx discovers source files."""
    _generate(os.path.join(app.srcdir, _OUTPUT_DIR))


def setup(app):
    app.connect("config-inited", _on_config_inited)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
