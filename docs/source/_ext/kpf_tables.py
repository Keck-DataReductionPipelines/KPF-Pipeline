"""Sphinx extension: derive the Data Products reference tables from the pipeline config.

The Data Products pages document extensions and header keywords. Rather than
restate them in prose (which drifts), the tables are rendered from
``kpfpipe/data_models/config/*.csv`` -- the same single source of truth
``extension_manifest`` and ``keyword_registry`` read at runtime. A documented
extension or keyword therefore cannot disagree with the one the code enforces.

Sphinx only reads files under ``source/``, so this extension derives the tables
into ``source/data_products/_generated/`` on ``config-inited`` (the same
build-time-copy pattern as conf.py's guidelines bridge). The generated CSVs are
gitignored; the config CSVs stay the sole tracked copy. Pages consume them with
a plain ``.. csv-table:: :file:`` directive.

Two table shapes are generated:

* ``{L0,L1,L2,L4}-extensions.csv`` -- one data model's extension manifest,
  passed through verbatim.
* ``{PRIMARY,QUALITY_CONTROL}-keywords.csv`` -- that extension's keywords
  across all four science levels, concatenated in level order with a ``Level``
  column marking where each is introduced. The levels register disjoint keyword
  sets, so the concatenation drops nothing.

The masters data models (``ML*``) are outside the Data Products scope and are
not generated here.
"""

import csv
import os

# Science data models documented under Data Products, in level order. The
# masters (ML1, ML2-flat, ML2-wls) are deliberately excluded -- see docstring.
LEVELS = ("L0", "L1", "L2", "L4")

# Extensions whose keyword tables are unioned across LEVELS, one page each.
KEYWORD_EXTENSIONS = ("PRIMARY", "QUALITY_CONTROL")

# Config column name -> table heading. Columns absent here keep their own name;
# columns present in a config CSV but absent from the source file are simply
# not emitted (the reader is per-file, so each table carries its own columns).
DISPLAY_NAMES = {
    "DataType": "Data Type",
    "BitDepth": "Bit Depth",
    "ExampleValue": "Example",
    "PopulatedBy": "Populated By",
}

_CONFIG_DIR = os.path.join("kpfpipe", "data_models", "config")
_OUTPUT_DIR = os.path.join("data_products", "_generated")


def _read(path):
    """``(header, rows)`` from a config CSV, both as lists of strings.

    Blank lines are dropped: the config CSVs use them as visual group
    separators, and ``pandas.read_csv`` -- what the runtime registry and
    manifest read them with -- skips them too.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader), [row for row in reader if row]


def _write(path, header, rows):
    """Write a generated table, translating the header to its display names."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(DISPLAY_NAMES.get(name, name) for name in header)
        writer.writerows(rows)


def _generate(config_dir, output_dir):
    """Derive every Data Products table from ``config_dir`` into ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)

    for level in LEVELS:
        name = f"{level}-extensions.csv"
        header, rows = _read(os.path.join(config_dir, name))
        _write(os.path.join(output_dir, name), header, rows)

    for extension in KEYWORD_EXTENSIONS:
        header, rows = None, []
        for level in LEVELS:
            source = os.path.join(config_dir, f"{level}-{extension}-keywords.csv")
            if not os.path.exists(source):
                continue
            level_header, level_rows = _read(source)
            # Level goes second: the keyword is the identifier the reader scans
            # for, the level an attribute of it.
            if header is None:
                header = [level_header[0], "Level", *level_header[1:]]
            rows.extend([row[0], level, *row[1:]] for row in level_rows)
        _write(os.path.join(output_dir, f"{extension}-keywords.csv"), header, rows)


def _on_config_inited(app, _config):
    """``config-inited`` handler: generate before Sphinx discovers source files."""
    repo_root = os.path.abspath(os.path.join(app.srcdir, "..", ".."))
    _generate(
        os.path.join(repo_root, _CONFIG_DIR), os.path.join(app.srcdir, _OUTPUT_DIR)
    )


def setup(app):
    app.connect("config-inited", _on_config_inited)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
