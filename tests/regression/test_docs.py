"""Tests for the hand-maintained Data Products tables (docs/source/data_products/).

Every table on those pages is derived from ``kpfpipe.data_models.config`` at
build time by ``docs/source/_ext/kpf_tables.py``, except the PRIMARY header
table: it is maintained by hand so the whole header can be presented in reading
order rather than in config-file order. That copy can drift, so it is asserted
here against the config registries it documents.
"""

import csv
import importlib.resources
from pathlib import Path

import pandas as pd

_CFG = importlib.resources.files("kpfpipe.data_models.config")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS_TABLE = _REPO_ROOT / "docs" / "source" / "data_products" / "PRIMARY-keywords.csv"

# The science data models registering PRIMARY keywords. Masters register theirs
# in ML* registries and are outside Data Products scope.
_LEVELS = ("L0", "L1", "L2", "L4")

# The docs table's header, in config column order: the config spelling for the
# columns a reader should see verbatim, a display spelling for the rest.
_DISPLAY_HEADER = [
    "Keyword",
    "Description",
    "Units",
    "Data Type",
    "Example",
    "Populated By",
]

_CONFIG_COLUMNS = ["Description", "Units", "DataType", "ExampleValue", "PopulatedBy"]


def _documented():
    """``{keyword: row}`` from the hand-maintained docs table, header dropped."""
    with _DOCS_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row][1:]
    return {row[0].strip(): tuple(cell.strip() for cell in row[1:]) for row in rows}


def _registered():
    """``{keyword: row}`` from the config PRIMARY registries.

    ``keep_default_na=False`` keeps the blank cells as empty strings; pandas
    would otherwise read them as NaN and the comparison would turn on a parsing
    artifact rather than on the text either file holds.
    """
    rows = {}
    for level in _LEVELS:
        table = pd.read_csv(
            _CFG / f"{level}-PRIMARY-keywords.csv", keep_default_na=False
        )
        for _, row in table.iterrows():
            rows[row["Keyword"].strip()] = tuple(
                row[column].strip() for column in _CONFIG_COLUMNS
            )
    return rows


class TestPrimaryHeaderTable:
    """The hand-maintained PRIMARY table against the config it documents."""

    def test_header_is_the_display_header(self):
        with _DOCS_TABLE.open(newline="", encoding="utf-8") as handle:
            assert next(csv.reader(handle)) == _DISPLAY_HEADER

    def test_every_registered_keyword_is_documented_exactly_once(self):
        documented, registered = _documented(), _registered()
        assert set(documented) == set(registered)
        with _DOCS_TABLE.open(newline="", encoding="utf-8") as handle:
            assert sum(1 for row in csv.reader(handle) if row) - 1 == len(documented)

    def test_documented_values_match_the_config(self):
        documented, registered = _documented(), _registered()
        assert {k: documented[k] for k in registered} == registered
