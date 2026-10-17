"""Independent oracle for the applicability tables (not a test module).

``test_qc_flags.py`` and ``test_diagnostics.py`` check the same drift contract
over their own classes, so the shared readers live here. The CSVs are read
straight off disk rather than through ``quality_control.applicability``: that
loader is part of what is under test, so the tests must not inherit its
normalization (the class-prefix split, the boolean coercion, the BOM handling).

``keep_default_na=False`` keeps a blank ``RequiredData`` as ``""`` rather than
NaN, so the comparison turns on the text the file holds.
"""

import itertools
import re

import pandas as pd

from kpfpipe.data_models.config import PATH as _KEYWORD_CFG_PATH
from kpfpipe.data_models.extension_manifest import extension_manifest
from kpfpipe.quality_control.config import PATH as _QC_CFG_PATH

# The boolean columns, spelled out rather than imported so a silent edit to
# ``applicability.FRAME_TYPES`` cannot pass unnoticed.
FRAME_TYPES = (
    "Target",
    "SoCal",
    "Bias",
    "Dark",
    "Flat",
    "LFC",
    "ThAr",
    "UNe",
    "Etalon",
)

QC_COLUMNS = ["Method", "Description", "RequiredData", *FRAME_TYPES]
DIAG_COLUMNS = ["Method", "Keywords", "Description", "RequiredData", *FRAME_TYPES]

# A QC row's Description is ``<KEYWORD>: <registry Description sans "QC: ">``.
DESCRIPTION = re.compile(r"^(?P<keyword>[A-Z0-9_-]{1,8}): (?P<text>.+)$")


def table(class_name):
    """``class_name``'s applicability CSV as a DataFrame, columns unmodified."""
    return pd.read_csv(
        _QC_CFG_PATH / f"{class_name}-applicability.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
    )


def tagged_methods(cls, tag):
    """``{method name: attribute}`` for every ``tag``-tagged method on ``cls``."""
    return {
        name: attr
        for klass in cls.__mro__
        for name, attr in klass.__dict__.items()
        if getattr(attr, tag, None) is not None
    }


def matches(entry, names):
    """Every name an entry declares: itself when literal, else its expansion.

    ``#`` is a trace index, 1 to 5, as everywhere else in the pipeline;
    ``{a/b/c}`` is one of the listed alternatives (``NANSCI{1/2/3}``,
    ``P{16/50/84}{G/R}AMP{1/2/3/4}``). A name spelled with ``#`` in ``names``
    (``EXSNR#``) is that literal name, never expanded. The caller checks the
    result against ``names``, so an entry declaring an unknown name fails.
    """
    if entry in names:
        return {entry}
    fields = [
        "12345"
        if token == "#"
        else token[1:-1].split("/")
        if token.startswith("{")
        else [token]
        for token in re.split(r"(#|\{[^{}]*\})", entry)
    ]
    return {"".join(parts) for parts in itertools.product(*fields)}


def unknown_extensions(required_data, data_model):
    """The ``|``-separated RequiredData entries naming no ``data_model`` extension."""
    names = extension_manifest.names(data_model)
    return [
        entry
        for entry in filter(None, required_data.split("|"))
        if not matches(entry, names) <= set(names)
    ]


def registered_keywords(populated_by):
    """``{keyword: Description}`` for every registry row ``populated_by`` writes.

    Read raw, so the keyword spelling matches the CSVs (``EXSNR#`` stays a
    template) and the oracle does not inherit the registry's expansion.
    """
    rows = {}
    for path in sorted(_KEYWORD_CFG_PATH.iterdir(), key=lambda p: p.name):
        if not path.name.endswith("-keywords.csv"):
            continue
        table = pd.read_csv(path, keep_default_na=False)
        for row in table.itertuples(index=False):
            if row.PopulatedBy == populated_by:
                rows[str(row.Keyword).strip()] = str(row.Description).strip()
    return rows
