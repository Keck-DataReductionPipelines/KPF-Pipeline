"""Independent oracle for the applicability tables (not a test module).

``test_qc_flags.py`` and ``test_diagnostics.py`` check the same drift contract
over their own classes, so the shared readers live here. The CSVs are read
straight off disk rather than through ``quality_control.applicability``: that
loader is part of what is under test, so the tests must not inherit its
normalization (the class-prefix split, the boolean coercion, the BOM handling).

``keep_default_na=False`` keeps a blank ``RequiredData`` as ``""`` rather than
NaN, so the comparison turns on the text the file holds.
"""

import re

import pandas as pd

from kpfpipe.data_models.config import PATH as _KEYWORD_CFG_PATH
from kpfpipe.data_models.extension_manifest import extension_manifest
from kpfpipe.quality_control.config import PATH as _QC_CFG_PATH

# The boolean columns, spelled out rather than imported so a silent edit to
# ``applicability.FRAME_TYPES`` cannot pass unnoticed.
FRAME_TYPES = ("Star", "Sun", "Bias", "Dark", "Flat", "LFC", "ThAr", "UNe", "Etalon")

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
    """The ``names`` an entry covers: itself when literal, else its ``#`` family.

    ``#`` stands for one varying field -- one or more characters -- so it covers
    both an index (``TRACE#_FLUX``, ``EXSNR#``) and a named field (``P#AMP#``,
    ``GDR#RMS``). A name spelled with ``#`` in the source table is that literal
    name, never a pattern, which is what keeps ``EXSNR#`` off ``EXSNRW#``.
    """
    if entry in names:
        return {entry}
    pattern = ".+".join(map(re.escape, entry.split("#")))
    return {name for name in names if re.fullmatch(pattern, name)}


def unknown_extensions(required_data, data_model):
    """The ``|``-separated RequiredData entries no ``data_model`` extension covers."""
    names = extension_manifest.names(data_model)
    return [
        entry
        for entry in filter(None, required_data.split("|"))
        if not matches(entry, names)
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
