"""Read the KPF header-keyword registry CSVs independently (not a test module).

The conformance test needs an oracle for the keyword->extension routing that is
*independent of the code under test*, so it reads the
``config/{prefix}-{EXTENSION}-keywords.csv`` tables directly here and compares
against the live table the data model exposes
(``KPFDataModel.keyword_registry.routing``). Tests therefore never import
``data_models.keyword_registry``: registry data reaches them through the model or
through this helper.

Two resolution rules must be replicated for the comparison to line up, and both
are written out by hand here rather than imported, so the rest of the oracle's
independence is real:

* a ``#`` in a keyword expands to ``1..DETECTOR["numtrace"]``;
* a filename whose extension part is a family stem names that family's members
  (``L2-TRACE_WAVE`` -> ``TRACE1_WAVE`` .. ``TRACE5_WAVE``; ``L4-CCF`` ->
  ``CCF1`` .. ``CCF5``).
"""

import importlib.resources

import pandas as pd

from kpfpipe import DETECTOR

_CFG = importlib.resources.files("kpfpipe.data_models.config")

# The science-chain profiles. The masters profiles are deliberately out: their
# keywords are not part of the science routing contract this oracle checks.
_SCIENCE_PROFILES = ("L0", "L1", "L2", "L4")

# Family stems, spelled out: filename extension part -> member template.
_FAMILY_STEMS = {
    "TRACE_FLUX": "TRACE{i}_FLUX",
    "TRACE_WAVE": "TRACE{i}_WAVE",
    "TRACE_VAR": "TRACE{i}_VAR",
    "TRACE_BLAZE": "TRACE{i}_BLAZE",
    "CCF": "CCF{i}",
    "RV": "RV{i}",
}

_INDICES = range(1, DETECTOR["numtrace"] + 1)


def _extensions_for(stem):
    """Concrete extension names for a filename's extension part."""
    template = _FAMILY_STEMS.get(stem)
    if template is None:
        return [stem]
    return [template.format(i=i) for i in _INDICES]


def _keywords_for(keyword):
    """Concrete keyword names for a CSV Keyword cell."""
    if "#" not in keyword:
        return [keyword]
    return [keyword.replace("#", str(i)) for i in _INDICES]


def read_kpf_header_registry():
    """Combined L0/L1/L2/L4 KPF keyword registry as a DataFrame.

    Columns: ``Keyword, Description, Units, DataType, ExampleValue, PopulatedBy,
    Extension, Profile``. ``Keyword`` and ``Extension`` are fully expanded, so
    one CSV row can yield several rows here.
    """
    rows = []
    for path in sorted(_CFG.iterdir(), key=lambda p: p.name):
        if not path.name.endswith("-keywords.csv"):
            continue
        profile, _, stem = path.name[: -len("-keywords.csv")].rpartition("-")
        if profile not in _SCIENCE_PROFILES:
            continue
        extensions = _extensions_for(stem)
        for _, row in pd.read_csv(path).iterrows():
            for keyword in _keywords_for(str(row["Keyword"]).strip()):
                for extension in extensions:
                    rows.append(
                        {
                            "Keyword": keyword,
                            "Description": (
                                ""
                                if pd.isna(row["Description"])
                                else str(row["Description"]).strip()
                            ),
                            "Units": (
                                ""
                                if pd.isna(row["Units"])
                                else str(row["Units"]).strip()
                            ),
                            "DataType": (
                                ""
                                if pd.isna(row["DataType"])
                                else str(row["DataType"]).strip()
                            ),
                            "ExampleValue": row["ExampleValue"],
                            "PopulatedBy": (
                                ""
                                if pd.isna(row["PopulatedBy"])
                                else str(row["PopulatedBy"]).strip()
                            ),
                            "Extension": extension,
                            "Profile": profile,
                        }
                    )
    return pd.DataFrame(rows)


def expected_routing(table=None):
    """``{keyword: home extension}`` implied by the CSVs, or absent when unrouted.

    The rule, restated independently: PRIMARY wins when the keyword is
    registered there; otherwise a keyword with exactly one home routes to it and
    a keyword with several does not route at all.
    """
    table = read_kpf_header_registry() if table is None else table
    homes = {}
    for _, row in table.iterrows():
        homes.setdefault(row["Keyword"], set()).add(row["Extension"])
    routing = {}
    for keyword, extensions in homes.items():
        if "PRIMARY" in extensions:
            routing[keyword] = "PRIMARY"
        elif len(extensions) == 1:
            routing[keyword] = next(iter(extensions))
    return routing


def expected_comment(description, units):
    """The FITS comment a row implies: ``Description [Units]``."""
    if not units or units.lower() == "n/a":
        return description
    return f"{description} [{units}]"
