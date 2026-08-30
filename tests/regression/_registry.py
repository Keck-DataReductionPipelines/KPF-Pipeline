"""Independent oracles for the keyword registry (not a test module).

``test_data_models_base.py`` needs an answer to "where does this keyword route?"
that is *independent of the code under test*, so the routing rule is re-derived
here from the ``config/{prefix}-{EXTENSION}-keywords.csv`` tables rather than read
off ``KPFDataModel.keyword_registry``. Tests therefore never import
``data_models.keyword_registry``: registry data reaches them through the model,
and the oracle it is checked against comes from here.

Two resolution rules are replicated by hand rather than imported, so the
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


def _homes():
    """``{keyword: set of extensions}`` implied by the science keyword CSVs."""
    homes = {}
    for path in sorted(_CFG.iterdir(), key=lambda p: p.name):
        if not path.name.endswith("-keywords.csv"):
            continue
        profile, _, stem = path.name[: -len("-keywords.csv")].rpartition("-")
        if profile not in _SCIENCE_PROFILES:
            continue
        template = _FAMILY_STEMS.get(stem)
        extensions = (
            [stem] if template is None else [template.format(i=i) for i in _INDICES]
        )
        for keyword in pd.read_csv(path)["Keyword"]:
            keyword = str(keyword).strip()
            members = (
                [keyword.replace("#", str(i)) for i in _INDICES]
                if "#" in keyword
                else [keyword]
            )
            for member in members:
                homes.setdefault(member, set()).update(extensions)
    return homes


def expected_routing():
    """``{keyword: home extension}`` implied by the CSVs, or absent when unrouted.

    The rule, restated independently: PRIMARY wins when the keyword is registered
    there; otherwise a keyword with exactly one home routes to it, and a keyword
    with several does not route at all.
    """
    routing = {}
    for keyword, extensions in _homes().items():
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
