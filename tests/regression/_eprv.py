"""The installed rvdata reference tables, for the EPRV compliance checks.

The EPRV standard's own keyword and extension tables ship with the pinned
``rv-data-standard`` release, so the ``TestEPRVCompliance`` classes in
``test_data_models_l{0,1,2,4}.py`` read them straight from the installed package.
Nothing is vendored, so there is no third copy to drift against.
"""

import importlib.resources

import pandas as pd

from kpfpipe import DETECTOR

_RVDATA = importlib.resources.files("rvdata.core.models.config")
_KPF = importlib.resources.files("kpfpipe.data_models.config")

# rvdata numbers its per-telescope families ``BASE1 ... BASE#``. KPF observes on
# one telescope, so those expand to index 1 alone.
_TELESCOPE_BASES = frozenset(
    {"TELEID", "TLST", "TRA", "TDEC", "TEL", "TZA", "TAZ", "THA", "PARST", "PAREND"}
)

_INDICES = range(1, DETECTOR["numtrace"] + 1)


def rvdata_table(name):
    """An installed rvdata reference table, by filename stem."""
    return pd.read_csv(_RVDATA / f"{name}.csv")


def kpf_table(name):
    """A KPF config table, by filename stem."""
    return pd.read_csv(_KPF / f"{name}.csv")


def expand(keyword):
    """The concrete keywords an rvdata table cell names.

    ``BASE1 ... BASE#`` and a bare ``#`` both run 1..``DETECTOR["numtrace"]``,
    except the single-telescope families, which stop at 1.
    """
    keyword = str(keyword).strip()
    if "..." in keyword:
        base = keyword.split("...")[0].strip().rstrip("0123456789")
        indices = [1] if base in _TELESCOPE_BASES else _INDICES
        return [f"{base}{i}" for i in indices]
    if "#" in keyword:
        return [keyword.replace("#", str(i)) for i in _INDICES]
    return [keyword]
