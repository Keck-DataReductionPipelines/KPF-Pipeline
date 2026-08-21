"""Read the KPF header-keyword registry CSVs independently (not a test module).

The conformance test needs an oracle for the keyword->extension routing that is
*independent of the code under test*, so it reads ``config/L{0,1,2,4}-headers.csv``
directly here and compares against the live table the data model exposes
(``KPFDataModel.keyword_registry.routing``). Tests therefore never import
``data_models.keyword_registry``: registry data reaches them through the model or
through this helper.
"""

import importlib.resources

import pandas as pd

_CFG = importlib.resources.files("kpfpipe.data_models.config")


def read_kpf_header_registry():
    """Combined L0/L1/L2/L4 KPF header registry as a DataFrame.

    Columns: ``Keyword, Description, Extension, DataType, PopulatedBy``.
    """
    return pd.concat(
        [pd.read_csv(_CFG / f"L{level}-headers.csv") for level in ("0", "1", "2", "4")],
        ignore_index=True,
    )
