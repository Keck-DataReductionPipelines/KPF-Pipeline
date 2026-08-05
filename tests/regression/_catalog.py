"""Synthetic CATALOG_RECORD tables for the data-model tests.

Built column by column rather than through ``AstroQuery._write_catalog_record``, so
the pass-through tests stay independent of the module that populates the extension.
The AstroQuery-built variant lives in conftest, for tests needing schema fidelity.
"""

import numpy as np
from astropy.table import Table

SOURCES = ("gaia", "kpf-drp")

_STR_CELLS = {
    "object": "12345",
    "radec_src": "gaia",
    "plx_src": "gaia",
    "rv_src": "gaia",
    "ra": "01:44:04.0000",
    "dec": "-15:56:14.900",
    "frame": "icrs",
    "color_name": "Gaia BP-RP",
}
_FLOAT_CELLS = {
    "pmra": -1.7,
    "pmdec": 0.85,
    "parallax": 273.8,
    "epoch": 2016.0,
    "equinox": 2000.0,
    "color": 1.2,
}


def catalog_record_table(rv=-16.6):
    """A two-row CATALOG_RECORD table: a 'gaia' source row + the merged 'kpf-drp'.

    Carries the full write schema, since KPF0.to_kpf1 reads every canonical column
    when overlaying the row onto the C*# cards. ``rv=None`` leaves rv (and the
    redshift derived from it) NaN -- the sentinel astropy's FITS reader hands back
    masked and ``KPFDataModel.from_fits`` normalizes.
    """
    table = Table()
    table["source"] = np.array(SOURCES, dtype=str)
    for column, value in _STR_CELLS.items():
        table[column] = np.array([value] * 2, dtype=str)
    for column, value in _FLOAT_CELLS.items():
        table[column] = np.array([value] * 2, dtype=float)
    missing = rv is None
    table["rv"] = np.array([np.nan if missing else rv] * 2, dtype=float)
    table["z"] = np.array([np.nan if missing else -5.5e-5] * 2, dtype=float)
    return table
