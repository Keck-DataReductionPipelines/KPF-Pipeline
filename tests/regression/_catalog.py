"""Synthetic CATALOG_RECORD tables for the data-model tests (not a test module).

Built column by column rather than through ``AstroQuery._write_catalog_record``, so
the pass-through tests stay independent of the module that populates the extension.
The AstroQuery-built variant lives in conftest, for tests needing schema fidelity.
"""

import numpy as np
from astropy.table import Table

SOURCES = ("gaia", "kpf-drp")

_STR_CELLS = {
    "object": "Gaia DR3 12345",
    "radec_src": "gaia",
    "plx_src": "gaia",
    "rv_src": "gaia",
    "ra": "12:00:00.0000",
    "dec": "+40:00:00.000",
    "frame": "icrs",
    "color_name": "Gaia BP-RP",
}
_FLOAT_CELLS = {
    "pmra": 0.5,
    "pmdec": -0.3,
    "parallax": 50.0,
    "epoch": 2016.0,
    "equinox": 2000.0,
    "color": 1.2,
}


def catalog_record_table(rv=10.0):
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
    table["z"] = np.array([np.nan if missing else 3.3357e-5] * 2, dtype=float)
    return table


def seed_sci2_cards(kpf2, *, sci_obj="Target", sky_obj="Sky", cal_obj="None"):
    """Seed the SCI2 catalog cards the CCF chain reads off a KPF2.

    These are the C*#3 cards ``KPF0.to_kpf1`` overlays for the SCI2 orderlet
    (colour + its band name, drive the CCF mask temperature; CRV3 centres the
    velocity grid) plus the per-fiber illumination sources. CRV3 is present so
    the module does not take its warn-and-default path.

    Takes the object rather than importing a data model, so this module stays
    import-light -- it is pulled in at collection through the data-model tests.
    """
    kpf2.headers["PRIMARY"]["CCLR3"] = 0.823  # G2V -> 5770 K
    kpf2.headers["PRIMARY"]["CCLRN3"] = "Gaia BP-RP"
    kpf2.headers["PRIMARY"]["CRV3"] = 0.0
    kpf2.headers["PRIMARY"]["CLSRC1"] = sky_obj
    for trace in (2, 3, 4):
        kpf2.headers["PRIMARY"][f"CLSRC{trace}"] = sci_obj
    kpf2.headers["PRIMARY"]["CLSRC5"] = cal_obj
    return kpf2
