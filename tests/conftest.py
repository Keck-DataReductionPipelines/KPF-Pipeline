"""
Shared pytest fixtures, helpers, and configuration for the KPF-Pipeline suite.

Synthetic-FITS builders that more than one test module needs live here so the
per-level modules (``test_data_models_l*.py`` and friends) don't each re-declare
them. Fixtures are seeded for run-to-run determinism, matching the convention
already used by ``test_quicklook_l0.py`` and ``test_master_*.py``.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

# Fixed seed so every synthetic-FITS fixture is byte-stable across runs.
_SEED = 20240113

# Real (gitignored) truth frames; tests marked ``requires_testdata`` are skipped
# when this directory is absent rather than erroring at collection.
TESTDATA_DIR = Path(__file__).parent / "testdata"


def image_hdu(name, shape, rng, dtype=np.float32):
    """Return a named ``ImageHDU`` filled with seeded random data."""
    return fits.ImageHDU(data=rng.random(shape).astype(dtype), name=name)


# ---------------------------------------------------------------------------
# Pytest configuration / markers
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_testdata: needs the gitignored tests/testdata truth frames; "
        "skipped when they are absent",
    )
    config.addinivalue_line(
        "markers",
        "slow: slow integration or heavy-compute test; excluded from the fast "
        "pre-commit subset (`-m 'not slow'`), run in the full suite",
    )
    config.addinivalue_line(
        "markers",
        "cli: scripts/CLI/tools-layer test (imports scripts.* / tools.*); "
        "excluded from the fast pre-commit subset, which covers recipes and "
        "below. Run in the full suite or focused with `-m cli`",
    )
    config.addinivalue_line(
        "markers",
        "quicklook: quicklook/QLP render test (slow PNG rendering, an offshoot "
        "from the production path); excluded from the fast pre-commit subset. "
        "Run in the full suite or focused with `-m quicklook`",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``requires_testdata`` tests when the truth frames are not present."""
    if TESTDATA_DIR.exists():
        return
    skip = pytest.mark.skip(reason="tests/testdata not present")
    for item in items:
        if "requires_testdata" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Synthetic-FITS fixtures (shared across data-model level modules)
# ---------------------------------------------------------------------------


def _catalog_record_hdu():
    """A CATALOG_RECORD BinTableHDU with the canonical 'kpf-drp' row, built via
    AstroQuery's writer for schema fidelity. A science L0's to_kpf1 needs this
    populated; single-source (gaia) so the overlay logs no mixed-source warning."""
    # Deferred, not for a cycle: this root conftest is imported at collection for
    # every session, and astro_query pulls in astroquery/astropy (~2 s).
    from kpfpipe.data_models.level0 import KPF0
    from kpfpipe.modules.astro_query import AstroQuery

    l0 = KPF0()
    l0.headers["PRIMARY"]["IMTYPE"] = "Object"
    AstroQuery(l0)._write_catalog_record(
        "kpf-drp",
        {
            "object": "Gaia_HD10700",
            "radec_src": "gaia",
            "plx_src": "gaia",
            "rv_src": "gaia",
            "ra": "01:44:04.0000",
            "dec": "-15:56:14.900",
            "pmra": -1.7,
            "pmdec": 0.85,
            "parallax": 273.8,
            "rv": -16.6,
            "frame": "icrs",
            "epoch": 2016.0,
            "equinox": 2000.0,
        },
    )
    return fits.BinTableHDU(data=l0.data["CATALOG_RECORD"], name="CATALOG_RECORD")


@pytest.fixture(scope="session")
def synthetic_l0_file(tmp_path_factory):
    """Create a minimal synthetic L0 FITS file (session-scoped read-only source:
    every consumer only from_fits() reads it and writes outputs to its own tmp_path)."""
    rng = np.random.default_rng(_SEED)
    fn = str(tmp_path_factory.mktemp("l0") / "KP.20240113.23249.10.fits")

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-01-13T10:26:56"
    primary.header["MJD-OBS"] = 60322.43537  # JD_UTC source (full JD = + 2400000.5)
    primary.header["EXPTIME"] = 300.0
    primary.header["ELAPSED"] = 300.0
    primary.header["OBJECT"] = "HD_10700"
    primary.header["IMTYPE"] = "Object"
    primary.header["GROBSERV"] = "Smith"
    primary.header["PROGNAME"] = "K123"
    primary.header["OFNAME"] = "KP.20240113.23249.10.fits"

    telemetry = Table({"keyword": ["TEMP1", "TEMP2"], "average": [20.0, 21.0]})
    telemetry_hdu = fits.BinTableHDU(data=telemetry, name="TELEMETRY")

    hdul = fits.HDUList(
        [
            primary,
            image_hdu("GREEN_AMP1", (32, 32), rng),
            image_hdu("GREEN_AMP2", (32, 32), rng),
            image_hdu("RED_AMP1", (32, 32), rng),
            image_hdu("CA_HK", (16, 16), rng),
            telemetry_hdu,
            _catalog_record_hdu(),
        ]
    )
    hdul.writeto(fn, overwrite=True)
    hdul.close()

    return fn


@pytest.fixture
def synthetic_l0_minimal(tmp_path):
    """Create an L0 file with only PRIMARY (no optional extensions)."""
    # Deferred like _catalog_record_hdu above: this root conftest is imported at
    # collection for every session, and _data_models pulls in kpfpipe.
    from .regression._data_models import write_minimal_l0

    return write_minimal_l0(
        tmp_path / "KP.20240113.00001.00.fits",
        primary_cards={"DATE-OBS": "2024-01-13T00:00:01", "PROGNAME": None},
    )


@pytest.fixture(scope="session")
def synthetic_l1_file(tmp_path_factory):
    """Create a minimal synthetic L1 FITS file (session-scoped read-only source:
    every consumer only from_fits() reads it and writes outputs to its own tmp_path)."""
    rng = np.random.default_rng(_SEED)
    fn = str(tmp_path_factory.mktemp("l1") / "kpf_L1_20240113T102656.fits")

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-01-13T10:26:56"
    primary.header["EXPTIME"] = 300.0
    primary.header["DATALVL"] = "L1"

    hdul = fits.HDUList(
        [
            primary,
            image_hdu("GREEN_CCD", (32, 32), rng),
            image_hdu("GREEN_VAR", (32, 32), rng),
            image_hdu("RED_CCD", (32, 32), rng),
            image_hdu("RED_VAR", (32, 32), rng),
        ]
    )
    hdul.writeto(fn, overwrite=True)
    hdul.close()

    return fn
