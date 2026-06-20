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


@pytest.fixture
def synthetic_l0_file(tmp_path):
    """Create a minimal synthetic L0 FITS file."""
    rng = np.random.default_rng(_SEED)
    fn = str(tmp_path / "KP.20240113.23249.10.fits")

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-01-13T10:26:56"
    primary.header["EXPTIME"] = 300.0
    primary.header["OBJECT"] = "HD_10700"
    primary.header["IMTYPE"] = "Object"

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
        ]
    )
    hdul.writeto(fn, overwrite=True)
    hdul.close()

    return fn


@pytest.fixture
def synthetic_l0_minimal(tmp_path):
    """Create an L0 file with only PRIMARY (no optional extensions)."""
    fn = str(tmp_path / "KP.20240113.00001.00.fits")

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-01-13T00:00:01"

    hdul = fits.HDUList([primary])
    hdul.writeto(fn, overwrite=True)
    hdul.close()

    return fn


@pytest.fixture
def synthetic_l1_file(tmp_path):
    """Create a minimal synthetic L1 FITS file."""
    rng = np.random.default_rng(_SEED)
    fn = str(tmp_path / "kpf_L1_20240113T102656.fits")

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
