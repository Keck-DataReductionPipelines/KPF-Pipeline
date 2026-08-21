"""
Shared pytest fixtures, helpers, and configuration for the KPF-Pipeline suite.

Synthetic-FITS builders that more than one test module needs live here so the
per-level modules (``test_data_models_l*.py`` and friends) don't each re-declare
them. Fixtures are seeded for run-to-run determinism, matching the convention
already used by ``test_quicklook_l0.py`` and ``test_master_*.py``.
"""

import socket
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

# ---------------------------------------------------------------------------
# Offline guard -- deliberately module-scope, not a fixture
# ---------------------------------------------------------------------------
#
# The regression suite is offline by contract: every external query is mocked.
# This makes that a rule the suite enforces rather than a claim its docstrings
# make -- a test that reaches for the wire now fails loudly, naming the address.
#
# It must run at import, not in a fixture: astroquery.gaia opens a connection to
# the ESA archive when `kpfpipe.modules.astro_query` is imported
# (astro_query.py:27, a filed production defect), and that import happens during
# COLLECTION, before any fixture -- even a session-scoped autouse one -- can run.
# Blocking it there is worth 1.4-2.7 s per worker process; astroquery survives
# the refusal, printing "Status messages could not be retrieved" and carrying on
# with the real `Gaia` object the tests patch.
#
# Loopback stays open so pytest-xdist, execnet and any local-socket test are
# unaffected.
_real_connect = socket.socket.connect


def _no_outbound_connect(self, address, *args, **kwargs):
    host = address[0] if isinstance(address, tuple) else address
    if isinstance(host, str) and (
        host.startswith("127.") or host in ("::1", "localhost")
    ):
        return _real_connect(self, address, *args, **kwargs)
    raise OSError(f"the regression suite is offline; blocked connect to {address!r}")


socket.socket.connect = _no_outbound_connect

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
        "slow: touches the real tests/testdata truth frames. Paired with "
        "requires_testdata rather than overlapping it: requires_testdata says "
        "the frames must be present, slow says the fast pre-commit subset "
        "(`-m 'not slow'`) skips the cost of reading them. Not a timing claim -- "
        "do not mark a synthetic test slow because it feels slow, and do not "
        "leave a real-data test unmarked because it happens to be quick.",
    )
    config.addinivalue_line(
        "markers",
        "cli: scripts/CLI/tools-layer test (imports scripts.* / tools.*); "
        "excluded from the fast pre-commit subset, which covers recipes and "
        "below. Run in the full suite or focused with `-m cli`",
    )
    config.addinivalue_line(
        "markers",
        "quicklook: exercises kpfpipe.quality_control.quicklook -- the "
        "PlotL0/L1/L2/L4 renderers; excluded from the fast pre-commit subset "
        "because the PNG rendering is slow. Names the MODULE under test, not "
        "the technique: a test elsewhere in the tree that happens to render a "
        "figure does not get this marker, so `-m quicklook` collects exactly "
        "the quicklook plots a developer is working on and nothing else. "
        "(scripts/quality_control/qlp.py drives these renderers, but its test "
        "is scripts-layer and so carries `cli`.) Run in the full suite or "
        "focused with `-m quicklook`",
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
