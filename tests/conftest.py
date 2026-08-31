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
# Network guard -- deliberately module-scope, not a fixture
# ---------------------------------------------------------------------------
#
# Every catalog query the pipeline makes is mocked, and this makes that a rule
# the suite enforces rather than a claim its docstrings make: a test that loses
# its patch fails loudly, naming the host.
#
# It must run at import, not in a fixture. `import barycorrpy` fetches IERS_B and
# leap seconds before it finishes executing, and that import happens during
# COLLECTION -- before any fixture, even a session-scoped autouse one, can run.
#
# The suite is NOT strictly offline. Reference data is allowed through, because
# there is no local cache to fall back on and no configuration that suppresses
# the fetch. A cold machine therefore downloads ~120 MB on its first run. That is
# the accepted price of not carrying a cache; it is paid by CI and by every fresh
# checkout.

# Mocked by every test that touches them. A connect here is a lost patch.
_CATALOG_HOSTS = (
    "gea.esac.esa.int",  # ESA Gaia TAP
    "simbad.cds.unistra.fr",  # SIMBAD, and its Harvard mirror below
    "simbad.harvard.edu",
)

# Reference data with no local alternative. Each entry is here because something
# fetches it unconditionally at import or first use -- not for convenience.
_REFERENCE_HOSTS = (
    "hpiers.obspm.fr",  # IERS_B + leap seconds; barycorrpy fetches at import
    "datacenter.iers.org",  # astropy IERS_A (Earth orientation)
    "maia.usno.navy.mil",  # IERS_A mirror, also barycorrpy's own
    "naif.jpl.nasa.gov",  # JPL solar-system ephemeris
    "data.astropy.org",  # astropy's data server, and its mirror below
    "www.astropy.org",
)

_real_getaddrinfo = socket.getaddrinfo
_real_connect = socket.socket.connect

# IPs that came back from resolving an allowed host. connect() is handed an
# address, not a name, so this is the only way it can tell them apart.
_allowed_addresses = set()


def _is_local(host):
    # host is None when binding; loopback must stay open or xdist/execnet die.
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    return isinstance(host, str) and (
        host.startswith("127.") or host in ("::1", "localhost", "0.0.0.0", "")
    )


def _blocked(host):
    why = (
        "the suite mocks this catalog -- a test lost its patch"
        if host in _CATALOG_HOSTS
        else f"not a known reference-data host; allowed: {', '.join(_REFERENCE_HOSTS)}"
    )
    return OSError(f"tests/conftest.py blocked a connection to {host!r}: {why}")


def _guarded_getaddrinfo(host, *args, **kwargs):
    if _is_local(host):
        return _real_getaddrinfo(host, *args, **kwargs)
    if host not in _REFERENCE_HOSTS:
        raise _blocked(host)
    infos = _real_getaddrinfo(host, *args, **kwargs)
    _allowed_addresses.update(info[4][0] for info in infos)
    return infos


def _guarded_connect(self, address, *args, **kwargs):
    # The backstop: catches anything that reaches a bare IP without resolving it.
    host = address[0] if isinstance(address, tuple) else address
    if _is_local(host) or host in _allowed_addresses:
        return _real_connect(self, address, *args, **kwargs)
    raise _blocked(host)


socket.getaddrinfo = _guarded_getaddrinfo
socket.socket.connect = _guarded_connect

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
    # AstroQuery reads the native instrument header, not the EPRV PRIMARY.
    l0.headers["INSTRUMENT_HEADER"]["IMTYPE"] = "Object"
    AstroQuery(l0)._write_catalog_record(
        "kpf-drp",
        {
            "object": "Gaia DR3 12345",
            "radec_src": "gaia",
            "plx_src": "gaia",
            "rv_src": "gaia",
            "ra": "12:00:00.0000",
            "dec": "+40:00:00.000",
            "pmra": 0.5,
            "pmdec": -0.3,
            "parallax": 50.0,
            "rv": 10.0,
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
    primary.header["OBJECT"] = "10700"
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
