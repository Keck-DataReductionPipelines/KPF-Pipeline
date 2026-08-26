"""Synthetic KPF data products for the test suite (not a test module).

Builders only -- nothing here asserts. ``_dtype_policy.py`` is the one contract
module; if a cross-cutting contract needs assertions, promote it to a new
contract module rather than growing one here.

Detector-derived constants are read from the production ``DETECTOR`` table so
they cannot drift. ``FIBERS`` is in canonical slicer order (``fiber_positions``
in ``reference/detector.toml``); a test asserting a production default's fiber
*ordering* is an oracle and must keep spelling the order out literally.

Deliberately absent: a shared array width. Every consumer picks its own ncol for
a stated reason (CCF clip margins, PNG size checks, overscan geometry), so width
is always an explicit argument.
"""

import os

import numpy as np
from astropy.io import fits
from astropy.table import Table

from kpfpipe import DETECTOR

# --- detector-derived constants --------------------------------------------

CHIPS = ("GREEN", "RED")
NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
NORDER_TOTAL = DETECTOR["numorder"]
NORDER = {"GREEN": NORDER_GREEN, "RED": NORDER_RED}

# Canonical slicer order: SKY=0, SCI1=1, SCI2=2, SCI3=3, CAL=4.
FIBERS = tuple(sorted(DETECTOR["fiber_positions"], key=DETECTOR["fiber_positions"].get))

# Self-consistent raw timing cards (END - BEG == ELAPSED), for the DATTIMOK check.
GOOD_DATES = {
    "DATE-BEG": "2024-09-23T09:12:09.484",
    "DATE-MID": "2024-09-23T09:12:15.519",
    "DATE-END": "2024-09-23T09:12:21.554",
    "ELAPSED": 12.07,
    "GRDATE-B": "2024-09-23T09:12:09.484",
    "GRDATE-E": "2024-09-23T09:12:21.554",
    "RDDATE-B": "2024-09-23T09:12:09.484",
    "RDDATE-E": "2024-09-23T09:12:21.554",
}

# Exposure-meter readings tiling the GOOD_DATES shutter window, and a clean flux
# array (4 readings x 25 wavelength channels -- more than the 20-channel negative
# run EMFLUXOK looks for).
EM_BEGS = [
    "2024-09-23T09:12:09.484",
    "2024-09-23T09:12:12.484",
    "2024-09-23T09:12:15.484",
    "2024-09-23T09:12:18.484",
]
EM_ENDS = [
    "2024-09-23T09:12:12.484",
    "2024-09-23T09:12:15.484",
    "2024-09-23T09:12:18.484",
    "2024-09-23T09:12:21.554",
]
EM_CLEAN_FLUX = np.full((4, 25), 1000.0)

# A physically-sane merged catalog row, matching the pointing below.
SCIENCE_POINTING = {"RA": "12:00:00.00", "DEC": "+40:00:00.0", "MJD-OBS": 60576.38}
CATALOG_RECORD = {
    "object": "synthetic",
    "ra": SCIENCE_POINTING["RA"],
    "dec": SCIENCE_POINTING["DEC"],
    "pmra": 0.0,  # zero PM keeps the propagated position on the pointing
    "pmdec": 0.0,
    "parallax": 100.0,
    "rv": 10.0,
    "frame": "icrs",
    "epoch": 2016.0,
    "equinox": 2000.0,
    "color": 0.823,
    "color_name": "Gaia BP-RP",
}

_DEFAULT_PRIMARY = {
    "INSTRUME": "KPF",
    "OBJECT": "synthetic",
    "IMTYPE": "Bias",
    "DATE-OBS": "2024-01-01T00:00:01",
    "PROGNAME": "K123",
    "TIMEERR": "NTP time correct to within 12.3 ms",
    "GRACFFLN": "regular-read-green.acf",
    "RDACFFLN": "regular-read-red.acf",
    "GRDATE-E": GOOD_DATES["GRDATE-E"],
    "RDDATE-E": GOOD_DATES["RDDATE-E"],
    "GRDATE": "2024-09-23T09:13:08.554",
    "RDDATE": "2024-09-23T09:13:08.554",
}


# --- L0 on-disk frames ------------------------------------------------------


def _primary_hdu(path, primary_cards):
    """PRIMARY with the cards every L0 needs; ``primary_cards`` overrides.

    OFNAME defaults to the file's own basename, which is what keeps the write
    from tripping ``check_filename_convention``. A card set to None is dropped,
    so a caller can build a frame that is deliberately missing one.
    """
    cards = dict(_DEFAULT_PRIMARY, OFNAME=os.path.basename(str(path)))
    cards.update(primary_cards or {})
    primary = fits.PrimaryHDU()
    for key, value in cards.items():
        if value is not None:
            primary.header[key] = value
    return primary


def write_amp_l0(
    path,
    *,
    namps=4,
    chips=CHIPS,
    shape=(10, 10),
    bias_level=None,
    seed=42,
    with_data=True,
    primary_cards=None,
    extra_hdus=(),
):
    """Write a synthetic raw L0 with a ``{chip}_AMP{n}`` HDU per amplifier.

    Returns the path. This is a plain builder, not a fixture: the caller owns
    its own fixture scope (a 4-amp frame at full detector size is ~140 MB, which
    one caller deliberately writes once per module) and decides whether to
    ``from_fits`` the result.

    ``bias_level=None`` fills every amp with a flat 1e6 D.N., a plausible raw
    level that clears the QCL0 pixel-quality thresholds -- adequate whenever the
    test only cares that data is present. A float instead fills with seeded
    Gaussian noise about that level, for tests that measure the assembled image.
    ``with_data=False`` writes ``data=None`` HDUs, which KPF0 stores as
    ``array(None, dtype=object)`` and treats as absent.
    """
    rng = np.random.default_rng(seed)
    hdus = [_primary_hdu(path, primary_cards)]
    for chip in chips:
        for amp in range(1, namps + 1):
            if not with_data:
                data = None
            elif bias_level is None:
                data = np.full(shape, 1.0e6, dtype=np.float32)
            else:
                data = (bias_level + rng.normal(0, 3.0, shape)).astype(np.float32)
            hdus.append(fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}"))
    hdus.extend(extra_hdus)
    hdul = fits.HDUList(hdus)
    hdul.writeto(str(path), overwrite=True)
    hdul.close()
    return str(path)


def write_minimal_l0(path, *, primary_cards=None, extra_hdus=()):
    """Write an L0 carrying only PRIMARY (plus any ``extra_hdus``).

    The no-optional-extensions counterpart to ``write_amp_l0``, for tests about
    the read path rather than the pixels.
    """
    return write_amp_l0(
        path,
        namps=0,
        with_data=False,
        primary_cards=primary_cards,
        extra_hdus=extra_hdus,
    )


# --- L2 in-memory -----------------------------------------------------------


def expmeter_hdus(flux=None, sky_flux=None):
    """EXPMETER_SCI/SKY BinTableHDUs tiling the GOOD_DATES shutter window."""

    def table(values):
        columns = {"Date-Beg": EM_BEGS, "Date-End": EM_ENDS}
        for i in range(values.shape[1]):
            columns[str(5000.0 + i)] = values[:, i]
        return Table(columns)

    return [
        fits.BinTableHDU(table(EM_CLEAN_FLUX if flux is None else flux), name=name)
        for name, flux in (
            ("EXPMETER_SCI", flux),
            ("EXPMETER_SKY", sky_flux),
        )
    ]


def telemetry_hdu(nrows=1):
    """A TELEMETRY BinTableHDU carrying ``nrows`` instrument readings."""
    return fits.BinTableHDU(
        Table({"keyword": ["TEMP1"] * nrows, "average": [20.0] * nrows}),
        name="TELEMETRY",
    )


def write_science_l0(path, *, primary_cards=None, **kwargs):
    """Write an ``IMTYPE='Object'`` L0 carrying everything QCL0 requires.

    A science frame, unlike the calibration default: pointing, self-consistent
    timing, both exposure-meter tables and telemetry. ``seed_catalog_record``
    supplies the astrometry AstroQuery would have resolved.
    """
    return write_amp_l0(
        path,
        primary_cards={
            "IMTYPE": "Object",
            "EXPTIME": GOOD_DATES["ELAPSED"],
            **SCIENCE_POINTING,
            **GOOD_DATES,
            **(primary_cards or {}),
        },
        extra_hdus=[*expmeter_hdus(), telemetry_hdu()],
        **kwargs,
    )


def seed_catalog_record(kpf0, record=None):
    """Write the wmko and merged kpf-drp CATALOG_RECORD rows.

    Stands in for a completed AstroQuery run without touching the network: the
    DCS target row (required by DiagL0's TCSOFF) and the merged row QCL0's
    ASTROMOK/COLOROK read, with Gaia and SIMBAD unmatched.
    """
    from kpfpipe.modules.astro_query import AstroQuery

    aq = AstroQuery(kpf0)
    for source in ("wmko", "kpf-drp"):
        aq._write_catalog_record(source, record or CATALOG_RECORD)
    return kpf0


def set_fiber_arrays(
    kpf2, suffix, value, *, ncol, chips=CHIPS, fibers=FIBERS, dtype=np.float32
):
    """Populate ``{chip}_{fiber}_{suffix}`` with a constant for the given fibers.

    ``ncol`` is required on purpose -- see the module docstring. ``dtype`` must be
    float64 for ``WAVE``, whose EPRV MinBitDepth the write path enforces.
    """
    for chip in chips:
        for fiber in fibers:
            kpf2.set_data(
                f"{chip}_{fiber}_{suffix}",
                np.full((NORDER[chip], ncol), value, dtype=dtype),
            )


def set_wave_bands(kpf2, *, ncol):
    """Give every fiber a WAVE grid dividing its chip's band across the orders.

    A constant WAVE satisfies the shape checks but carries no order-to-wavelength
    map, which the DiagL2 SNR and flux-ratio metrics need. Each order takes a
    contiguous slice of the chip's band [Angstroms], so every wavelength those
    metrics ask for lands in exactly one order, and all fibers share the grid,
    making the inter-fiber interpolation exact.
    """
    for chip, (lo, hi) in (("GREEN", (4450.0, 6000.0)), ("RED", (6000.0, 8800.0))):
        edges = np.linspace(lo, hi, NORDER[chip] + 1)
        wave = np.stack(
            [np.linspace(edges[o], edges[o + 1], ncol) for o in range(NORDER[chip])]
        )
        for fiber in FIBERS:
            kpf2.set_data(f"{chip}_{fiber}_WAVE", wave)


# --- L4 in-memory -----------------------------------------------------------


def make_l4(
    *,
    sci=True,
    rv_filled=True,
    jitter=0.0,
    berv=0.0,
    bervrng=None,
    bjdrng=None,
    seed=3,
):
    """KPF4 with science CCF cubes and per-order RV tables.

    ``jitter`` is the per-order scatter applied to BJD_TDB and BERV. It defaults
    to 0 (every order identical). **A caller exercising the BJDOK/BERVOK gates
    must pass 1e-7** -- that magnitude is tuned to sit well inside those gates
    (about 0.03 s in BJD, 3e-4 m/s in BERV), so a good product passes; re-rolling it
    larger silently turns a passing test into a failing one. Non-zero jitter also
    scatters RV itself by 1e-3 km/s, so the product looks like a real one.

    ``rv_filled=False`` seeds NaN RVs, as a CrossCorrelation-only L4 has before
    RadialVelocity runs.

    Does not seed the required PRIMARY keywords; call ``seed_required_primary``
    when the test needs KWRDPRL4 to pass.
    """
    from kpfpipe.data_models.level4 import KPF4

    l4 = KPF4()
    if sci:
        rng = np.random.default_rng(seed)

        def scatter(width):
            # Always an array: a scalar here would make astropy reject the
            # mixed scalar/array column dict.
            if not jitter:
                return np.zeros(NORDER_TOTAL)
            return rng.normal(0, width, NORDER_TOTAL)

        for fiber in ("SCI1", "SCI2", "SCI3"):
            if not rv_filled:
                rv_col = np.full(NORDER_TOTAL, np.nan)
            else:
                rv_col = np.zeros(NORDER_TOTAL) + scatter(1e-3)
            l4.set_data(f"{fiber}_CCF", np.ones((NORDER_TOTAL, 5)))
            l4.set_data(f"{fiber}_CCF_VAR", np.ones((NORDER_TOTAL, 5)))
            l4.set_data(
                f"{fiber}_RV",
                Table(
                    {
                        "ORDER_INDEX": np.arange(NORDER_TOTAL),
                        "RV": rv_col,
                        "RV_ERR": np.full(NORDER_TOTAL, 1e-3),
                        "BJD_TDB": 2460000.0 + scatter(jitter),
                        "BERV": berv + scatter(jitter),
                        "WAVE_START": np.full(NORDER_TOTAL, 4500.0),
                        "WAVE_END": np.full(NORDER_TOTAL, 8700.0),
                        "WEIGHT": np.ones(NORDER_TOTAL),
                    }
                ),
            )
    if bervrng is not None:
        l4.headers["QUALITY_CONTROL"]["BERVRNG"] = bervrng
    if bjdrng is not None:
        l4.headers["QUALITY_CONTROL"]["BJDRNG"] = bjdrng
    return l4


def seed_required_primary(kpf, qc_cls):
    """Seed every PRIMARY keyword ``qc_cls`` requires, skipping ones already set.

    KWRDPR* is presence-only, so sentinel values suffice; reusing the production
    ``_required_primary_keywords`` avoids drift. ``qc_cls`` is a parameter rather
    than an import, so this module stays free of quality_control.
    """
    for kw in qc_cls(kpf)._required_primary_keywords():
        if kw not in kpf.headers["PRIMARY"]:
            kpf.headers["PRIMARY"][kw] = ("UNKNOWN", "seeded for test")
