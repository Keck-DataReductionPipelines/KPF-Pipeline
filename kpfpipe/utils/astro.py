"""Astronomy-related helper functions, in the style of ``astropy``."""

import logging

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.constants import c
from astropy.coordinates import EarthLocation

from kpfpipe import OBSERVATORY, REPO_ROOT

logger = logging.getLogger(__name__)

# Lives here, not in the package root, to keep that root free of astropy. It is
# the only reader of OBSERVATORY: everything else wants the site as a location.
KECK_LOCATION = EarthLocation.from_geodetic(
    lat=OBSERVATORY["latitude"] * u.deg,
    lon=OBSERVATORY["longitude"] * u.deg,
    height=OBSERVATORY["altitude"] * u.m,
    ellipsoid=OBSERVATORY["geosys"],
)


def compute_doppler_factor(v):
    """Relativistic Doppler factor ``f = lambda_obs / lambda_rest`` for a source
    with radial velocity ``v``.

    Standard astronomical convention: a receding source (``v > 0``) is
    redshifted, so ``f > 1``. The factor relates to the redshift by
    ``f = 1 + z``.

    Parameters
    ----------
    v : astropy.units.Quantity
        Radial velocity [km/s] (any velocity unit accepted); positive =
        receding. Passing a bare (unitless) value raises, so units stay
        explicit.

    Returns
    -------
    float or ndarray
        Dimensionless Doppler factor ``lambda_obs / lambda_rest``.
    """
    beta = (v / c).to(u.dimensionless_unscaled).value
    return ((1.0 + beta) / (1.0 - beta)) ** 0.5


def compute_redshift(v):
    """Relativistic redshift ``z = lambda_obs / lambda_rest - 1`` for a source
    with radial velocity ``v``. A receding source (``v > 0``) gives ``z > 0``;
    computed as ``compute_doppler_factor(v) - 1``.

    Parameters
    ----------
    v : astropy.units.Quantity
        Radial velocity [km/s] (any velocity unit accepted); positive =
        receding. Passing a bare (unitless) value raises, so units stay
        explicit.

    Returns
    -------
    float or ndarray
        Dimensionless redshift.
    """
    return compute_doppler_factor(v) - 1.0


def air_to_vac(wave_air):
    """Convert air wavelengths to vacuum via the Edlén 1953 formula (two
    iterations; only wavelengths > 2000 Å are modified).

    Not yet wired into a pipeline module: retained for the forthcoming
    vacuum-wavelength work and used as a test oracle for wavelength-calibration
    output.

    Parameters
    ----------
    wave_air : array-like
        Wavelengths [Å, air].

    Returns
    -------
    ndarray
        Wavelengths [Å, vacuum].
    """
    wave_vac = np.asarray(wave_air, dtype=np.float64).copy()
    modify = wave_vac > 2000.0
    if np.any(modify):
        wave_new = wave_vac[modify]
        for _ in range(2):
            sigma2 = (1e4 / wave_vac[modify]) ** 2
            fact = (
                1.0 + 5.792105e-2 / (238.0185 - sigma2) + 1.67917e-3 / (57.362 - sigma2)
            )
            wave_vac[modify] = wave_new * fact
    return wave_vac


# The empirical main-sequence colour/temperature table of Pecaut & Mamajek (2013),
# "A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence", held in
# reference/ verbatim as distributed. Only these columns are read; G-J is not
# tabulated and is formed below as M_G - M_J.
_EEM_PATH = f"{REPO_ROOT}/reference/EEM_dwarf_UBVIJHK_colors_Teff.txt"
_EEM_COLUMNS = ("Teff", "B-V", "Bp-Rp", "M_G", "M_J")


def _read_eem_table():
    """Read the tabulated dwarf sequence from ``_EEM_PATH``.

    The upstream file is a '#'-commented preamble, a '#SpT ...' column header, the
    data block, then that header again and a notes footer -- so the data block is
    every line between the first header and the next commented one. Cells absent
    from the table are runs of dots and a few carry a trailing ':' (uncertain);
    stripping the ':' and coercing leaves the dot-runs as NaN.

    Returns
    -------
    pandas.DataFrame
        The ``_EEM_COLUMNS`` columns as floats, ordered hot to cool.
    """
    with open(_EEM_PATH) as handle:
        lines = handle.read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("#SpT"))
    names = lines[start].lstrip("#").split()
    rows = []
    for line in lines[start + 1 :]:
        if line.startswith("#"):
            break
        rows.append(line.split())
    table = pd.DataFrame(rows, columns=names)[list(_EEM_COLUMNS)]
    for column in _EEM_COLUMNS:
        table[column] = pd.to_numeric(table[column].str.rstrip(":"), errors="coerce")
    return table


_EEM = _read_eem_table()


def _monotone_sequence(color):
    """Strictly increasing ``(colour, Teff)`` arrays for one tabulated colour.

    Interpolation needs an increasing colour, but the sequence reverses in a few
    places -- for G-J only in the M_G - M_J difference, never in either magnitude
    alone. Walking hot to cool and keeping a row only when its colour exceeds the
    last kept one censors those rows without naming spectral types, so a refreshed
    table stays valid.

    Parameters
    ----------
    color : array-like
        One colour index [mag] per table row, ordered hot to cool.

    Returns
    -------
    tuple of ndarray
        ``(colour, Teff)``, colour strictly increasing and Teff [K] decreasing.
    """
    color = np.asarray(color, dtype=np.float64)
    teff = np.asarray(_EEM["Teff"], dtype=np.float64)
    usable = np.isfinite(color) & np.isfinite(teff)
    color, teff = color[usable], teff[usable]

    keep = np.zeros(color.size, dtype=bool)
    reddest = -np.inf
    for i, value in enumerate(color):
        if value > reddest:
            keep[i] = True
            reddest = value
    return color[keep], teff[keep]


_B_V = _monotone_sequence(_EEM["B-V"])
_BP_RP = _monotone_sequence(_EEM["Bp-Rp"])
_G_J = _monotone_sequence(_EEM["M_G"] - _EEM["M_J"])

# Colour index -> its tabulated sequence, keyed by CCLRN#
_COLOR_SEQUENCES = {
    "B-V": _B_V,
    "Gaia BP-RP": _BP_RP,
    "G-J": _G_J,
}


def color_to_teff(color, color_name):
    """Effective temperature from a catalog colour index.

    Interpolates the empirical main-sequence colour/temperature sequence of
    Pecaut & Mamajek (2013), held in ``reference/`` as distributed. The sequence is
    solar-metallicity dwarfs and the colour is taken as already dereddened, so the
    temperature is indicative rather than a measurement.

    Beyond either end of the tabulated colour range the two outermost rows set a
    linear extrapolation, so a very hot or very cool star still gets a temperature
    rather than no answer at all. That extrapolation is uncalibrated, hence the
    warning.

    Parameters
    ----------
    color : float
        Colour index [mag], bluer band minus redder band.
    color_name : str
        Which index, as AstroQuery labels it on ``CCLRN#``: ``'B-V'`` (SIMBAD),
        ``'Gaia BP-RP'`` (Gaia), or ``'G-J'`` (WMKO).

    Returns
    -------
    float
        Effective temperature [K].

    Raises
    ------
    ValueError
        ``color_name`` is not one of the three recognized indices, ``color`` is
        not a finite number, or the colour is so far outside the tabulated range
        that it extrapolates to a non-physical temperature.
    """
    if color_name not in _COLOR_SEQUENCES:
        raise ValueError(
            f"unrecognized colour index {color_name!r}; expected one of "
            f"{list(_COLOR_SEQUENCES)}"
        )
    try:
        value = float(color)
    except (TypeError, ValueError):
        value = np.nan
    if not np.isfinite(value):
        raise ValueError(f"{color_name} colour {color!r} is not a finite number")

    colors, teffs = _COLOR_SEQUENCES[color_name]
    if colors[0] <= value <= colors[-1]:
        return float(np.interp(value, colors, teffs))

    lo, hi = (0, 1) if value < colors[0] else (-2, -1)
    slope = (teffs[hi] - teffs[lo]) / (colors[hi] - colors[lo])
    teff = float(teffs[lo] + slope * (value - colors[lo]))
    if teff <= 0.0:
        raise ValueError(
            f"{color_name} = {value} extrapolates to a non-physical "
            f"Teff = {teff:.0f} K; "
            "the colour is outside any plausible range"
        )
    logger.warning(
        "%s = %.3f lies outside the tabulated range %.3f to %.3f; extrapolating "
        "linearly to Teff = %.0f K",
        color_name,
        value,
        colors[0],
        colors[-1],
        teff,
    )
    return teff
