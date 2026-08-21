"""Synthetic spectra and line masks for the CCF chain (not a test module).

Shared by the two modules that consume them, ``cross_correlation.py`` and
``radial_velocity.py``. Builders only -- nothing here asserts.

The narrow velocity grid is the reason these constants are shared rather than
per-file: both integration suites need a grid small enough to be fast, wide
enough for RadialVelocity's second-pass +/-3 sigma window (sigma ~ 4 km/s) to
stay on-grid, and an injected RV that lands exactly on a grid point.
"""

import numpy as np
from astropy.constants import c

SPEED_OF_LIGHT_KMS = np.float64(c.to("km/s").value)

RANGE_KMS = [-15.0, 15.0]
STEP_KMS = 0.25  # matches the module default
NVEL = round((RANGE_KMS[1] - RANGE_KMS[0]) / STEP_KMS) + 1
V_INJECT = 1.5  # injected RV [km/s], on the grid
MASK_CENTERS = np.linspace(5015.0, 5035.0, 30)  # vacuum line centers [Angstrom]
# Wide enough that the default CCF clip (clip_edge_pixels=(500, 500)) trims the
# order edges but leaves the 5015-5035 A mask lines well inside.
NCOL = 2000


def make_mask(centers, weights=None, width=1.0):
    """Build a line-mask dict matching CrossCorrelation._build_line_mask."""
    centers = np.asarray(centers, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(centers)
    half_width = centers * (width / 2.0 / SPEED_OF_LIGHT_KMS)
    return {
        "center": centers,
        "weight": np.asarray(weights, dtype=np.float64),
        "start": centers - half_width,
        "end": centers + half_width,
    }


def absorption_spectrum(wave, centers, weights=None, depth=0.6, sigma_kms=4.0):
    """Unit continuum with Gaussian absorption lines at `centers`."""
    if weights is None:
        weights = np.ones_like(centers)
    flux = np.ones_like(wave)
    for center, weight in zip(centers, weights, strict=False):
        sigma_a = center * sigma_kms / SPEED_OF_LIGHT_KMS
        flux -= (
            depth
            * (weight / np.max(weights))
            * np.exp(-0.5 * ((wave - center) / sigma_a) ** 2)
        )
    return flux
