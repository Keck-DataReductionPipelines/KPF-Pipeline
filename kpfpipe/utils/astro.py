"""Air/vacuum wavelength conversion and relativistic Doppler/redshift helpers."""

import astropy.units as u
import numpy as np
from astropy.constants import c


def compute_doppler_factor(v):
    """
    Relativistic Doppler factor ``f = lambda_obs / lambda_rest`` for a source
    with radial velocity `v`.

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
    """
    Relativistic redshift ``z = lambda_obs / lambda_rest - 1`` for a source
    with radial velocity `v`.

    Standard astronomical convention: a receding source (``v > 0``) gives
    ``z > 0``, so `z` carries the same sign as `v`. Related to the Doppler
    factor by ``z = f - 1``, which is how it is computed here.

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
    """
    Convert air wavelengths to vacuum via the Edlén 1953 formula.

    Two iterations are used, and only wavelengths > 2000 Å are modified.

    Not yet wired into a pipeline module: retained for the forthcoming
    vacuum-wavelength work (the EPRV standard mandates vacuum wavelengths), and
    already used as a test oracle for wavelength-calibration output.

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
