import numpy as np
from astropy.constants import c


def compute_doppler_shift(v):
    # TODO: sanitize input to ensure unit consistency
    beta = v / c
    z = ((1 - beta) / (1 + beta))**0.5
    return z


def air_to_vac(wave_air):
    """
    Convert air wavelengths [Å] to vacuum via the Edlén 1953 formula
    (two iterations, only wavelengths > 2000 Å are modified).
    """
    wave_vac = np.asarray(wave_air, dtype=np.float64).copy()
    modify = wave_vac > 2000.0
    if np.any(modify):
        wave_new = wave_vac[modify]
        for _ in range(2):
            sigma2 = (1e4 / wave_vac[modify]) ** 2
            fact = (1.0 + 5.792105e-2 / (238.0185 - sigma2)
                    + 1.67917e-3 / (57.362 - sigma2))
            wave_vac[modify] = wave_new * fact
    return wave_vac
