from astropy.constants import c


def compute_doppler_shift(v):
    # TODO: sanitize input to ensure unit consistency
    beta = v / c
    z = ((1 - beta) / (1 + beta))**0.5
    return z
