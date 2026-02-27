from astropy.stats import mad_std
from astropy.stats import mad_std
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from scipy.optimize import least_squares


def gaussian_dist(theta, x):
    b, a, mu, sigma = theta
    return b + a * np.exp(-(x-mu)**2/(2*sigma**2))


def gaussian_jac(theta, x):
    b, a, mu, sigma = theta
    dx = x - mu
    e = np.exp(-dx**2 / (2*sigma**2))

    J = np.empty((x.size, 4))
    J[:, 0] = a * e * dx / sigma**2
    J[:, 1] = a * e * dx**2 / sigma**3
    J[:, 2] = e
    J[:, 3] = 1.0
    
    return J


def gaussian_theta0(x, y):
    b0 = 0.25 * np.sum(y[:2] + y[-2:])
    a0 = np.max(y) - b0
    mu0 = x[np.argmax(y)]
    sigma0 = np.std(x)
    
    return [b0, a0, mu0, sigma0]


def _res_wrapper(theta, x, y, func):
    """
    Helper function for optimize_lsq
    """
    return func(theta, x) - y


def _jac_wrapper(theta, x, y, jac):
    """
    Helper function for optimize_lsq
    """
    return jac(theta, x)


def optimize_lsq(theta0, x, y, func, jac):
    """
    Wrapper function for scipy.optimize.least_squares
    """
    result = least_squares(_res_wrapper, 
                           theta0, 
                           args = (x, y, func),
                           jac = _jac_wrapper,
                           method = 'lm', 
                           )
    
    theta, rms = result.x, np.std(result.fun)
    
    return theta, rms


def flag_outliers(x, sigma, method='median', kernel_size=None):
    """
    Flag outliers in an array above some sigma threshold
    """
    if method == 'median':
        med = np.nanmedian(x)
        mad = mad_std(x, ignore_nan=True)
        out = np.abs(x - med) / mad > sigma

    elif method == 'trend':
        trend = gaussian_filter(median_filter(x, size=kernel_size), sigma=kernel_size)
        out = np.abs(x - trend) / mad_std(x - trend, ignore_nan=True) > sigma

    else:
        raise ValueError(f"method must be 'median' or 'trend'; {method} not supported")

    return out