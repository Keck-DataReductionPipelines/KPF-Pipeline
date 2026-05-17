from astropy.stats import mad_std
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import median_filter, gaussian_filter, convolve, distance_transform_edt
from scipy.optimize import least_squares


def gaussian_dist(theta, x):
    b, a, mu, sigma = theta
    return b + a * np.exp(-(x-mu)**2/(2*sigma**2))


def gaussian_jac(theta, x):
    b, a, mu, sigma = theta
    dx = x - mu
    e = np.exp(-dx**2 / (2*sigma**2))

    J = np.empty((x.size, 4))
    J[:, 0] = 1.0
    J[:, 1] = e
    J[:, 2] = a * e * dx / sigma**2
    J[:, 3] = a * e * dx**2 / sigma**3

    return J


def gaussian_theta0(x, y):
    b0 = 0.25 * np.sum(y[:2] + y[-2:])
    a0 = np.max(y) - b0
    mu0 = x[np.argmax(y)]
    sigma0 = np.std(x)
    
    return [b0, a0, mu0, sigma0]


_FUNCTIONS = {
    'gaussian': (gaussian_dist, gaussian_jac, gaussian_theta0),
}


def optimize_lsq(x, y, linemodel):
    """
    Fit a 1D line model to (x, y) by non-linear least squares.

    Looks up the model function, Jacobian, and theta0 initializer for the
    given lineprofile name and dispatches scipy.optimize.least_squares.
    """
    try:
        func, jac, theta0_func = _FUNCTIONS[linemodel]
    except KeyError:
        raise ValueError(f"Unsupported line function: {linemodel}")

    def residual(theta):
        return func(theta, x) - y

    def jacobian(theta):
        return jac(theta, x)

    theta0 = theta0_func(x, y)

    result = least_squares(residual,
                           theta0,
                           jac=jacobian,
                           method='lm',
                           )

    theta, rms = result.x, np.std(result.fun)

    return theta, rms


def flag_outliers(x, sigma, method='median', axis=None, kernel_size=None):
    """
    Flag outliers in an array above some sigma threshold
    """
    eps = 1e-12

    if method == 'median':
        med = np.nanmedian(x)
        mad = mad_std(x, ignore_nan=True)
        out = np.abs(x - med) / (mad + eps) > sigma

    elif method == 'trend':
        trend = gaussian_filter(median_filter(x, size=kernel_size), sigma=kernel_size)
        mad = mad_std(x - trend, ignore_nan=True)
        out = np.abs(x - trend) / (mad + eps) > sigma

    else:
        raise ValueError(f"method must be 'median' or 'trend'; {method} not supported")

    return out


def interpolate_bad_pixels(data, mask, method='local', fill_outside=True):
    """
    Interpolate over bad pixels.
    """

    good = mask.astype(bool)
    bad = ~good

    data_interp = data.copy()

    # local convolution-based method (assumes isolated bad pixels).
    # Cast kernel and weight mask to the input dtype so scipy.convolve does
    # not promote float32 image data to float64 in its intermediates.
    if method == 'local':
        kernel = np.array([[1,2,1],
                           [2,0,2],
                           [1,2,1]], dtype=data.dtype) / 12.0

        neighbor_sum = convolve(data * good, kernel, mode='mirror')
        weight = convolve(good.astype(data.dtype), kernel, mode='mirror')

        valid = bad & (weight > 0)
        data_interp[valid] = neighbor_sum[valid] / weight[valid]

        if fill_outside:
            remaining = bad & (weight == 0)
            if np.any(remaining):
                _, indices = distance_transform_edt(
                    remaining,
                    return_indices=True
                )
                data_interp[remaining] = data_interp[
                    tuple(indices[:, remaining])
                ]

    # global linear interpolation (robust to clumps of bad pixels)
    elif method == 'global':

        ny, nx = data.shape
        y = np.arange(ny)
        x = np.arange(nx)

        interp = RegularGridInterpolator(
            (y, x),
            data,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        coords = np.column_stack(np.nonzero(bad))
        values = interp(coords)

        data_interp[bad] = values

        if fill_outside:
            nan_mask = np.isnan(data_interp)
            if np.any(nan_mask):
                _, indices = distance_transform_edt(
                    nan_mask,
                    return_indices=True
                )
                data_interp[nan_mask] = data_interp[
                    tuple(indices[:, nan_mask])
                ]

    else:
        raise ValueError("method must be 'local' or 'global'")

    return data_interp