"""Statistical helpers, in the style of ``scipy`` & ``numpy``."""

import numpy as np
from astropy.stats import mad_std
from numpy.polynomial import legendre, polynomial
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import (
    convolve,
    distance_transform_edt,
    gaussian_filter,
    median_filter,
)
from scipy.optimize import leastsq


def strictly_increasing(x):
    """Return True if the 1D array is strictly increasing."""
    x = np.asarray(x)
    return bool(np.all(x[:-1] < x[1:]))


def _gaussian_dist(theta, x):
    """Gaussian model at ``x`` for theta = [b, a, mu, sigma]."""
    b, a, mu, sigma = theta
    return b + a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def _gaussian_jac(theta, x):
    """Analytic Jacobian of ``_gaussian_dist`` w.r.t. theta; shape (x.size, 4)."""
    b, a, mu, sigma = theta
    dx = x - mu
    exp_term = np.exp(-(dx**2) / (2 * sigma**2))

    J = np.empty((x.size, 4))
    J[:, 0] = 1.0
    J[:, 1] = exp_term
    J[:, 2] = a * exp_term * dx / sigma**2
    J[:, 3] = a * exp_term * dx**2 / sigma**3

    return J


def _gaussian_theta0_generator(x, y):
    """Initial-guess theta = [b, a, mu, sigma] for a Gaussian fit to (x, y)."""
    b0 = 0.25 * np.sum(y[:2] + y[-2:])
    a0 = np.max(y) - b0
    mu0 = x[np.argmax(y)]

    # Seed sigma from the line's FWHM: the span of x clearing the half-maximum
    # level (b0 + a0/2), floored at one grid step so a barely-resolved line
    # still gets a positive width. FWHM = 2*sqrt(2*ln2) * sigma.
    above = x[y >= b0 + 0.5 * a0]
    fwhm = max(above.max() - above.min(), (x.max() - x.min()) / (x.size - 1))
    sigma0 = fwhm / 2.3548200450309493

    return [b0, a0, mu0, sigma0]


def _gaussian_untransform(theta):
    """Report fitted [b, a, mu, sigma], forcing sigma >= 0.

    The model depends only on ``sigma**2``, so the fit may converge to a negative
    sigma; ``abs`` picks the physical (positive) width.
    """
    b, a, mu, sigma = theta
    return np.array([b, a, mu, np.abs(sigma)])


# Each entry: (model, jacobian, theta0 initializer, untransform). untransform
# maps the fitted parameters back to reported ones (identity if not needed).
_FUNCTIONS = {
    "gaussian": (
        _gaussian_dist,
        _gaussian_jac,
        _gaussian_theta0_generator,
        _gaussian_untransform,
    ),
}


def optimize_lsq(x, y, linemodel):
    """Fit a 1D line model to (x, y) by non-linear least squares.

    Looks up the model function, analytic Jacobian, and initial-guess
    generator registered under ``linemodel``, then fits via MINPACK's lmder
    (``scipy.optimize.leastsq``, not ``least_squares`` -- see the implementation
    note below), and maps the solution back with the model's untransform.

    Parameters
    ----------
    x : ndarray
        1D independent variable (e.g. velocity or pixel grid).
    y : ndarray
        1D dependent variable to fit, same shape as ``x``.
    linemodel : str
        Registered line-profile name. Currently only 'gaussian' is supported;
        an unknown name raises.

    Returns
    -------
    theta : ndarray
        Fitted parameters in the model's reported convention. For 'gaussian':
        ``[b, a, mu, sigma]`` (baseline, amplitude, center, positive width).
    rms : float
        RMS of the fit residuals (``std`` of ``model - y`` at the solution).

    Raises
    ------
    ValueError
        If ``linemodel`` is not a registered line-profile name.
    """
    try:
        func, jac, theta0_generator, untransform = _FUNCTIONS[linemodel]
    except KeyError:
        raise ValueError(f"Unsupported line function: {linemodel}") from None

    def residual(theta):
        return func(theta, x) - y

    def jacobian(theta):
        return jac(theta, x)

    theta0 = theta0_generator(x, y)

    # Call MINPACK's lmder directly via ``leastsq`` instead of ``least_squares``.
    # Both bottom out in the same Fortran routine, but for ``method="lm"`` with no
    # bounds the ``least_squares`` wrapper adds per-call overhead (input checks, an
    # extra residual evaluation, building OptimizeResult) that dominates for the
    # many tiny line fits here. The tolerances and ``maxfev`` below are exactly what
    # ``least_squares`` forwards to lmder, so the fit is bit-for-bit identical.
    sol, _cov, info, _msg, _ier = leastsq(
        residual,
        theta0,
        Dfun=jacobian,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        maxfev=100 * len(theta0),
        full_output=True,
    )

    theta, rms = untransform(sol), np.std(info["fvec"])

    return theta, rms


def robust_polyfit(x, y, deg, sigma=4.0, maxiter=8, min_valid_fraction=0.5, full=False):
    """Least-squares polynomial fit that rejects outliers by median/MAD.

    A drop-in replacement for ``numpy.polynomial.polynomial.polyfit`` that
    refits, dropping samples whose residual lies more than ``sigma`` robust
    deviations from the median residual, until the accepted set stops changing.
    Non-finite samples are never accepted.

    Each fit is solved in the Legendre basis over the sample span, whose
    design matrix stays well conditioned where the power basis' Vandermonde
    matrix does not, and the result is converted back to power-basis
    coefficients.

    Parameters
    ----------
    x : array_like
        Sample positions.
    y : array_like
        Sample values, same shape as ``x``.
    deg : int
        Degree of the fitted polynomial.
    sigma : float, default 4.0
        Rejection threshold, in robust deviations (``astropy.stats.mad_std``) of
        the accepted residuals. The threshold is floored at 0.25 in the units of
        ``y``, so a fit whose scatter is near zero does not reject samples that
        are, in absolute terms, on the curve.
    maxiter : int, default 8
        Maximum number of rejection iterations.
    min_valid_fraction : float, default 0.5
        Fraction of the samples that must stay accepted. A rejection that would
        fall below it is not taken; failing it at the outset raises.
    full : bool, default False
        If True, also return the accepted-sample mask and the residual RMS.

    Returns
    -------
    coeffs : ndarray
        Fitted coefficients, lowest order first, as ``polyfit`` returns them.
    good : ndarray
        Boolean mask of the samples the final fit accepted. Returned only if
        ``full`` is True.
    rms : float
        RMS of the accepted samples' residuals. Returned only if ``full`` is
        True.

    Raises
    ------
    ValueError
        If fewer samples are finite than the fit requires.
    """
    x, y = np.asarray(x), np.asarray(y)
    good = np.isfinite(x) & np.isfinite(y)
    min_good = max(deg + 1, int(np.ceil(x.size * min_valid_fraction)))
    if good.sum() < min_good:
        raise ValueError(
            f"only {good.sum()} of {x.size} samples are valid; "
            f"at least {min_good} are required"
        )

    # Hold the domain fixed across iterations so rejecting a sample near an end
    # cannot rescale the basis and perturb the coefficients on its own.
    domain = [x[good].min(), x[good].max()]

    for _ in range(maxiter):
        series = legendre.Legendre.fit(x[good], y[good], deg, domain=domain)
        residual = y - series(x)
        median_residual = np.nanmedian(residual[good])
        residual_scatter = mad_std(residual[good], ignore_nan=True)
        rejection_limit = max(0.25, sigma * residual_scatter)
        still_valid = np.isfinite(y) & (
            np.abs(residual - median_residual) <= rejection_limit
        )
        if still_valid.sum() < min_good:
            break
        if np.array_equal(still_valid, good):
            good = still_valid
            break
        good = still_valid

    series = legendre.Legendre.fit(x[good], y[good], deg, domain=domain)
    # The conversion trims trailing zero coefficients; ``polyfit`` always
    # returns deg + 1 of them, and callers index by order.
    coeffs = np.zeros(deg + 1)
    converted = series.convert(kind=polynomial.Polynomial).coef
    coeffs[: converted.size] = converted
    if not full:
        return coeffs
    residual = y[good] - series(x[good])
    return coeffs, good, float(np.sqrt(np.mean(residual**2)))


def _smooth_filter(x, size=None, *, axes=None):
    """Median- then Gaussian-smooth ``x`` (chains scipy's ``median_filter`` and
    ``gaussian_filter``).

    ``size`` sets both the median window and the Gaussian sigma; ``axes`` restricts
    smoothing to those axes (all when None), so the trend follows structure along
    them without blurring across the others.
    """
    return gaussian_filter(
        median_filter(x, size=size, axes=axes), sigma=size, axes=axes
    )


def flag_outliers(x, sigma, axis=None, kernel_size=None, method="median"):
    """Flag elements of ``x`` more than ``sigma`` robust deviations from their peers.

    ``axis`` selects the axis along which each element is compared to its peers:

    - ``method="median"`` computes the median and MAD *reducing over* ``axis``,
      so each element is judged against the others sharing its remaining
      indices (e.g. ``axis=0`` on a (frame, row, col) cube flags, per pixel,
      the frames that deviate across the stack).
    - ``method="trend"`` smooths ``x`` *along* ``axis`` (see ``_smooth_filter``) and
      judges each element against that local trend, so it tolerates structure
      that varies smoothly along ``axis`` (e.g. illumination along dispersion). The
      deviation is scaled by a local (rolling) MAD rather than one global MAD, so
      it tolerates heteroscedasticity (scatter that grows with the local level).

    ``axis=None`` compares every element to a single global statistic.
    """
    eps = 1e-12

    if method == "median":
        med = np.nanmedian(x, axis=axis, keepdims=True)
        dev = np.abs(x - med)
        mad = 1.482602218505602 * np.nanmedian(dev, axis=axis, keepdims=True)
        # Reuse one abs-dev compute to avoid extra temporary arrays.
        dev /= mad + eps
        out = dev > sigma

    elif method == "trend":
        trend = _smooth_filter(x, size=kernel_size, axes=axis)
        dev = np.abs(x - trend)
        mad = 1.482602218505602 * _smooth_filter(dev, size=kernel_size, axes=axis)
        # Reuse one abs-dev compute to avoid extra temporary arrays.
        dev /= mad + eps
        out = dev > sigma

    else:
        raise ValueError(f"method must be 'median' or 'trend'; {method} not supported")

    return out


def interpolate_bad_pixels(data, mask, method="local", fill_outside=True):
    """Interpolate over bad pixels of a 2D image, replacing each with a value
    inferred from its good neighbors.

    Parameters
    ----------
    data : ndarray
        2D image; bad pixels are filled, good pixels pass through unchanged.
        Not modified in place (a copy is returned).
    mask : ndarray
        Good-pixel mask broadcastable to ``data``, truthy where a pixel is good.
        Its logical complement marks the pixels to interpolate.
    method : {'local', 'global'}, default 'local'
        - ``'local'``: fill each bad pixel from a 3x3 weighted mean of its good
          neighbors (assumes isolated bad pixels).
        - ``'global'``: bilinearly interpolate every bad pixel from the full
          good grid (robust to clumps of adjacent bad pixels).
    fill_outside : bool, default True
        Fill any bad pixels the chosen method leaves unset -- ``'local'``
        pixels with no good neighbor in their 3x3 window, ``'global'`` pixels
        outside the good-data convex hull -- with the value of the nearest
        filled pixel (a Euclidean distance-transform lookup). If False, such
        pixels are left untouched: they keep their original value under
        ``'local'`` and become NaN under ``'global'``.

    Returns
    -------
    ndarray
        Copy of ``data`` with bad pixels interpolated.

    Raises
    ------
    ValueError
        If ``method`` is not 'local' or 'global'.
    """

    good = mask.astype(bool)
    bad = ~good

    data_interp = data.copy()

    # The local convolution-based method assumes isolated bad pixels. Cast the
    # kernel and weight mask to the input dtype so ``scipy.convolve`` does not
    # promote float32 image data to float64 in its intermediates.
    if method == "local":
        kernel = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]], dtype=data.dtype) / 12.0

        neighbor_sum = convolve(data * good, kernel, mode="mirror")
        weight = convolve(good.astype(data.dtype), kernel, mode="mirror")

        valid = bad & (weight > 0)
        data_interp[valid] = neighbor_sum[valid] / weight[valid]

        if fill_outside:
            remaining = bad & (weight == 0)
            if np.any(remaining):
                indices = distance_transform_edt(
                    remaining, return_distances=False, return_indices=True
                )
                data_interp[remaining] = data_interp[tuple(indices[:, remaining])]

    # Global linear interpolation is robust to clumps of bad pixels.
    elif method == "global":
        ny, nx = data.shape
        y = np.arange(ny)
        x = np.arange(nx)

        interp = RegularGridInterpolator(
            (y, x), data, method="linear", bounds_error=False, fill_value=np.nan
        )

        coords = np.column_stack(np.nonzero(bad))
        values = interp(coords)

        data_interp[bad] = values

        if fill_outside:
            nan_mask = np.isnan(data_interp)
            if np.any(nan_mask):
                indices = distance_transform_edt(
                    nan_mask, return_distances=False, return_indices=True
                )
                data_interp[nan_mask] = data_interp[tuple(indices[:, nan_mask])]

    else:
        raise ValueError("method must be 'local' or 'global'")

    return data_interp
