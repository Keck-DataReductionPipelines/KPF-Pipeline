"""
polly

misc

Contains miscellaneous functions that are used in various places in the package. For
now, it only contains the `savitzky_golay` function, which is a Python implementation
of the Savitzky-Golay filter.
"""

from math import factorial

import numpy as np
from numpy.typing import ArrayLike


def savitzky_golay(
    y: ArrayLike,
    window_size: int,
    order: int,
    deriv: int = 0,
    rate: float = 1,
) -> ArrayLike:
    # FROM: https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    """
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape andf features of the signal
    better than other types of filtering approaches, such as moving averages techniques.

    Args:
        y (array_like, shape (N,)): The y values of the time history of the signal.
        window_size (int): the length of the window. Must be an odd integer number.
        order (int): the order of the polynomial used in the filtering. Must be less
            than (`window_size` - 1).
        deriv (int): the order of the derivative to compute. Default: 0, which
            corresponds to only smoothing

    Returns:
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).

    Notes:
    The Savitzky-Golay is a type of low-pass filter, particularly suited for smoothing
    noisy data. The main idea behind this approach is to make for each point a least
    squares fit with a polynomial of high order over a odd-sized window centered at the
    point.


    [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
    [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = abs(int(window_size))
        order = abs(int(order))
    except ValueError as e:
        raise ValueError("window_size and order have to be of type int") from e
    if window_size % 2 != 1:
        print("window_size size must be a positive odd number. Adding 1")
        window_size += 1
    if window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.asmatrix(
        [[k**i for i in order_range] for k in range(-half_window, half_window + 1)]
    )
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")
