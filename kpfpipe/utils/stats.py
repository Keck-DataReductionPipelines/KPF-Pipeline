import numpy as np
from scipy.optimize import least_squares


def gaussian(theta, x):
    mu, sigma, a, b = theta
    return b + a * np.exp(-(x-mu)**2/(2*sigma**2))


def gaussian_jac(theta, x):
    mu, sigma, a, b = theta
    dx = x - mu
    e = np.exp(-dx**2 / (2*sigma**2))

    J = np.empty((x.size, 4))
    J[:, 0] = a * e * dx / sigma**2
    J[:, 1] = a * e * dx**2 / sigma**3
    J[:, 2] = e
    J[:, 3] = 1.0
    
    return J


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
                           args = (x, y, func)
                           jac = _jac_wrapper,
                           method = 'lm', 
                           )
    
    theta, rms = result.x, np.std(result.fun)
    
    return theta, rms