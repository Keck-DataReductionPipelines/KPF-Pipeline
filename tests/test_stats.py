"""
Tests for kpfpipe.utils.stats helpers.
"""

import numpy as np
import pytest

from kpfpipe.utils.stats import (
    interpolate_bad_pixels,
    optimize_lsq,
    gaussian_dist,
    gaussian_jac,
)


class TestGaussianFit:
    """The Gaussian width is fit as log(sigma); optimize_lsq untransforms it
    back to sigma, which must therefore always be positive."""

    def test_recovers_known_gaussian(self):
        x = np.arange(-10, 11, dtype=float)
        for sigma in (2.7, 1.1):
            theta_true = [2.0, 50.0, 1.3, sigma]
            y = gaussian_dist([2.0, 50.0, 1.3, np.log(sigma)], x)
            theta, _ = optimize_lsq(x, y, "gaussian")
            np.testing.assert_allclose(theta, theta_true, rtol=1e-5, atol=1e-5)
            assert theta[3] > 0

    def test_sigma_is_positive(self):
        # Sigma enters only as sigma**2, so the fit must never return a negative width.
        x = np.arange(-8, 9, dtype=float)
        y = gaussian_dist([1.0, 25.0, -0.6, np.log(2.0)], x)
        theta, _ = optimize_lsq(x, y, "gaussian")
        assert theta[3] > 0

    def test_jacobian_matches_finite_difference(self):
        # Guards the d/d(log_sigma) chain-rule term in gaussian_jac.
        x = np.linspace(-5, 5, 21)
        theta = np.array([1.0, 4.0, 0.5, np.log(1.8)])  # [b, a, mu, log_sigma]
        J = gaussian_jac(theta, x)
        eps = 1e-6
        for k in range(4):
            tp, tm = theta.copy(), theta.copy()
            tp[k] += eps
            tm[k] -= eps
            fd = (gaussian_dist(tp, x) - gaussian_dist(tm, x)) / (2 * eps)
            np.testing.assert_allclose(J[:, k], fd, rtol=1e-4, atol=1e-6)


class TestInterpolateBadPixels:
    def test_preserves_float32_dtype(self):
        data = np.ones((8, 8), dtype=np.float32)
        mask = np.ones((8, 8), dtype=bool)
        mask[3, 3] = False
        data[3, 3] = 1e6  # bad pixel
        out = interpolate_bad_pixels(data, mask)
        assert out.dtype == np.float32

    def test_preserves_float64_dtype(self):
        data = np.ones((8, 8), dtype=np.float64)
        mask = np.ones((8, 8), dtype=bool)
        mask[3, 3] = False
        data[3, 3] = 1e6
        out = interpolate_bad_pixels(data, mask)
        assert out.dtype == np.float64

    def test_replaces_bad_pixel_with_neighbor_mean(self):
        data = np.ones((5, 5), dtype=np.float32) * 2.0
        mask = np.ones((5, 5), dtype=bool)
        mask[2, 2] = False
        data[2, 2] = 1e6
        out = interpolate_bad_pixels(data, mask)
        # 8 neighbors all = 2.0 → interpolated value should be ~2.0
        assert np.isclose(out[2, 2], 2.0, atol=1e-5)

    def test_good_pixels_unchanged(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0.0, 1.0, (10, 10)).astype(np.float32)
        mask = np.ones((10, 10), dtype=bool)
        mask[5, 5] = False
        original = data.copy()
        out = interpolate_bad_pixels(data, mask)
        good_pixels = mask
        np.testing.assert_array_equal(out[good_pixels], original[good_pixels])
