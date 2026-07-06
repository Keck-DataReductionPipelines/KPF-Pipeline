"""Tests for kpfpipe.utils.stats: fits, bad-pixel interp, outliers, monotonicity."""

import numpy as np
import pytest

from kpfpipe.utils.stats import (
    _gaussian_dist,
    _gaussian_jac,
    flag_outliers,
    interpolate_bad_pixels,
    optimize_lsq,
    strictly_increasing,
)


class TestGaussianFit:
    """The Gaussian width is fit directly as sigma; optimize_lsq reports
    abs(sigma), which must therefore always be positive (the model depends
    only on sigma**2)."""

    def test_recovers_known_gaussian(self):
        x = np.arange(-10, 11, dtype=float)
        for sigma in (2.7, 1.1):
            theta_true = [2.0, 50.0, 1.3, sigma]
            y = _gaussian_dist(theta_true, x)
            theta, _ = optimize_lsq(x, y, "gaussian")
            np.testing.assert_allclose(theta, theta_true, rtol=1e-5, atol=1e-5)
            assert theta[3] > 0

    def test_sigma_is_positive(self):
        # Sigma enters only as sigma**2, so the fit must never return a negative width.
        x = np.arange(-8, 9, dtype=float)
        y = _gaussian_dist([1.0, 25.0, -0.6, 2.0], x)
        theta, _ = optimize_lsq(x, y, "gaussian")
        assert theta[3] > 0

    def test_jacobian_matches_finite_difference(self):
        # Guards the d/d(sigma) term (a*E*dx**2/sigma**3) in _gaussian_jac.
        x = np.linspace(-5, 5, 21)
        theta = np.array([1.0, 4.0, 0.5, 1.8])  # [b, a, mu, sigma]
        J = _gaussian_jac(theta, x)
        eps = 1e-6
        for k in range(4):
            tp, tm = theta.copy(), theta.copy()
            tp[k] += eps
            tm[k] -= eps
            fd = (_gaussian_dist(tp, x) - _gaussian_dist(tm, x)) / (2 * eps)
            np.testing.assert_allclose(J[:, k], fd, rtol=1e-4, atol=1e-6)


class TestInterpolateBadPixels:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_preserves_dtype(self, dtype):
        data = np.ones((8, 8), dtype=dtype)
        mask = np.ones((8, 8), dtype=bool)
        mask[3, 3] = False
        data[3, 3] = 1e6  # bad pixel
        out = interpolate_bad_pixels(data, mask)
        assert out.dtype == dtype

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

    def test_global_method_fills_bad_pixel_clump(self):
        # Global linear interpolation handles clumps; NaN-flagged pixels are filled.
        data = np.full((6, 6), 5.0, dtype=np.float64)
        mask = np.ones((6, 6), dtype=bool)
        mask[2:4, 2:4] = False  # 2x2 clump
        data[2:4, 2:4] = np.nan
        out = interpolate_bad_pixels(data, mask, method="global")
        assert np.all(np.isfinite(out))
        assert out[0, 0] == 5.0  # good pixel untouched

    def test_unsupported_method_raises(self):
        data = np.ones((4, 4), dtype=np.float32)
        mask = np.ones((4, 4), dtype=bool)
        with pytest.raises(ValueError, match="method must be 'local' or 'global'"):
            interpolate_bad_pixels(data, mask, method="bogus")


class TestOptimizeLsqErrors:
    def test_unsupported_line_model_raises(self):
        x = np.arange(5.0)
        y = np.zeros(5)
        with pytest.raises(ValueError, match="Unsupported line function"):
            optimize_lsq(x, y, "not_a_model")


class TestFlagOutliers:
    def test_median_method_flags_spike(self):
        x = np.full(50, 10.0)
        x[25] = 1000.0
        out = flag_outliers(x, sigma=5.0, method="median")
        assert out[25] and not out[0]

    def test_trend_method_flags_spike(self):
        # The trend method detrends with a median+gaussian filter before flagging.
        x = np.full(50, 10.0)
        x[25] = 1000.0
        out = flag_outliers(x, sigma=5.0, kernel_size=5, method="trend")
        assert out[25]

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError, match="method must be 'median' or 'trend'"):
            flag_outliers(np.arange(10.0), sigma=5.0, method="bogus")


class TestStrictlyIncreasing:
    def test_true_for_increasing(self):
        assert strictly_increasing([1.0, 2.0, 3.0]) is True

    def test_false_for_non_increasing(self):
        assert strictly_increasing([1.0, 1.0, 2.0]) is False
        assert strictly_increasing([3.0, 2.0, 1.0]) is False
