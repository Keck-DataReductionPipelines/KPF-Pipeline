"""
Tests for kpfpipe.utils.stats helpers.
"""

import numpy as np
import pytest

from kpfpipe.utils.stats import interpolate_bad_pixels


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
