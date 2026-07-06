"""
Tests for kpfpipe.utils helpers: astro (astro.py), KPF timestamp/obs_id
conversions (kpf_utils.py), statistics (stats.py), and config loading (config.py).
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c

from kpfpipe.utils.astro import air_to_vac, compute_doppler_factor, compute_redshift
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.kpf_utils import (
    get_datecode,
    get_obs_id,
    get_timestamp,
    hst_to_utc,
    is_datecode,
    is_obs_id,
    is_timestamp,
    kpf_timestamp_to_datetime,
    kpf_timestamp_to_eprv_timestamp,
    utc_to_hst,
)
from kpfpipe.utils.stats import (
    _gaussian_dist,
    _gaussian_jac,
    flag_outliers,
    interpolate_bad_pixels,
    optimize_lsq,
    strictly_increasing,
)

C_KMS = c.to("km/s").value


# ===========================================================================
# astro.py — Doppler/redshift helpers and air->vacuum
#
# Convention under test: positive radial velocity = receding = redshift (z > 0),
# so the Doppler factor f = lambda_obs / lambda_rest = 1 + z, and z carries the
# same sign as the velocity. This is the convention BarycentricCorrection relies
# on when storing BARYCORR_Z for RadialVelocity._compute_ccf_1d.
# ===========================================================================


class TestComputeRedshift:
    def test_sign_matches_velocity(self):
        # Receding (v > 0) -> positive redshift; approaching -> negative.
        assert compute_redshift(+18.508 * u.km / u.s) > 0
        assert compute_redshift(-18.508 * u.km / u.s) < 0

    def test_nonrelativistic_magnitude(self):
        # For v << c, z ~ v/c.
        v = -18.508 * u.km / u.s
        assert compute_redshift(v) == pytest.approx(v.value / C_KMS, rel=1e-4)

    def test_factor_is_one_plus_z(self):
        v = 30.0 * u.km / u.s
        assert compute_doppler_factor(v) == pytest.approx(1.0 + compute_redshift(v))

    def test_unit_agnostic(self):
        # Same physical velocity in different units -> same result.
        assert compute_redshift(-18.508 * u.km / u.s) == pytest.approx(
            compute_redshift(-18508.0 * u.m / u.s)
        )

    def test_zero_velocity(self):
        assert compute_redshift(0.0 * u.km / u.s) == pytest.approx(0.0)
        assert compute_doppler_factor(0.0 * u.km / u.s) == pytest.approx(1.0)

    def test_array_input(self):
        v = np.array([-10.0, 0.0, 10.0]) * u.km / u.s
        z = compute_redshift(v)
        assert z.shape == (3,)
        assert z[0] < 0 < z[2] and z[1] == pytest.approx(0.0)

    def test_bare_value_raises(self):
        # Units must stay explicit; a unitless argument fails loudly.
        with pytest.raises(u.UnitsError):
            compute_redshift(-18508.0)

    def test_factor_direction(self):
        # Receding source is redshifted (f > 1); approaching is blueshifted.
        assert compute_doppler_factor(+18.508 * u.km / u.s) > 1.0
        assert compute_doppler_factor(-18.508 * u.km / u.s) < 1.0


class TestAirToVac:
    def test_vacuum_longer_than_air(self):
        wave_air = np.array([5000.0, 6000.0, 7000.0])
        wave_vac = air_to_vac(wave_air)
        assert np.all(wave_vac > wave_air)

    def test_below_2000A_unchanged(self):
        wave_air = np.array([1500.0, 1800.0])
        np.testing.assert_array_equal(air_to_vac(wave_air), wave_air)


# ===========================================================================
# kpf_utils.py — timestamp conversion utilities
# ===========================================================================


class TestUtcToHst:
    def test_midday_no_rollover(self):
        # 12:00 UTC = 02:00 HST same day (43200 - 36000 = 7200)
        assert utc_to_hst("20240405.43200.00") == "20240405.07200.00"

    def test_rollover_to_previous_day(self):
        # 03:00 UTC = 17:00 HST previous day (10800 - 36000 < 0)
        assert utc_to_hst("20240405.10800.00") == "20240404.61200.00"

    def test_frame_str_preserved(self):
        assert utc_to_hst("20240405.43200.71").endswith(".71")

    def test_exact_offset_boundary(self):
        # 10:00 UTC = 00:00 HST (36000 - 36000 = 0)
        assert utc_to_hst("20240405.36000.00") == "20240405.00000.00"


class TestHstToUtc:
    def test_midday_no_rollover(self):
        # 02:00 HST = 12:00 UTC same day (7200 + 36000 = 43200)
        assert hst_to_utc("20240405.07200.00") == "20240405.43200.00"

    def test_rollover_to_next_day(self):
        # 17:00 HST = 03:00 UTC next day
        # (61200 + 36000 = 97200 -> 97200 - 86400 = 10800)
        assert hst_to_utc("20240404.61200.00") == "20240405.10800.00"

    def test_frame_str_preserved(self):
        assert hst_to_utc("20240405.07200.71").endswith(".71")

    def test_exact_offset_boundary(self):
        # 00:00 HST = 10:00 UTC (0 + 36000 = 36000)
        assert hst_to_utc("20240405.00000.00") == "20240405.36000.00"

    def test_roundtrip(self):
        ts = "20240405.40113.57"
        assert hst_to_utc(utc_to_hst(ts)) == ts


class TestKpfTimestampToEprv:
    def test_basic_conversion(self):
        # 40113s = 11:08:33
        assert kpf_timestamp_to_eprv_timestamp("20240405.40113.57") == "20240405T110833"

    def test_midnight(self):
        assert kpf_timestamp_to_eprv_timestamp("20240405.00000.00") == "20240405T000000"

    def test_end_of_day(self):
        # 86399s = 23:59:59
        assert kpf_timestamp_to_eprv_timestamp("20240405.86399.00") == "20240405T235959"

    def test_one_hour(self):
        assert kpf_timestamp_to_eprv_timestamp("20240405.03600.00") == "20240405T010000"

    def test_frame_field_dropped(self):
        # Frame field should not appear in output
        result = kpf_timestamp_to_eprv_timestamp("20240405.40113.57")
        assert "57" not in result
        assert "." not in result


class TestGetObsId:
    def test_extracts_from_bare_obs_id(self):
        assert get_obs_id("KP.20240405.40113.57") == "KP.20240405.40113.57"

    def test_extracts_from_path(self):
        assert (
            get_obs_id("/data/L0/20240405/KP.20240405.40113.57.fits")
            == "KP.20240405.40113.57"
        )

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="No obs_id found"):
            get_obs_id("not_an_obs_id.fits")


class TestGetDatecode:
    def test_extracts_from_obs_id(self):
        assert get_datecode("KP.20240405.40113.57") == "20240405"

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="Cannot extract datecode"):
            get_datecode("not_an_obs_id")


class TestGetTimestamp:
    def test_extracts_from_obs_id(self):
        assert get_timestamp("KP.20240405.40113.57") == "20240405.40113.57"

    def test_extracts_from_path(self):
        assert (
            get_timestamp("/data/L0/20240405/KP.20240405.40113.57.fits")
            == "20240405.40113.57"
        )

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="No KPF timestamp found"):
            get_timestamp("notimestamp.fits")


# ---------------------------------------------------------------------------
# Validation: predicates reject semantically invalid input
# ---------------------------------------------------------------------------


class TestIsObsId:
    def test_valid_obs_id(self):
        assert is_obs_id("KP.20240405.40113.57") is True

    def test_rejects_bad_date(self):
        # Format matches but month 99 is not a real date.
        assert is_obs_id("KP.20249999.40113.57") is False

    def test_rejects_seconds_out_of_range(self):
        # SSSSS = 99999 > 86399.
        assert is_obs_id("KP.20240405.99999.57") is False

    def test_rejects_wrong_format(self):
        assert is_obs_id("20240405.40113.57") is False

    def test_rejects_non_string(self):
        assert is_obs_id(None) is False
        assert is_obs_id(12345) is False


class TestIsDatecode:
    def test_valid_datecode(self):
        assert is_datecode("20240405") is True

    def test_rejects_bad_date(self):
        # Format matches but month 99 is not a real date.
        assert is_datecode("20249999") is False

    def test_rejects_wrong_format(self):
        assert is_datecode("2024-04-05") is False

    def test_rejects_non_string(self):
        assert is_datecode(None) is False
        assert is_datecode(20240405) is False


class TestIsTimestamp:
    def test_valid_timestamp(self):
        assert is_timestamp("20240405.40113.57") is True

    def test_rejects_bad_date(self):
        assert is_timestamp("20249999.40113.57") is False

    def test_rejects_seconds_out_of_range(self):
        assert is_timestamp("20240405.99999.57") is False

    def test_rejects_wrong_format(self):
        assert is_timestamp("20240405T110833") is False

    def test_rejects_non_string(self):
        assert is_timestamp(None) is False


# ---------------------------------------------------------------------------
# Validation: extractors raise on semantically invalid embedded timestamps
# ---------------------------------------------------------------------------


class TestGetObsIdValidation:
    def test_raises_on_invalid_date(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            get_obs_id("KP.20249999.40113.57.fits")

    def test_raises_on_seconds_out_of_range(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            get_obs_id("KP.20240405.99999.57.fits")

    def test_raises_on_non_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            get_obs_id(None)


class TestGetDatecodeValidation:
    def test_raises_on_invalid_date(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            get_datecode("KP.20249999.40113.57")

    def test_raises_on_seconds_out_of_range(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            get_datecode("KP.20240405.99999.57")

    def test_raises_on_non_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            get_datecode(None)


class TestGetTimestampValidation:
    def test_raises_on_invalid_date(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            get_timestamp("KP.20249999.40113.57.fits")

    def test_raises_on_seconds_out_of_range(self):
        # The path contains a syntactically-matching but invalid timestamp.
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            get_timestamp("KP.20240405.99999.57.fits")

    def test_raises_on_non_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            get_timestamp(None)


# ---------------------------------------------------------------------------
# Validation: converters reject malformed input rather than silently
# producing wrong output (the day-rollover and hh>=24 slips).
# ---------------------------------------------------------------------------


class TestKpfTimestampToDatetimeValidation:
    def test_raises_on_seconds_out_of_range(self):
        # Previously: silently rolled over into the next day.
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            kpf_timestamp_to_datetime("20240405.99999.57")

    def test_raises_on_invalid_date(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            kpf_timestamp_to_datetime("20249999.40113.57")

    def test_raises_on_wrong_format(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            kpf_timestamp_to_datetime("not-a-timestamp")


class TestUtcToHstValidation:
    def test_raises_on_seconds_out_of_range(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            utc_to_hst("20240405.99999.57")

    def test_raises_on_wrong_format(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            utc_to_hst("20240405T110833")


class TestHstToUtcValidation:
    def test_raises_on_seconds_out_of_range(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            hst_to_utc("20240405.99999.57")

    def test_raises_on_wrong_format(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            hst_to_utc("not-a-timestamp")


class TestKpfTimestampToEprvValidation:
    def test_raises_on_seconds_out_of_range(self):
        # Previously: produced 'YYYYMMDDT273739' with hh=27.
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            kpf_timestamp_to_eprv_timestamp("20240405.99999.57")

    def test_raises_on_invalid_date(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp"):
            kpf_timestamp_to_eprv_timestamp("20249999.40113.57")


# ===========================================================================
# stats.py — Gaussian fitting and bad-pixel interpolation
# ===========================================================================


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


# ===========================================================================
# config.py — ConfigHandler (TOML loading + section overrides)
# ===========================================================================


def _write_toml(tmp_path, text, name="cfg.toml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


class TestConfigHandler:
    def test_loads_sections(self, tmp_path):
        cfg = _write_toml(tmp_path, '[DATA_DIRS]\nroot = "/data"\n[KPFPIPE]\nn = 3\n')
        handler = ConfigHandler(cfg)
        assert handler.config["DATA_DIRS"]["root"] == "/data"

    def test_get_params_default_sections(self, tmp_path):
        cfg = _write_toml(tmp_path, '[DATA_DIRS]\nroot = "/data"\n[KPFPIPE]\nn = 3\n')
        params = ConfigHandler(cfg).get_params()  # sections=None -> defaults
        assert params == {"root": "/data", "n": 3}

    def test_get_params_flattens_nested_dict(self, tmp_path):
        toml = "[KPFPIPE]\nsimple = 1\n[KPFPIPE.nested]\nsub = 2\n"
        cfg = _write_toml(tmp_path, toml)
        params = ConfigHandler(cfg).get_params(["KPFPIPE"])
        assert params["simple"] == 1
        assert params["nested_sub"] == 2

    def test_override_merges_into_dict_section(self, tmp_path):
        cfg = _write_toml(tmp_path, '[DATA_DIRS]\nroot = "/data"\n')
        handler = ConfigHandler(cfg, overrides={"DATA_DIRS": {"out": "/out"}})
        assert handler.config["DATA_DIRS"] == {"root": "/data", "out": "/out"}

    def test_override_replaces_when_section_absent_or_non_dict(self, tmp_path):
        cfg = _write_toml(tmp_path, '[DATA_DIRS]\nroot = "/data"\n')
        handler = ConfigHandler(cfg, overrides={"NEW_SECTION": 42})
        assert handler.config["NEW_SECTION"] == 42

    def test_load_config_with_explicit_path_switches_file(self, tmp_path):
        cfg1 = _write_toml(tmp_path, "[KPFPIPE]\nn = 1\n", name="a.toml")
        cfg2 = _write_toml(tmp_path, "[KPFPIPE]\nn = 2\n", name="b.toml")
        handler = ConfigHandler(cfg1)
        handler.load_config(cfg2)
        assert str(handler.path).endswith("b.toml")
        assert handler.config["KPFPIPE"]["n"] == 2

    def test_get_params_reloads_when_config_empty(self, tmp_path):
        cfg = _write_toml(tmp_path, "[KPFPIPE]\nn = 7\n")
        handler = ConfigHandler(cfg)
        handler.config = {}  # force the reload branch in get_params
        params = handler.get_params(["KPFPIPE"])
        assert params["n"] == 7
