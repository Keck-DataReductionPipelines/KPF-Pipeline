"""
Tests for kpfpipe.utils.astro Doppler/redshift helpers and air->vacuum.

Convention under test: positive radial velocity = receding = redshift (z > 0),
so the Doppler factor f = lambda_obs / lambda_rest = 1 + z, and z carries the
same sign as the velocity. This is the convention BarycentricCorrection relies
on when storing BARYCORR_Z for RadialVelocity._compute_ccf_1d.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c

from kpfpipe.utils.astro import air_to_vac, compute_doppler_factor, compute_redshift

C_KMS = c.to('km/s').value


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
            compute_redshift(-18508.0 * u.m / u.s))

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
