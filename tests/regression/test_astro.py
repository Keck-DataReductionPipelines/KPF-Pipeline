"""Tests for kpfpipe.utils.astro: Doppler/redshift helpers, air->vacuum, colour->Teff.

Convention under test: positive radial velocity = receding = redshift (z > 0), so
the Doppler factor f = lambda_obs / lambda_rest = 1 + z and z carries the sign of
the velocity -- what BarycentricCorrection assumes when it stores BARYCORR_Z for
CrossCorrelation._compute_ccf_1d.

The colour->Teff tests are anchored on the Sun: all three catalog colour indices
must land on the tabulated G2V temperature, since CrossCorrelation picks the same
stellar line mask whichever catalog supplied the colour.
"""

import logging
import tomllib

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c

from kpfpipe import REPO_ROOT
from kpfpipe.utils.astro import (
    _B_V,
    _BP_RP,
    _EEM,
    _G_J,
    KECK_LOCATION,
    air_to_vac,
    color_to_teff,
    compute_doppler_factor,
    compute_redshift,
)

C_KMS = c.to("km/s").value

# The tabulated G2V row, in the three colours the pipeline can be handed.
SUN_TEFF = 5770.0
SUN_COLORS = {"B-V": 0.650, "Gaia BP-RP": 0.823, "G-J": 4.635 - 3.60}


class TestKeckLocation:
    """KECK_LOCATION is built from observatory.toml, never a second literal."""

    @staticmethod
    def _config():
        return tomllib.loads((REPO_ROOT / "reference/observatory.toml").read_text())

    def test_matches_the_observatory_config(self):
        config = self._config()
        assert KECK_LOCATION.lat.deg == pytest.approx(config["latitude"])
        assert KECK_LOCATION.lon.deg == pytest.approx(config["longitude"])
        assert KECK_LOCATION.height.to(u.m).value == pytest.approx(config["altitude"])

    def test_geosys_is_the_ellipsoid_astropy_uses(self):
        assert KECK_LOCATION.ellipsoid == self._config()["geosys"]


class TestComputeRedshift:
    def test_sign_matches_velocity(self):
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
        # A unitless argument is ambiguous (km/s? m/s?), so it must fail loudly.
        with pytest.raises(u.UnitsError):
            compute_redshift(-18508.0)

    def test_factor_direction(self):
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


class TestColorToTeff:
    @pytest.mark.parametrize("color_name", list(SUN_COLORS))
    def test_solar_colors_agree(self, color_name):
        assert color_to_teff(SUN_COLORS[color_name], color_name) == pytest.approx(
            SUN_TEFF
        )

    def test_redder_is_cooler(self):
        assert color_to_teff(1.5, "Gaia BP-RP") < color_to_teff(0.823, "Gaia BP-RP")

    @pytest.mark.parametrize("sequence", [_B_V, _BP_RP, _G_J])
    def test_sequence_is_strictly_increasing(self, sequence):
        colors, teffs = sequence
        assert np.all(np.diff(colors) > 0)
        assert np.all(np.diff(teffs) < 0)

    def test_censoring_is_driven_by_the_difference(self):
        # M_G and M_J are each monotonic on their own, but their difference is not,
        # so the censoring has to be applied to M_G - M_J, not to either magnitude.
        m_g, m_j = _EEM["M_G"].to_numpy(), _EEM["M_J"].to_numpy()
        both = np.isfinite(m_g) & np.isfinite(m_j)
        assert np.all(np.diff(m_g[both]) >= 0)
        assert np.all(np.diff(m_j[both]) >= 0)
        assert np.sum(np.diff((m_g - m_j)[both]) <= 0) > 0
        assert _G_J[0].size < both.sum()

    @pytest.mark.parametrize(
        ("color", "color_name"), [(-1.0, "B-V"), (5.5, "Gaia BP-RP")]
    )
    def test_extrapolates_beyond_the_table_with_a_warning(
        self, color, color_name, caplog
    ):
        colors = {"B-V": _B_V, "Gaia BP-RP": _BP_RP}[color_name][0]
        with caplog.at_level(logging.WARNING):
            teff = color_to_teff(color, color_name)
        assert teff > 0
        assert "extrapolating" in caplog.text
        # Bluer than the table -> hotter than its hottest row, and vice versa.
        assert (teff > color_to_teff(colors[0], color_name)) == (color < colors[0])

    def test_unrecognized_name_raises(self):
        with pytest.raises(ValueError, match="unrecognized colour index"):
            color_to_teff(1.0, "V-Ks")

    @pytest.mark.parametrize("color", [np.nan, None, "bright"])
    def test_non_finite_color_raises(self, color):
        with pytest.raises(ValueError, match="not a finite number"):
            color_to_teff(color, "B-V")

    def test_non_physical_extrapolation_raises(self):
        # The red end of B-V falls ~4000 K/mag, so an absurd colour runs past 0 K.
        with pytest.raises(ValueError, match="non-physical"):
            color_to_teff(3.0, "B-V")
