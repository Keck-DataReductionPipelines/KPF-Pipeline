"""
Regression tests for image assembly (L0 → L1).

Uses real L0 FITS files from ~/analysis/kpf/L0/. Tests are skipped
if the test data directory is not available.
"""

import os
import tempfile

import numpy as np
import pytest

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.modules.image_assembly import ImageAssembly

L0_DIR = os.path.expanduser("~/analysis/kpf/L0")
L0_BIAS = os.path.join(L0_DIR, "KP.20240923.03637.97.fits")
L0_FLAT = os.path.join(L0_DIR, "KP.20240923.00022.16.fits")

needs_l0_data = pytest.mark.skipif(
    not os.path.isdir(L0_DIR) or not os.path.isfile(L0_BIAS),
    reason="L0 test data not available",
)


@needs_l0_data
class TestImageAssemblyBias:
    """Regression tests using a bias frame (no signal, 2-amp mode)."""

    @pytest.fixture(scope="class")
    def l1_bias(self):
        l0 = KPF0.from_fits(L0_BIAS)
        ia = ImageAssembly(l0)
        return ia.perform(), ia

    def test_returns_kpf1(self, l1_bias):
        l1, _ = l1_bias
        assert isinstance(l1, KPF1)
        assert l1.level == 1

    def test_green_ccd_shape(self, l1_bias):
        l1, _ = l1_bias
        assert l1.data["GREEN_CCD"].shape == (4080, 4080)
        assert l1.data["GREEN_CCD"].dtype == np.float32

    def test_red_ccd_shape(self, l1_bias):
        l1, _ = l1_bias
        assert l1.data["RED_CCD"].shape == (4080, 4080)
        assert l1.data["RED_CCD"].dtype == np.float32

    def test_variance_frames_exist(self, l1_bias):
        l1, _ = l1_bias
        assert l1.data["GREEN_VAR"].shape == (4080, 4080)
        assert l1.data["RED_VAR"].shape == (4080, 4080)

    def test_variance_positive(self, l1_bias):
        l1, _ = l1_bias
        assert np.all(l1.data["GREEN_VAR"] >= 0)
        assert np.all(l1.data["RED_VAR"] >= 0)

    def test_bias_near_zero(self, l1_bias):
        """After overscan subtraction, a bias frame should be near zero."""
        l1, _ = l1_bias
        assert abs(np.nanmedian(l1.data["GREEN_CCD"])) < 5.0
        assert abs(np.nanmedian(l1.data["RED_CCD"])) < 5.0

    def test_primary_header_carried_forward(self, l1_bias):
        l1, _ = l1_bias
        assert l1.headers["PRIMARY"]["INSTRUME"] == "KPF"

    def test_obs_id_carried_forward(self, l1_bias):
        l1, _ = l1_bias
        assert l1.obs_id == "KP.20240923.03637.97"

    def test_datalvl_set(self, l1_bias):
        l1, _ = l1_bias
        datalvl = l1.headers["PRIMARY"]["DATALVL"]
        val = datalvl[0] if isinstance(datalvl, tuple) else datalvl
        assert val == "L1"

    def test_read_noise_in_header(self, l1_bias):
        l1, _ = l1_bias
        # 2-amp mode: expect RNGRN1, RNGRN2, RNRED1, RNRED2
        assert "RNGRN1" in l1.headers["PRIMARY"]
        assert "RNRED1" in l1.headers["PRIMARY"]

    def test_read_noise_reasonable(self, l1_bias):
        """Read noise should be between 1 and 20 electrons for KPF."""
        _, ia = l1_bias
        for channel_ext, rn in ia.readnoise.items():
            assert 1.0 < rn < 20.0, f"Read noise for {channel_ext} = {rn} e-"

    def test_overscan_method_in_header(self, l1_bias):
        l1, _ = l1_bias
        oscan = l1.headers["PRIMARY"]["OSCANMET"]
        val = oscan[0] if isinstance(oscan, tuple) else oscan
        assert val == "rowmedian"

    def test_receipt_chain(self, l1_bias):
        l1, _ = l1_bias
        modules = l1.receipt["Module_Name"].values
        assert "from_fits" in modules
        assert "to_l1" in modules
        assert "image_assembly" in modules

    def test_passthrough_telemetry(self, l1_bias):
        l1, _ = l1_bias
        assert "TELEMETRY" in l1.extensions

    def test_no_nans_in_ccd(self, l1_bias):
        l1, _ = l1_bias
        assert not np.any(np.isnan(l1.data["GREEN_CCD"]))
        assert not np.any(np.isnan(l1.data["RED_CCD"]))


@needs_l0_data
class TestImageAssemblyFlat:
    """Regression tests using a flat lamp frame (has signal)."""

    @pytest.fixture(scope="class")
    def l1_flat(self):
        if not os.path.isfile(L0_FLAT):
            pytest.skip("Flat L0 file not available")
        l0 = KPF0.from_fits(L0_FLAT)
        ia = ImageAssembly(l0)
        return ia.perform()

    def test_flat_has_signal(self, l1_flat):
        """A flat lamp should have significant positive signal."""
        assert np.nanmedian(l1_flat.data["GREEN_CCD"]) > 100.0
        assert np.nanmedian(l1_flat.data["RED_CCD"]) > 100.0

    def test_flat_variance_exceeds_readnoise(self, l1_flat):
        """Variance should include photon noise (larger than read noise alone)."""
        assert np.nanmedian(l1_flat.data["GREEN_VAR"]) > 10.0
        assert np.nanmedian(l1_flat.data["RED_VAR"]) > 10.0


@needs_l0_data
class TestImageAssemblyRoundTrip:
    """Test that L1 can be written to FITS and read back."""

    def test_write_and_read_back(self):
        l0 = KPF0.from_fits(L0_BIAS)
        ia = ImageAssembly(l0)
        l1 = ia.perform()

        with tempfile.TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, "test_l1.fits")
            l1.to_fits(fn)

            l1_read = KPF1.from_fits(fn)

            assert l1_read.data["GREEN_CCD"].shape == (4080, 4080)
            assert l1_read.data["RED_CCD"].shape == (4080, 4080)
            np.testing.assert_array_almost_equal(
                l1_read.data["GREEN_CCD"], l1.data["GREEN_CCD"], decimal=4
            )
            np.testing.assert_array_almost_equal(
                l1_read.data["RED_CCD"], l1.data["RED_CCD"], decimal=4
            )

    def test_roundtrip_preserves_header(self):
        l0 = KPF0.from_fits(L0_BIAS)
        ia = ImageAssembly(l0)
        l1 = ia.perform()

        with tempfile.TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, "test_l1.fits")
            l1.to_fits(fn)

            l1_read = KPF1.from_fits(fn)
            assert l1_read.headers["PRIMARY"]["INSTRUME"] == "KPF"
