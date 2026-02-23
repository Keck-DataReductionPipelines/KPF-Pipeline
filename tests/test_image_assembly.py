"""
Regression tests for image assembly (L0 → L1).

Uses real L0 FITS files for regression tests (skipped if data unavailable).
Set KPF_TESTDATA env var to your L0 data directory, or defaults to
/data/kpf/L0 (Keck server) then ~/analysis/kpf/L0 (local).
"""

import os
import tempfile

import numpy as np
import pytest
from astropy.io import fits

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.modules.image_assembly import ImageAssembly


def _find_l0_dir():
    """Resolve L0 test data directory from env var or known locations."""
    candidates = [
        os.environ.get("KPF_TESTDATA", ""),
        "/data/kpf/L0",
        os.path.expanduser("~/analysis/kpf/L0"),
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return None


L0_DIR = _find_l0_dir()
L0_BIAS = os.path.join(L0_DIR, "KP.20240923.03637.97.fits") if L0_DIR else ""
L0_FLAT = os.path.join(L0_DIR, "KP.20240923.00022.16.fits") if L0_DIR else ""

needs_l0_data = pytest.mark.skipif(
    L0_DIR is None or not os.path.isfile(L0_BIAS),
    reason="L0 test data not available (set KPF_TESTDATA env var)",
)


# ---------------------------------------------------------------------------
# Synthetic 4-amp L0 fixture (no real data needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_4amp_l0(tmp_path):
    """Create a synthetic L0 FITS file with 4-amp readout on both CCDs."""
    fn = str(tmp_path / "KP.20240101.00001.00.fits")
    rng = np.random.default_rng(42)

    # 4-amp dimensions: 2040 imaging rows + 30 parallel overscan,
    # 4 prescan + 2040 imaging cols + 50 serial overscan
    nrow, ncol = 2070, 2094
    bias_level = 1000.0

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["OBJECT"] = "synthetic-4amp"
    primary.header["IMTYPE"] = "Bias"
    primary.header["DATE-OBS"] = "2024-01-01T00:00:01"

    hdus = [primary]
    for chip in ["GREEN", "RED"]:
        for amp in range(1, 5):
            data = (bias_level + rng.normal(0, 3.0, (nrow, ncol))).astype(np.float32)
            hdu = fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}")
            hdus.append(hdu)

    hdul = fits.HDUList(hdus)
    hdul.writeto(fn, overwrite=True)
    hdul.close()
    return fn


# ---------------------------------------------------------------------------
# 2-amp regression tests (real data)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 4-amp mode tests (synthetic data)
# ---------------------------------------------------------------------------

class TestImageAssembly4Amp:
    """Test 4-amp mode assembly using synthetic data."""

    def test_4amp_produces_valid_l1(self, synthetic_4amp_l0):
        l0 = KPF0.from_fits(synthetic_4amp_l0)
        ia = ImageAssembly(l0)
        l1 = ia.perform()

        assert isinstance(l1, KPF1)
        assert l1.data["GREEN_CCD"].shape == (4080, 4080)
        assert l1.data["RED_CCD"].shape == (4080, 4080)

    def test_4amp_detects_four_amplifiers(self, synthetic_4amp_l0):
        l0 = KPF0.from_fits(synthetic_4amp_l0)
        ia = ImageAssembly(l0)
        ia.count_amplifiers("GREEN")
        ia.count_amplifiers("RED")
        assert ia.namp["GREEN"] == 4
        assert ia.namp["RED"] == 4
        assert ia.dims["GREEN"] == (2040, 2040)

    def test_4amp_read_noise_all_amps(self, synthetic_4amp_l0):
        l0 = KPF0.from_fits(synthetic_4amp_l0)
        ia = ImageAssembly(l0)
        l1 = ia.perform()

        # 4-amp mode: should have 8 read noise measurements
        assert len(ia.readnoise) == 8
        for channel_ext in ["GREEN_AMP1", "GREEN_AMP2", "GREEN_AMP3", "GREEN_AMP4",
                            "RED_AMP1", "RED_AMP2", "RED_AMP3", "RED_AMP4"]:
            assert channel_ext in ia.readnoise

        # All 8 RN keywords in header
        for key in ["RNGRN1", "RNGRN2", "RNGRN3", "RNGRN4",
                     "RNRED1", "RNRED2", "RNRED3", "RNRED4"]:
            assert key in l1.headers["PRIMARY"]

    def test_4amp_bias_near_zero(self, synthetic_4amp_l0):
        """Synthetic bias with known noise should be near zero after overscan."""
        l0 = KPF0.from_fits(synthetic_4amp_l0)
        ia = ImageAssembly(l0)
        l1 = ia.perform()

        assert abs(np.nanmedian(l1.data["GREEN_CCD"])) < 10.0
        assert abs(np.nanmedian(l1.data["RED_CCD"])) < 10.0

    def test_4amp_no_nans(self, synthetic_4amp_l0):
        l0 = KPF0.from_fits(synthetic_4amp_l0)
        ia = ImageAssembly(l0)
        l1 = ia.perform()

        assert not np.any(np.isnan(l1.data["GREEN_CCD"]))
        assert not np.any(np.isnan(l1.data["RED_CCD"]))


# ---------------------------------------------------------------------------
# FITS round-trip tests (real data)
# ---------------------------------------------------------------------------

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
