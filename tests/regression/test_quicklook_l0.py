"""Tests for L0 quicklook plots."""

import os

import matplotlib
import numpy as np
import pytest
from astropy.io import fits
from PIL import Image

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from kpfpipe.data_models.level0 import KPF0

# Quicklook/QLP render suite: slow PNG rendering, so excluded from
# `make test-fast`. Run in the full suite or `make test-qlp`.
pytestmark = pytest.mark.quicklook


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figures a test left open (the output_dir=None path returns them)."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_4amp_l0(tmp_path):
    fn = str(tmp_path / "KP.20240405.00001.00.fits")
    rng = np.random.default_rng(42)

    nrow, ncol = 2070, 2094
    bias_level = 1000.0

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["OBJECT"] = "synthetic-4amp"
    primary.header["IMTYPE"] = "Bias"
    primary.header["DATE-OBS"] = "2024-04-05T01:00:37"
    primary.header["OFNAME"] = os.path.basename(fn)
    primary.header["PROGNAME"] = "K123"

    hdus = [primary]
    for chip in ["GREEN", "RED"]:
        for amp in range(1, 5):
            data = (bias_level + rng.normal(0, 3.0, (nrow, ncol))).astype(np.float32)
            hdus.append(fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}"))

    hdul = fits.HDUList(hdus)
    hdul.writeto(fn, overwrite=True)
    hdul.close()
    return KPF0.from_fits(fn)


@pytest.fixture
def synthetic_2amp_l0(tmp_path):
    fn = str(tmp_path / "KP.20240405.00002.00.fits")
    rng = np.random.default_rng(99)

    nrow, ncol = 4080, 2094
    bias_level = 1000.0

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["OBJECT"] = "synthetic-2amp"
    primary.header["IMTYPE"] = "Bias"
    primary.header["DATE-OBS"] = "2024-04-05T01:00:38"
    primary.header["OFNAME"] = os.path.basename(fn)
    primary.header["PROGNAME"] = "K123"

    hdus = [primary]
    for chip in ["GREEN", "RED"]:
        for amp in range(1, 3):
            data = (bias_level + rng.normal(0, 3.0, (nrow, ncol))).astype(np.float32)
            hdus.append(fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}"))

    hdul = fits.HDUList(hdus)
    hdul.writeto(fn, overwrite=True)
    hdul.close()
    return KPF0.from_fits(fn)


@pytest.fixture
def small_2amp_l0(tmp_path):
    """Small enough that a native-resolution PNG can be size-checked exactly."""
    fn = str(tmp_path / "KP.20240405.00004.00.fits")
    rng = np.random.default_rng(123)

    nrow, ncol = 32, 24

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["OBJECT"] = "small-2amp"
    primary.header["IMTYPE"] = "Bias"
    primary.header["DATE-OBS"] = "2024-04-05T01:00:39"
    primary.header["OFNAME"] = os.path.basename(fn)
    primary.header["PROGNAME"] = "K123"

    hdus = [primary]
    for chip in ["GREEN", "RED"]:
        for amp in range(1, 3):
            data = rng.normal(1000.0, 3.0, (nrow, ncol)).astype(np.float32)
            hdus.append(fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}"))

    hdul = fits.HDUList(hdus)
    hdul.writeto(fn, overwrite=True)
    hdul.close()
    return KPF0.from_fits(fn)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestPlotL0Constructor:
    def test_init_with_l0(self, synthetic_4amp_l0):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(synthetic_4amp_l0)
        assert qlp.kpf_obj is synthetic_4amp_l0
        assert qlp.obs_id == "KP.20240405.00001.00"
        assert qlp.name == "synthetic-4amp"
        assert qlp.output_dir is None

    def test_init_with_output_dir(self, synthetic_4amp_l0, tmp_path):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(synthetic_4amp_l0, output_dir=str(tmp_path))
        assert qlp.output_dir == str(tmp_path)


# ---------------------------------------------------------------------------
# stitched_image
# ---------------------------------------------------------------------------


class TestStitchedImage4Amp:
    def test_title_format(self, synthetic_4amp_l0):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(synthetic_4amp_l0)
        fig = qlp.stitched_image("green")
        ax = fig.axes[0]
        title = ax.get_title()
        assert "L0 - Green CCD" in title
        assert "KP.20240405.00001.00" in title
        assert "synthetic-4amp" in title
        plt.close(fig)

    def test_red_chip(self, synthetic_4amp_l0):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(synthetic_4amp_l0)
        fig = qlp.stitched_image("red")
        ax = fig.axes[0]
        assert "L0 - Red CCD" in ax.get_title()
        plt.close(fig)

    def test_colorbar_label_adu(self, synthetic_4amp_l0):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(synthetic_4amp_l0)
        fig = qlp.stitched_image("green")
        # image + colorbar
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_image_shape_4amp(self, synthetic_4amp_l0):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(synthetic_4amp_l0)
        fig = qlp.stitched_image("green")
        ax = fig.axes[0]
        images = ax.get_images()
        assert len(images) == 1
        # 4-amp: oriented and stripped to imaging area (4080 x 4080)
        assert images[0].get_array().shape == (4080, 4080)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 2-amp mode and 2^16 scaling
# ---------------------------------------------------------------------------


class TestStitchedImage2Amp:
    def test_image_shape_2amp(self, synthetic_2amp_l0):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(synthetic_2amp_l0)
        fig = qlp.stitched_image("red")
        ax = fig.axes[0]
        images = ax.get_images()
        # 2-amp: full height (4080) x 2 amps side by side (2094*2)
        assert images[0].get_array().shape == (4080, 2094 * 2)
        plt.close(fig)

    def test_title_2amp(self, synthetic_2amp_l0):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(synthetic_2amp_l0)
        fig = qlp.stitched_image("green")
        ax = fig.axes[0]
        assert "synthetic-2amp" in ax.get_title()
        plt.close(fig)


class TestStitchedImage2To16:
    def test_scales_high_values(self, tmp_path):
        # A median above the 200 * 2^16 threshold triggers the 2^16 rescale.
        fn = str(tmp_path / "KP.20240405.00003.00.fits")
        high_val = 300 * 2**16

        primary = fits.PrimaryHDU()
        primary.header["OBJECT"] = "high-value"
        primary.header["OFNAME"] = os.path.basename(fn)
        primary.header["PROGNAME"] = "K123"
        hdus = [primary]
        for chip in ["GREEN", "RED"]:
            for amp in range(1, 5):
                data = np.full((100, 100), high_val, dtype=np.float64)
                hdus.append(fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}"))

        hdul = fits.HDUList(hdus)
        hdul.writeto(fn, overwrite=True)
        hdul.close()

        l0 = KPF0.from_fits(fn)

        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(l0)
        fig = qlp.stitched_image("green")

        ax = fig.axes[0]
        # imshow keeps the stitched image masked, so use the mask-aware median.
        img_data = ax.get_images()[0].get_array()
        assert np.ma.median(img_data) == pytest.approx(300, abs=1)
        plt.close(fig)


# ---------------------------------------------------------------------------
# File saving and run()
# ---------------------------------------------------------------------------


class TestPlotL0FileSaving:
    def test_saves_png_when_output_dir_set(self, synthetic_4amp_l0, tmp_path):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(synthetic_4amp_l0, output_dir=str(tmp_path))
        fig = qlp.stitched_image("green")
        expected_path = (
            tmp_path / "KP.20240405.00001.00_L0_stitched_image_green_zoomable.png"
        )
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0
        plt.close(fig)

    def test_no_file_when_output_dir_none(self, synthetic_4amp_l0, tmp_path):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(synthetic_4amp_l0)
        fig = qlp.stitched_image("green")
        pngs = list(tmp_path.glob("*.png"))
        assert pngs == []
        plt.close(fig)

    def test_full_res_saves_native_dimensions(self, small_2amp_l0, tmp_path):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(small_2amp_l0, output_dir=str(tmp_path), full_res=True)
        fig = qlp.stitched_image("green")
        expected_path = (
            tmp_path / "KP.20240405.00004.00_L0_stitched_image_green_full_res.png"
        )
        default_path = (
            tmp_path / "KP.20240405.00004.00_L0_stitched_image_green_zoomable.png"
        )
        assert expected_path.exists()
        assert not default_path.exists()
        with Image.open(expected_path) as png:
            assert png.size == (48, 32)
        plt.close(fig)


class TestPlotL0Run:
    def test_run_all_returns_dict_of_figures(self, synthetic_4amp_l0):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(synthetic_4amp_l0)
        figs = qlp.run("all")
        assert isinstance(figs, dict)
        assert "stitched_image_green" in figs
        assert "stitched_image_red" in figs
        assert all(isinstance(f, plt.Figure) for f in figs.values())

    def test_run_single_plot_name(self, synthetic_4amp_l0):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(synthetic_4amp_l0)
        figs = qlp.run("stitched_image")
        assert set(figs.keys()) == {"stitched_image_green", "stitched_image_red"}

    def test_run_full_res_override_saves_native(self, small_2amp_l0, tmp_path):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(small_2amp_l0, output_dir=str(tmp_path))
        qlp.run("stitched_image", full_res=True)
        expected_path = (
            tmp_path / "KP.20240405.00004.00_L0_stitched_image_red_full_res.png"
        )
        with Image.open(expected_path) as png:
            assert png.size == (48, 32)

    def test_run_unknown_which_raises(self, synthetic_4amp_l0):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(synthetic_4amp_l0)
        with pytest.raises(ValueError, match="unknown plot"):
            qlp.run("bogus")

    def test_run_requires_which(self, synthetic_4amp_l0):
        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(synthetic_4amp_l0)
        with pytest.raises(TypeError):
            qlp.run()

    def test_run_skips_missing_chip(self, tmp_path):
        fn = str(tmp_path / "KP.20240405.00004.00.fits")
        rng = np.random.default_rng(7)

        primary = fits.PrimaryHDU()
        primary.header["OBJECT"] = "green-only"
        primary.header["OFNAME"] = os.path.basename(fn)
        primary.header["PROGNAME"] = "K123"
        hdus = [primary]
        for amp in range(1, 5):
            data = rng.normal(1000, 3, (100, 100)).astype(np.float32)
            hdus.append(fits.ImageHDU(data=data, name=f"GREEN_AMP{amp}"))

        hdul = fits.HDUList(hdus)
        hdul.writeto(fn, overwrite=True)
        hdul.close()

        l0 = KPF0.from_fits(fn)

        from kpfpipe.quality_control.quicklook.level0 import PlotL0

        qlp = PlotL0(l0)
        figs = qlp.run("all")
        assert "stitched_image_green" in figs
        assert "stitched_image_red" not in figs
