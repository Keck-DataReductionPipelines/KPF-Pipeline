"""Tests for L0 quicklook plots."""

import os
import numpy as np
import pytest
from astropy.io import fits

import matplotlib
matplotlib.use('Agg')

from kpfpipe.data_models.level0 import KPF0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_4amp_l0(tmp_path):
    """Create a synthetic KPF0 with 4-amp green and red data."""
    fn = str(tmp_path / "KP.20240405.00001.00.fits")
    rng = np.random.default_rng(42)

    nrow, ncol = 2070, 2094
    bias_level = 1000.0

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["OBJECT"] = "synthetic-4amp"
    primary.header["IMTYPE"] = "Bias"
    primary.header["DATE-OBS"] = "2024-04-05T01:00:37"

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
    """Create a synthetic KPF0 with 2-amp green and red data."""
    fn = str(tmp_path / "KP.20240405.00002.00.fits")
    rng = np.random.default_rng(99)

    nrow, ncol = 4080, 2094
    bias_level = 1000.0

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["OBJECT"] = "synthetic-2amp"
    primary.header["IMTYPE"] = "Bias"
    primary.header["DATE-OBS"] = "2024-04-05T01:00:38"

    hdus = [primary]
    for chip in ["GREEN", "RED"]:
        for amp in range(1, 3):
            data = (bias_level + rng.normal(0, 3.0, (nrow, ncol))).astype(np.float32)
            hdus.append(fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}"))

    hdul = fits.HDUList(hdus)
    hdul.writeto(fn, overwrite=True)
    hdul.close()
    return KPF0.from_fits(fn)


# ---------------------------------------------------------------------------
# Task 1: Constructor
# ---------------------------------------------------------------------------

class TestPlotL0Constructor:

    def test_init_with_l0(self, synthetic_4amp_l0):
        from kpfpipe.qlp.plot_l0 import PlotL0
        qlp = PlotL0(synthetic_4amp_l0)
        assert qlp.l0 is synthetic_4amp_l0
        assert qlp.obs_id == "KP.20240405.00001.00"
        assert qlp.name == "synthetic-4amp"
        assert qlp.output_dir is None

    def test_init_with_output_dir(self, synthetic_4amp_l0, tmp_path):
        from kpfpipe.qlp.plot_l0 import PlotL0
        qlp = PlotL0(synthetic_4amp_l0, output_dir=str(tmp_path))
        assert qlp.output_dir == str(tmp_path)


# ---------------------------------------------------------------------------
# Task 2: stitched_image
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt


class TestStitchedImage4Amp:

    def test_returns_figure(self, synthetic_4amp_l0):
        from kpfpipe.qlp.plot_l0 import PlotL0
        qlp = PlotL0(synthetic_4amp_l0)
        fig = qlp.stitched_image('green')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_title_format(self, synthetic_4amp_l0):
        from kpfpipe.qlp.plot_l0 import PlotL0
        qlp = PlotL0(synthetic_4amp_l0)
        fig = qlp.stitched_image('green')
        ax = fig.axes[0]
        title = ax.get_title()
        assert 'L0 - Green CCD' in title
        assert 'KP.20240405.00001.00' in title
        assert 'synthetic-4amp' in title
        plt.close(fig)

    def test_red_chip(self, synthetic_4amp_l0):
        from kpfpipe.qlp.plot_l0 import PlotL0
        qlp = PlotL0(synthetic_4amp_l0)
        fig = qlp.stitched_image('red')
        ax = fig.axes[0]
        assert 'L0 - Red CCD' in ax.get_title()
        plt.close(fig)

    def test_colorbar_label_adu(self, synthetic_4amp_l0):
        from kpfpipe.qlp.plot_l0 import PlotL0
        qlp = PlotL0(synthetic_4amp_l0)
        fig = qlp.stitched_image('green')
        # Figure should have 2 axes: image + colorbar
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_image_shape_4amp(self, synthetic_4amp_l0):
        from kpfpipe.qlp.plot_l0 import PlotL0
        qlp = PlotL0(synthetic_4amp_l0)
        fig = qlp.stitched_image('green')
        ax = fig.axes[0]
        images = ax.get_images()
        assert len(images) == 1
        # 4-amp: 2 amps stacked vertically (2070*2) x 2 amps side by side (2094*2)
        assert images[0].get_array().shape == (2070 * 2, 2094 * 2)
        plt.close(fig)
