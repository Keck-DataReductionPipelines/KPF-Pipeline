"""Tests for L1 quicklook plots."""

import matplotlib
import numpy as np
import pytest
from astropy.io import fits
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kpfpipe.data_models.level1 import KPF1

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 300x300 is large enough to exercise the 100-pixel border stripping used by
# image()'s percentile scaling without paying for full 4080x4080 arrays.
_FIXTURE_SHAPE = (300, 300)


def _build_synthetic_l1(
    tmp_path,
    *,
    obs_id="KP.20240405.00001.00",
    object_name="synthetic-l1",
    shape=_FIXTURE_SHAPE,
):
    """Create a synthetic L1 FITS file with assembled CCDs (no read-noise)."""
    fn = str(tmp_path / f"{obs_id}_L1.fits")
    rng = np.random.default_rng(42)

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["OBJECT"] = object_name
    primary.header["DATALVL"] = "L1"
    primary.header["DATE-OBS"] = "2024-04-05T01:00:37"
    primary.header["EXPTIME"] = 300.0

    hdus = [primary]
    for chip in ["GREEN", "RED"]:
        ccd = (1000.0 + rng.normal(0, 3.0, shape)).astype(np.float32)
        var = np.abs(ccd).astype(np.float32)
        ccd_hdu = fits.ImageHDU(data=ccd, name=f"{chip}_CCD")
        var_hdu = fits.ImageHDU(data=var, name=f"{chip}_VAR")
        hdus += [ccd_hdu, var_hdu]

    hdul = fits.HDUList(hdus)
    hdul.writeto(fn, overwrite=True)
    hdul.close()
    return fn


def _seed_read_noise(l1):
    """Seed RN*/RNNG* via set_keyword, exactly as ImageAssembly does.

    set_keyword routes these to QUALITY_CONTROL (their registry home), so this
    exercises the real read path the quicklook annotation depends on — writing
    them onto PRIMARY here would let the annotation test pass while production
    (which reads QUALITY_CONTROL) silently rendered nothing.
    """
    for i in range(1, 5):
        l1.set_keyword(f"RNGREEN{i}", 3.5 + 0.05 * i)
        l1.set_keyword(f"RNNGGR{i}", 1.0 + 0.001 * i)
        l1.set_keyword(f"RNRED{i}", 4.0 + 0.05 * i)
        l1.set_keyword(f"RNNGRD{i}", 1.002 + 0.001 * i)


@pytest.fixture
def synthetic_l1(tmp_path):
    obs_id = "KP.20240405.00001.00"
    l1 = KPF1.from_fits(_build_synthetic_l1(tmp_path, obs_id=obs_id))
    l1.obs_id = obs_id
    _seed_read_noise(l1)
    return l1


@pytest.fixture
def synthetic_l1_no_rn(tmp_path):
    obs_id = "KP.20240405.00002.00"
    l1 = KPF1.from_fits(_build_synthetic_l1(tmp_path, obs_id=obs_id))
    l1.obs_id = obs_id
    return l1


# ---------------------------------------------------------------------------
# Task 1: Constructor
# ---------------------------------------------------------------------------


class TestPlotL1Constructor:
    def test_init(self, synthetic_l1):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1)
        assert qlp.l1_obj is synthetic_l1
        assert qlp.obs_id == "KP.20240405.00001.00"
        assert qlp.name == "synthetic-l1"
        assert qlp.output_dir is None

    def test_init_with_output_dir(self, synthetic_l1, tmp_path):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1, output_dir=str(tmp_path))
        assert qlp.output_dir == str(tmp_path)


class TestImage:
    def test_title_green(self, synthetic_l1):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1)
        fig = qlp.image("green")
        title = fig.axes[0].get_title()
        assert "L1 - Green CCD" in title
        assert "KP.20240405.00001.00" in title
        assert "synthetic-l1" in title
        plt.close(fig)

    def test_title_red(self, synthetic_l1):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1)
        fig = qlp.image("red")
        assert "L1 - Red CCD" in fig.axes[0].get_title()
        plt.close(fig)

    def test_has_colorbar(self, synthetic_l1):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1)
        fig = qlp.image("green")
        # image axis + colorbar axis
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_image_shape(self, synthetic_l1):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1)
        fig = qlp.image("green")
        img = fig.axes[0].get_images()[0].get_array()
        assert img.shape == _FIXTURE_SHAPE
        plt.close(fig)

    def test_read_noise_annotation_present(self, synthetic_l1):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1)
        fig = qlp.image("green")
        texts = [t.get_text() for t in fig.axes[0].texts]
        # Should have a read noise annotation containing 'RN:' prefix
        assert any("RN:" in t for t in texts), f"texts found: {texts}"
        plt.close(fig)

    def test_no_read_noise_annotation_when_headers_missing(self, synthetic_l1_no_rn):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1_no_rn)
        fig = qlp.image("green")
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert not any("RN:" in t for t in texts)
        plt.close(fig)


class TestFileSaving:
    def test_saves_png(self, synthetic_l1, tmp_path):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1, output_dir=str(tmp_path))
        fig = qlp.image("green")
        expected = tmp_path / "KP.20240405.00001.00_L1_image_green_zoomable.png"
        assert expected.exists()
        assert expected.stat().st_size > 0
        plt.close(fig)

    def test_no_file_when_output_dir_none(self, synthetic_l1, tmp_path):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1)
        fig = qlp.image("green")
        assert list(tmp_path.glob("*.png")) == []
        plt.close(fig)

    def test_full_res_saves_native_dimensions(self, synthetic_l1, tmp_path):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1, output_dir=str(tmp_path), full_res=True)
        fig = qlp.image("green")
        expected = tmp_path / "KP.20240405.00001.00_L1_image_green_full_res.png"
        default = tmp_path / "KP.20240405.00001.00_L1_image_green_zoomable.png"
        assert expected.exists()
        assert not default.exists()
        with Image.open(expected) as png:
            assert png.size == (_FIXTURE_SHAPE[1], _FIXTURE_SHAPE[0])
        plt.close(fig)


class TestRun:
    def test_run_all_returns_dict(self, synthetic_l1):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1)
        figs = qlp.run("all")
        assert isinstance(figs, dict)
        assert "image_green" in figs
        assert "image_red" in figs

    def test_run_single_plot_name(self, synthetic_l1):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1)
        figs = qlp.run("image")
        assert set(figs.keys()) == {"image_green", "image_red"}

    def test_run_full_res_override_saves_native(self, synthetic_l1, tmp_path):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1, output_dir=str(tmp_path))
        qlp.run("image", full_res=True)
        expected = tmp_path / "KP.20240405.00001.00_L1_image_red_full_res.png"
        with Image.open(expected) as png:
            assert png.size == (_FIXTURE_SHAPE[1], _FIXTURE_SHAPE[0])

    def test_run_unknown_which_raises(self, synthetic_l1):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1)
        with pytest.raises(ValueError, match="unknown plot"):
            qlp.run("bogus")

    def test_run_requires_which(self, synthetic_l1):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1)
        with pytest.raises(TypeError):
            qlp.run()

    def test_run_skips_missing_chip(self, tmp_path):
        # KPF1 with only green CCD, no red
        rng = np.random.default_rng(42)
        fn = str(tmp_path / "KP.20240405.00003.00_L1.fits")
        primary = fits.PrimaryHDU()
        primary.header["INSTRUME"] = "KPF"
        primary.header["OBJECT"] = "green-only"
        primary.header["DATALVL"] = "L1"
        primary.header["DATE-OBS"] = "2024-04-05T01:00:37"
        primary.header["EXPTIME"] = 300.0
        green_ccd = fits.ImageHDU(
            data=rng.random((100, 100)).astype(np.float32), name="GREEN_CCD"
        )
        green_var = fits.ImageHDU(
            data=rng.random((100, 100)).astype(np.float32), name="GREEN_VAR"
        )
        hdul = fits.HDUList([primary, green_ccd, green_var])
        hdul.writeto(fn, overwrite=True)
        hdul.close()

        l1 = KPF1.from_fits(fn)
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(l1)
        figs = qlp.run("all")
        assert "image_green" in figs
        assert "image_red" not in figs


class TestStubs:
    @pytest.mark.parametrize(
        "method_name",
        [
            "histogram",
            "column_cut",
            "zoom_3x3",
            "order_trace_overlay",
            "bias_subtracted",
            "dark_subtracted",
        ],
    )
    def test_stub_raises_not_implemented(self, synthetic_l1, method_name):
        from kpfpipe.quality_control.quicklook.level1 import PlotL1

        qlp = PlotL1(synthetic_l1)
        method = getattr(qlp, method_name)
        with pytest.raises(NotImplementedError):
            method("green")
