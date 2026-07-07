"""Tests for L2 quicklook plots (wavelength-aware extracted spectra)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from kpfpipe import DETECTOR
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.quality_control.quicklook.level2 import PlotL2

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
NCOL = 32  # small detector width for fast tests

_FIBERS = ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]
_CHIPS = ["GREEN", "RED"]
_PLOT_METHODS = (
    "snr_per_order",
    "peak_flux",
    "spectrum_single_order",
    "spectrum_one_row",
    "orderlet_flux_ratios",
)

_OBS_ID = "KP.20240405.40113.57"


def _make_l2(*, with_wave=True, object_name="Tau Ceti"):
    """Build a KPF2 with FLUX/VAR (and optionally WAVE) for all fibers/chips.

    obs_id is set on the model attribute, as the pipeline populates it.
    """
    l2 = KPF2()
    l2.obs_id = _OBS_ID
    l2.headers["PRIMARY"]["INSTRUME"] = "KPF"
    l2.headers["PRIMARY"]["OBJECT"] = object_name
    l2.headers["PRIMARY"]["DATE-OBS"] = "2024-04-05T11:08:33"

    rng = np.random.default_rng(42)
    for chip in _CHIPS:
        norder = NORDER_GREEN if chip == "GREEN" else NORDER_RED
        for fiber in _FIBERS:
            flux = rng.uniform(100.0, 1000.0, size=(norder, NCOL)).astype(np.float32)
            l2.set_data(f"{chip}_{fiber}_FLUX", flux)
            l2.set_data(f"{chip}_{fiber}_VAR", np.abs(flux) + 1.0)
            if with_wave:
                wave = np.linspace(
                    4000.0, 8000.0, norder * NCOL, dtype=np.float64
                ).reshape(norder, NCOL)
                l2.set_data(f"{chip}_{fiber}_WAVE", wave)
    return l2


@pytest.fixture
def l2():
    return _make_l2(with_wave=True)


@pytest.fixture
def l2_no_wave():
    return _make_l2(with_wave=False)


# ---------------------------------------------------------------------------
# Constructor / obs_id resolution
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_obs_id_from_attribute(self, l2):
        assert PlotL2(l2).obs_id == _OBS_ID

    def test_explicit_obs_id_overrides(self, l2):
        assert (
            PlotL2(l2, obs_id="KP.20240405.99999.99").obs_id == "KP.20240405.99999.99"
        )

    def test_name_from_object_header(self, l2):
        assert PlotL2(l2).name == "Tau Ceti"


# ---------------------------------------------------------------------------
# Individual plots (wavelength present)
# ---------------------------------------------------------------------------


class TestPlots:
    @pytest.mark.parametrize("method", _PLOT_METHODS)
    @pytest.mark.parametrize("chip", ["green", "red"])
    def test_method_returns_figure(self, l2, method, chip):
        fig = getattr(PlotL2(l2), method)(chip)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_spectrum_single_order_explicit_order(self, l2):
        fig = PlotL2(l2).spectrum_single_order("green", order=3)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_spectrum_single_order_out_of_range_raises(self, l2):
        with pytest.raises(ValueError, match="out of range"):
            PlotL2(l2).spectrum_single_order("green", order=NORDER_GREEN + 5)

    def test_snr_title_has_obs_id_and_object(self, l2):
        fig = PlotL2(l2).snr_per_order("green")
        title = fig.axes[0].get_title()
        assert _OBS_ID in title
        assert "Tau Ceti" in title
        plt.close(fig)


# ---------------------------------------------------------------------------
# Fail loudly when wavelength solution is absent
# ---------------------------------------------------------------------------


class TestRequiresWave:
    @pytest.mark.parametrize("method", _PLOT_METHODS)
    def test_method_raises_without_wave(self, l2_no_wave, method):
        with pytest.raises(ValueError, match="WAVE is not populated"):
            getattr(PlotL2(l2_no_wave), method)("green")

    def test_run_all_raises_without_wave(self, l2_no_wave):
        with pytest.raises(ValueError, match="WAVE is not populated"):
            PlotL2(l2_no_wave).run("all")


# ---------------------------------------------------------------------------
# run(which) dispatch
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_all_returns_all_plots_both_chips(self, l2):
        figs = PlotL2(l2).run("all")
        assert isinstance(figs, dict)
        expected = {f"{m}_{c}" for m in _PLOT_METHODS for c in ("green", "red")}
        assert set(figs) == expected
        assert all(isinstance(f, plt.Figure) for f in figs.values())

    def test_run_single_plot(self, l2):
        figs = PlotL2(l2).run("snr_per_order")
        assert set(figs) == {"snr_per_order_green", "snr_per_order_red"}

    def test_run_unknown_raises(self, l2):
        with pytest.raises(ValueError, match="unknown plot"):
            PlotL2(l2).run("bogus")

    def test_run_requires_which(self, l2):
        with pytest.raises(TypeError):
            PlotL2(l2).run()


# ---------------------------------------------------------------------------
# File saving
# ---------------------------------------------------------------------------


class TestFileSaving:
    def test_run_writes_pngs(self, l2, tmp_path):
        PlotL2(l2, output_dir=str(tmp_path)).run("all")
        pngs = sorted(p.name for p in tmp_path.glob("*.png"))
        expected = sorted(
            f"{_OBS_ID}_L2_{m}_{c}_zoomable.png"
            for m in _PLOT_METHODS
            for c in ("green", "red")
        )
        assert pngs == expected
        assert all((tmp_path / n).stat().st_size > 0 for n in pngs)

    def test_no_files_when_output_dir_none(self, l2, tmp_path):
        PlotL2(l2).run("all")
        assert list(tmp_path.glob("*.png")) == []
