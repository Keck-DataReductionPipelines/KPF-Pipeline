"""Tests for L4 quicklook plots (CCFs and radial velocities)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.table import Table
from PIL import Image

from kpfpipe import DETECTOR
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.quality_control.quicklook.level4 import PlotL4

# Quicklook/QLP render suite: slow PNG rendering, so it is excluded from
# `make test-fast`. Run in the full suite or `make test-qlp`.
# The Agg pin and the close-figures teardown come from tests/regression/conftest.py.
pytestmark = pytest.mark.quicklook


NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
NORDER = NORDER_GREEN + NORDER_RED
NVEL = 21  # small velocity grid for fast tests

_FIBERS = ["SCI1", "SCI2", "SCI3", "CAL", "SKY"]
_CHIPS = ["GREEN", "RED"]
_VELSTART = -10.0
_VELSTEP = 1.0
_OBS_ID = "KP.20240405.40113.57"

_RV_SFX = {"SCI1": "1", "SCI2": "2", "SCI3": "3", "CAL": "C", "SKY": "S"}


def _gaussian_ccf(rng, depth=0.5):
    """A single-order CCF: a downward Gaussian dip plus noise."""
    v = np.arange(NVEL) * _VELSTEP + _VELSTART
    dip = 1.0 - depth * np.exp(-0.5 * (v / 3.0) ** 2)
    return dip + rng.normal(0.0, 0.01, size=NVEL)


def _make_l4(
    *, chips=("GREEN", "RED"), object_name="Tau Ceti", with_rv=True, with_weight=True
):
    """Build a KPF4 with CCF cubes + velocity headers + per-CCD RV keywords.

    `chips` selects which detector halves carry real CCF data; an omitted chip is
    left as the zero-filled concatenation slice ("no CCF there"). `with_rv` adds
    per-order RV tables for the science + sky orderlets, as RadialVelocity writes
    them, with weight 0 on the three bluest green orders as the real CCF
    order-weight table has. `with_weight=False` omits the WEIGHT column to
    exercise PlotL4's fail-loud path.
    """
    l4 = KPF4()
    l4.obs_id = _OBS_ID
    l4.headers["PRIMARY"]["INSTRUME"] = "KPF"
    l4.headers["PRIMARY"]["OBJECT"] = object_name

    rng = np.random.default_rng(7)
    for fiber in _FIBERS:
        # Real data goes only into the requested chips' order ranges.
        cube = np.zeros((NORDER, NVEL), dtype=np.float64)
        green_sl = slice(0, NORDER_GREEN)
        red_sl = slice(NORDER_GREEN, NORDER)
        if "GREEN" in chips:
            for o in range(NORDER_GREEN):
                cube[green_sl][o] = _gaussian_ccf(rng)
        if "RED" in chips:
            for o in range(NORDER_RED):
                cube[red_sl][o] = _gaussian_ccf(rng)
        l4.set_data(f"{fiber}_CCF", cube)

        # Velocity grid + mask live in the CCF extension header.
        ext = l4.data._resolve(f"{fiber}_CCF")
        l4.headers[ext]["VELSTART"] = _VELSTART
        l4.headers[ext]["VELSTEP"] = _VELSTEP
        l4.headers[ext]["VELNSTEP"] = NVEL
        l4.headers[ext]["CCFMASK"] = "G2_espresso"

    # Per-order RV tables (green orders near CCD1RV=0.5, red near CCD2RV=0.6),
    # plus the combined-RV keyword on each fiber's own RV extension header.
    if with_rv:
        rng2 = np.random.default_rng(11)
        base = np.where(np.arange(NORDER) < NORDER_GREEN, 0.5, 0.6)
        weight = np.ones(NORDER)
        weight[:3] = 0.0  # bluest green orders excluded (as the CCF weight table does)
        for fiber in ("SCI1", "SCI2", "SCI3", "SKY"):
            cols = {
                "ORDER_INDEX": np.arange(NORDER),
                "RV": base + rng2.normal(0.0, 0.002, NORDER),
                "RV_ERR": np.full(NORDER, 0.0015),
            }
            if with_weight:
                cols["WEIGHT"] = weight
            l4.set_data(f"{fiber}_RV", Table(cols))
            rv_ext = l4.data._resolve(f"{fiber}_RV")
            l4.headers[rv_ext][f"CCD1RV{_RV_SFX[fiber]}"] = 0.5
            l4.headers[rv_ext][f"CCD2RV{_RV_SFX[fiber]}"] = 0.6
    return l4


@pytest.fixture
def l4():
    return _make_l4()


# ---------------------------------------------------------------------------
# Constructor / obs_id resolution
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_obs_id_from_attribute(self, l4):
        assert PlotL4(l4).obs_id == _OBS_ID

    def test_explicit_obs_id_overrides(self, l4):
        assert PlotL4(l4, obs_id="KP.1.2.3").obs_id == "KP.1.2.3"

    def test_name_from_object_header(self, l4):
        assert PlotL4(l4).name == "Tau Ceti"


# ---------------------------------------------------------------------------
# ccf_grid plot
# ---------------------------------------------------------------------------


class TestCcfGrid:
    @pytest.mark.parametrize("chip", ["green", "red"])
    def test_returns_figure(self, l4, chip):
        fig = PlotL4(l4).ccf_grid(chip)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_one_panel_per_fiber_with_data(self, l4):
        fig = PlotL4(l4).ccf_grid("green")
        assert len(fig.axes) == len(_FIBERS)
        plt.close(fig)

    def test_suptitle_has_obs_id_and_object(self, l4):
        fig = PlotL4(l4).ccf_grid("green")
        suptitle = fig.get_suptitle()
        assert _OBS_ID in suptitle
        assert "Tau Ceti" in suptitle
        plt.close(fig)

    def test_panel_title_has_fiber_and_mask(self, l4):
        fig = PlotL4(l4).ccf_grid("green")
        titles = " | ".join(ax.get_title() for ax in fig.axes)
        assert "SCI2" in titles
        assert "G2_espresso" in titles
        plt.close(fig)

    def test_returns_none_when_chip_has_no_ccf(self):
        # Only green carries CCF data; the red slice is all zeros -> skipped.
        l4 = _make_l4(chips=("GREEN",))
        assert PlotL4(l4).ccf_grid("red") is None


# ---------------------------------------------------------------------------
# Per-order annotations (parity with v2.12 plot_CCF_grid)
# ---------------------------------------------------------------------------


def _sci2_panel(fig):
    return next(a for a in fig.axes if "SCI2" in a.get_title())


class TestCcfGridAnnotations:
    def test_sci_panel_annotations(self, l4):
        """Every read-only property of the SCI2 panel, on one render.

        These were six tests re-rendering an identical 5-panel x 35-order figure
        to make six read-only assertions; one render answers all of them. A
        class-scoped figure fixture would not work here -- conftest's autouse
        teardown closes every figure after each test.
        """
        import re

        fig = PlotL4(l4).ccf_grid("green")
        panel = _sci2_panel(fig)
        txt = " ".join(t.get_text() for t in panel.texts)

        # At least one text annotation per order (the order-index labels).
        assert len(panel.texts) >= NORDER_GREEN
        # The delta-RV and weight column headers.
        assert "(this - avg)" in txt
        assert "weight" in txt
        # Green SCI2 combined RV is CCD1RV2 = 0.5 km/s, shown to 5 dp.
        assert "0.50000" in txt and "km s" in txt
        # Per-order delta-RV is km/s to 4 decimals, e.g. "0.0020 km s^-1".
        assert re.search(r"\d\.\d{4}", txt)
        # The three bluest green orders carry weight 0 in the CCF weight table;
        # the panel must report them as "0.00", not a fabricated nonzero value.
        assert "0.00" in txt
        # Orders follow the default color cycle.
        assert len({ln.get_color() for ln in panel.lines}) > 1
        plt.close(fig)

    def test_renders_without_rv_tables(self):
        # No RV tables -> annotations are omitted but the plot still renders.
        fig = PlotL4(_make_l4(with_rv=False)).ccf_grid("green")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_unilluminated_fiber_panel_says_not_illuminated(self, l4):
        # CAL and SKY carry no CCF on many science frames. Draw the panel
        # directly on a bare axes: the 5-panel grid render is not needed to see
        # which branch _draw_ccf_panel takes.
        l4.set_data("CAL_CCF", np.zeros((NORDER, NVEL), dtype=np.float64))
        fig, ax = plt.subplots()
        PlotL4(l4)._draw_ccf_panel(
            ax, "green", "CAL", norder=NORDER_GREEN, vref=np.array([-10.0, 10.0])
        )
        assert any("not illuminated" in t.get_text() for t in ax.texts)
        plt.close(fig)

    def test_missing_weight_column_raises(self):
        # PlotL4 must never substitute weights the pipeline did not actually use.
        l4 = _make_l4(with_weight=False)
        with pytest.raises(ValueError, match="WEIGHT column"):
            PlotL4(l4).ccf_grid("green")


class TestVelocityGrid:
    def test_missing_velstep_raises(self, l4):
        # CrossCorrelation writes VELSTART/VELSTEP/VELNSTEP on every CCF, so
        # their absence is a malformed product. A hdr.get(..., 0.0) here would
        # fabricate a velocity axis instead -- the silent substitution the
        # method's docstring forbids. No render.
        del l4.headers[l4.data._resolve("SCI2_CCF")]["VELSTEP"]
        with pytest.raises(KeyError):
            PlotL4(l4)._velocity_grid("SCI2", NVEL)


# ---------------------------------------------------------------------------
# run() driver
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_all_both_chips(self, l4):
        figs = PlotL4(l4).run("all")
        assert set(figs) == {"ccf_grid_green", "ccf_grid_red"}

    def test_run_single_plot(self, l4):
        figs = PlotL4(l4).run("ccf_grid")
        assert set(figs) == {"ccf_grid_green", "ccf_grid_red"}

    def test_run_unknown_raises(self, l4):
        with pytest.raises(ValueError, match="unknown plot"):
            PlotL4(l4).run("bogus")

    def test_run_skips_chip_without_data(self):
        figs = PlotL4(_make_l4(chips=("GREEN",))).run("all")
        assert set(figs) == {"ccf_grid_green"}


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


class TestOutput:
    def test_writes_png_per_chip(self, l4, tmp_path):
        PlotL4(l4, output_dir=str(tmp_path), obs_id=_OBS_ID).run("all")
        pngs = sorted(p.name for p in tmp_path.glob("*.png"))
        assert pngs == sorted(
            f"{_OBS_ID}_L4_ccf_grid_{chip}_zoomable.png" for chip in ("green", "red")
        )
        # The name set is the real contract, but a blank canvas would satisfy it;
        # check one file actually has ink on it.
        with Image.open(tmp_path / pngs[0]) as png:
            assert png.convert("L").getextrema() != (255, 255)
