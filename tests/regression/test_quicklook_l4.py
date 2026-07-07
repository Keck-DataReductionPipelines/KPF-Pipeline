"""Tests for L4 quicklook plots (CCFs and radial velocities)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.table import Table

from kpfpipe import DETECTOR
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.quality_control.quicklook.level4 import PlotL4

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
    """A single-order CCF: a downward Gaussian dip plus noise (one per order)."""
    v = np.arange(NVEL) * _VELSTEP + _VELSTART
    dip = 1.0 - depth * np.exp(-0.5 * (v / 3.0) ** 2)
    return dip + rng.normal(0.0, 0.01, size=NVEL)


def _make_l4(
    *, chips=("GREEN", "RED"), object_name="Tau Ceti", with_rv=True, with_weight=True
):
    """Build a KPF4 with CCF cubes + velocity headers + per-CCD RV keywords.

    `chips` selects which detector halves carry real CCF data; a chip omitted
    here is left as the zero-filled concatenation slice (i.e. "no CCF there").
    `with_rv` also attaches per-order RV tables (ORDER_INDEX/RV/RV_ERR/WEIGHT) for
    the science + sky orderlets, as RadialVelocity writes them; the first three
    green orders carry weight 0 (excluded from the combined RV, as the real CCF
    order-weight table does). `with_weight=False` omits the WEIGHT column to
    exercise PlotL4's fail-loud path. obs_id is set on the model attribute, as
    the pipeline populates it.
    """
    l4 = KPF4()
    l4.obs_id = _OBS_ID
    l4.headers["PRIMARY"]["INSTRUME"] = "KPF"
    l4.headers["PRIMARY"]["OBJECT"] = object_name

    rng = np.random.default_rng(7)
    for fiber in _FIBERS:
        # Build the full concatenated cube, writing real data only into the
        # requested chips' order ranges (the rest stays zero).
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

        # Velocity grid + mask live in the CCF extension header (resolved alias).
        ext = l4.data._resolve(f"{fiber}_CCF")
        l4.headers[ext]["VELSTART"] = _VELSTART
        l4.headers[ext]["VELSTEP"] = _VELSTEP
        l4.headers[ext]["VELNSTEP"] = NVEL
        l4.headers[ext]["CCFMASK"] = "G2_espresso"

    # Per-order RV tables (green orders near CCD1RV=0.5, red near CCD2RV=0.6),
    # plus the legacy combined-RV keyword on each fiber's own RV extension header
    # (CCD{n}RV{sfx} -> RV#), as RadialVelocity writes them via set_keyword.
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
    def test_sci_panel_labels_every_order(self, l4):
        # At least one text annotation per order (the order-index labels).
        fig = PlotL4(l4).ccf_grid("green")
        assert len(_sci2_panel(fig).texts) >= NORDER_GREEN
        plt.close(fig)

    def test_sci_panel_has_delta_rv_and_weight_headers(self, l4):
        fig = PlotL4(l4).ccf_grid("green")
        txt = " ".join(t.get_text() for t in _sci2_panel(fig).texts)
        assert "(this - avg)" in txt  # the delta-RV column header (mathtext Delta)
        assert "weight" in txt
        plt.close(fig)

    def test_orderlet_rv_value_annotated(self, l4):
        # Green SCI2 combined RV is CCD1RV2 = 0.5 km/s; shown (5 dp, km s^-1) at
        # the vertical RV line.
        fig = PlotL4(l4).ccf_grid("green")
        txt = " ".join(t.get_text() for t in _sci2_panel(fig).texts)
        assert "0.50000" in txt and "km s" in txt
        plt.close(fig)

    def test_per_order_delta_rv_values_present(self, l4):
        import re

        fig = PlotL4(l4).ccf_grid("green")
        txt = " ".join(t.get_text() for t in _sci2_panel(fig).texts)
        # Per-order delta-RV is km/s to 4 decimals, e.g. "0.0020 km s^-1".
        assert re.search(r"\d\.\d{4}", txt)
        plt.close(fig)

    def test_per_order_distinct_colors(self, l4):
        fig = PlotL4(l4).ccf_grid("green")
        colors = {ln.get_color() for ln in _sci2_panel(fig).lines}
        assert len(colors) > 1  # orders follow the default color cycle

    def test_renders_without_rv_tables(self):
        # No RV tables -> annotations are omitted but the plot still renders.
        fig = PlotL4(_make_l4(with_rv=False)).ccf_grid("green")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_zero_weight_orders_annotated_as_zero(self, l4):
        # The three bluest green orders carry weight 0 in the CCF weight table;
        # the panel must report them as "0.00", not a fabricated nonzero value.
        fig = PlotL4(l4).ccf_grid("green")
        txt = " ".join(t.get_text() for t in _sci2_panel(fig).texts)
        assert "0.00" in txt
        plt.close(fig)

    def test_missing_weight_column_raises(self):
        # An RV table without WEIGHT must fail loudly -- PlotL4 never substitutes
        # weights the pipeline did not actually use (e.g. inverse-variance).
        l4 = _make_l4(with_weight=False)
        with pytest.raises(ValueError, match="WEIGHT column"):
            PlotL4(l4).ccf_grid("green")


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
        for chip in ("green", "red"):
            assert (tmp_path / f"{_OBS_ID}_L4_ccf_grid_{chip}_zoomable.png").is_file()
