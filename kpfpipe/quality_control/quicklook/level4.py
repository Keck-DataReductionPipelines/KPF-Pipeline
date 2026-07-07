"""L4 quicklook plots for KPF cross-correlation functions (CCFs) and RVs.

Ports the per-order CCF grid from the v2.12 ``AnalyzeL2`` class (the old
pipeline's "L2" level is vNext's L4: CCFs and RVs live in the L4 product).
The CCF grid shows, for each illuminated orderlet, every order's CCF stacked
vertically so the per-order consistency of the dip is visible at a glance.

Pure visualization — no science computation is written back to the product.
"""

import os
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np

from kpfpipe.quality_control.quicklook._save_png import save_png

_DPI = 200
# Orderlet panels, left-to-right, in the canonical KPF order.
_FIBERS = ["SCI1", "SCI2", "SCI3", "CAL", "SKY"]
_SCI_FIBERS = ["SCI1", "SCI2", "SCI3"]
# Per-order CCF normalization percentile (science vs. cal/sky), matching the
# v2.12 AnalyzeL2 convention.
_SCI_NORM_PCTILE = 99
_CALSKY_NORM_PCTILE = 90
# Vertical offset between successive orders' normalized CCFs in the grid.
_ORDER_OFFSET = 0.5
# Per-fiber suffix for the legacy combined-RV keyword CCD{1|2}RV{sfx}, which
# RadialVelocity homes on each fiber's own RV-table extension header.
_RV_SFX = {"SCI1": "1", "SCI2": "2", "SCI3": "3", "CAL": "C", "SKY": "S"}


class PlotL4:
    """
    Quicklook plots for KPF L4 (RVs and CCFs) data.

    Args:
        l4_obj: KPF4 data object (post-CrossCorrelation + RadialVelocity).
        output_dir: directory to save PNG files. None = return Figure only.
        obs_id: observation ID for titles/filenames. If None, falls back to
            the l4_obj.obs_id attribute (populated on every construction path).
    """

    _PLOT_METHODS = ("ccf_grid",)

    def __init__(self, l4_obj, output_dir=None, obs_id=None):
        self.l4_obj = l4_obj
        self.output_dir = output_dir
        self.fibers = _FIBERS

        primary = (
            l4_obj.headers.get("PRIMARY", {}) if hasattr(l4_obj, "headers") else {}
        )
        self.obs_id = obs_id or getattr(l4_obj, "obs_id", None) or ""
        self.name = primary.get("OBJECT", "") if primary else ""

    # ------------------------------------------------------------------
    # Data access helpers
    # ------------------------------------------------------------------

    def _ccf(self, chip, fiber):
        """Return the (norder, nvel) CCF cube for one chip+fiber, or None.

        Returns None when the extension is absent/empty OR contains no real
        signal (all zero/NaN) — e.g. the zero-filled half of the concatenated
        cube when RVs were only computed for the other chip.
        """
        arr = self.l4_obj.data.get(f"{chip.upper()}_{fiber.upper()}_CCF")
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.size == 0:
            return None
        if not np.any(np.isfinite(arr) & (arr != 0.0)):
            return None
        return arr

    def _ext_header(self, fiber, suffix):
        """Return the {fiber}_{suffix} extension header (resolving its alias)."""
        key = f"{fiber.upper()}_{suffix}"
        if hasattr(self.l4_obj.data, "_resolve"):
            key = self.l4_obj.data._resolve(key)
        return self.l4_obj.headers.get(key, {}) or {}

    def _velocity_grid(self, fiber, nvel):
        """Velocity axis [km/s] from the CCF header, or sample index fallback."""
        hdr = self._ext_header(fiber, "CCF")
        start = hdr.get("VELSTART") if hdr else None
        step = hdr.get("VELSTEP") if hdr else None
        nstep = hdr.get("VELNSTEP") if hdr else None
        if start is None or step is None:
            return np.arange(nvel)
        n = int(nstep) if nstep is not None else nvel
        return np.arange(n) * float(step) + float(start)

    def _ccf_mask(self, fiber):
        return self._ext_header(fiber, "CCF").get("CCFMASK", "") or ""

    def _combined_rv(self, chip, fiber):
        """Per-CCD orderlet-combined RV [km/s], or None.

        The legacy CCD{1|2}RV{sfx} keyword is routed by set_keyword to the
        fiber's own RV-table extension header (e.g. CCD1RV2 -> RV3), so it is
        read from there, not PRIMARY/INSTRUMENT_HEADER.
        """
        n = "1" if chip.upper() == "GREEN" else "2"
        val = self._ext_header(fiber, "RV").get(f"CCD{n}RV{_RV_SFX[fiber.upper()]}")
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _has_chip(self, chip):
        return any(self._ccf(chip, fiber) is not None for fiber in self.fibers)

    @staticmethod
    def _norm_order(row, is_sci):
        """Normalize one order's CCF, matching v2.12 AnalyzeL2.plot_CCF_grid.

        SCI orderlets divide by the 99th percentile; CAL/SKY by the 90th, with
        the old all-negative-CCF shift handling.
        """
        if is_sci:
            denom = np.nanpercentile(row, _SCI_NORM_PCTILE)
            return row / denom if denom != 0 else row
        if np.nanpercentile(row, 99) < 0:
            shift = np.nanpercentile(row, 0.1)
            denom = np.nanpercentile(row + shift, _CALSKY_NORM_PCTILE)
            return (row + shift) / denom if denom != 0 else row + shift
        denom = np.nanpercentile(row, _CALSKY_NORM_PCTILE)
        return row if denom == 0 else row / denom

    @staticmethod
    def _cycle_colors(n):
        """Default Matplotlib color cycle, one color per order (repeating every
        10), matching the v2.12 per-order coloring."""
        base = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        return [base[i % len(base)] for i in range(n)]

    # ------------------------------------------------------------------
    # Common decoration
    # ------------------------------------------------------------------

    def _decorate_and_save(self, fig, plot_name, chip):
        """Add the standard QLP timestamp and save if output_dir is set."""
        current_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.005,
            f"KPF QLP: {current_time} UT",
            fontsize=8,
            color="darkgray",
            ha="right",
            va="bottom",
        )
        if self.output_dir is not None:
            prefix = f"{self.obs_id}_" if self.obs_id else ""
            path = os.path.join(
                self.output_dir, f"{prefix}L4_{plot_name}_{chip.lower()}_zoomable.png"
            )
            save_png(fig, path, dpi=_DPI, compress_level=6)
        return fig

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _rv_table(self, chip, fiber):
        """Return the per-order RV table (chip-sliced) for a fiber, or None."""
        tab = self.l4_obj.data.get(f"{chip.upper()}_{fiber.upper()}_RV")
        if tab is None or len(tab) == 0:
            return None
        return tab

    def _order_weights(self, rvtab):
        """Per-order CCF-combination weights for the SCI annotations, read from
        the RV table's WEIGHT column (written by CrossCorrelation).

        These are the weights the pipeline actually uses to combine the per-order
        CCFs (see RadialVelocity._combine_ccfs); orders with weight 0 are excluded
        from the combined RV. Returns None when the fiber has no RV table, but
        *raises* when a table is present yet lacks WEIGHT: there is no meaningful
        substitute, and silently inventing one (e.g. from RV_ERR) would annotate
        weights the pipeline never used and mask the missing column.
        """
        if rvtab is None:
            return None
        if "WEIGHT" not in rvtab.colnames:
            raise ValueError(
                "L4 RV table has no WEIGHT column; cannot annotate per-order CCF "
                "weights. The L4 must come from a CrossCorrelation step that "
                "writes the WEIGHT column."
            )
        return np.asarray(rvtab["WEIGHT"], dtype=float)

    def _draw_ccf_panel(self, ax, chip, fiber, norder, vref):
        """Draw one orderlet panel of stacked per-order CCFs (v2.12 layout).

        Every panel shares the same `norder`-based y-range so the five orderlet
        panels line up. An unilluminated orderlet (e.g. etalon/LFC CAL, no CCF)
        gets a framed panel over the shared `vref` velocity range with a
        'not illuminated' note instead of a blank default axis.
        """
        is_sci = fiber in _SCI_FIBERS
        top = norder * _ORDER_OFFSET
        ax.set_xlabel("RV (km/s)", fontsize=18)
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        ax.grid(False)
        ax.set_ylim(0, top + 1)

        ccf = self._ccf(chip, fiber)
        if ccf is None:
            ax.set_xlim(vref[0], vref[-1])
            ax.text(
                0.5 * (vref[0] + vref[-1]),
                0.5 * (top + 1),
                f"{fiber}: not illuminated\n(no CCF)",
                ha="center",
                va="center",
                color="darkgray",
                fontsize=14,
            )
            ax.set_title(f"{fiber} CCF", fontsize=18)
            return

        vgrid = self._velocity_grid(fiber, ccf.shape[1])
        colors = self._cycle_colors(ccf.shape[0])
        combined_rv = self._combined_rv(chip, fiber) if is_sci else None
        rvtab = self._rv_table(chip, fiber)
        order_rv = (
            np.asarray(rvtab["RV"], dtype=float)
            if rvtab is not None and "RV" in rvtab.colnames
            else None
        )
        weights = self._order_weights(rvtab)

        # SCI orderlets: combined-RV line + value, and the column headers.
        if is_sci and combined_rv is not None and np.isfinite(combined_rv):
            ax.plot([combined_rv, combined_rv], [0, top + 0.5], color="k")
            ax.text(
                combined_rv,
                top + 0.7,
                f"{combined_rv:.5f}" + r" km s$^{-1}$",
                color="k",
                ha="center",
                fontsize=11,
            )
            ax.text(
                vgrid[2],
                top + 0.55,
                r"$\Delta$RV (this - avg)",
                va="center",
                ha="left",
                color="k",
                fontsize=11,
            )
            ax.text(
                vgrid[-1] - 3,
                top + 0.55,
                "weight",
                va="center",
                ha="right",
                color="k",
                fontsize=11,
            )

        for o in range(ccf.shape[0]):
            row = ccf[o]
            if not np.any(row):  # order with no computed CCF -> skip (match v2.12)
                continue
            ax.plot(
                vgrid,
                self._norm_order(row, is_sci) + o * _ORDER_OFFSET,
                color=colors[o],
            )
            ax.text(
                vgrid[-1] + 0.75,
                1 + o * _ORDER_OFFSET - 0.10,
                str(o),
                color=colors[o],
                va="center",
                ha="left",
                fontsize=11,
            )
            if (
                is_sci
                and order_rv is not None
                and combined_rv is not None
                and o < order_rv.size
                and np.isfinite(order_rv[o])
            ):
                drv = order_rv[o] - combined_rv  # km/s
                ax.text(
                    vgrid[2],
                    1 + o * _ORDER_OFFSET - 0.3,
                    f"{drv:.4f}" + r" km s$^{-1}$",
                    color=colors[o],
                    va="center",
                    fontsize=11,
                )
                if weights is not None and o < weights.size:
                    ax.text(
                        vgrid[-1] - 3,
                        1 + o * _ORDER_OFFSET - 0.3,
                        f"{weights[o]:.2f}",
                        color=colors[o],
                        va="center",
                        ha="right",
                        fontsize=11,
                    )

        mask = self._ccf_mask(fiber)
        ax.set_title(f"{fiber} CCF" + (f" ({mask} mask)" if mask else ""), fontsize=18)

    def ccf_grid(self, chip):
        """Per-orderlet CCF grid for one chip, matching the v2.12 plot_CCF_grid
        layout: five panels (SCI1, SCI2, SCI3, CAL, SKY), each order's CCF
        normalized and offset, colored by the default cycle and labeled by order
        index. SCI panels add the combined-RV line + value and per-order
        delta-RV / weight columns. Returns None if the chip has no CCF data.
        """
        if not self._has_chip(chip):
            return None

        # Shared order count and a reference velocity grid (uniform panel
        # heights; the empty CAL panel reuses the velocity range).
        populated = [f for f in self.fibers if self._ccf(chip, f) is not None]
        norder = self._ccf(chip, populated[0]).shape[0]
        vref = self._velocity_grid(populated[0], self._ccf(chip, populated[0]).shape[1])

        fig, axes = plt.subplots(1, len(self.fibers), figsize=(25, 15), squeeze=False)
        for ax, fiber in zip(axes[0], self.fibers, strict=True):
            self._draw_ccf_panel(ax, chip, fiber, norder, vref)

        chip_title = "Green" if chip.upper() == "GREEN" else "Red"
        title = f"L4 - {chip_title} CCD: {self.obs_id}"
        if self.name:
            title += f" - {self.name}"
        fig.suptitle(title, fontsize=30)
        fig.tight_layout()
        return self._decorate_and_save(fig, "ccf_grid", chip)

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run(self, which):
        """Generate the requested plot(s) for every chip that has CCF data,
        saving each to ``output_dir``. In that save-to-disk mode the figure is
        closed so callers don't accumulate them; when ``output_dir`` is None the
        figures are returned open, so they display when the caller renders them
        (e.g. interactively in a notebook).

        Args:
            which: 'all' to run every implemented plot, or the name of a
                single plot method (one of ``self._PLOT_METHODS``).

        Returns:
            dict mapping ``{method_name}_{chip}`` to matplotlib.Figure (closed
            only when saved to ``output_dir``; useful for tests/introspection).
        """
        if which == "all":
            names = self._PLOT_METHODS
        elif which in self._PLOT_METHODS:
            names = (which,)
        else:
            raise ValueError(
                f"unknown plot {which!r}; expected 'all' or one of {self._PLOT_METHODS}"
            )

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

        figures = {}
        for chip in ("green", "red"):
            if not self._has_chip(chip):
                continue
            for name in names:
                fig = getattr(self, name)(chip)
                if fig is None:
                    continue
                figures[f"{name}_{chip}"] = fig
                # Closing frees memory in save-to-disk mode; when returning
                # figures for interactive display, leave them open.
                if self.output_dir is not None:
                    plt.close(fig)
        return figures
