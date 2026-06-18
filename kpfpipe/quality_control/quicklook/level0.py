"""L0 quicklook plots for raw KPF detector images."""

import os
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt

from kpfpipe.modules.image_assembly import ImageAssembly


class PlotL0:
    """
    Quicklook plots for KPF L0 (raw CCD) data.

    Takes a KPF0 object and generates plots of the raw detector images.
    Pure visualization — no science computation. Amplifier counting and
    orientation delegate to ImageAssembly so detector-geometry knowledge
    has a single home.

    Args:
        l0_obj: KPF0 data object.
        output_dir: Directory to save PNG files. None = return Figure only.
    """

    _PLOT_METHODS = ("stitched_image",)

    def __init__(self, l0_obj, output_dir=None):
        self.l0 = l0_obj
        self.output_dir = output_dir
        self.obs_id = getattr(l0_obj, "obs_id", None) or ""
        self.name = ""
        if "PRIMARY" in l0_obj.headers:
            self.name = l0_obj.headers["PRIMARY"].get("OBJECT", "")

    def _has_chip(self, chip):
        """Return True if any AMP extension for the chip holds data."""
        for i in range(1, 5):
            ext = f"{chip.upper()}_AMP{i}"
            arr = self.l0.data.get(ext)
            if arr is not None and np.size(arr) > 0:
                return True
        return False

    def _stitch(self, chip):
        """Concatenate raw amplifier arrays into a single display image.

        Uses ImageAssembly to count amps and (for 4-amp mode) apply per-amp
        orientation, then applies the same blue -> red FFI orientation as
        ImageAssembly so the L0 display matches the assembled output. Per-amp
        orientation is performed on a deepcopy of the L0 so the caller's object
        is not mutated.
        """
        chip = chip.upper()

        # Count amplifiers via ImageAssembly (non-destructive).
        ia = ImageAssembly(self.l0)
        ia.count_amplifiers(chip)
        namp = ia.namp[chip]

        if namp == 2:
            image = np.concatenate(
                (self.l0.data[f"{chip}_AMP1"], self.l0.data[f"{chip}_AMP2"]),
                axis=1,
            )
        elif namp == 4:
            # orient_channels mutates l0.data, so operate on a copy.
            l0_copy = deepcopy(self.l0)
            ia = ImageAssembly(l0_copy)
            ia.count_amplifiers(chip)
            ia.orient_channels(chip)

            prescan = ia.prescan
            nrow_img = ia.nrow // 2
            ncol_img = ia.ncol // 2

            panels = {}
            for i in range(1, 5):
                panels[i] = l0_copy.data[f"{chip}_AMP{i}"][
                    :nrow_img, prescan : prescan + ncol_img
                ]

            bot = np.concatenate((panels[1], panels[2]), axis=1)
            top = np.concatenate((panels[3], panels[4]), axis=1)
            image = np.concatenate((bot, top), axis=0)

        return ImageAssembly.orient_ffi(image, chip)

    def stitched_image(self, chip):
        """
        Plot the stitched raw detector image for one CCD.

        Replicates v2.12 plot_L0_stitched_image.

        Args:
            chip: 'green' or 'red'.

        Returns:
            matplotlib.Figure
        """
        chip_upper = chip.upper()

        image = self._stitch(chip_upper)

        # Legacy data format: values stored with extra 2^16 factor
        twotosixteen = False
        if np.nanmedian(image) > 200 * 2**16:
            twotosixteen = True
            image = image / 2**16

        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        plt.imshow(
            image,
            cmap="viridis",
            origin="lower",
            vmin=np.percentile(image, 1),
            vmax=np.percentile(image, 99.5),
        )

        plt.title(
            f"L0 - {chip_upper.capitalize()} CCD: {self.obs_id} - {self.name}",
            fontsize=14,
        )
        plt.xlabel("Column (pixel number)", fontsize=14)
        plt.ylabel("Row (pixel number)", fontsize=14)

        cbar_label = "ADU"
        if twotosixteen:
            cbar_label += r" / $2^{16}$"
        cbar = plt.colorbar(shrink=0.95, label=cbar_label)
        cbar.ax.yaxis.label.set_size(14)
        cbar.ax.tick_params(labelsize=12)
        plt.grid(False)

        # Timestamp annotation
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        timestamp_label = f"KPF QLP: {current_time} UT"
        plt.annotate(
            timestamp_label,
            xy=(1, 0),
            xycoords="axes fraction",
            fontsize=8,
            color="darkgray",
            ha="right",
            va="top",
            xytext=(100, -21),
            textcoords="offset points",
        )
        plt.subplots_adjust(bottom=0.1)

        if self.output_dir is not None:
            fig_path = os.path.join(
                self.output_dir,
                f"{self.obs_id}_L0_stitched_image_{chip.lower()}_zoomable.png",
            )
            fig.savefig(fig_path, dpi=150, facecolor="w")

        return fig

    def run(self, which):
        """
        Generate the requested plot(s) for every chip that has data,
        saving each to `output_dir` and closing the matplotlib figure
        so callers don't accumulate them.

        Args:
            which: 'all' to run every implemented plot, or the name of a
                   single plot method (one of self._PLOT_METHODS).

        Returns:
            dict mapping `{method_name}_{chip}` to matplotlib.Figure
            (closed; useful for tests/introspection).
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
        for chip in ["green", "red"]:
            if not self._has_chip(chip):
                continue
            for name in names:
                fig = getattr(self, name)(chip)
                figures[f"{name}_{chip}"] = fig
                plt.close(fig)
        return figures
