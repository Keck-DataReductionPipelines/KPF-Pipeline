"""L0 quicklook plots for raw KPF detector images."""

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.quality_control.quicklook._save_png import save_image_png, save_png
from kpfpipe.quality_control.quicklook.base import Plot


class PlotL0(Plot):
    """Quicklook plots for KPF L0 (raw CCD) data.

    Takes a KPF0 object and generates plots of the raw detector images.
    Pure visualization -- no science computation. Amplifier counting and
    orientation delegate to ImageAssembly so detector-geometry knowledge
    has a single home.

    Parameters
    ----------
    l0_obj : KPF0
        KPF0 data object.
    output_dir : str or None
        Directory to save PNG files. None returns the Figure only.
    full_res : bool
        Save full 1:1 pixel-resolution PNGs instead of the default downsampled
        figure PNGs. Full-resolution files can be tens of MB per chip.
    """

    LEVEL = "L0"
    _PLOT_METHODS = ("stitched_image",)

    def __init__(self, l0_obj, output_dir=None, full_res=False):
        super().__init__(l0_obj, output_dir)
        self.full_res = full_res

    def _has_chip(self, chip):
        """Return True if any AMP extension for the chip holds data."""
        for i in range(1, 5):
            ext = f"{chip.upper()}_AMP{i}"
            arr = self.kpf_obj.data.get(ext)
            if arr is not None and np.size(arr) > 0:
                return True
        return False

    def _stitch(self, chip):
        """Concatenate raw amplifier arrays into a single display image.

        Delegates amp counting/orientation to ImageAssembly (on a deepcopy, since
        orient_channels mutates l0.data), then applies the same blue -> red FFI
        orientation so the L0 display matches the assembled output.
        """
        chip = chip.upper()

        # Count amplifiers via ImageAssembly (non-destructive).
        ia = ImageAssembly(self.kpf_obj)
        ia.count_amplifiers(chip)
        namp = ia.namp[chip]

        if namp == 2:
            image = np.concatenate(
                (self.kpf_obj.data[f"{chip}_AMP1"], self.kpf_obj.data[f"{chip}_AMP2"]),
                axis=1,
            )
        elif namp == 4:
            # orient_channels mutates l0.data, so operate on a copy.
            l0_copy = deepcopy(self.kpf_obj)
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

    def stitched_image(self, chip, *, full_res=None):
        """Plot the stitched raw detector image for one CCD.

        Replicates v2.12 plot_L0_stitched_image.

        Parameters
        ----------
        chip : str
            'green' or 'red'.
        full_res : bool or None
            Save a native-size, one-output-pixel-per-CCD-pixel PNG when true.
            None uses the constructor's ``full_res`` setting.

        Returns
        -------
        matplotlib.Figure
            The stitched-image figure.
        """
        if full_res is None:
            full_res = self.full_res

        chip_upper = chip.upper()

        image = self._stitch(chip_upper)
        # Stats run on the unmasked pixels (mask -> NaN), mirroring the L1 path;
        # nan* over the masked array would otherwise ignore the mask and warn.
        # imshow keeps the masked ``image`` so masked pixels still render as blank.
        finite = np.ma.filled(image, np.nan)

        # Legacy data format: values stored with extra 2^16 factor
        twotosixteen = False
        if np.nanmedian(finite) > 200 * 2**16:
            twotosixteen = True
            image = image / 2**16
            finite = finite / 2**16

        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        cmap = "viridis"
        vmin = np.nanpercentile(finite, 1)
        vmax = np.nanpercentile(finite, 99.5)
        plt.imshow(
            image,
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
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

        timestamp_label = f"KPF QLP: {self._timestamp()} UT"
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
            if full_res:
                fig_path = os.path.join(
                    self.output_dir,
                    f"{self.obs_id}_L0_stitched_image_{chip.lower()}_full_res.png",
                )
                save_image_png(
                    image,
                    fig_path,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    compress_level=1,
                )
            else:
                fig_path = os.path.join(
                    self.output_dir,
                    f"{self.obs_id}_L0_stitched_image_{chip.lower()}_zoomable.png",
                )
                save_png(fig, fig_path, dpi=150, compress_level=1)

        return fig
