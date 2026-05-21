"""L1 quicklook plots for assembled FFI."""

import os
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt

from kpfpipe.modules.image_assembly import _RN_KEYS


def _unwrap(val):
    return val[0] if isinstance(val, tuple) else val


class PlotL1:
    """
    Quicklook plots for KPF L1 (assembled FFI) data.

    Takes a KPF1 object and generates plots of the assembled detector image.
    Pure visualization — no science computation. Read-noise header mapping
    is imported from ImageAssembly so there is one source of truth for which
    FITS keywords hold which amplifier's read noise.

    Args:
        l1_obj: KPF1 data object (post-ImageAssembly).
        output_dir: Directory to save PNG files. None = return Figure only.
    """

    def __init__(self, l1_obj, output_dir=None):
        self.l1 = l1_obj
        self.output_dir = output_dir
        self.obs_id = getattr(l1_obj, 'obs_id', None) or ''
        self.name = ''
        if 'PRIMARY' in l1_obj.headers:
            self.name = l1_obj.headers['PRIMARY'].get('OBJECT', '')

    def _read_noise_values(self, chip):
        """Return (rn_list, rnng_list) from PRIMARY header, or ([], []) if absent."""
        primary = self.l1.headers.get('PRIMARY', {})
        rn_values = []
        rnng_values = []
        for i in range(1, 5):
            channel_ext = f'{chip.upper()}_AMP{i}'
            rn_key, rnng_key = _RN_KEYS[channel_ext]
            if rn_key in primary and rnng_key in primary:
                rn_values.append(float(_unwrap(primary[rn_key])))
                rnng_values.append(float(_unwrap(primary[rnng_key])))
        return rn_values, rnng_values

    def image(self, chip):
        """
        Plot the assembled 2D detector image for one CCD.

        Replicates v2.12 plot_2D_image for the basic science-frame case.

        Args:
            chip: 'green' or 'red'.

        Returns:
            matplotlib.Figure
        """
        chip_upper = chip.upper()
        ext = f'{chip_upper}_CCD'
        image = np.asarray(self.l1.data[ext])

        # Interior percentile (strip 100-pixel border to avoid edge effects)
        interior = image[100:-100, 100:-100] if min(image.shape) > 200 else image
        vmin = np.nanpercentile(interior, 0.1)
        vmax = np.nanpercentile(interior, 99.0)

        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        plt.imshow(
            image, cmap='viridis', origin='lower', interpolation='None',
            vmin=vmin, vmax=vmax,
        )

        plt.title(
            f'L1 - {chip_upper.capitalize()} CCD: {self.obs_id} - {self.name}',
            fontsize=14,
        )
        plt.xlabel('Column (pixel number)', fontsize=18)
        plt.ylabel('Row (pixel number)', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        cbar = plt.colorbar(label='Counts (e-)')
        cbar.ax.yaxis.label.set_size(18)
        cbar.ax.tick_params(labelsize=14)
        plt.grid(False)

        # Read noise annotation (from L1 headers populated by ImageAssembly)
        rn_values, rnng_values = self._read_noise_values(chip_upper)
        if rn_values:
            rn_text = 'RN: ' + ', '.join(f'{v:.2f}' for v in rn_values)
            rn_text += r' e- (stdev(overscan); 10-$\sigma$ outlier rej.)'
            rn_text += '\nNon-Gaussian RN: ' + ', '.join(f'{v:.3f}' for v in rnng_values)
            rn_text += r' (0.80$\times$stdev/mad in overscan)'
            plt.annotate(
                rn_text, xy=(0, 0), xycoords='axes fraction',
                fontsize=8, color='darkgray', ha='left', va='top',
                xytext=(-50, -21), textcoords='offset points',
            )

        # Timestamp
        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        plt.annotate(
            f'KPF QLP: {current_time} UT', xy=(1, 0), xycoords='axes fraction',
            fontsize=8, color='darkgray', ha='right', va='top',
            xytext=(100, -21), textcoords='offset points',
        )
        plt.subplots_adjust(bottom=0.1)

        if self.output_dir is not None:
            fig_path = os.path.join(
                self.output_dir,
                f'{self.obs_id}_L1_image_{chip.lower()}_zoomable.png',
            )
            fig.savefig(fig_path, dpi=600, facecolor='w')

        return fig

    def histogram(self, chip):
        """Flux distribution histogram. Matches v2.12 plot_2D_image_histogram."""
        raise NotImplementedError("PlotL1.histogram not yet implemented")

    def column_cut(self, chip):
        """Median column profiles at 10th/50th/90th percentiles.
        Matches v2.12 plot_2D_column_cut."""
        raise NotImplementedError("PlotL1.column_cut not yet implemented")

    def zoom_3x3(self, chip):
        """3x3 grid of zoomed detector regions. Matches v2.12 plot_2D_image_zoom_3x3."""
        raise NotImplementedError("PlotL1.zoom_3x3 not yet implemented")

    def order_trace_overlay(self, chip):
        """2x2 grid with order trace overlaid. Matches v2.12 plot_2D_order_trace2x2.
        Requires calibration DB lookup for order trace file (not yet in v3)."""
        raise NotImplementedError("PlotL1.order_trace_overlay not yet implemented")

    def bias_subtracted(self, chip):
        """Image with master bias subtracted. Matches v2.12 plot_2D_image(subtract_master_bias=True)."""
        raise NotImplementedError("PlotL1.bias_subtracted not yet implemented")

    def dark_subtracted(self, chip):
        """Image with master dark subtracted, displayed in e-/hr.
        Matches v2.12 plot_2D_image(subtract_master_dark=True)."""
        raise NotImplementedError("PlotL1.dark_subtracted not yet implemented")

    def _has_chip(self, chip):
        ext = f'{chip.upper()}_CCD'
        if ext not in self.l1.data or self.l1.data[ext] is None:
            return False
        return np.size(self.l1.data[ext]) > 0

    def all(self):
        """
        Generate all implemented L1 plots for chips present in the data.

        Returns:
            dict mapping plot name to matplotlib.Figure.
        """
        figures = {}
        for chip in ['green', 'red']:
            if self._has_chip(chip):
                figures[f'L1_image_{chip}'] = self.image(chip)
        return figures
