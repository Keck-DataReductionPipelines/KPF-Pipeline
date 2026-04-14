"""L0 quicklook plots for raw KPF detector images."""

import os
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt


class PlotL0:
    """
    Quicklook plots for KPF L0 (raw CCD) data.

    Takes a KPF0 object and generates plots of the raw detector images.
    Pure visualization — no science computation.

    Args:
        l0_obj: KPF0 data object.
        output_dir: Directory to save PNG files. None = return Figure only.
    """

    def __init__(self, l0_obj, output_dir=None):
        self.l0 = l0_obj
        self.output_dir = output_dir
        self.obs_id = getattr(l0_obj, 'obs_id', None) or ''
        self.name = ''
        if 'PRIMARY' in l0_obj.headers:
            self.name = l0_obj.headers['PRIMARY'].get('OBJECT', '')

    def _count_amps(self, chip):
        """Count amplifier extensions present for a chip."""
        chip = chip.upper()
        count = 0
        for i in range(1, 5):
            ext = f'{chip}_AMP{i}'
            if ext in self.l0.data and self.l0.data[ext] is not None:
                if np.size(self.l0.data[ext]) > 0:
                    count += 1
        return count

    def _stitch(self, chip):
        """Concatenate raw amplifier arrays into a single display image."""
        chip = chip.upper()
        namp = self._count_amps(chip)

        if namp == 2:
            image = np.concatenate(
                (self.l0.data[f'{chip}_AMP1'], self.l0.data[f'{chip}_AMP2']),
                axis=1,
            )
            if chip == 'GREEN':
                image = np.flipud(image)
        elif namp == 4:
            bot = np.concatenate(
                (self.l0.data[f'{chip}_AMP1'], self.l0.data[f'{chip}_AMP2']),
                axis=1,
            )
            top = np.concatenate(
                (self.l0.data[f'{chip}_AMP3'], self.l0.data[f'{chip}_AMP4']),
                axis=1,
            )
            image = np.concatenate((bot, top), axis=0)
        else:
            raise ValueError(
                f"Unsupported amplifier count ({namp}) for {chip} CCD. "
                f"Expected 2 or 4."
            )

        return image

    def stitched_image(self, chip):
        """
        Plot the stitched raw detector image for one CCD.

        Replicates plot_L0_stitched_image from v2.12.

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
            image, cmap='viridis', origin='lower',
            vmin=np.percentile(image, 1),
            vmax=np.percentile(image, 99.5),
        )

        plt.title(
            f'L0 - {chip_upper.capitalize()} CCD: {self.obs_id} - {self.name}',
            fontsize=14,
        )
        plt.xlabel('Column (pixel number)', fontsize=14)
        plt.ylabel('Row (pixel number)', fontsize=14)

        cbar_label = 'ADU'
        if twotosixteen:
            cbar_label += r' / $2^{16}$'
        cbar = plt.colorbar(shrink=0.95, label=cbar_label)
        cbar.ax.yaxis.label.set_size(14)
        cbar.ax.tick_params(labelsize=12)
        plt.grid(False)

        # Timestamp annotation
        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f'KPF QLP: {current_time} UT'
        plt.annotate(
            timestamp_label, xy=(1, 0), xycoords='axes fraction',
            fontsize=8, color='darkgray', ha='right', va='top',
            xytext=(100, -21), textcoords='offset points',
        )
        plt.subplots_adjust(bottom=0.1)

        if self.output_dir is not None:
            fig_path = os.path.join(
                self.output_dir,
                f'{self.obs_id}_L0_stitched_image_{chip.lower()}_zoomable.png',
            )
            fig.savefig(fig_path, dpi=600, facecolor='w')

        return fig
