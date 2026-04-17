"""L1 quicklook plots for assembled KPF 2D frames."""

import os
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt


class PlotL1:
    """
    Quicklook plots for KPF L1 (assembled 2D) data.

    Takes a KPF1 object and generates plots of the assembled detector image.
    Pure visualization — no science computation.

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

    _RN_KEYS = {
        'GREEN': [('RNGREEN1', 'RNNGGR1'),
                  ('RNGREEN2', 'RNNGGR2'),
                  ('RNGREEN3', 'RNNGGR3'),
                  ('RNGREEN4', 'RNNGGR4')],
        'RED':   [('RNRED1', 'RNNGRD1'),
                  ('RNRED2', 'RNNGRD2'),
                  ('RNRED3', 'RNNGRD3'),
                  ('RNRED4', 'RNNGRD4')],
    }

    def _read_noise_values(self, chip):
        """Return (rn_list, rnng_list) from PRIMARY header, or ([], []) if absent."""
        primary = self.l1.headers.get('PRIMARY', {})
        rn_values = []
        rnng_values = []
        for rn_key, rnng_key in self._RN_KEYS[chip.upper()]:
            if rn_key in primary and rnng_key in primary:
                rn_values.append(float(primary[rn_key]))
                rnng_values.append(float(primary[rnng_key]))
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
