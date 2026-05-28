"""
KPF Master Bias construction module.
"""
import numpy as np

from kpfpipe import DEFAULTS
from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.modules.masters.base import BaseMasterModule
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import flag_outliers, interpolate_bad_pixels

DEFAULTS.update({'stack_sigma': 5.0})


class Bias(BaseMasterModule):
    """
    Construct a master bias frame from a stack of L0 bias exposures.

    Stacks frames using sigma-clipped statistics, interpolates bad pixels,
    and performs a final outlier pass on the combined image. Outputs a
    KPFMasterL1 containing per-chip IMG, SNR, and MASK extensions.

    Parameters
    ----------
    l0_file_list : list of str
        Sorted list of L0 FITS file paths to stack.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: nframe_stream, stack_sigma,
        exptime_tolerance, chips.
    """
    def __init__(self, l0_file_list, config=None):
        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS", "KPFPIPE", "BIAS"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")
        super().__init__(l0_file_list, params)

        self._results = None  # populated by make_master_l1()


    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def make_master_l1(self, l0_file_list=None, nstream=None, sigma=None,
                       filepath=None, verbose=True):
        """
        Build master bias from stack.

        The constructed KPFMasterL1 is returned and cached on
        `self.ml1_obj`; pass `filepath` to also persist it to disk
        via `save_master('L1', ...)`.

        Parameters
        ----------
        l0_file_list : list of str, optional
            L0 files to stack. Defaults to self.l0_file_list.
        nstream : int, optional
            Stream threshold passed to stack_frames.
        sigma : float, optional
            Outlier rejection threshold passed to stack_frames.
        filepath : str, optional
            If provided, calls `self.save_master('L1', filepath)` at
            the end to persist the master L1 to a FITS file at this filepath.
        verbose : bool, optional
            If True (default), emit per-frame progress prints during stacking.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if nstream is None:
            nstream = self.nframe_stream
        if sigma is None:
            sigma = self.stack_sigma

        l1_arrays = self.stack_frames(
            l0_file_list=l0_file_list,
            nstream=nstream,
            sigma=sigma,
            verbose=verbose,
        )

        for chip in self.chips:
            img = l1_arrays[f'{chip}_IMG']
            snr = l1_arrays[f'{chip}_SNR']
            mask = l1_arrays[f'{chip}_MASK']

            l1_arrays[f'{chip}_IMG'] = interpolate_bad_pixels(img, mask)
            l1_arrays[f'{chip}_SNR'] = interpolate_bad_pixels(snr, mask)

            out = flag_outliers(l1_arrays[f'{chip}_IMG'], sigma, axis=0)
            bad = ((l1_arrays[f'{chip}_SNR'] <= 0) | (l1_arrays[f'{chip}_IMG'] == 0))

            l1_arrays[f'{chip}_MASK'] = ~(bad | out)

        self.ml1_obj = KPFMasterL1()

        for chip in self.chips:
            self.ml1_obj.set_data(f'{chip}_IMG',  l1_arrays[f'{chip}_IMG'])
            self.ml1_obj.set_data(f'{chip}_SNR',  l1_arrays[f'{chip}_SNR'])
            self.ml1_obj.set_data(f'{chip}_MASK', l1_arrays[f'{chip}_MASK'])

        self.ml1_obj.set_input_files(l0_file_list)
        self.ml1_obj.receipt_add_entry('master_bias', 'PASS')

        self._results = {
            chip: {
                'num_bad':   int(np.sum(~l1_arrays[f'{chip}_MASK'])),
                'pct_bad': float(100.0 * np.mean(~l1_arrays[f'{chip}_MASK'])),
                'median':  float(np.nanmedian(l1_arrays[f'{chip}_IMG'])),
                'rms':     float(np.nanstd(l1_arrays[f'{chip}_IMG'])),
            }
            for chip in self.chips
        }

        if filepath is not None:
            self.save_master('L1', filepath, overwrite=True)

        return self.ml1_obj

    def info(self):
        """Print a summary of the module configuration and stacking results."""
        print("Bias")
        print(f"  l0_file_list:")
        for fn in self.l0_file_list:
            print(f"    {fn}")
        print(f"  chips:  {self.chips}")

        if self._results is None:
            print("  make_master_l1() has not been called")
            return

        print(f"\n  {'chip':<8s} {'median [e-]':<15s} {'rms [e-]':<10s} {'bad pixels'}")
        print("  " + "-" * 56)
        for chip, stats in self._results.items():
            print(f"  {chip:<8s} {stats['median']:<15.4f} {stats['rms']:<10.4f} {stats['num_bad']} ({stats['pct_bad']:.3f}%)")