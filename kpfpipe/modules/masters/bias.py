"""
KPF Master Bias construction module.
"""

from kpfpipe.modules.masters.base import BaseMasterModule
from kpfpipe.utils.config import ConfigHandler


class Bias(BaseMasterModule):
    """
    Construct a master bias frame from a stack of L0 bias exposures.

    Stacks frames using sigma-clipped statistics, interpolates bad pixels,
    and performs a final outlier pass on the combined image. Outputs a
    KPFMasterL1 containing per-chip IMG, SNR, and MASK extensions.

    Standard reduction: a bias receives no calibration
    (`_STANDARD_CALIBRATIONS` is empty), so raw frames are stacked as-is.

    Parameters
    ----------
    l0_file_list : list of str
        Sorted list of L0 FITS file paths to stack.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: stack_sigma,
        exptime_tolerance, chips.
    """

    def __init__(self, l0_file_list, config=None):
        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(
                ["DATA_DIRS", "KPFPIPE", "BIAS", "MODULE_IMAGE_PROCESSING"]
            )
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")
        super().__init__(l0_file_list, params)

        self._info = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def make_master_l1(
        self,
        l0_file_list=None,
        *,
        nstream=6,
        sigma=None,
        filepath=None,
        verbose=True,
    ):
        """
        Build master bias from stack.

        The constructed KPFMasterL1 is returned and cached on
        `self.ml1_obj`; pass `filepath` to also persist it to disk
        via `save_master('L1', ...)`.

        A bias receives no calibrations, so there are no bias/dark/
        flat overrides (unlike Dark/WLS).

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
        if sigma is None:
            sigma = self.stack_sigma

        l1_arrays = self.stack_frames(
            l0_file_list=l0_file_list,
            nstream=nstream,
            sigma=sigma,
            verbose=verbose,
            cal_type="bias",
        )

        self.ml1_obj = self._build_ml1_obj(l1_arrays, l0_file_list, master_type="bias")
        self._info = self._populate_info(l1_arrays)

        if filepath is not None:
            self.save_master("L1", filepath, overwrite=True)

        return self.ml1_obj

    def info(self):
        """Print a summary of the module configuration and stacking results."""
        print("Bias")
        print("  l0_file_list:")
        for fn in self.l0_file_list:
            print(f"    {fn}")
        print(f"  chips:  {self.chips}")

        if self._info is None:
            print("  make_master_l1() has not been called")
            return

        print(f"\n  {'chip':<8s} {'median [e-]':<15s} {'rms [e-]':<10s} {'bad pixels'}")
        print("  " + "-" * 56)
        for chip, stats in self._info.items():
            print(
                f"  {chip:<8s} {stats['median']:<15.4f} {stats['rms']:<10.4f} "
                f"{stats['num_bad']} ({stats['pct_bad']:.3f}%)"
            )
