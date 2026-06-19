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

    Standard reduction: a bias receives no calibration correction
    (`_STANDARD_CORRECTIONS` is empty), so raw frames are stacked as-is.

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
            params = config.get_params(
                ["DATA_DIRS", "KPFPIPE", "BIAS", "MODULE_IMAGE_PROCESSING"]
            )
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")
        super().__init__(l0_file_list, params)

        self._results = None  # populated by make_master_l1()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def make_master_l1(
        self,
        l0_file_list=None,
        *,
        nstream=None,
        sigma=None,
        bias=None,
        dark=None,
        flat=None,
        filepath=None,
        verbose=True,
    ):
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
        bias, dark, flat : bool | str | KPFMasterL1, optional
            Per-call correction overrides (same forms as ImageProcessing.perform:
            bool, a master filepath, or a KPFMasterL1 object), clamped by the
            master's standard. A bias has no standard corrections, so these are
            no-ops here.
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

        self._active_corrections = self._resolve_corrections(
            bias=bias, dark=dark, flat=flat
        )

        l1_arrays = self.stack_frames(
            l0_file_list=l0_file_list,
            nstream=nstream,
            sigma=sigma,
            verbose=verbose,
        )
        l1_arrays = self.clean_l1_arrays(l1_arrays, sigma)

        self.ml1_obj = self.build_ml1_obj(
            l1_arrays, l0_file_list, receipt_key="master_bias"
        )
        self._results = self._compute_results(l1_arrays)

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

        if self._results is None:
            print("  make_master_l1() has not been called")
            return

        print(f"\n  {'chip':<8s} {'median [e-]':<15s} {'rms [e-]':<10s} {'bad pixels'}")
        print("  " + "-" * 56)
        for chip, stats in self._results.items():
            print(
                f"  {chip:<8s} {stats['median']:<15.4f} {stats['rms']:<10.4f} "
                f"{stats['num_bad']} ({stats['pct_bad']:.3f}%)"
            )
