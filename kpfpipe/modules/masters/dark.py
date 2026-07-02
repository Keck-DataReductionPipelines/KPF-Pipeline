"""
KPF Master Dark construction module.
"""

import logging

from kpfpipe.modules.masters.base import BaseMasterModule
from kpfpipe.utils.config import ConfigHandler

logger = logging.getLogger(__name__)


class Dark(BaseMasterModule):
    """
    Construct a master dark frame from a stack of L0 dark exposures.

    Standard reduction: a dark is bias-subtracted (`_STANDARD_CALIBRATIONS =
    ("bias",)`). The associated master bias is subtracted from each frame via
    the shared `_process_frame` hook (CalibrationAssociation + ImageProcessing)
    before stacking with sigma-clipped statistics, interpolating bad pixels,
    and performing a final outlier pass. Outputs a KPFMasterL1 containing
    per-chip IMG, SNR, and MASK extensions, with IMG in electrons/sec.

    Parameters
    ----------
    l0_file_list : list of str
        Sorted list of L0 FITS file paths to stack.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: stack_sigma,
        exptime_tolerance, chips.
    """

    _STANDARD_CALIBRATIONS = ("bias",)

    def __init__(self, l0_file_list, config=None):
        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(
                [
                    "DATA_DIRS",
                    "KPFPIPE",
                    "DARK",
                    "MODULE_CALIBRATION_ASSOCIATION",
                    "MODULE_IMAGE_PROCESSING",
                ]
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
        bias=None,
        filepath=None,
        verbose=True,
    ):
        """
        Build master dark from stack.

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
        bias : bool | str | KPFMasterL1, optional
            Per-call master-bias override (same forms as ImageProcessing.perform:
            bool, a master filepath, or a KPFMasterL1 object). A dark's standard
            is bias-only, so `bias=False` skips bias subtraction and
            `bias="/path/master_bias.fits"` uses a specific master. Dark/flat are
            never applied to a dark, so they are not accepted.
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

        self._active_calibrations = self._resolve_calibrations(bias=bias)

        l1_arrays = self.stack_frames(
            l0_file_list=l0_file_list,
            nstream=nstream,
            sigma=sigma,
            verbose=verbose,
            cal_type="dark",
        )

        # Dark current is a rate: stack_frames normalizes each frame by its
        # exposure time, so the master dark IMG is in electrons/sec (BUNIT is
        # derived from master_type in _build_ml1_obj).
        self.ml1_obj = self._build_ml1_obj(l1_arrays, l0_file_list, master_type="dark")
        self._info = self._populate_info(l1_arrays)

        if filepath is not None:
            self.save_master("L1", filepath, overwrite=True)

        logger.info("summary:\n%s", self._info_text())

        return self.ml1_obj

    def _info_text(self):
        """Build the info() report text."""
        lines = []
        lines.append("Dark")
        lines.append("  l0_file_list:")
        for fn in self.l0_file_list:
            lines.append(f"    {fn}")
        lines.append(f"  chips:  {self.chips}")

        if self._info is None:
            lines.append("  make_master_l1() has not been called")
            return "\n".join(lines)

        lines.append(
            f"\n  {'chip':<8s} {'median [e-/s]':<15s} "
            f"{'rms [e-/s]':<10s} {'bad pixels'}"
        )
        lines.append("  " + "-" * 56)
        for chip, stats in self._info.items():
            lines.append(
                f"  {chip:<8s} {stats['median']:<15.4f} {stats['rms']:<10.4f} "
                f"{stats['num_bad']} ({stats['pct_bad']:.3f}%)"
            )
        return "\n".join(lines)

    def info(self):
        """Print a summary of the module configuration and stacking results."""
        print(self._info_text())
