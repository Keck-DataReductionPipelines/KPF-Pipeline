"""KPF Master Flat construction module."""

import logging

from kpfpipe.modules.masters.base import BaseMasterModule
from kpfpipe.utils.config import ConfigHandler

logger = logging.getLogger(__name__)


class Flat(BaseMasterModule):
    """
    Construct a master flat frame from a stack of L0 flat exposures.

    Stacks frames using sigma-clipped statistics, interpolates bad pixels,
    and performs a final outlier pass on the combined image. Outputs a
    KPFMasterL1 containing per-chip IMG, SNR, and MASK extensions; the IMG is
    the total electrons summed over the stack (BUNIT 'electrons').

    Standard reduction: a flat is bias- and dark-subtracted before stacking.

    Parameters
    ----------
    l0_file_list : list of str
        Sorted list of L0 FITS file paths to stack.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: stack_sigma, min_stack_size,
        chips, bias and dark (master-calibration overrides), and the
        calibration-association keys masters_search_window_days and
        KPF_MASTERS_OUTPUT.
    """

    _STANDARD_CALIBRATIONS = ("bias", "dark")

    def __init__(self, l0_file_list, config=None):
        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(
                [
                    "DATA_DIRS",
                    "TRACES",
                    "FLAT",
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
        dark=None,
        master_path=None,
    ):
        """
        Build master flat from stack.

        The constructed KPFMasterL1 is returned and cached on
        ``self.ml1_obj``; pass ``master_path`` to also persist it to disk
        via ``save_master('L1', ...)``.

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
            bool, a master filepath, or a KPFMasterL1 object). ``bias=False``
            skips bias subtraction and ``bias="/path/master_bias.fits"`` uses a
            specific master.
        dark : bool | str | KPFMasterL1, optional
            Per-call master-dark override (same forms as ``bias``). A flat's
            standard is bias- and dark-subtraction; flat division is never
            applied when building a master flat, so no flat override is accepted.
        master_path : str, optional
            If provided, persist the master L1 to a FITS file at this path.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if sigma is None:
            sigma = self.stack_sigma

        self._active_calibrations = self._resolve_calibrations(bias=bias, dark=dark)

        l1_arrays = self.stack_frames(
            l0_file_list=l0_file_list,
            nstream=nstream,
            sigma=sigma,
            cal_type="flat",
        )

        self.ml1_obj = self._build_ml1_obj(l1_arrays, l0_file_list, master_type="flat")
        self._populate_stack_info(l1_arrays)
        self._track_info()

        if master_path is not None:
            self.save_master("L1", master_path, overwrite=True)

        logger.info("%s", self._info)

        return self.ml1_obj

    def _track_info(self):
        """Build and cache the info() summary text from _stack_info."""
        lines = ["Flat", "  l0_file_list:"]
        for fn in self.l0_file_list:
            lines.append(f"    {fn}")
        lines.append(f"  chips:  {self.chips}")
        lines.append(f"\n  {'chip':<8s} {'median':<15s} {'rms':<10s} {'bad pixels'}")
        lines.append("  " + "-" * 56)
        for chip, stats in self._stack_info.items():
            lines.append(
                f"  {chip:<8s} {stats['median']:<15.4f} {stats['rms']:<10.4f} "
                f"{stats['num_bad']} ({stats['pct_bad']:.3f}%)"
            )
        self._info = "\n\n" + "\n".join(lines) + "\n\n"

    def info(self):
        """Print a summary of the module configuration and stacking results."""
        if self._info is None:
            print(f"{type(self).__name__}: make_master_l1() has not been called")
        else:
            print(self._info)
