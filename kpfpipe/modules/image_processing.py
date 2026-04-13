"""
KPF Image Processing module.
"""
import os

from kpfpipe import DEFAULTS
from kpfpipe.data_models.masters.level1 import KPFMasterL1
from kpfpipe.utils.config import ConfigHandler


class ImageProcessing:
    """
    Apply calibration corrections to an assembled KPF L1 frame.

    Currently implements bias subtraction only. Dark subtraction and
    flat division will be added in future updates.

    Parameters
    ----------
    l1_obj : KPF1
        Assembled L1 frame. The PRIMARY header must contain BIASFILE
        and BIASDIR keywords (written by CalibrationAssociation).
    config : None | dict | ConfigHandler
        Module configuration. No module-specific keys at this time.
    """

    def __init__(self, l1_obj, config=None):
        self.l1_obj = l1_obj

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS", "KPFPIPE", "MODULE_IMAGE_PROCESSING"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self.chips = params.get('chips', ['GREEN', 'RED'])
        self._bias_path = None  # set by load_bias()
        self._results = None    # populated by perform()

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def load_bias(self, bias_path=None):
        """
        Load the master bias frame from disk.

        If bias_path is provided it is used directly, bypassing the header
        lookup. Otherwise the path is constructed from BIASDIR and BIASFILE
        in the L1 PRIMARY header (written by CalibrationAssociation).

        Parameters
        ----------
        bias_path : str, optional
            Explicit path to the master bias FITS file. When given, BIASFILE
            and BIASDIR headers are ignored.

        Returns
        -------
        KPFMasterL1
            Master bias frame loaded from disk.

        Raises
        ------
        FileNotFoundError
            If BIASFILE or BIASDIR is absent from the PRIMARY header (when
            bias_path is not provided), or if the file does not exist on disk.
        """
        if bias_path is None:
            header = self.l1_obj.headers['PRIMARY']
            bias_file = header.get('BIASFILE')
            bias_dir  = header.get('BIASDIR')

            if not bias_file or not bias_dir:
                raise FileNotFoundError(
                    "BIASFILE and BIASDIR must be present in the L1 PRIMARY header. "
                    "Run CalibrationAssociation before ImageProcessing."
                )

            bias_path = os.path.join(bias_dir, bias_file)

        if not os.path.isfile(bias_path):
            raise FileNotFoundError(f"Master bias file not found: {bias_path}")

        self._bias_path = bias_path
        return KPFMasterL1.from_fits(bias_path)

    def subtract_bias(self, master_bias, chip):
        """
        Subtract master bias image from the CCD data for a single chip.

        Parameters
        ----------
        master_bias : KPFMasterL1
            Master bias frame loaded from disk.
        chip : str
            CCD identifier, e.g. 'GREEN' or 'RED'.

        Returns
        -------
        None
            Modifies l1_obj.data['{chip}_CCD'] in-place.
        """
        chip = chip.upper()
        self.l1_obj.data[f'{chip}_CCD'] -= master_bias.data[f'{chip}_IMG']

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, chips=None):
        """
        Run image processing corrections on the L1 frame.

        Parameters
        ----------
        chips : list of str, optional
            CCD chips to process. Defaults to self.chips.

        Returns
        -------
        KPF1
            The input L1 frame with calibrations applied in-place and
            a receipt entry added.

        Raises
        ------
        FileNotFoundError
            Propagated from load_bias() if the master bias cannot be located.
        """
        if chips is None:
            chips = self.chips

        master_bias = self.load_bias()

        self._results = {}
        for chip in chips:
            self.subtract_bias(master_bias, chip)
        self._results['bias'] = self._bias_path

        self.l1_obj.headers['PRIMARY']['BIASUB'] = (True, 'Bias subtraction applied')
        self.l1_obj.receipt_add_entry('image_processing', 'PASS')

        return self.l1_obj

    def info(self):
        """Print a summary of the module configuration and processing results."""
        print("ImageProcessing")
        print(f"  obs_id: {self.l1_obj.obs_id}")
        print(f"  chips:  {self.chips}")

        if self._results is None:
            print("  perform() has not been called")
            return

        print(f"\n  {'cal_type':<10s} {'master file'}")
        print("  " + "-" * 60)
        for cal_type, path in self._results.items():
            print(f"  {cal_type:<10s} {path}")
