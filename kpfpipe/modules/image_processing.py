"""
KPF Image Processing module.
"""
import os

from kpfpipe import DEFAULTS
from kpfpipe.data_models.masters.level1 import KPFMasterL1
from kpfpipe.utils.config import ConfigHandler

_DEFAULTS = {**DEFAULTS,
    'bias': True,
    'dark': False,
    'flat': False,
}


class ImageProcessing:
    """
    Apply calibration corrections to an assembled KPF L1 frame.

    Currently implements bias subtraction only. Dark subtraction and
    flat division will be added in future updates; requesting them now
    raises NotImplementedError.

    Parameters
    ----------
    l1_obj : KPF1
        Assembled L1 frame. The PRIMARY header must contain BIASFILE
        and BIASDIR keywords (written by CalibrationAssociation).
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: bias, dark, flat
        (boolean flags toggling each correction).
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

        for k, v in _DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self._bias_path = None  # set by load_bias()
        self._results = None    # populated by perform()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_bias(self, value):
        """Resolve a `bias` kwarg into a KPFMasterL1 instance.

        See perform() for accepted input types. Updates self._bias_path
        so downstream reporting reflects what was actually used.
        """
        if isinstance(value, KPFMasterL1):
            dirname = getattr(value, 'dirname', '') or ''
            filename = getattr(value, 'filename', None)
            self._bias_path = os.path.join(dirname, filename) if filename else None
            return value
        if isinstance(value, str):
            return self.load_bias(bias_path=value)
        if value is True:
            return self.load_bias()
        raise TypeError(
            f"bias must be bool, filepath str, or KPFMasterL1; got {type(value).__name__}"
        )

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

    def subtract_bias(self, bias_l1, chip):
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
        self.l1_obj.data[f'{chip}_CCD'] -= bias_l1.data[f'{chip}_IMG']

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, chips=None, bias=None, dark=None, flat=None):
        """
        Run image processing corrections on the L1 frame.

        Parameters
        ----------
        chips : list of str, optional
            CCD chips to process. Defaults to self.chips.
        bias : bool | str | KPFMasterL1, optional
            How to source the master bias. Falsy → skip. True → load via
            BIASFILE/BIASDIR in the PRIMARY header. str → treat as an
            explicit filepath. KPFMasterL1 → use this object directly
            (no disk I/O). Defaults to self.bias.
        dark : bool | str | KPFMasterL1, optional
            Same shape as `bias`. Any truthy value raises
            NotImplementedError until dark subtraction is built out.
        flat : bool | str | KPFMasterL1, optional
            Same shape as `bias`. Any truthy value raises
            NotImplementedError until flat division is built out.

        Returns
        -------
        KPF1
            The input L1 frame with calibrations applied in-place and
            a receipt entry added.

        Raises
        ------
        FileNotFoundError
            Propagated from load_bias() if the master bias cannot be located.
        NotImplementedError
            If dark or flat is truthy.
        TypeError
            If `bias` is not bool, str, or KPFMasterL1.
        """
        if chips is None:
            chips = self.chips
        if bias is None:
            bias = self.bias
        if dark is None:
            dark = self.dark
        if flat is None:
            flat = self.flat

        if dark:
            raise NotImplementedError("dark subtraction not yet implemented")
        if flat:
            raise NotImplementedError("flat division not yet implemented")

        self._results = {}
        if bias:
            master_bias = self._resolve_bias(bias)
            for chip in chips:
                self.subtract_bias(master_bias, chip)
            self._results['bias'] = self._bias_path

        self.l1_obj.headers['PRIMARY']['BIASUB'] = (bool(bias), 'Bias subtraction applied')
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
