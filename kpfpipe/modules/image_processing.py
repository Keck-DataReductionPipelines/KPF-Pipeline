"""
KPF Image Processing module.

Applies the standard CCD calibration sequence to an assembled L1 frame, in
order: bias subtraction, then dark subtraction (an exposure-scaled rate), then
flat division. Which steps run is set by the bias/dark/flat flags, resolved
DEFAULTS < [MODULE_IMAGE_PROCESSING] config < perform() kwargs. The masters
modules drive these same flags per file type (see kpfpipe/modules/masters);
science applies the full sequence. Flat division is not yet implemented.
"""

import os

from kpfpipe import DEFAULTS
from kpfpipe.data_models.masters.level1 import KPFMasterL1
from kpfpipe.utils.config import ConfigHandler

_DEFAULTS = {
    **DEFAULTS,
    "bias": True,
    "dark": True,
    "flat": False,  # flat division not yet implemented
}


class ImageProcessing:
    """
    Apply calibrations to an assembled KPF L1 frame.

    Implements bias and dark subtraction. Flat division will be added in a
    future update; requesting it now raises NotImplementedError.

    Parameters
    ----------
    l1_obj : KPF1
        Assembled L1 frame. The PRIMARY header must contain the {BIAS,DARK}FILE
        and {BIAS,DARK}DIR keywords (written by CalibrationAssociation) for any
        calibration requested via the header lookup.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: bias, dark, flat
        (boolean flags toggling each calibration).
    """

    def __init__(self, l1_obj, config=None):
        self.l1_obj = l1_obj

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(
                ["DATA_DIRS", "KPFPIPE", "MODULE_IMAGE_PROCESSING"]
            )
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in _DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        # Resolved masters and their paths, cached per instance by
        # _resolve_master() during perform() so a master is read at most once.
        self._bias_ml1 = None
        self._dark_ml1 = None
        self._bias_path = None
        self._dark_path = None
        self._results = None  # populated by perform()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _load_master(cls, master_path):
        """
        Load a master frame from an explicit path.

        The single FITS-read chokepoint for masters; `BaseMasterModule` also
        delegates here.

        Parameters
        ----------
        master_path : str
            Path to the master L1 FITS file.

        Returns
        -------
        KPFMasterL1
            The loaded master.

        Raises
        ------
        FileNotFoundError
            If `master_path` does not exist on disk.
        """
        if not os.path.isfile(master_path):
            raise FileNotFoundError(f"Master file not found: {master_path}")

        return KPFMasterL1.from_fits(master_path)

    def _header_master_path(self, cal_type):
        """
        Build a master path from the {PREFIX}FILE/{PREFIX}DIR PRIMARY headers.

        `cal_type` is the lowercase calibration name ('bias' or 'dark'); the
        header prefix is its uppercase form (BIAS, DARK), written by
        CalibrationAssociation.
        """
        prefix = cal_type.upper()
        header = self.l1_obj.headers["PRIMARY"]
        master_file = header.get(f"{prefix}FILE")
        master_dir = header.get(f"{prefix}DIR")

        if not master_file or not master_dir:
            raise FileNotFoundError(
                f"{prefix}FILE and {prefix}DIR must be present in the L1 PRIMARY "
                "header. Run CalibrationAssociation before ImageProcessing."
            )

        return os.path.join(master_dir, master_file)

    def _resolve_master(self, cal_type, value):
        """
        Resolve a `bias`/`dark` kwarg into a `KPFMasterL1` instance.

        See `perform` for accepted input types. The resolved master is cached
        on `self._{cal_type}_ml1` (with its path on `self._{cal_type}_path`) so
        repeat calls — e.g. once per chip — reuse it instead of re-reading.
        """
        cached = getattr(self, f"_{cal_type}_ml1")
        if cached is not None:
            return cached

        if isinstance(value, KPFMasterL1):
            dirname = getattr(value, "dirname", "") or ""
            filename = getattr(value, "filename", None)
            path = os.path.join(dirname, filename) if filename else None
            master = value
        elif isinstance(value, str):
            path = value
            master = self._load_master(path)
        elif value is True:
            path = self._header_master_path(cal_type)
            master = self._load_master(path)
        else:
            raise TypeError(
                f"{cal_type} must be bool, filepath str, or KPFMasterL1; "
                f"got {type(value).__name__}"
            )

        setattr(self, f"_{cal_type}_path", path)
        setattr(self, f"_{cal_type}_ml1", master)
        return master

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def subtract_bias(self, chip, bias=None):
        """
        Subtract the master bias image from the CCD data for a single chip.

        Parameters
        ----------
        chip : str
            CCD identifier, e.g. 'GREEN' or 'RED'.
        bias : bool | str | KPFMasterL1, optional
            Master bias source, resolved via `_resolve_master` (see `perform`
            for the accepted input types). Defaults to self.bias.

        Returns
        -------
        None
            Modifies `l1_obj.data['{chip}_CCD']` in-place.
        """
        if bias is None:
            bias = self.bias
        bias_l1 = self._resolve_master("bias", bias)
        chip = chip.upper()
        self.l1_obj.data[f"{chip}_CCD"] -= bias_l1.data[f"{chip}_IMG"]

    def subtract_dark(self, chip, dark=None):
        """
        Subtract the master dark from the CCD data for a single chip.

        The master dark is a rate (electrons/sec), so it is scaled by the
        frame's exposure time before subtraction. Otherwise identical to
        `subtract_bias`.

        Parameters
        ----------
        chip : str
            CCD identifier, e.g. 'GREEN' or 'RED'.
        dark : bool | str | KPFMasterL1, optional
            Master dark source, resolved via `_resolve_master` (see `perform`
            for the accepted input types). Defaults to self.dark.

        Returns
        -------
        None
            Modifies `l1_obj.data['{chip}_CCD']` in-place.
        """
        if dark is None:
            dark = self.dark
        dark_l1 = self._resolve_master("dark", dark)
        chip = chip.upper()
        exptime = self.l1_obj.headers["PRIMARY"]["EXPTIME"]
        self.l1_obj.data[f"{chip}_CCD"] -= dark_l1.data[f"{chip}_IMG"] * exptime

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, chips=None, *, bias=None, dark=None, flat=None):
        """
        Run image processing calibrations on the L1 frame.

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
            Same shape as `bias`, sourced from DARKFILE/DARKDIR. Applied
            after bias subtraction and scaled by the frame's exposure time.
            Defaults to self.dark.
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
            Propagated from _resolve_master() if a master cannot be located.
        NotImplementedError
            If flat is truthy.
        TypeError
            If `bias` or `dark` is not bool, str, or KPFMasterL1.
        """
        # Per-call kwargs override the instance config; subtract_bias/
        # subtract_dark then read the resolved sources from self.
        if chips is not None:
            self.chips = chips
        if bias is not None:
            self.bias = bias
        if dark is not None:
            self.dark = dark
        if flat is not None:
            self.flat = flat

        if self.flat:
            raise NotImplementedError("flat division not yet implemented")

        self._results = {}
        if self.bias:
            for chip in self.chips:
                self.subtract_bias(chip)
            self._results["bias"] = self._bias_path

        if self.dark:
            for chip in self.chips:
                self.subtract_dark(chip)
            self._results["dark"] = self._dark_path

        self.l1_obj.headers["PRIMARY"]["BIASUB"] = (
            bool(self.bias),
            "Bias subtraction applied",
        )
        self.l1_obj.headers["PRIMARY"]["DARKSUB"] = (
            bool(self.dark),
            "Dark subtraction applied",
        )
        self.l1_obj.receipt_add_entry("image_processing", "PASS")

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
