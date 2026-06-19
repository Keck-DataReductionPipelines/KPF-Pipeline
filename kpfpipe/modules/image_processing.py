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
    Apply calibration corrections to an assembled KPF L1 frame.

    Implements bias and dark subtraction. Flat division will be added in a
    future update; requesting it now raises NotImplementedError.

    Parameters
    ----------
    l1_obj : KPF1
        Assembled L1 frame. The PRIMARY header must contain the {BIAS,DARK}FILE
        and {BIAS,DARK}DIR keywords (written by CalibrationAssociation) for any
        correction requested via the header lookup.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: bias, dark, flat
        (boolean flags toggling each correction).
    """

    # Loaded masters, keyed by absolute path and shared across instances.
    # Masters are read-only after loading, so caching them avoids re-reading
    # the same file for every frame in a stack. Cleared via clear_master_cache.
    _master_cache = {}

    @classmethod
    def clear_master_cache(cls):
        """Drop all cached master frames (used for test isolation)."""
        cls._master_cache.clear()

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

        self._bias_path = None  # set by load_bias()
        self._dark_path = None  # set by load_dark()
        self._results = None  # populated by perform()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _load_master(cls, master_path):
        """
        Load a master frame from an explicit path, caching by absolute path.

        The single FITS-read chokepoint for masters; `BaseMasterModule` also
        delegates here so the cache is shared across the pipeline.

        Parameters
        ----------
        master_path : str
            Path to the master L1 FITS file.

        Returns
        -------
        KPFMasterL1
            The loaded master (a cached instance on a repeat path).

        Raises
        ------
        FileNotFoundError
            If `master_path` does not exist on disk.
        """
        if not os.path.isfile(master_path):
            raise FileNotFoundError(f"Master file not found: {master_path}")

        key = os.path.abspath(master_path)
        if key not in cls._master_cache:
            cls._master_cache[key] = KPFMasterL1.from_fits(master_path)
        return cls._master_cache[key]

    def _header_master_path(self, cal_type):
        """
        Build a master path from the {PREFIX}FILE/{PREFIX}DIR PRIMARY headers.

        `cal_type` is the lowercase correction name ('bias' or 'dark'); the
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

        See `perform` for accepted input types. Records `self._{cal_type}_path`
        (after a successful load) so downstream reporting reflects what was used.
        """
        if isinstance(value, KPFMasterL1):
            dirname = getattr(value, "dirname", "") or ""
            filename = getattr(value, "filename", None)
            path = os.path.join(dirname, filename) if filename else None
            setattr(self, f"_{cal_type}_path", path)
            return value

        if isinstance(value, str):
            path = value
        elif value is True:
            path = self._header_master_path(cal_type)
        else:
            raise TypeError(
                f"{cal_type} must be bool, filepath str, or KPFMasterL1; "
                f"got {type(value).__name__}"
            )

        master = self._load_master(path)
        setattr(self, f"_{cal_type}_path", path)
        return master

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
        return self._resolve_master(
            "bias", bias_path if bias_path is not None else True
        )

    def load_dark(self, dark_path=None):
        """
        Load the master dark frame from disk.

        Mirrors `load_bias`: uses `dark_path` if given, otherwise builds the
        path from DARKDIR and DARKFILE in the L1 PRIMARY header.

        Parameters
        ----------
        dark_path : str, optional
            Explicit path to the master dark FITS file. When given, DARKFILE
            and DARKDIR headers are ignored.

        Returns
        -------
        KPFMasterL1
            Master dark frame loaded from disk.

        Raises
        ------
        FileNotFoundError
            If DARKFILE or DARKDIR is absent from the PRIMARY header (when
            dark_path is not provided), or if the file does not exist on disk.
        """
        return self._resolve_master(
            "dark", dark_path if dark_path is not None else True
        )

    def subtract_bias(self, bias_l1, chip):
        """
        Subtract master bias image from the CCD data for a single chip.

        Parameters
        ----------
        bias_l1 : KPFMasterL1
            Master bias frame loaded from disk.
        chip : str
            CCD identifier, e.g. 'GREEN' or 'RED'.

        Returns
        -------
        None
            Modifies `l1_obj.data['{chip}_CCD']` in-place.
        """
        chip = chip.upper()
        self.l1_obj.data[f"{chip}_CCD"] -= bias_l1.data[f"{chip}_IMG"]

    def subtract_dark(self, dark_l1, chip):
        """
        Subtract the master dark from the CCD data for a single chip.

        The master dark is a rate (electrons/sec), so it is scaled by the
        frame's exposure time before subtraction. Otherwise identical to
        `subtract_bias`.

        Parameters
        ----------
        dark_l1 : KPFMasterL1
            Master dark frame loaded from disk (in electrons/sec).
        chip : str
            CCD identifier, e.g. 'GREEN' or 'RED'.

        Returns
        -------
        None
            Modifies `l1_obj.data['{chip}_CCD']` in-place.
        """
        chip = chip.upper()
        exptime = self.l1_obj.headers["PRIMARY"]["EXPTIME"]
        self.l1_obj.data[f"{chip}_CCD"] -= dark_l1.data[f"{chip}_IMG"] * exptime

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, chips=None, *, bias=None, dark=None, flat=None):
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
            Propagated from load_bias()/load_dark() if a master cannot be located.
        NotImplementedError
            If flat is truthy.
        TypeError
            If `bias` or `dark` is not bool, str, or KPFMasterL1.
        """
        if chips is None:
            chips = self.chips
        if bias is None:
            bias = self.bias
        if dark is None:
            dark = self.dark
        if flat is None:
            flat = self.flat

        if flat:
            raise NotImplementedError("flat division not yet implemented")

        self._results = {}
        if bias:
            master_bias = self._resolve_master("bias", bias)
            for chip in chips:
                self.subtract_bias(master_bias, chip)
            self._results["bias"] = self._bias_path

        # Dark subtraction follows bias: the bias offset must be removed before
        # the (exposure-scaled) dark current is subtracted.
        if dark:
            master_dark = self._resolve_master("dark", dark)
            for chip in chips:
                self.subtract_dark(master_dark, chip)
            self._results["dark"] = self._dark_path

        self.l1_obj.headers["PRIMARY"]["BIASUB"] = (
            bool(bias),
            "Bias subtraction applied",
        )
        self.l1_obj.headers["PRIMARY"]["DARKSUB"] = (
            bool(dark),
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
