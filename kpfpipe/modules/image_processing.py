"""
KPF Image Processing module.

Applies the standard CCD calibration sequence to an assembled L1 frame, in
order: bias subtraction, then dark subtraction (an exposure-scaled rate), then
flat division. Which steps run is set by the bias/dark/flat flags. Flat
division is not yet implemented.
"""

import logging
import os

import numpy as np

from kpfpipe import DEFAULTS
from kpfpipe.data_models.masters.level1 import KPFMasterL1
from kpfpipe.utils.config import ConfigHandler

logger = logging.getLogger(__name__)

_DEFAULTS = {
    **DEFAULTS,
    "bias": True,
    "dark": True,
    "flat": False,  # flat division not yet implemented
}

_CALIBRATION_APPLIED_KEYS = {
    "bias": "biassub",
    "dark": "darksub",
    "flat": "flatdiv",
}


class ImageProcessing:
    """
    Apply calibrations to an assembled KPF L1 frame.

    Implements bias and dark subtraction. Flat division will be added in a
    future update; requesting it now raises NotImplementedError.

    Parameters
    ----------
    l1_obj : KPF1
        Assembled L1 frame. The RECEIPT table must carry the {bias,dark}file
        argument (the master's full path, written by CalibrationAssociation) for
        any calibration requested via the receipt lookup.
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
                ["DATA_DIRS", "TRACES", "MODULE_IMAGE_PROCESSING"]
            )
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in _DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        # Resolved masters + paths, cached by _resolve_master() so each master
        # is read at most once.
        self._bias_ml1 = None
        self._dark_ml1 = None
        self._flat_ml1 = None
        self._bias_path = None
        self._dark_path = None
        self._flat_path = None
        self._biassub = None  # applied flags for the receipt entry
        self._darksub = None
        self._flatdiv = None
        self._info = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _load_master(cls, master_path):
        """
        Load a master L1 frame from an explicit path.

        The single FITS-read chokepoint for masters (``BaseMasterModule`` also
        delegates here); raises FileNotFoundError if the path is absent.
        """
        if not os.path.isfile(master_path):
            raise FileNotFoundError(f"Master file not found: {master_path}")

        return KPFMasterL1.from_fits(master_path)

    def _resolve_master(self, cal_type, value):
        """
        Resolve a ``bias``/``dark`` kwarg into a ``KPFMasterL1`` instance.

        See ``perform`` for accepted input types. The resolved master is cached
        on ``self._{cal_type}_ml1`` (with its path on ``self._{cal_type}_path``) so
        repeat calls -- e.g. once per chip -- reuse it instead of re-reading.
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
            path = self._get_master_path(cal_type)
            master = self._load_master(path)
        else:
            raise TypeError(
                f"{cal_type} must be bool, filepath str, or KPFMasterL1; "
                f"got {type(value).__name__}"
            )

        setattr(self, f"_{cal_type}_path", path)
        setattr(self, f"_{cal_type}_ml1", master)
        return master

    def _get_master_path(self, cal_type):
        """
        Read the master path for ``cal_type`` from the L1 RECEIPT table.

        Returns the ``{cal_type}file`` argument of the calibration_association
        receipt entry, the master's full path. Raises FileNotFoundError if it is
        absent -- a ``True`` calibration was requested on a frame that has not
        been through CalibrationAssociation.
        """
        entry = self.l1_obj.receipt_read_entry("calibration_association")
        master_file = entry.get(f"{cal_type}file")
        if not master_file:
            raise FileNotFoundError(
                f"{cal_type}file must be present in the RECEIPT table. "
                "Run CalibrationAssociation before ImageProcessing."
            )
        return master_file

    @staticmethod
    def _master_image_variance(img, snr):
        """
        Recover the per-pixel variance of a master IMG from its stored SNR.

        A master stores IMG = counts / exptime and SNR = |counts| / sqrt(var),
        so var(IMG) = (IMG / SNR)**2 (= var / exptime**2) -- the master's
        contribution to the science VAR, small because a master averages many
        frames. SNR is exactly zero only at bad / zero-flux pixels, which
        contribute zero variance rather than inf/NaN.
        """
        ratio = np.zeros_like(img, dtype=np.float32)
        np.divide(img, snr, out=ratio, where=snr > 0)
        return ratio**2

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
            Master bias source, resolved via ``_resolve_master`` (see ``perform``
            for the accepted input types). Defaults to self.bias.

        Returns
        -------
        None
            Modifies ``l1_obj.data['{chip}_CCD']`` and ``['{chip}_VAR']`` in-place:
            the bias is subtracted from CCD and its variance added to VAR.
        """
        if bias is None:
            bias = self.bias
        bias_l1 = self._resolve_master("bias", bias)
        chip = chip.upper()
        img = bias_l1.data[f"{chip}_IMG"]
        snr = bias_l1.data[f"{chip}_SNR"]
        self.l1_obj.data[f"{chip}_CCD"] -= img
        self.l1_obj.data[f"{chip}_VAR"] += self._master_image_variance(img, snr)

    def subtract_dark(self, chip, dark=None):
        """
        Subtract the master dark from the CCD data for a single chip.

        The master dark is a rate (electrons/sec), so it is scaled by the
        frame's exposure time before subtraction. Otherwise identical to
        ``subtract_bias``.

        Parameters
        ----------
        chip : str
            CCD identifier, e.g. 'GREEN' or 'RED'.
        dark : bool | str | KPFMasterL1, optional
            Master dark source, resolved via ``_resolve_master`` (see ``perform``
            for the accepted input types). Defaults to self.dark.

        Returns
        -------
        None
            Modifies ``l1_obj.data['{chip}_CCD']`` and ``['{chip}_VAR']`` in-place:
            the exposure-scaled dark is subtracted from CCD and its variance
            (scaled by exptime**2) added to VAR.
        """
        if dark is None:
            dark = self.dark
        dark_l1 = self._resolve_master("dark", dark)
        chip = chip.upper()
        exptime = self.l1_obj.headers["PRIMARY"]["EXPTIME"]
        img = dark_l1.data[f"{chip}_IMG"]
        snr = dark_l1.data[f"{chip}_SNR"]
        self.l1_obj.data[f"{chip}_CCD"] -= img * exptime
        self.l1_obj.data[f"{chip}_VAR"] += exptime**2 * self._master_image_variance(
            img, snr
        )

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------

    def _track_info(self):
        """Build and cache the info() summary text from instance attributes."""
        lines = [
            "ImageProcessing",
            f"  obs_id: {self.l1_obj.obs_id}",
            f"  chips:  {self.chips}",
            f"\n  {'cal_type':<10s} {'master file'}",
            "  " + "-" * 60,
        ]
        if self.bias:
            lines.append(f"  {'bias':<10s} {self._bias_path}")
        if self.dark:
            lines.append(f"  {'dark':<10s} {self._dark_path}")
        self._info = "\n\n" + "\n".join(lines) + "\n\n"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    def calibration_applied(l1_obj, cal_type):
        """
        Return True if ``cal_type`` is already flagged applied on ``l1_obj``.

        Reads the applied flag (``_CALIBRATION_APPLIED_KEYS``: biassub/darksub/
        flatdiv) from the image_processing receipt entry, written by a prior
        ``perform``. Lets callers -- and ``perform`` itself -- avoid applying a
        calibration twice (e.g. a cached frame revisited during stacking).

        Parameters
        ----------
        l1_obj : KPF1
            Frame whose RECEIPT table is inspected.
        cal_type : str
            Calibration name: 'bias', 'dark', or 'flat'.

        Returns
        -------
        bool
            True if the calibration's flag was recorded applied.
        """
        entry = l1_obj.receipt_read_entry("image_processing")
        return entry.get(_CALIBRATION_APPLIED_KEYS[cal_type]) == "1"

    def perform(self, chips=None, *, bias=None, dark=None, flat=None):
        """
        Run image processing calibrations on the L1 frame.

        Parameters
        ----------
        chips : list of str, optional
            CCD chips to process. Defaults to self.chips.
        bias : bool | str | KPFMasterL1, optional
            How to source the master bias. Falsy → skip. True → load via
            biasfile (the master's full path) in the RECEIPT table. str →
            treat as an explicit filepath. KPFMasterL1 → use this object
            directly (no disk I/O). Defaults to self.bias.
        dark : bool | str | KPFMasterL1, optional
            Same shape as ``bias``, sourced from darkfile. Applied
            after bias subtraction and scaled by the frame's exposure time.
            Defaults to self.dark.
        flat : bool | str | KPFMasterL1, optional
            Same shape as ``bias``. Any truthy value raises
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
            If ``bias`` or ``dark`` is not bool, str, or KPFMasterL1.
        RuntimeError
            If a requested calibration is already flagged applied on the frame
            (biassub/darksub), guarding against double subtraction.
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

        # Guard against re-applying a calibration already subtracted from this
        # frame (e.g. a double perform() call); checked before any mutation.
        prior_bias = self.calibration_applied(self.l1_obj, "bias")
        prior_dark = self.calibration_applied(self.l1_obj, "dark")
        prior_flat = self.calibration_applied(self.l1_obj, "flat")
        if self.bias and prior_bias:
            raise RuntimeError("bias already subtracted from this frame (biassub=1)")
        if self.dark and prior_dark:
            raise RuntimeError("dark already subtracted from this frame (darksub=1)")

        if self.bias:
            for chip in self.chips:
                logger.debug("subtracting bias from %s CCD", chip)
                self.subtract_bias(chip)

        if self.dark:
            for chip in self.chips:
                logger.debug("subtracting dark from %s CCD", chip)
                self.subtract_dark(chip)

        # OR with the prior flag so applying one calibration never clears
        # another already recorded on the frame.
        self._biassub = bool(self.bias) or prior_bias
        self._darksub = bool(self.dark) or prior_dark
        self._flatdiv = bool(self.flat) or prior_flat

        self._track_info()
        self.l1_obj.receipt_add_entry(
            "image_processing",
            f"biassub={int(self._biassub)}, darksub={int(self._darksub)}, "
            f"flatdiv={int(self._flatdiv)}",
            "PASS",
        )

        logger.info("%s", self._info)
        return self.l1_obj

    def info(self):
        """Print a summary of the module configuration and processing results."""
        if self._info is None:
            print(f"{type(self).__name__}: perform() has not been called")
        else:
            print(self._info)
