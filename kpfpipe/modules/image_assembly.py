"""
KPF Image Assembly module.

Assembles a raw L0 readout into a single L1 full-frame image (FFI). Reads the
2- or 4-amplifier readout mode from the L0 header (NAMPGRN/NAMPRED), subtracts
per-amplifier overscan bias, measures read noise, and stitches together the FFI
(4080 x 4080 arrays) for both GREEN and RED CCDs.

ImageAssembly applies no external calibrations: bias/dark/flat are handled
downstream by ImageProcessing following the standard CCD reduction sequence
(bias subtraction, then exposure-scaled dark subtraction, then flat division),
gated per file type by the masters modules.
"""

import logging

import numpy as np
import pandas as pd

from kpfpipe import DEFAULT_CFG, DETECTOR
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import flag_outliers

logger = logging.getLogger(__name__)

_DEFAULT_CFG = {
    **DEFAULT_CFG,
    "overscan_method": "rowmedian",
    "readnoise_sigma": 10.0,
}

# Public by design (special case): per-amplifier read-noise header keywords.
# ImageAssembly is the first module to touch raw L0 and owns detector read-noise
# metadata, so QC/Quicklook import this table rather than re-deriving it. This is
# the one sanctioned public constant in kpfpipe/modules/.
RN_KEYS = {
    "GREEN_AMP1": ["RNGREEN1", "RNNGGR1"],
    "GREEN_AMP2": ["RNGREEN2", "RNNGGR2"],
    "GREEN_AMP3": ["RNGREEN3", "RNNGGR3"],
    "GREEN_AMP4": ["RNGREEN4", "RNNGGR4"],
    "RED_AMP1": ["RNRED1", "RNNGRD1"],
    "RED_AMP2": ["RNRED2", "RNNGRD2"],
    "RED_AMP3": ["RNRED3", "RNNGRD3"],
    "RED_AMP4": ["RNRED4", "RNNGRD4"],
}


class ImageAssembly:
    """
    Assemble a raw L0 readout into an L1 full-frame image.

    Orients amplifier channels, applies gain conversion (ADU --> photo-electrons),
    measures read noise, subtracts overscan bias, and stitches the per-chip FFI;
    also converts EXPMETER_SCI/SKY wavelengths from nm to Angstroms.

    Parameters
    ----------
    l0_obj : KPF0
        Raw L0 readout to assemble. Its per-amplifier extensions
        (``{CHIP}_AMP{n}``) are read and modified in place during assembly.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: overscan_method, readnoise_sigma.
    """

    def __init__(self, l0_obj, config=None):
        self.l0_obj = l0_obj

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS", "TRACES", "MODULE_IMAGE_ASSEMBLY"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in _DEFAULT_CFG.items():
            setattr(self, k, params.get(k, v))
        self.chips = tuple(self.chips)
        self.fibers = tuple(self.fibers)

        for k, v in DETECTOR.items():
            setattr(self, k, v)
        self.nrow, self.ncol = self.ccd["nrow"], self.ccd["ncol"]

        self._info = None
        self.namp = {}  # number of amplifiers; set by _infer_amplifier_settings()
        self.dims = {}  # amplifier dimensions; set by _infer_amplifier_settings()
        self.gain = {}  # per-amp gain; set by _infer_amplifier_settings()
        self.orientation = {}  # amp ext -> flip; set by _infer_amplifier_settings()
        self.readnoise = {}  # set by measure_read_noise()
        self.rn_nongauss = {}  # non-gaussian read noise ~ std/mad

        self._infer_amplifier_settings()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer_amplifier_settings(self):
        """
        Cache amplifier count (namp) and per-channel dimensions, orientation,
        and gain. Orientation maps each channel to standard orientation
        (serial overscan right, parallel overscan bottom).
        """
        primary = self.l0_obj.headers["PRIMARY"]
        self.namp = {"GREEN": primary["NAMPGRN"], "RED": primary["NAMPRED"]}

        for chip, namp in self.namp.items():
            if namp == 0:  # chip not read out; nothing to assemble
                continue
            if namp == 2:
                self.dims[chip] = (self.nrow, self.ncol // 2)
            elif namp == 4:
                self.dims[chip] = (self.nrow // 2, self.ncol // 2)
            else:
                raise ValueError(
                    f"Only 2-amp and 4-amp mode supported, "
                    f"detected {namp} on {chip} CCD"
                )

        for chip in self.chips:
            chip = chip.upper()
            df = pd.DataFrame(self.amplifiers[chip]).set_index("channel_id")
            self.orientation.update(dict(zip(df["ext_name"], df["flip"], strict=False)))
            self.gain.update(dict(zip(df["ext_name"], df["gain"], strict=False)))

    def _get_overscan_pixels(self, chip, amp_no, buffer=(0, 0)):
        """
        Return the (serial, parallel) overscan pixel arrays for one amplifier,
        with optional edge-buffer trimming. Assumes standard orientation.
        """
        chip = chip.upper()
        full_amplifier = self.l0_obj.data[f"{chip}_AMP{amp_no}"]

        ncol_prescan = self.ccd["prescan"]
        nrow_imaging, ncol_imaging = self.dims[chip]

        oscan_pix_srl = full_amplifier[:nrow_imaging, ncol_prescan + ncol_imaging :]
        oscan_pix_prl = full_amplifier[nrow_imaging:, : ncol_prescan + ncol_imaging]

        start = buffer[0] if buffer[0] > 0 else None
        end = -buffer[1] if buffer[1] > 0 else None

        oscan_pix_srl = oscan_pix_srl[:, start:end]
        oscan_pix_prl = oscan_pix_prl[start:end, :]

        return oscan_pix_srl, oscan_pix_prl

    def _get_imaging_pixels(self, chip, amp_no):
        """
        Return the active-imaging-area pixels for one amplifier (prescan and
        overscan stripped). Assumes standard orientation.
        """
        chip = chip.upper()
        full_amplifier = self.l0_obj.data[f"{chip}_AMP{amp_no}"]

        ncol_prescan = self.ccd["prescan"]
        nrow_imaging, ncol_imaging = self.dims[chip]

        image_pix = full_amplifier[
            :nrow_imaging, ncol_prescan : ncol_prescan + ncol_imaging
        ]

        return image_pix

    def _oscan_zero(self, chip, amp_no, **kwargs):
        """
        Returns overscan bias level of zero. chip/amp_no/kwargs are unused here
        but kept for the uniform ``_oscan_*`` dispatch signature (see
        subtract_overscan()); ``del`` marks them intentionally discarded.
        """
        del chip, amp_no, kwargs
        return 0.0

    def _oscan_median(self, chip, amp_no, **kwargs):
        """
        Calculates single-value median of serial overscan region
        """
        oscan_srl, _ = self._get_overscan_pixels(chip, amp_no, **kwargs)
        oscan_bias = np.nanmedian(oscan_srl)
        return oscan_bias

    def _oscan_rowmedian(self, chip, amp_no, **kwargs):
        """
        Calculates row-by-row median of serial overscan region
        """
        oscan_srl, _ = self._get_overscan_pixels(chip, amp_no, **kwargs)
        oscan_bias = np.nanmedian(oscan_srl, axis=1)[:, None]
        return oscan_bias

    @staticmethod
    def _convert_expmeter_wavelengths_to_angstroms(l1_obj):
        """
        Rename EXPMETER_SCI/SKY wavelength column labels from nm to Å at the
        L0 → L1 boundary, so the whole L1+ pipeline uses one wavelength unit
        (RVData L2 and KPF WAVE arrays are in Angstroms). Non-numeric columns
        (e.g. 'Date-Beg') are skipped; flux values are unchanged.
        """
        for ext_name in ("EXPMETER_SCI", "EXPMETER_SKY"):
            if ext_name not in l1_obj.data:
                continue
            table = l1_obj.data[ext_name]
            if table is None or not hasattr(table, "rename_column"):
                continue
            for col in list(table.colnames):
                try:
                    wave_nm = float(col)
                except (ValueError, TypeError):
                    continue
                new_name = format(wave_nm * 10, "g")
                if new_name != col:
                    table.rename_column(col, new_name)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def orient_channels(self, chip):
        """
        Reorient amplifier channels to a standard orientation in-place.
        (serial overscan on right, parallel overscan on bottom)

        Parameters
        ----------
        chip : str
            CCD identifier, e.g., 'GREEN' or 'RED'.

        Notes
        -----
        The transformations are flips, so each is its own inverse -- calling
        twice restores the input. measure_read_noise and subtract_overscan use
        this to orient to standard, do their work, then restore the original
        orientation, so a non-standard orientation never propagates between
        steps; it is also called directly by Quicklook for its L0 display.
        """
        chip = chip.upper()

        for i in range(self.namp[chip]):
            channel_ext = f"{chip.upper()}_AMP{i + 1}"
            flip = self.orientation[channel_ext]
            image = self.l0_obj.data[channel_ext]

            if flip == "rows":
                image_reoriented = np.flip(image, axis=0)
            elif flip == "cols":
                image_reoriented = np.flip(image, axis=1)
            elif flip == "both":
                image_reoriented = np.flip(image, axis=(0, 1))
            elif flip == "none":
                image_reoriented = image
            else:
                raise ValueError(
                    "unexpected 'flip' entry found in orientation reference"
                )

            self.l0_obj.data[channel_ext] = image_reoriented

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def apply_gain_conversion(self, chip):
        """
        Convert pixel values from ADU to photo-electrons using amplifier-specific gain.
        Amplifier channels are modified in-place.

        Parameters
        ----------
        chip : str
            CCD identifier, e.g., 'GREEN' or 'RED'.
        """
        chip = chip.upper()

        for i in range(self.namp[chip]):
            channel_ext = f"{chip}_AMP{i + 1}"
            self.l0_obj.data[channel_ext] *= self.gain[channel_ext] / (2**16)

    def measure_read_noise(self, chip, sigma=None, buffer=(5, 5)):
        """
        Estimate read noise for each amplifier from overscan pixels.

        Parameters
        ----------
        chip : str
            CCD identifier, e.g., 'GREEN' or 'RED'.
        sigma : float, optional
            Threshold for sigma clipping overscan pixels.
        buffer : tuple of int, optional
            Number of pixels to ignore at the edges (start, end). Defaults to (5, 5).

        Notes
        -----
        Stores ``self.readnoise[channel_ext]`` (std of cleaned overscan) and
        ``self.rn_nongauss[channel_ext]`` (``sqrt(2/pi) * std / mad``, normalized to
        be ~1 for a Gaussian).
        """
        if sigma is None:
            sigma = self.readnoise_sigma

        chip = chip.upper()

        # Overscan extraction assumes standard orientation; restore the original
        # afterward so a flipped channel never propagates to the next step.
        self.orient_channels(chip)
        for i in range(self.namp[chip]):
            channel_ext = f"{chip}_AMP{i + 1}"

            oscan_srl, _ = self._get_overscan_pixels(chip, i + 1, buffer)

            out = flag_outliers(oscan_srl, sigma, method="median")
            std = np.nanstd(oscan_srl[~out])
            mad = np.nanmean(np.abs(oscan_srl[~out] - np.nanmean(oscan_srl[~out])))

            self.readnoise[channel_ext] = std
            self.rn_nongauss[channel_ext] = np.sqrt(2 / np.pi) * std / mad
        self.orient_channels(chip)

    def subtract_overscan(self, chip, method=None, buffer=(0, 0)):
        """
        Subtract overscan bias from imaging pixels for each amplifier. Also
        removes overscan region from amplifier channel, leaving only active
        imaging area pixels. Amplifier channels are modified in-place.

        Parameters
        ----------
        chip : str
            CCD identifier, e.g., 'GREEN' or 'RED'.
        method : str
            Overscan subtraction method ('zero', 'median', 'rowmedian').
        buffer : tuple of int, optional
            Number of pixels to ignore at edges. Defaults to (0, 0).
        """
        if method is None:
            method = self.overscan_method

        try:
            oscan_fxn = getattr(self, f"_oscan_{method}")
        except AttributeError:
            raise AttributeError(
                f"Unsupported overscan subtraction method: '{method}'"
            ) from None

        # Imaging/overscan extraction assumes standard orientation; restore the
        # original afterward so stitch_ffi consumes channels as it always has.
        self.orient_channels(chip)
        for i in range(self.namp[chip]):
            image = self._get_imaging_pixels(chip, i + 1)
            oscan_bias = oscan_fxn(chip, i + 1, buffer=buffer)
            self.l0_obj.data[f"{chip.upper()}_AMP{i + 1}"] = image - oscan_bias
        self.orient_channels(chip)

    def stitch_ffi(self, chip):
        """
        Combine individual amplifier channels into a full-frame image (FFI).

        Parameters
        ----------
        chip : str
            CCD identifier, e.g., 'GREEN' or 'RED'.

        Returns
        -------
        ccd_ffi : ndarray
            Full-frame data image.
        var_ffi : ndarray
            Full-frame variance image, incorporating read noise.

        Notes
        -----
        Supports 2-amp and 4-amp CCD configurations. Raises an error if
        any other number of amplifiers is detected.
        """
        chip = chip.upper()

        nrow, ncol = self.nrow, self.ncol
        row_mid, col_mid = nrow // 2, ncol // 2

        ccd_ffi = np.zeros((nrow, ncol), dtype=np.float32)
        var_ffi = np.zeros((nrow, ncol), dtype=np.float32)

        if self.namp[chip] == 2:
            ccd_ffi[:, :col_mid] = self.l0_obj.data[f"{chip}_AMP1"]
            ccd_ffi[:, col_mid:] = self.l0_obj.data[f"{chip}_AMP2"]
            var_ffi[:, :col_mid] = (
                np.abs(ccd_ffi[:, :col_mid]) + self.readnoise[f"{chip}_AMP1"]
            )
            var_ffi[:, col_mid:] = (
                np.abs(ccd_ffi[:, col_mid:]) + self.readnoise[f"{chip}_AMP2"]
            )

        elif self.namp[chip] == 4:
            ccd_ffi[:row_mid, :col_mid] = self.l0_obj.data[f"{chip}_AMP1"]
            ccd_ffi[:row_mid, col_mid:] = self.l0_obj.data[f"{chip}_AMP2"]
            ccd_ffi[row_mid:, :col_mid] = self.l0_obj.data[f"{chip}_AMP3"]
            ccd_ffi[row_mid:, col_mid:] = self.l0_obj.data[f"{chip}_AMP4"]
            var_ffi[:row_mid, :col_mid] = (
                np.abs(ccd_ffi[:row_mid, :col_mid]) + self.readnoise[f"{chip}_AMP1"]
            )
            var_ffi[:row_mid, col_mid:] = (
                np.abs(ccd_ffi[:row_mid, col_mid:]) + self.readnoise[f"{chip}_AMP2"]
            )
            var_ffi[row_mid:, :col_mid] = (
                np.abs(ccd_ffi[row_mid:, :col_mid]) + self.readnoise[f"{chip}_AMP3"]
            )
            var_ffi[row_mid:, col_mid:] = (
                np.abs(ccd_ffi[row_mid:, col_mid:]) + self.readnoise[f"{chip}_AMP4"]
            )

        else:
            raise ValueError(
                f"Only 2-amp and 4-amp mode supported, "
                f"detected {self.namp[chip]} on {chip} CCD"
            )

        ccd_ffi = self.orient_ffi(ccd_ffi, chip, self.namp[chip])
        var_ffi = self.orient_ffi(var_ffi, chip, self.namp[chip])

        return ccd_ffi, var_ffi

    @staticmethod
    def orient_ffi(image, chip, namp):
        """
        Flip an assembled image into the standard FFI orientation.
        """
        image = np.flip(image, axis=1)
        if chip.upper() == "GREEN" and namp == 2:
            image = np.flip(image, axis=0)
        return image

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------

    def _track_info(self, chips):
        """Build and cache the info() summary text from instance attributes."""
        lines = [
            "ImageAssembly",
            f"  obs_id:           {self.l0_obj.obs_id}",
            f"  overscan_method:  {self.overscan_method}",
            f"  readnoise_sigma:  {self.readnoise_sigma}",
            f"\n  {'channel':<14s} {'read noise [e-]':<18s} {'non-gaussian'}",
            "  " + "-" * 48,
        ]
        for chip in chips:
            for ch in self.readnoise:
                if not ch.startswith(chip.upper()):
                    continue
                rn = round(float(self.readnoise[ch]), 4)
                rnng = round(float(self.rn_nongauss[ch]), 4)
                lines.append(f"  {ch:<14s} {rn:<18} {rnng}")
            lines.append("")
        self._info = "\n\n" + "\n".join(lines) + "\n\n"

    def _set_headers(self, l1_obj):
        """
        Write assembly metadata to ``l1_obj``: per-amplifier read noise
        (RN_KEYS) and the non-Gaussian factor.
        """
        for channel_ext, rn in self.readnoise.items():
            key_read, key_rnng = RN_KEYS[channel_ext]
            l1_obj.set_keyword(key_read, round(float(rn), 4))
            l1_obj.set_keyword(key_rnng, round(float(self.rn_nongauss[channel_ext]), 4))

    def _receipt_args(self):
        """Whether overscan was subtracted; "zero" strips it but subtracts none."""
        return f"oscansub={int(self.overscan_method != 'zero')}"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, chips=None, *, overscan_method=None, readnoise_sigma=None):
        """
        Execute the image assembly algorithm. Optional keyword arguments
        default to config settings.

        Parameters
        ----------
        chips : list of str, optional
            CCD identifiers to process, i.e. 'GREEN', 'RED'
        overscan_method : str, optional
            Method for overscan subtraction ('zero', 'median', 'rowmedian').
        readnoise_sigma : float, optional
            Sigma threshold for clipping overscan pixels when measuring read noise.

        Returns
        -------
        l1_obj : KPF1
            L1 data object containing assembled full frame images (FFIs)
            for data and variance.

        Notes
        -----
        Amplifier-channel orientation is handled on-the-fly inside
        measure_read_noise and subtract_overscan (each restores the original
        orientation afterward), not as a separate top-level step.
        """
        if chips is None:
            chips = self.chips
        if overscan_method is None:
            overscan_method = self.overscan_method
        if readnoise_sigma is None:
            readnoise_sigma = self.readnoise_sigma

        self.chips = chips
        self.overscan_method = overscan_method
        self.readnoise_sigma = readnoise_sigma

        l1_obj = self.l0_obj.to_kpf1()

        for chip in chips:
            self.apply_gain_conversion(chip)
            self.measure_read_noise(chip, readnoise_sigma)
            self.subtract_overscan(chip, overscan_method)

            ccd_ffi, var_ffi = self.stitch_ffi(chip)
            l1_obj.set_data(f"{chip}_CCD", ccd_ffi)
            l1_obj.set_data(f"{chip}_VAR", var_ffi)

        self._convert_expmeter_wavelengths_to_angstroms(l1_obj)
        self._set_headers(l1_obj)
        self._track_info(chips)
        l1_obj.receipt_add_entry("image_assembly", self._receipt_args(), "PASS")

        logger.info("%s", self._info)
        return l1_obj

    def info(self):
        """Print a summary of the module configuration and processing results."""
        if self._info is None:
            print(f"{type(self).__name__}: perform() has not been called")
        else:
            print(self._info)
