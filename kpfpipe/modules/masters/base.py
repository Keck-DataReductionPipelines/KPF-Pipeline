"""
Base class for KPF Masters modules.
"""

import os
import warnings

import numpy as np

from kpfpipe import DEFAULTS
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.masters.level1 import KPFMasterL1
from kpfpipe.modules.calibration_association import CalibrationAssociation
from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.modules.image_processing import ImageProcessing
from kpfpipe.modules.spectral_extraction import SpectralExtraction
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import flag_outliers, interpolate_bad_pixels

# TODO: throw out first frame in stack?
# TODO: use start, middle, end of stack for initial datacube


class BaseMasterModule:
    """
    Base class for KPF masters generation.

    The class should not be called directly, but is used for inheritance
    of masters subclasses: Bias, Dark, Flat, WLS. Masters modules read a
    stack of L0 files from disk and output a masters L1 object.

    Each frame is calibrated before stacking/extraction following standard CCD
    reduction: bias gets no correction, dark is bias-subtracted, flat is
    bias+dark-subtracted, and WLS (like science) is bias+dark-subtracted then
    flat-divided. Each subclass declares its standard set via
    `_STANDARD_CORRECTIONS`. Which of those actually run is the standard set
    intersected with the resolved bias/dark/flat flags
    (DEFAULTS < [MODULE_IMAGE_PROCESSING] config < make_master kwargs): a flag
    can only turn a standard correction off, never enable one a master type
    does not use.
    """

    # Module defaults; subclasses extend via `{**BaseMasterModule._DEFAULTS, ...}`.
    # bias/dark/flat are the globally-enabled corrections (the no-config
    # fallback; in practice resolved from the shared [MODULE_IMAGE_PROCESSING]
    # config and make_master kwargs). Keep in sync with the same keys in
    # image_processing._DEFAULTS.
    _DEFAULTS = {
        **DEFAULTS,
        "nframe_stream": 6,
        "stack_sigma": 5.0,
        "bias": True,
        "dark": True,
        "flat": False,
    }

    # The corrections that are standard for this master type (the ceiling).
    # `_process_frame` applies the intersection of this set with the resolved
    # bias/dark/flat flags. Subclasses override (e.g. Dark -> ("bias",)).
    _STANDARD_CORRECTIONS = ()

    def __init__(self, l0_file_list, config=None):
        if l0_file_list != sorted(l0_file_list):
            raise ValueError("l0_file_list must be sorted in ascending order")
        self.l0_file_list = l0_file_list

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS", "KPFPIPE"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in self._DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self._l1_obj_cache = {}

        # Masters output root; CalibrationAssociation reads masters from here
        # when `_process_frame` associates a calibration for a stacked frame.
        self._masters_root = params.get("KPF_MASTERS_OUTPUT")

        # populated by subclass make_master_l1(); used by save_master('L1', ...)
        self.ml1_obj = None
        # populated by subclass make_master_l2(); used by save_master('L2', ...)
        self.ml2_obj = None

        # Effective per-frame corrections (standard set masked by the resolved
        # flags); make_master_l1/l2 re-resolve this with any kwarg overrides.
        self._active_corrections = self._resolve_corrections()

    # ------------------------------------------------------------------
    # Private helpers for masters.
    # ------------------------------------------------------------------

    def _load_master(self, fn, verbose=True):
        """
        Load a master file into an L1 object.

        Parameters
        ----------
        fn : str
            Path to master L1 FITS file.
        verbose : bool, optional
            If True (default), emit a progress print and propagate load /
            exptime-check failures as UserWarnings. If False, all such
            output is suppressed; the (None, False) failure return value
            still signals the caller.

        Returns
        -------
        l1_obj : KPF1 or None
            Assembled L1 data object if successful, otherwise None.
        success : bool
            True if file was successfully loaded and processed, False otherwise.

        Notes
        -----
        Delegates the FITS read to `ImageProcessing._load_master`, which caches
        masters by path so a file is not re-read once loaded.
        """
        if verbose:
            print(f"loading {fn}")

        success = True
        failure = False

        try:
            l1_obj = ImageProcessing._load_master(fn)

        except (FileNotFoundError, OSError) as e:
            if verbose:
                warnings.warn(f"Failed to load {fn}: {e}", stacklevel=2)
            return None, failure

        return l1_obj, success

    # ------------------------------------------------------------------
    # Private helpers for frame handling (loading, calibration, etc.).
    # ------------------------------------------------------------------

    def _load_frame(self, fn, ncache=None, exptime_tolerance=0.1, verbose=True):
        """
        Load an L0 file and perform image assembly to produce an L1 object.

        Parameters
        ----------
        fn : str
            Path to L0 FITS file.
        ncache : int, optional
            Maximum number of L1 objects to retain in internal cache.
        exptime_tolerance : float
            Maximum allowed excess of elapsed time over requested exposure time,
            in seconds (default = 0.1).
        verbose : bool, optional
            If True (default), emit a progress print and propagate load /
            exptime-check failures as UserWarnings. If False, all such
            output is suppressed; the (None, False) failure return value
            still signals the caller.

        Returns
        -------
        l1_obj : KPF1 or None
            Assembled L1 data object if successful, otherwise None.
        success : bool
            True if file was successfully loaded and processed, False otherwise.

        Notes
        -----
        Successfully processed frames may be cached to reduce redundant I/O and
        recomputation. Cache size is limited by `ncache`, which defaults to
        `nframe_stream - 1`.
        """
        if verbose:
            print(f"loading {fn}")

        if ncache is None:
            ncache = self.nframe_stream - 1

        success = True
        failure = False

        if fn in self._l1_obj_cache:
            l1_obj = self._l1_obj_cache[fn]

        else:
            try:
                l0_obj = KPF0.from_fits(fn)
                l1_obj = ImageAssembly(l0_obj).perform()

                if len(self._l1_obj_cache) < ncache:
                    self._l1_obj_cache[fn] = l1_obj

            except (FileNotFoundError, OSError) as e:
                if verbose:
                    warnings.warn(f"Failed to load {fn}: {e}", stacklevel=2)
                return None, failure

        try:
            self._check_exptime_vs_elapsed(l1_obj, exptime_tolerance)
        except ValueError as e:
            if verbose:
                warnings.warn(f"Exptime check failed for {fn}: {e}", stacklevel=2)
            return None, failure

        return l1_obj, success

    def _resolve_corrections(self, *, bias=None, dark=None, flat=None):
        """
        Resolve which corrections to apply to each frame, as {name: bool}.

        For each correction, the effective value is the per-master standard
        (`_STANDARD_CORRECTIONS`) AND the resolved flag, where the flag is the
        make_master kwarg if given, else the config-resolved `self.<name>`
        (DEFAULTS < [MODULE_IMAGE_PROCESSING]). A flag can only turn a standard
        correction off; it can never enable one outside the master's standard.

        Parameters
        ----------
        bias, dark, flat : bool, optional
            Per-call overrides; None means "use the resolved config value".

        Returns
        -------
        dict
            {'bias': bool, 'dark': bool, 'flat': bool}.
        """
        overrides = {"bias": bias, "dark": dark, "flat": flat}
        return {
            name: bool(
                name in self._STANDARD_CORRECTIONS
                and (
                    overrides[name]
                    if overrides[name] is not None
                    else getattr(self, name)
                )
            )
            for name in ("bias", "dark", "flat")
        }

    def _process_frame(self, l1_obj):
        """
        Apply this module's active calibrations to an assembled frame.

        Associates the active masters (writing their paths into the PRIMARY
        header) and subtracts them via ImageProcessing, reusing the standard
        science-path modules rather than reimplementing the math. Modules with
        no active corrections receive the frame unchanged.

        Parameters
        ----------
        l1_obj : KPF1
            Assembled L1 frame to calibrate, mutated in place.

        Returns
        -------
        KPF1
            The same frame with the active calibrations applied.
        """
        corrections = self._active_corrections
        cal_types = [name for name in ("bias", "dark", "flat") if corrections[name]]
        if not cal_types:
            return l1_obj

        calibration_association = CalibrationAssociation(
            l1_obj, {"KPF_MASTERS_OUTPUT": self._masters_root}
        )
        l1_obj = calibration_association.perform(cal_types)

        image_processing = ImageProcessing(l1_obj)
        l1_obj = image_processing.perform(
            bias=corrections["bias"],
            dark=corrections["dark"],
            flat=corrections["flat"],
        )

        return l1_obj

    def _extract_frame(self, l1_obj, verbose=True):
        """
        Extract an assembled frame to L2 (for spectral masters, e.g. WLS).

        Performs spectral extraction only. Calibration is not implicit here:
        callers that need bias/dark applied must run `_process_frame` on the
        frame first.

        Parameters
        ----------
        l1_obj : KPF1
            Assembled (and, where needed, already calibrated) L1 frame.
        verbose : bool, optional
            If True (default), emit progress prints during extraction.

        Returns
        -------
        KPF2
            The extracted L2 spectra.
        """
        spectral_extraction = SpectralExtraction(l1_obj)
        return spectral_extraction.perform(verbose=verbose)

    @staticmethod
    def _check_exptime_vs_elapsed(l1_obj, exptime_tolerance):
        """
        Validate that elapsed readout time is consistent with requested exposure time.

        Parameters
        ----------
        l1_obj : KPF1
            Assembled L1 object whose PRIMARY header contains EXPTIME and ELAPSED.
        exptime_tolerance : float
            Maximum allowed excess of elapsed time over requested exposure time,
            in seconds.

        Raises
        ------
        ValueError
            If elapsed time is less than requested (premature readout), or if the
            excess exceeds exptime_tolerance.
        """
        exptime = l1_obj.headers["PRIMARY"]["EXPTIME"]
        elapsed = l1_obj.headers["PRIMARY"]["ELAPSED"]

        delta = elapsed - exptime
        if delta < 0:
            raise ValueError("premature frame readout detected")
        if delta > exptime_tolerance:
            raise ValueError(f"elapsed time - requested time > {exptime_tolerance}")

    # ------------------------------------------------------------------
    # Private helpers for frame stacking.
    # ------------------------------------------------------------------

    def _compute_stats_from_datacube(
        self, l0_file_list=None, nframe=None, sigma=None, verbose=True
    ):
        """
        Compute stacked statistics using an in-memory data cube.

        Parameters
        ----------
        l0_file_list : list of str, optional
            List of L0 FITS filenames to process.
        nframe : int, optional
            Maximum number of successfully loaded frames to include.
        sigma : float, optional
            Sigma threshold for outlier rejection across frames.
        verbose : bool, optional
            If True (default), emit per-frame progress prints and load
            failure warnings from `_load_frame`.

        Returns
        -------
        stats : dict
            Per-extension statistics including:
            - 'nframe'     : number of valid frames per pixel
            - 'total_sum'  : summed counts across valid frames
            - 'rate_mean'  : exposure-time-normalized mean
            - 'rate_rms'   : frame-to-frame sample RMS
        exptime_total : float
            Total integrated exposure time across included frames.

        Notes
        -----
        Outlier rejection is performed jointly on CCD and VAR extensions.
        Exposure times must be either all zero or all strictly positive.
        Raises an error if more than 20% of frames fail to load.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if nframe is None:
            nframe = self.nframe_stream - 1
        if sigma is None:
            sigma = self.stack_sigma

        nframe = np.min([nframe, len(l0_file_list)])

        if nframe < 2:
            raise ValueError(f"Stacking requires at least two frames, got {nframe}")

        nrow = self.ccd["nrow"]
        ncol = self.ccd["ncol"]

        data_cube = {}
        exptime = np.zeros(nframe, dtype=np.float32)

        for chip in self.chips:
            data_cube[f"{chip}_CCD"] = np.zeros((nframe, nrow, ncol), dtype=np.float32)
            data_cube[f"{chip}_VAR"] = np.zeros((nframe, nrow, ncol), dtype=np.float32)

        i = 0
        failure = 0

        for fn in l0_file_list:
            if i >= nframe:
                break

            l1_obj, success = self._load_frame(fn, verbose=verbose)

            if not success:
                failure += 1
                if failure / len(l0_file_list) > 0.2:
                    raise ValueError("more than 20% of frames in stack failed to load")
                continue

            l1_obj = self._process_frame(l1_obj)

            exptime[i] = l1_obj.headers["PRIMARY"]["EXPTIME"]

            for chip in self.chips:
                data_cube[f"{chip}_CCD"][i] = l1_obj.data[f"{chip}_CCD"]
                data_cube[f"{chip}_VAR"][i] = l1_obj.data[f"{chip}_VAR"]

            i += 1

        if i < nframe:
            exptime = exptime[:i]
            for k in data_cube.keys():
                data_cube[k] = data_cube[k][:i]

        if np.any(exptime < 0):
            raise ValueError(f"Exposure times cannot be negative; exptime = {exptime}")

        if np.all(exptime > 0):
            T = exptime[:, None, None]
        elif np.all(exptime == 0):
            T = np.ones_like(exptime)[:, None, None]
        else:
            raise ValueError(
                f"Exposure times must be all zero or all non-zero; exptime = {exptime}"
            )

        stats = {}
        exptime_total = np.sum(exptime)

        for chip in self.chips:
            stats[f"{chip}_CCD"] = {}
            stats[f"{chip}_VAR"] = {}

            out = flag_outliers(
                data_cube[f"{chip}_CCD"] / T, sigma, axis=0
            ) | flag_outliers(data_cube[f"{chip}_VAR"] / T, sigma, axis=0)

            valid = ~out
            N = np.sum(~out, axis=0)
            good = N > 1

            for suffix in ["CCD", "VAR"]:
                ext = f"{chip}_{suffix}"
                frame_data = data_cube[ext]
                R = frame_data / T

                total_sum = np.sum(frame_data, axis=0, where=valid)

                rate_mean = np.zeros_like(R[0])
                rate_mean[good] = np.sum(R, axis=0, where=valid)[good] / N[good]

                diff2 = (R - rate_mean) ** 2
                sum_sq_dev = np.sum(diff2, axis=0, where=valid)

                rate_rms = np.zeros_like(R[0])
                rate_rms[good] = np.sqrt(sum_sq_dev[good] / (N[good] - 1))

                stats[ext]["nframe"] = N
                stats[ext]["total_sum"] = total_sum
                stats[ext]["rate_mean"] = rate_mean
                stats[ext]["rate_rms"] = rate_rms

        return stats, exptime_total

    def _compute_stats_from_stream(
        self, l0_file_list=None, ndirect=None, sigma=None, verbose=True
    ):
        """
        Compute stacked statistics using streaming Welford accumulation.

        Parameters
        ----------
        l0_file_list : list of str, optional
            List of L0 FITS filenames to process.
        ndirect : int, optional
            Number of initial frames used to estimate approximate statistics
            for defining clipping thresholds.
        sigma : float, optional
            Sigma threshold for outlier rejection.
        verbose : bool, optional
            If True (default), emit per-frame progress prints and load
            failure warnings from `_load_frame`.

        Returns
        -------
        exact_stats : dict
            Per-extension statistics including:
            - 'nframe'     : number of valid frames per pixel
            - 'total_sum'  : summed counts across valid frames
            - 'rate_mean'  : per-pixel rate mean, normalized by exposure time
            - 'rate_rms'   : frame-to-frame rate rms deviation
        exptime_total : float
            Total integrated exposure time across included frames.

        Notes
        -----
        An initial subset of frames is processed using the direct data cube
        method to estimate approximate mean and RMS. These estimates define
        per-pixel clipping bounds for the streaming pass.

        Optimized to reduce memory usage at the expense of compute speed.
        Raises an error if more than 20% of frames fail to load or if exposure
        times are inconsistent.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if ndirect is None:
            ndirect = self.nframe_stream - 1
        if sigma is None:
            sigma = self.stack_sigma

        approx_stats, exptime_direct = self._compute_stats_from_datacube(
            l0_file_list=l0_file_list,
            nframe=ndirect,
            sigma=sigma,
            verbose=verbose,
        )

        if len(l0_file_list) <= ndirect:
            return approx_stats, exptime_direct

        exact_stats = {}
        exptime_total = 0.0
        zero_exptime = exptime_direct == 0

        nrow = self.ccd["nrow"]
        ncol = self.ccd["ncol"]

        for chip in self.chips:
            for suffix in ["CCD", "VAR"]:
                ext = f"{chip}_{suffix}"

                exact_stats[ext] = {}
                exact_stats[ext]["nframe"] = np.zeros((nrow, ncol), dtype=np.int32)
                exact_stats[ext]["total_sum"] = np.zeros((nrow, ncol), dtype=np.float32)
                exact_stats[ext]["rate_mean"] = np.zeros((nrow, ncol), dtype=np.float32)
                exact_stats[ext]["rate_M2"] = np.zeros((nrow, ncol), dtype=np.float32)

                approx_mean = approx_stats[ext]["rate_mean"]
                approx_rms = approx_stats[ext]["rate_rms"]

                approx_stats[ext]["rate_lower"] = approx_mean - approx_rms * sigma
                approx_stats[ext]["rate_upper"] = approx_mean + approx_rms * sigma

        failure = 0
        clipping_mask = np.ones((nrow, ncol), dtype=bool)

        for fn in l0_file_list:
            l1_obj, success = self._load_frame(fn, verbose=verbose)

            if not success:
                failure += 1
                if failure / len(l0_file_list) > 0.2:
                    raise ValueError("more than 20% of frames in stack failed to load")
                continue

            l1_obj = self._process_frame(l1_obj)

            exptime = l1_obj.headers["PRIMARY"]["EXPTIME"]

            if zero_exptime != (exptime == 0):
                raise ValueError("Exposure times must be all zero or all non-zero")

            if exptime < 0:
                raise ValueError("Exposure times cannot be negative")
            elif exptime == 0:
                T = 1.0
            else:
                T = exptime

            exptime_total += exptime

            for chip in self.chips:
                clipping_mask[:] = True
                R = {}

                for suffix in ["CCD", "VAR"]:
                    ext = f"{chip}_{suffix}"
                    frame_data = l1_obj.data[ext]
                    R[ext] = frame_data / T

                    lower = approx_stats[ext]["rate_lower"]
                    upper = approx_stats[ext]["rate_upper"]
                    clipping_mask &= (R[ext] >= lower) & (R[ext] <= upper)

                for suffix in ["CCD", "VAR"]:
                    ext = f"{chip}_{suffix}"
                    frame_data = l1_obj.data[ext]
                    rate = R[ext]

                    N = exact_stats[ext]["nframe"]
                    N += clipping_mask

                    total_sum = exact_stats[ext]["total_sum"]
                    total_sum += frame_data * clipping_mask

                    # Welford algorithm accumulation begins
                    mean = exact_stats[ext]["rate_mean"]
                    safe_N = np.maximum(N, 1)
                    delta = (rate - mean) * clipping_mask
                    mean += delta / safe_N
                    delta2 = (rate - mean) * clipping_mask
                    M2 = exact_stats[ext]["rate_M2"]
                    M2 += delta * delta2
                    # Welford algorithm accumulation ends

                    exact_stats[ext]["total_sum"] = total_sum
                    exact_stats[ext]["rate_mean"] = mean
                    exact_stats[ext]["rate_M2"] = M2

        for chip in self.chips:
            for suffix in ["CCD", "VAR"]:
                ext = f"{chip}_{suffix}"

                N = exact_stats[ext]["nframe"]
                mean = exact_stats[ext]["rate_mean"]
                M2 = exact_stats[ext]["rate_M2"]
                rms = np.sqrt(np.where(N > 1, M2 / (N - 1), 0))

                exact_stats[ext]["rate_rms"] = rms
                del exact_stats[ext]["rate_M2"]

        return exact_stats, exptime_total

    # ------------------------------------------------------------------
    # Private helpers for tracking results.
    # ------------------------------------------------------------------

    def _compute_results(self, l1_arrays):
        """
        Summarize per-chip master statistics for `info()` and tests.

        Returns
        -------
        dict
            Per-chip {'num_bad', 'pct_bad', 'median', 'rms'}.
        """
        return {
            chip: {
                "num_bad": int(np.sum(~l1_arrays[f"{chip}_MASK"])),
                "pct_bad": float(100.0 * np.mean(~l1_arrays[f"{chip}_MASK"])),
                "median": float(np.nanmedian(l1_arrays[f"{chip}_IMG"])),
                "rms": float(np.nanstd(l1_arrays[f"{chip}_IMG"])),
            }
            for chip in self.chips
        }

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def stack_frames(self, l0_file_list=None, nstream=None, sigma=None, verbose=True):
        """
        Stack full-frame images to produce masters L1.

        Parameters
        ----------
        l0_file_list : list of str, optional
            List of L0 FITS filenames to stack.
        nstream : int, optional
            Threshold number of frames above which streaming statistics are used.
        sigma : float, optional
            Sigma threshold for frame-to-frame outlier rejection.
        verbose : bool, optional
            If True (default), emit per-frame progress prints and load
            failure warnings during stacking.

        Returns
        -------
        l1_arrays : dict
            Dictionary containing per-chip stacked products:
            - '{chip}_IMG'  : mean count rate FFI
            - '{chip}_SNR'  : signal-to-noise ratio FFI
            - '{chip}_MASK' : boolean bad pixel mask (1 = good, 0 = bad)

        Notes
        -----
        If number of frames is less than `nstream`, statistics are computed
        directly from a full data cube. Otherwise, streaming Welford statistics
        are used to reduce memory usage.

        An initial subset of frames is processed using the direct data cube
        method to estimate approximate mean and rms. These estimates define
        per-pixel clipping bounds for the streaming pass.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if nstream is None:
            nstream = self.nframe_stream
        if sigma is None:
            sigma = self.stack_sigma

        nframe = len(l0_file_list)

        if nframe < 2:
            raise ValueError(f"Stacking requires at least two frames, got {nframe}")

        if nframe < nstream:
            stats, exptime = self._compute_stats_from_datacube(
                l0_file_list, nstream - 1, sigma, verbose=verbose
            )
        else:
            stats, exptime = self._compute_stats_from_stream(
                l0_file_list, nstream - 1, sigma, verbose=verbose
            )

        # TODO: add check that nframe is consistent between CCD and VAR
        for chip in self.chips:
            if np.any(stats[f"{chip}_CCD"]["nframe"] != stats[f"{chip}_VAR"]["nframe"]):
                raise ValueError(
                    f"mismatched frame count between {chip}_CCD and {chip}_VAR"
                )

        l1_arrays = {}
        for chip in self.chips:
            img = stats[f"{chip}_CCD"]["rate_mean"]
            tot = stats[f"{chip}_CCD"]["total_sum"]
            var = stats[f"{chip}_VAR"]["total_sum"]

            good = var > 0

            for suffix in ["CCD", "VAR"]:
                ext = f"{chip}_{suffix}"
                good &= stats[ext]["nframe"] > 0.5 * nframe

            snr = np.zeros_like(img)
            snr[good] = np.abs(tot[good]) / np.sqrt(var[good])

            # Welford accumulators run in float64 for numerical stability;
            # the stored master image fits comfortably in float32 (bias signal
            # is a few-ADU offset with ~5 e- read noise) and halves the
            # on-disk size of the IMG and SNR extensions.
            l1_arrays[f"{chip}_IMG"] = img.astype(np.float32)
            l1_arrays[f"{chip}_SNR"] = snr.astype(np.float32)
            l1_arrays[f"{chip}_MASK"] = good

        return l1_arrays

    def finalize_l1_arrays(self, l1_arrays, sigma):
        """
        Interpolate bad pixels and recompute the bad-pixel mask per chip.

        Parameters
        ----------
        l1_arrays : dict
            Per-chip '{chip}_IMG/_SNR/_MASK' arrays from `stack_frames`.
        sigma : float
            Sigma threshold for the final outlier pass on the combined image.

        Returns
        -------
        dict
            The same dict with IMG/SNR interpolated over bad pixels and MASK
            recomputed (True = good).
        """
        for chip in self.chips:
            img = l1_arrays[f"{chip}_IMG"]
            snr = l1_arrays[f"{chip}_SNR"]
            mask = l1_arrays[f"{chip}_MASK"]

            l1_arrays[f"{chip}_IMG"] = interpolate_bad_pixels(img, mask)
            l1_arrays[f"{chip}_SNR"] = interpolate_bad_pixels(snr, mask)

            out = flag_outliers(l1_arrays[f"{chip}_IMG"], sigma, axis=0)
            bad = (l1_arrays[f"{chip}_SNR"] <= 0) | (l1_arrays[f"{chip}_IMG"] == 0)

            l1_arrays[f"{chip}_MASK"] = ~(bad | out)

        return l1_arrays

    def build_master_l1(self, l1_arrays, l0_file_list, *, receipt_key, bunit=None):
        """
        Assemble a KPFMasterL1 from finalized per-chip arrays.

        Parameters
        ----------
        l1_arrays : dict
            Finalized '{chip}_IMG/_SNR/_MASK' arrays.
        l0_file_list : list of str
            L0 files that went into the stack; recorded via set_input_files.
        receipt_key : str
            Receipt entry name (e.g. 'master_bias', 'master_dark').
        bunit : str, optional
            If given, written as the BUNIT header on each '{chip}_IMG'.

        Returns
        -------
        KPFMasterL1
            The populated master L1 object.
        """
        ml1_obj = KPFMasterL1()

        for chip in self.chips:
            ml1_obj.set_data(f"{chip}_IMG", l1_arrays[f"{chip}_IMG"])
            ml1_obj.set_data(f"{chip}_SNR", l1_arrays[f"{chip}_SNR"])
            ml1_obj.set_data(f"{chip}_MASK", l1_arrays[f"{chip}_MASK"])

            if bunit is not None:
                ml1_obj.headers[f"{chip}_IMG"]["BUNIT"] = bunit

        ml1_obj.set_input_files(l0_file_list)
        ml1_obj.receipt_add_entry(receipt_key, "PASS")

        return ml1_obj

    def save_master(self, level, path, *, overwrite=False):
        """
        Write the cached master object to a FITS file at `path`.

        Parameters
        ----------
        level : str
            Data level of the master to save; selects which cached
            object to write. One of 'L1' or 'L2', matching the
            subclass's `make_master_l1` / `make_master_l2` entry point.
        path : str
            Output FITS path. Parent directories are created as needed.
        overwrite : bool, optional
            If False (default), refuse to clobber an existing file and
            raise FileExistsError. If True, replace any existing file
            at `path`. Defaults to False to protect against accidental
            overwrites when called directly; entry points that pass an
            output path through `make_master_lN()` should set True
            explicitly.

        Raises
        ------
        ValueError
            If `level` is not a recognized data level.
        FileExistsError
            If `path` already exists and `overwrite` is False.
        RuntimeError
            If the corresponding make_master_lN() has not been run yet,
            or raised before constructing the master.
        """
        if level not in ("L1", "L2"):
            raise ValueError(f"level must be 'L1' or 'L2'; got {level!r}")

        attr = f"ml{level[1]}_obj"
        obj = getattr(self, attr, None)
        if obj is None:
            raise RuntimeError(
                f"No master available; run make_master_{level.lower()}() first"
            )

        if not overwrite and os.path.exists(path):
            raise FileExistsError(
                f"{path} already exists; pass overwrite=True to replace it"
            )

        os.makedirs(os.path.dirname(path), exist_ok=True)
        obj.to_fits(path)
