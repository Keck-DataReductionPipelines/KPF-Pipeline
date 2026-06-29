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
from kpfpipe.utils.io import build_master_path_from_fits_header
from kpfpipe.utils.stats import flag_outliers, interpolate_bad_pixels


class BaseMasterModule:
    """
    Base class for KPF masters generation.

    The class should not be called directly, but is used for inheritance
    of masters subclasses: Bias, Dark, Flat, WLS. Masters modules read a
    stack of L0 files from disk and output a masters L1 object.

    Each frame is calibrated before stacking/extraction following standard CCD
    reduction: bias gets no calibration, dark is bias-subtracted, flat is
    bias+dark-subtracted, and WLS (like science) is bias+dark-subtracted then
    flat-divided. Each subclass declares its standard set via
    `_STANDARD_CALIBRATIONS`. Which of those actually run is the standard set
    intersected with the resolved bias/dark/flat flags
    (DEFAULTS < [MODULE_IMAGE_PROCESSING] config < make_master kwargs): a flag
    can only turn a standard calibration off, never enable one a master type
    does not use.
    """

    # Module defaults; subclasses extend via `{**BaseMasterModule._DEFAULTS, ...}`.
    # bias/dark/flat are the globally-enabled calibrations (the no-config
    # fallback; in practice resolved from the shared [MODULE_IMAGE_PROCESSING]
    # config and make_master kwargs). Keep in sync with the same keys in
    # image_processing._DEFAULTS.
    _DEFAULTS = {
        **DEFAULTS,
        "stack_sigma": 5.0,
        "bias": True,
        "dark": True,
        "flat": False,
    }

    # The calibrations that are standard for this master type (the ceiling).
    # `_process_frame` applies the intersection of this set with the resolved
    # bias/dark/flat flags. Subclasses override (e.g. Dark -> ("bias",)).
    _STANDARD_CALIBRATIONS = ()

    # Physical pixel units (BUNIT) by master type, written to the IMG
    # extensions in `_build_ml1_obj`. A bias master is a stacked count image
    # (electrons); a dark is normalized to a rate (electrons/sec); a flat is a
    # unitless relative throughput (no BUNIT). Unknown types get no BUNIT.
    _BUNIT_BY_TYPE = {
        "bias": "electrons",
        "dark": "electrons/sec",
        "flat": None,
    }

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

        # Forwarded to CalibrationAssociation in `_process_frame` so an operator's
        # configured search window is honored, not silently reset to the default.
        self._masters_search_window_days = params.get("masters_search_window_days")

        # One cached master (KPFMasterL1) and its path per calibration type.
        # Frames in a stack are taken close in time and so almost always
        # associate the same nearest master; caching lets `_process_frame`
        # read each master from disk once rather than once per frame.
        self._master_ml1 = {}
        self._master_paths = {}

        # populated by subclass make_master_l1(); used by save_master('L1', ...)
        self.ml1_obj = None
        # populated by subclass make_master_l2(); used by save_master('L2', ...)
        self.ml2_obj = None

        # Effective per-frame calibrations (standard set masked by the resolved
        # flags); make_master_l1/l2 re-resolve this with any kwarg overrides.
        self._active_calibrations = self._resolve_calibrations()

    # ------------------------------------------------------------------
    # Private helpers for frame handling (loading, calibration, etc.).
    # ------------------------------------------------------------------

    def _resolve_calibrations(self, *, bias=None, dark=None, flat=None):
        """
        Resolve which calibrations to apply to each frame.

        For each calibration, the value is the per-call override (the make_master
        kwarg if not None, else the config-resolved `self.<name>`, i.e.
        DEFAULTS < [MODULE_IMAGE_PROCESSING]) — but only if the calibration is in
        the per-master standard (`_STANDARD_CALIBRATIONS`); otherwise it is forced
        off. A flag/path can only turn a standard calibration off (or aim it at a
        specific master); it can never enable one outside the master's standard.

        Parameters
        ----------
        bias, dark, flat : bool | str | KPFMasterL1, optional
            Per-call overrides (same accepted forms as `ImageProcessing.perform`:
            True → header-associated master, str → filepath, KPFMasterL1 → that
            object). None means "use the resolved config value".

        Returns
        -------
        dict
            {name: False | True | str | KPFMasterL1} for name in bias/dark/flat.
        """
        overrides = {"bias": bias, "dark": dark, "flat": flat}
        resolved = {}
        for name in ("bias", "dark", "flat"):
            if name not in self._STANDARD_CALIBRATIONS:
                resolved[name] = False
                continue
            request = (
                overrides[name] if overrides[name] is not None else getattr(self, name)
            )
            resolved[name] = request
        return resolved

    def _load_calibration(self, l1_obj, cal_type):
        """
        Resolve one active calibration to a master, caching one per type.

        The source comes from `self._active_calibrations[cal_type]`: True → the
        master associated into the frame header, str → a filepath, KPFMasterL1 →
        an in-memory master. A disk-backed master is read only when its path
        differs from the one already cached for `cal_type`; frames in a stack
        almost always share the same master, so each is read from disk once.
        Falsy (inactive) values and in-memory KPFMasterL1 objects are returned
        unchanged.

        Parameters
        ----------
        l1_obj : KPF1
            The frame being calibrated; its PRIMARY header supplies the
            associated master path for a True calibration.
        cal_type : str
            Calibration name: 'bias', 'dark', or 'flat'.

        Returns
        -------
        bool | str | KPFMasterL1
            A cached KPFMasterL1 for a disk-backed calibration; otherwise the
            source value unchanged (so ImageProcessing handles falsy/invalid).
        """
        value = self._active_calibrations[cal_type]
        if not value or isinstance(value, KPFMasterL1):
            return value

        if isinstance(value, str):
            path = value
        elif value is True:
            path = build_master_path_from_fits_header(l1_obj, cal_type)
        else:
            return value  # let ImageProcessing raise the TypeError

        if self._master_paths.get(cal_type) != path:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Master file not found: {path}")
            self._master_ml1[cal_type] = KPFMasterL1.from_fits(path)
            self._master_paths[cal_type] = path
        return self._master_ml1[cal_type]

    def _load_frame(self, fn, cache, exptime_tolerance=0.1, verbose=True):
        """
        Load an L0 file and perform image assembly to produce an L1 object.

        Parameters
        ----------
        fn : str
            Path to L0 FITS file.
        cache : bool
            If True, retain the assembled L1 in the internal cache for reuse.
            The caller decides which frames are worth caching (see the streaming
            stats path). A frame already cached is returned from the cache
            regardless of this flag.
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
        When `cache=True`, the successfully processed frame is retained to
        reduce redundant I/O and recomputation. The streaming stats path caches
        its approximation-pass frames so the exact pass reuses them.
        """
        if verbose:
            print(f"loading {fn}")

        success = True
        failure = False

        if fn in self._l1_obj_cache:
            l1_obj = self._l1_obj_cache[fn]

        else:
            try:
                l0_obj = KPF0.from_fits(fn)
                l1_obj = ImageAssembly(l0_obj).perform()

                if cache:
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

    def _process_frame(self, l1_obj):
        """
        Apply this module's active calibrations to an assembled frame.

        Subtracts the active masters via ImageProcessing, reusing the standard
        science-path modules rather than reimplementing the math. Calibrations
        requested as `True` are associated first (CalibrationAssociation writes
        their nearest-in-time master into the PRIMARY header); calibrations given
        as an explicit filepath or KPFMasterL1 object skip association and are
        passed straight through. Modules with no active calibrations receive the
        frame unchanged.

        Parameters
        ----------
        l1_obj : KPF1
            Assembled L1 frame to calibrate, mutated in place.

        Returns
        -------
        KPF1
            The same frame with the active calibrations applied.
        """
        calibrations = self._active_calibrations
        active = [name for name, value in calibrations.items() if value]
        if not active:
            return l1_obj

        # Skip frames already calibrated (e.g. a cached frame revisited by the
        # streaming pass) so calibrations are never subtracted twice.
        if all(ImageProcessing.calibration_applied(l1_obj, name) for name in active):
            return l1_obj

        # Only header-driven (True) calibrations need association; explicit
        # filepath / KPFMasterL1 overrides are loaded directly by ImageProcessing.
        cal_types = [name for name, value in calibrations.items() if value is True]
        if cal_types:
            ca_config = {"KPF_MASTERS_OUTPUT": self._masters_root}
            if self._masters_search_window_days is not None:
                ca_config["masters_search_window_days"] = (
                    self._masters_search_window_days
                )
            calibration_association = CalibrationAssociation(l1_obj, ca_config)
            l1_obj = calibration_association.perform(cal_types)

        # Load (and cache) each active master here, then hand the in-memory
        # masters to ImageProcessing so a shared master is read once per stack.
        image_processing = ImageProcessing(l1_obj)
        l1_obj = image_processing.perform(
            bias=self._load_calibration(l1_obj, "bias"),
            dark=self._load_calibration(l1_obj, "dark"),
            flat=self._load_calibration(l1_obj, "flat"),
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
            Assembled L1 object whose INSTRUMENT_HEADER contains EXPTIME and ELAPSED.
        exptime_tolerance : float
            Maximum allowed excess of elapsed time over requested exposure time,
            in seconds.

        Raises
        ------
        ValueError
            If elapsed time is less than requested (premature readout), or if the
            excess exceeds exptime_tolerance.
        """
        exptime = l1_obj.headers["INSTRUMENT_HEADER"]["EXPTIME"]
        elapsed = l1_obj.headers["INSTRUMENT_HEADER"]["ELAPSED"]

        delta = elapsed - exptime
        if delta < 0:
            raise ValueError("premature frame readout detected")
        if delta > exptime_tolerance:
            raise ValueError(f"elapsed time - requested time > {exptime_tolerance}")

    # ------------------------------------------------------------------
    # Private helpers for frame stacking.
    # ------------------------------------------------------------------

    @staticmethod
    def _check_load_failures(failure, n_total, max_fail_fraction, max_fail_number):
        """
        Raise if cumulative frame-load failures exceed either limit.

        Parameters
        ----------
        failure : int
            Number of frames that have failed to load so far.
        n_total : int
            Total number of frames in the stack.
        max_fail_fraction : float
            Maximum allowed fraction of failed frames.
        max_fail_number : int
            Maximum allowed absolute number of failed frames.

        Raises
        ------
        ValueError
            If either limit is exceeded.
        """
        if failure / n_total > max_fail_fraction or failure > max_fail_number:
            raise ValueError(
                f"too many frames failed to load "
                f"({failure} of {n_total}); limits are "
                f"max_fail_fraction={max_fail_fraction:.0%}, "
                f"max_fail_number={max_fail_number}"
            )

    def _compute_stats_from_datacube(
        self,
        l0_file_list=None,
        *,
        sigma=None,
        verbose=True,
        cache=False,
        max_fail_fraction=0.2,
        max_fail_number=2,
        exptime_tolerance=0.1,
    ):
        """
        Compute stacked statistics using an in-memory data cube.

        Parameters
        ----------
        l0_file_list : list of str, optional
            List of L0 FITS filenames to process. The full list is stacked;
            the caller is responsible for passing the desired subset (e.g. the
            streaming path passes only its approximation-pass frames).
        sigma : float, optional
            Sigma threshold for outlier rejection across frames.
        verbose : bool, optional
            If True (default), emit per-frame progress prints and load
            failure warnings from `_load_frame`.
        cache : bool, optional
            If True, cache each loaded frame so a later pass (the streaming
            exact pass) can reuse it without re-reading from disk. Defaults to
            False, since the standalone datacube path reads each frame once.
        max_fail_fraction : float, optional
            Maximum fraction of frames allowed to fail loading before raising.
            Defaults to 0.2.
        max_fail_number : int, optional
            Maximum absolute number of frames allowed to fail loading before
            raising. Defaults to 2. Stacking raises when either limit is
            exceeded.
        exptime_tolerance : float, optional
            Exposure-time tolerance in seconds (default 0.1): the per-frame
            elapsed-vs-requested bound in `_load_frame`, and the threshold at or
            below which a frame's exposure counts as zero (a bias).

        Returns
        -------
        stats : dict
            Per-extension statistics including:
            - 'nframe'      : number of valid frames per pixel
            - 'counts_sum'  : summed counts across valid frames
            - 'rate_mean'   : equal-weight mean of per-frame rates (the center
                              used for streaming clip bounds, not the master IMG)
            - 'rate_rms'    : frame-to-frame sample RMS of the per-frame rates
            On the '{chip}_CCD' entry only:
            - 'exptime_sum' : per-pixel total exposure time over valid frames
                              (the denominator of the exposure-weighted rate in
                              stack_frames; equals the valid-frame count for a
                              bias stack, where T = 1)
        zero_exptime : bool
            True if this is a bias stack (all frames have exposure at or below
            exptime_tolerance), used by the streaming path to validate per-frame
            exposure consistency.

        Notes
        -----
        Exposure time is read from the EPRV-standard PRIMARY EXPTIME (the actual
        elapsed time, mapped from the native WMKO ELAPSED), not the nominal
        requested EXPTIME. Outlier rejection is performed jointly on CCD and VAR
        extensions, in rate space, so frames of differing exposure are
        comparable. Exposure times must be either all zero (≤ tolerance) or all
        above tolerance. Raises an error if more than `max_fail_fraction` of
        frames fail to load.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if sigma is None:
            sigma = self.stack_sigma

        nframe = len(l0_file_list)

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
            l1_obj, success = self._load_frame(
                fn, cache=cache, exptime_tolerance=exptime_tolerance, verbose=verbose
            )

            if not success:
                failure += 1
                self._check_load_failures(
                    failure, len(l0_file_list), max_fail_fraction, max_fail_number
                )
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

        if np.all(exptime > exptime_tolerance):
            zero_exptime = False
            T = exptime[:, None, None]
        elif np.all(exptime <= exptime_tolerance):
            zero_exptime = True
            T = np.ones_like(exptime)[:, None, None]
        else:
            raise ValueError(
                f"Exposure times must be all zero or all non-zero; exptime = {exptime}"
            )

        stats = {}

        for chip in self.chips:
            stats[f"{chip}_CCD"] = {}
            stats[f"{chip}_VAR"] = {}

            # Per-pixel frame-to-frame rejection: at each pixel, flag the frames
            # whose rate deviates from the across-frame median (axis=0 is the
            # frame axis). A frame is rejected if it is an outlier in either the
            # counts rate or the variance rate.
            out = flag_outliers(
                data_cube[f"{chip}_CCD"] / T, sigma, axis=0
            ) | flag_outliers(data_cube[f"{chip}_VAR"] / T, sigma, axis=0)

            valid = ~out
            N = np.sum(~out, axis=0)
            good = N > 1

            # Per-pixel total exposure time over the surviving frames; this is
            # the denominator of the exposure-weighted rate estimate (see
            # stack_frames). T is 1 for a bias stack, so it reduces to N.
            stats[f"{chip}_CCD"]["exptime_sum"] = np.sum(
                np.where(valid, T, 0.0), axis=0
            )

            for suffix in ["CCD", "VAR"]:
                ext = f"{chip}_{suffix}"
                frame_data = data_cube[ext]
                R = frame_data / T

                counts_sum = np.sum(frame_data, axis=0, where=valid)

                rate_mean = np.zeros_like(R[0])
                rate_mean[good] = np.sum(R, axis=0, where=valid)[good] / N[good]

                diff2 = (R - rate_mean) ** 2
                sum_sq_dev = np.sum(diff2, axis=0, where=valid)

                rate_rms = np.zeros_like(R[0])
                rate_rms[good] = np.sqrt(sum_sq_dev[good] / (N[good] - 1))

                stats[ext]["nframe"] = N
                stats[ext]["counts_sum"] = counts_sum
                stats[ext]["rate_mean"] = rate_mean
                stats[ext]["rate_rms"] = rate_rms

        return stats, zero_exptime

    def _compute_stats_from_stream(
        self,
        l0_file_list=None,
        *,
        nstream,
        sigma=None,
        verbose=True,
        max_fail_fraction=0.2,
        max_fail_number=2,
        exptime_tolerance=0.1,
    ):
        """
        Compute stacked statistics with a single streaming pass over the frames.

        Parameters
        ----------
        l0_file_list : list of str, optional
            List of L0 FITS filenames to process.
        nstream : int
            Streaming threshold; the first `nstream - 1` frames form the
            approximation pass that estimates the per-pixel clipping bounds.
        sigma : float, optional
            Sigma threshold for outlier rejection.
        verbose : bool, optional
            If True (default), emit per-frame progress prints and load
            failure warnings from `_load_frame`.
        max_fail_fraction : float, optional
            Maximum fraction of frames allowed to fail loading before raising.
            Defaults to 0.2.
        max_fail_number : int, optional
            Maximum absolute number of frames allowed to fail loading before
            raising. Defaults to 2. Stacking raises when either limit is
            exceeded.
        exptime_tolerance : float, optional
            Exposure-time tolerance in seconds (default 0.1): the per-frame
            elapsed-vs-requested bound in `_load_frame`, and the threshold at or
            below which a frame's exposure counts as zero (a bias).

        Returns
        -------
        exact_stats : dict
            Per-extension statistics needed by `stack_frames`:
            - 'nframe'     : number of valid frames per pixel
            - 'counts_sum' : summed counts across valid frames
            On the '{chip}_CCD' entry only:
            - 'exptime_sum' : per-pixel total exposure time over valid frames
            (Unlike `_compute_stats_from_datacube`, the streaming pass does not
            produce 'rate_mean'/'rate_rms': clip bounds come from the datacube
            approximation below, and the master IMG is counts_sum / exptime_sum.)
        zero_exptime : bool
            True if this is a bias stack (all frames have exposure at or below
            exptime_tolerance).

        Notes
        -----
        An initial subset of frames is processed using the direct data cube
        method to estimate approximate rate mean and RMS. These estimates define
        the per-pixel clipping bounds for the streaming pass.

        Optimized to reduce memory usage at the expense of compute speed.
        Raises an error if frame load failures exceed `max_fail_fraction` or
        `max_fail_number`, or if exposure times are inconsistent.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if sigma is None:
            sigma = self.stack_sigma

        ndirect = nstream - 1

        # The approximation pass stacks only the first `ndirect` frames; cache
        # them so the streaming exact pass below reuses them instead of
        # re-reading those files from disk.
        approx_stats, zero_exptime = self._compute_stats_from_datacube(
            l0_file_list=l0_file_list[:ndirect],
            sigma=sigma,
            verbose=verbose,
            cache=True,
            max_fail_fraction=max_fail_fraction,
            max_fail_number=max_fail_number,
            exptime_tolerance=exptime_tolerance,
        )

        if len(l0_file_list) <= ndirect:
            return approx_stats, zero_exptime

        exact_stats = {}

        nrow = self.ccd["nrow"]
        ncol = self.ccd["ncol"]

        for chip in self.chips:
            for suffix in ["CCD", "VAR"]:
                ext = f"{chip}_{suffix}"

                exact_stats[ext] = {}
                exact_stats[ext]["nframe"] = np.zeros((nrow, ncol), dtype=np.int32)
                exact_stats[ext]["counts_sum"] = np.zeros(
                    (nrow, ncol), dtype=np.float32
                )

                approx_mean = approx_stats[ext]["rate_mean"]
                approx_rms = approx_stats[ext]["rate_rms"]

                approx_stats[ext]["rate_lower"] = approx_mean - approx_rms * sigma
                approx_stats[ext]["rate_upper"] = approx_mean + approx_rms * sigma

            # Total exposure time per pixel, tracked once per chip (CCD and VAR
            # share a survivor mask); the denominator of the exposure-weighted
            # rate in stack_frames. Reduces to the survivor count for a bias
            # stack, where T = 1.
            exact_stats[f"{chip}_CCD"]["exptime_sum"] = np.zeros(
                (nrow, ncol), dtype=np.float32
            )

        failure = 0
        clipping_mask = np.ones((nrow, ncol), dtype=bool)

        for fn in l0_file_list:
            # The first `ndirect` frames are cache hits from the approximation
            # pass above; the rest are read once here and not worth caching.
            l1_obj, success = self._load_frame(
                fn, cache=False, exptime_tolerance=exptime_tolerance, verbose=verbose
            )

            if not success:
                failure += 1
                self._check_load_failures(
                    failure, len(l0_file_list), max_fail_fraction, max_fail_number
                )
                continue

            l1_obj = self._process_frame(l1_obj)

            exptime = l1_obj.headers["PRIMARY"]["EXPTIME"]

            if exptime < 0:
                raise ValueError("Exposure times cannot be negative")

            is_zero = exptime <= exptime_tolerance
            if zero_exptime != is_zero:
                raise ValueError("Exposure times must be all zero or all non-zero")

            T = 1.0 if is_zero else exptime

            for chip in self.chips:
                # Clip each pixel against the rate bounds (joint over CCD/VAR),
                # then accumulate counts and exposure time over the survivors.
                clipping_mask[:] = True

                for suffix in ["CCD", "VAR"]:
                    ext = f"{chip}_{suffix}"
                    rate = l1_obj.data[ext] / T
                    lower = approx_stats[ext]["rate_lower"]
                    upper = approx_stats[ext]["rate_upper"]
                    clipping_mask &= (rate >= lower) & (rate <= upper)

                exact_stats[f"{chip}_CCD"]["exptime_sum"] += T * clipping_mask

                for suffix in ["CCD", "VAR"]:
                    ext = f"{chip}_{suffix}"
                    exact_stats[ext]["nframe"] += clipping_mask
                    exact_stats[ext]["counts_sum"] += l1_obj.data[ext] * clipping_mask

        return exact_stats, zero_exptime

    def _clean_l1_arrays(self, l1_arrays, sigma, cal_type=None):
        """
        Interpolate bad pixels and recompute the bad-pixel mask per chip.

        Parameters
        ----------
        l1_arrays : dict
            Per-chip '{chip}_IMG/_SNR/_MASK' arrays from `stack_frames`.
        sigma : float
            Sigma threshold for the final outlier pass on the combined image.
        cal_type : {'bias', 'dark', 'flat', None}, optional
            Selects the final outlier-flagging mode. The detector is illuminated
            only for flats, so each cal type compares a pixel to a different
            notion of "normal" (in the assembled image dispersion runs along
            axis=1, cross-dispersion along axis=0):

            - 'bias' : per-column median (axis=0) -- each pixel is judged against
              its own cross-dispersion column, removing the CCD's column-wise
              bias structure.
            - 'dark' : global median (no illumination or column structure to
              preserve). `None` falls back to this conservative mode.
            - 'flat' : residual deviation from the smooth illumination trend
              along dispersion (axis=1), tolerating the blaze while catching
              pixels that depart from it.

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

            img_fixed = interpolate_bad_pixels(img, mask)

            if cal_type == "bias":
                out = flag_outliers(img_fixed, sigma, axis=0, method="median")
            elif cal_type == "flat":
                out = flag_outliers(
                    img_fixed, sigma, axis=1, kernel_size=32, method="trend"
                )
            elif cal_type in ("dark", None):
                out = flag_outliers(img_fixed, sigma, method="median")
            else:
                raise ValueError(
                    f"unknown cal_type {cal_type!r}; expected 'bias', 'dark', or 'flat'"
                )

            bad = (img == 0) | (snr <= 0)

            good = mask & ~(bad | out)
            l1_arrays[f"{chip}_MASK"] = good
            l1_arrays[f"{chip}_IMG"] = interpolate_bad_pixels(img, good)
            l1_arrays[f"{chip}_SNR"] = interpolate_bad_pixels(snr, good)

        return l1_arrays

    # ------------------------------------------------------------------
    # Private helpers for building outputs and tracking results.
    # ------------------------------------------------------------------

    def _build_ml1_obj(self, l1_arrays, l0_file_list, *, master_type):
        """
        Assemble a KPFMasterL1 from finalized per-chip arrays.

        Parameters
        ----------
        l1_arrays : dict
            Finalized '{chip}_IMG/_SNR/_MASK' arrays.
        l0_file_list : list of str
            L0 files that went into the stack; recorded via set_input_files.
        master_type : str
            WMKO filename token for the product ('bias', 'dark', 'flat').
            Recorded via set_input_files for the compliant output filename, and
            the single source for both the receipt key (`master_{master_type}`)
            and the BUNIT units (`_BUNIT_BY_TYPE`).

        Returns
        -------
        KPFMasterL1
            The populated master L1 object.
        """
        receipt_key = f"master_{master_type}"
        bunit = self._BUNIT_BY_TYPE.get(master_type)

        ml1_obj = KPFMasterL1()

        for chip in self.chips:
            ml1_obj.set_data(f"{chip}_IMG", l1_arrays[f"{chip}_IMG"])
            ml1_obj.set_data(f"{chip}_SNR", l1_arrays[f"{chip}_SNR"])
            ml1_obj.set_data(f"{chip}_MASK", l1_arrays[f"{chip}_MASK"])

            if bunit is not None:
                ml1_obj.headers[f"{chip}_IMG"]["BUNIT"] = bunit

        ml1_obj.set_input_files(l0_file_list, master_type)
        ml1_obj.receipt_add_entry(receipt_key, "PASS")

        return ml1_obj

    def _populate_info(self, l1_arrays):
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

    def stack_frames(
        self,
        l0_file_list=None,
        nstream=6,
        sigma=None,
        verbose=True,
        cal_type=None,
        max_fail_fraction=0.2,
        max_fail_number=2,
        exptime_tolerance=0.1,
    ):
        """
        Stack full-frame images to produce masters L1.

        Parameters
        ----------
        l0_file_list : list of str, optional
            List of L0 FITS filenames to stack.
        nstream : int, optional
            Threshold number of frames at or above which the memory-light
            streaming path is used instead of the in-memory data cube.
            Defaults to 6.
        sigma : float, optional
            Sigma threshold for frame-to-frame outlier rejection.
        verbose : bool, optional
            If True (default), emit per-frame progress prints and load
            failure warnings during stacking.
        cal_type : {'bias', 'dark', 'flat', None}, optional
            Calibration type, forwarded to `_clean_l1_arrays` to select the
            final outlier-flagging mode. Defaults to the conservative
            global-median mode.
        max_fail_fraction : float, optional
            Maximum fraction of frames allowed to fail loading before raising.
            Defaults to 0.2.
        max_fail_number : int, optional
            Maximum absolute number of frames allowed to fail loading before
            raising. Defaults to 2. Stacking raises when either limit is
            exceeded.
        exptime_tolerance : float, optional
            Exposure-time tolerance in seconds (default 0.1), threaded to the
            stats sub-methods. It bounds the allowed excess of elapsed over
            requested time in `_check_exptime_vs_elapsed`, and is the threshold
            at or below which a frame's exposure counts as zero (a bias).

        Returns
        -------
        l1_arrays : dict
            Dictionary containing per-chip stacked products, with bad pixels
            interpolated and the bad-pixel mask recomputed (`_clean_l1_arrays`):
            - '{chip}_IMG'  : mean count rate FFI
            - '{chip}_SNR'  : signal-to-noise ratio FFI
            - '{chip}_MASK' : boolean bad pixel mask (1 = good, 0 = bad)

        Notes
        -----
        If number of frames is less than `nstream`, statistics are computed
        directly from a full data cube. Otherwise, a single-pass streaming
        accumulation is used to bound memory usage.

        An initial subset of frames is processed using the direct data cube
        method to estimate approximate mean and rms. These estimates define
        per-pixel clipping bounds for the streaming pass.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if sigma is None:
            sigma = self.stack_sigma

        nframe = len(l0_file_list)

        if nframe < 2:
            raise ValueError(f"Stacking requires at least two frames, got {nframe}")

        if nframe < nstream:
            stats, _ = self._compute_stats_from_datacube(
                l0_file_list,
                sigma=sigma,
                verbose=verbose,
                max_fail_fraction=max_fail_fraction,
                max_fail_number=max_fail_number,
                exptime_tolerance=exptime_tolerance,
            )
        else:
            stats, _ = self._compute_stats_from_stream(
                l0_file_list,
                nstream=nstream,
                sigma=sigma,
                verbose=verbose,
                max_fail_fraction=max_fail_fraction,
                max_fail_number=max_fail_number,
                exptime_tolerance=exptime_tolerance,
            )

        for chip in self.chips:
            if np.any(stats[f"{chip}_CCD"]["nframe"] != stats[f"{chip}_VAR"]["nframe"]):
                raise ValueError(
                    f"mismatched frame count between {chip}_CCD and {chip}_VAR"
                )

        l1_arrays = {}
        for chip in self.chips:
            counts = stats[f"{chip}_CCD"]["counts_sum"]
            var_sum = stats[f"{chip}_VAR"]["counts_sum"]
            exptime_sum = stats[f"{chip}_CCD"]["exptime_sum"]

            good = var_sum > 0

            for suffix in ["CCD", "VAR"]:
                ext = f"{chip}_{suffix}"
                good &= stats[ext]["nframe"] > 0.5 * nframe

            good &= exptime_sum > 0

            # Exposure-weighted rate estimate: total counts over total exposure
            # time across the surviving frames (the ML rate under Poisson
            # statistics, correct for mixed exposures). For a bias stack
            # exptime_sum is the survivor count, so this is the mean in electrons.
            img = np.zeros_like(counts)
            img[good] = counts[good] / exptime_sum[good]

            # SNR of the rate is invariant to the exposure normalization: the
            # total exposure cancels in |sum(counts)| / sqrt(sum(var)).
            snr = np.zeros_like(img)
            snr[good] = np.abs(counts[good]) / np.sqrt(var_sum[good])

            # The stored master image fits comfortably in float32 (bias signal
            # is a few-ADU offset with ~5 e- read noise), which halves the
            # on-disk size of the IMG and SNR extensions.
            l1_arrays[f"{chip}_IMG"] = img.astype(np.float32)
            l1_arrays[f"{chip}_SNR"] = snr.astype(np.float32)
            l1_arrays[f"{chip}_MASK"] = good

        # Interpolate bad pixels and recompute the bad-pixel mask before return.
        return self._clean_l1_arrays(l1_arrays, sigma, cal_type=cal_type)

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

        # obj.to_fits creates the parent directory as needed.
        obj.to_fits(path)
