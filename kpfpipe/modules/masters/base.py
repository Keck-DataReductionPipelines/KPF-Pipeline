"""
Base class for KPF Masters modules.
"""

import logging
import os

import numpy as np

from kpfpipe import DEFAULTS
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.masters.level1 import KPFMasterL1
from kpfpipe.modules.calibration_association import CalibrationAssociation
from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.modules.image_processing import ImageProcessing
from kpfpipe.modules.spectral_extraction import SpectralExtraction
from kpfpipe.quality_control.diagnostics import DiagL0
from kpfpipe.quality_control.qc_flags import QCL0
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import flag_outliers, interpolate_bad_pixels

logger = logging.getLogger(__name__)


class BaseMasterModule:
    """
    Base class for KPF masters generation.

    Not called directly; the concrete masters subclasses (Bias, Dark, Flat,
    WLS) inherit it. A masters module reads a stack of L0 files from disk and
    outputs a masters L1 object. Each frame is calibrated before
    stacking/extraction following standard CCD reduction: a bias gets no
    calibration, a dark is bias-subtracted, and a flat and WLS are bias- and
    dark-subtracted.

    Parameters
    ----------
    l0_file_list : list of str
        Sorted list of L0 FITS file paths to stack (ascending order is
        required).
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: stack_sigma, min_stack_size,
        bias, dark, flat, masters_search_window_days, KPF_MASTERS_OUTPUT.
    """

    # Module defaults; subclasses extend via ``{**BaseMasterModule._DEFAULTS, ...}``.
    # bias/dark/flat are the globally-enabled calibrations (the no-config
    # fallback; in practice resolved from the shared [MODULE_IMAGE_PROCESSING]
    # config and make_master kwargs). Keep in sync with the same keys in
    # image_processing._DEFAULTS.
    _DEFAULTS = {
        **DEFAULTS,
        "stack_sigma": 5.0,
        "min_stack_size": 5,
        "bias": True,
        "dark": True,
        "flat": False,
    }

    # The calibrations that are standard for this master type (the ceiling).
    # ``_process_frame`` applies the intersection of this set with the resolved
    # bias/dark/flat flags. Subclasses override (e.g. Dark -> ("bias",)).
    _STANDARD_CALIBRATIONS = ()

    # Physical pixel units (BUNIT) by master type, written to the IMG
    # extensions in ``_build_ml1_obj``. A bias master is a stacked count image
    # (electrons); a dark is normalized to a rate (electrons/sec); a flat is the
    # total electrons summed over the stack (electrons). Unknown types get no BUNIT.
    _BUNIT_BY_TYPE = {
        "bias": "electrons",
        "dark": "electrons/sec",
        "flat": "electrons",
    }

    # QCL0 flags a frame must pass to enter a stack: data present and not
    # observer-junk. A frame failing either is dropped in ``_load_frame`` and
    # counted as a load failure. Deliberately loosened while the QC suite is
    # overhauled: KWRDPRL0 writes no card (its check is stubbed) and EXPTIMOK
    # leaves the tuple with it, since a flag with no card raises a bare KeyError
    # at the `qc[kw][0]` read below rather than rejecting the frame.
    _REQUIRED_L0_QC_FLAGS = ("DATAPRL0", "NOTJUNK")

    # Exposure-time threshold (seconds) for the bias/zero-exposure decision in
    # the stats methods: a frame whose exposure is <= this counts as zero
    # (a bias, stacked with T=1), else as a real exposure (T=exptime).
    _ZERO_EXPTIME_TOL_SECONDS = 0.1

    def __init__(self, l0_file_list, config=None):
        if l0_file_list != sorted(l0_file_list):
            raise ValueError("l0_file_list must be sorted in ascending order")
        self.l0_file_list = l0_file_list

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS", "TRACES"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in self._DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self._l1_obj_cache = {}

        # Masters output root; CalibrationAssociation reads masters from here
        # when ``_process_frame`` associates a calibration for a stacked frame.
        self._masters_output = params.get("KPF_MASTERS_OUTPUT")

        # Forwarded to CalibrationAssociation in ``_process_frame`` so an operator's
        # configured search window is honored, not silently reset to the default.
        self._masters_search_window_days = params.get("masters_search_window_days")

        # One cached master (KPFMasterL1) and its path per calibration type.
        # Frames in a stack are taken close in time and so almost always
        # associate the same nearest master; caching lets ``_process_frame``
        # read each master from disk once rather than once per frame.
        self._master_ml1 = {}
        self._master_paths = {}

        # populated by subclass make_master_l1(); used by save_master('L1', ...)
        self.ml1_obj = None
        # populated by subclass make_master_l2(); used by save_master('L2', ...)
        self.ml2_obj = None

        # L0 files that actually stacked; recorded as the master's INPUT_FILES.
        self._stacked_files = []

        # Per-chip stack statistics cached by _populate_stack_info() (dict);
        # consumed by the subclass _track_info() when it builds the info() text.
        self._stack_info = None

        # Effective per-frame calibrations (standard set masked by the resolved
        # flags); make_master_l1/l2 re-resolve this with any kwarg overrides.
        self._active_calibrations = self._resolve_calibrations()

    # ------------------------------------------------------------------
    # Private helpers for frame handling (loading, calibration, etc.).
    # ------------------------------------------------------------------

    def _resolve_calibrations(self, *, bias=None, dark=None, flat=None):
        """
        Resolve which calibrations apply per frame: {name: False|True|str|KPFMasterL1}.

        For bias/dark/flat, take the per-call override (make_master kwarg if not
        None, else config-resolved ``self.<name>``) only if the calibration is in
        this master's ``_STANDARD_CALIBRATIONS``, else force it off. A flag/path can
        only disable a standard calibration (or aim it at a specific master), never
        enable one outside the standard. Override forms match
        ``ImageProcessing.perform``.
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

        Source is ``self._active_calibrations[cal_type]``: True → master associated
        into the frame header, str → filepath, KPFMasterL1 → in-memory master. A
        disk-backed master is read only when its path differs from the cached one
        (frames in a stack almost always share a master, so each is read once).
        Falsy and in-memory values are returned unchanged.
        """
        value = self._active_calibrations[cal_type]
        if not value or isinstance(value, KPFMasterL1):
            return value

        if isinstance(value, str):
            path = value
        elif value is True:
            path = l1_obj.headers["RECEIPT"][f"{cal_type.upper()}FILE"]
        else:
            return value  # let ImageProcessing raise the TypeError

        if self._master_paths.get(cal_type) != path:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Master file not found: {path}")
            self._master_ml1[cal_type] = KPFMasterL1.from_fits(path)
            self._master_paths[cal_type] = path
        return self._master_ml1[cal_type]

    def _load_frame(self, fn, cache=False):
        """
        Load an L0 file and assemble it to an L1 object; None on failure.

        Failure means the frame could not be read/assembled or failed a required
        QCL0 flag (``_REQUIRED_L0_QC_FLAGS``, incl. the EXPTIME/ELAPSED consistency
        check); a warning names the cause and callers detect it via ``is None``.
        With ``cache=True`` the assembled L1 is retained for reuse (the streaming
        stats path caches its approximation-pass frames so the exact pass reuses
        them); an already-cached frame is returned regardless of the flag.
        """
        if fn in self._l1_obj_cache:
            return self._l1_obj_cache[fn]

        try:
            l0_obj = KPF0.from_fits(fn, standardize=True)

            DiagL0(l0_obj).run()
            qc = QCL0(l0_obj).run()
            failed = [kw for kw in self._REQUIRED_L0_QC_FLAGS if not qc[kw][0]]
            if failed:
                logger.warning("QC failed for %s: %s", fn, ", ".join(failed))
                return None

            l1_obj = ImageAssembly(l0_obj).perform()

            if cache:
                self._l1_obj_cache[fn] = l1_obj

        except (FileNotFoundError, OSError) as e:
            logger.warning("Failed to load %s: %s", fn, e)
            return None

        return l1_obj

    def _process_frame(self, l1_obj):
        """
        Apply this module's active calibrations to an assembled frame (in place).

        Subtracts active masters via ImageProcessing (reusing the science-path
        modules). ``True`` calibrations are associated first (nearest-in-time
        master written to PRIMARY); explicit filepath/KPFMasterL1 overrides pass
        straight through. No active calibrations → frame returned unchanged.
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
            ca_config = {"KPF_MASTERS_OUTPUT": self._masters_output}
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

    def _extract_frame(self, l1_obj):
        """
        Extract an assembled frame to L2 (for spectral masters, e.g. WLS).

        Spectral extraction only -- calibration is not implicit; callers needing
        bias/dark applied must run ``_process_frame`` first.
        """
        spectral_extraction = SpectralExtraction(l1_obj)
        return spectral_extraction.perform()

    # ------------------------------------------------------------------
    # Private helpers for frame stacking.
    # ------------------------------------------------------------------

    @staticmethod
    def _check_load_failures(failure, n_total, max_fail_fraction, max_fail_number):
        """
        Raise ValueError if cumulative frame-load failures exceed either the
        fraction (``max_fail_fraction``) or absolute (``max_fail_number``) limit.
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
        cache=False,
        max_fail_fraction=0.2,
        max_fail_number=2,
    ):
        """
        Compute stacked statistics using an in-memory data cube.

        Stacks the full ``l0_file_list`` (caller passes the desired subset). With
        ``cache=True`` each loaded frame is retained so a later pass (the streaming
        exact pass) reuses it. Raises when frame-load failures exceed
        ``max_fail_fraction`` or ``max_fail_number``.

        Returns ``(stats, zero_exptime)``. ``stats`` is per-extension with 'nframe',
        'counts_sum', 'rate_mean' (equal-weight mean of per-frame rates -- the clip
        center, not the master IMG), 'rate_rms' (frame-to-frame sample RMS), and on
        '{chip}_CCD' only 'exptime_sum' (per-pixel total exposure over valid frames;
        the exposure-weighted-rate denominator in stack_frames, = valid-frame count
        for a bias stack where T=1). ``zero_exptime`` is True for a bias stack (all
        exposures ≤ ``_ZERO_EXPTIME_TOL_SECONDS``).

        Notes
        -----
        Exposure time is the EPRV PRIMARY EXPTIME (elapsed, mapped from native
        ELAPSED), not the nominal requested EXPTIME. Rejection is on the CCD rate
        (VAR = |CCD| + RN adds no independent info); rates make differing-exposure
        frames comparable, and VAR is still summed for the SNR. Exposures must be
        all zero (≤ ``_ZERO_EXPTIME_TOL_SECONDS``) or all above it.
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
        failure_count = 0
        self._stacked_files = []

        for fn in l0_file_list:
            l1_obj = self._load_frame(fn, cache=cache)

            if l1_obj is None:
                failure_count += 1
                self._check_load_failures(
                    failure_count, len(l0_file_list), max_fail_fraction, max_fail_number
                )
                continue

            self._stacked_files.append(fn)
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

        if np.all(exptime > self._ZERO_EXPTIME_TOL_SECONDS):
            zero_exptime = False
            T = exptime[:, None, None]
        elif np.all(exptime <= self._ZERO_EXPTIME_TOL_SECONDS):
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

            # Per-pixel frame-to-frame rejection on the CCD rate (axis=0 is the
            # frame axis). VAR = |CCD| + RN is monotonic in |CCD|, so clipping it
            # adds no independent info; VAR is only summed below for the SNR.
            out = flag_outliers(data_cube[f"{chip}_CCD"] / T, sigma, axis=0)

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
        max_fail_fraction=0.2,
        max_fail_number=2,
    ):
        """
        Compute stacked statistics with a single streaming pass over the frames
        (memory-light, at the cost of compute speed).

        The first ``nstream - 1`` frames are stacked via the datacube method to
        estimate per-pixel rate mean/RMS, which set the clipping bounds for the
        streaming pass. Raises when load failures exceed
        ``max_fail_fraction``/``max_fail_number`` or exposures are inconsistent.

        Returns ``(exact_stats, zero_exptime)``. ``exact_stats`` carries only
        'nframe', 'counts_sum', and (on '{chip}_CCD') 'exptime_sum' -- no
        'rate_mean'/'rate_rms', since clip bounds come from the approximation and
        the master IMG is counts_sum / exptime_sum.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if sigma is None:
            sigma = self.stack_sigma

        ndirect = nstream - 1

        # The approximation pass stacks only the first ``ndirect`` frames; cache
        # them so the streaming exact pass below reuses them instead of
        # re-reading and re-assembling those files from disk.
        approx_stats, zero_exptime = self._compute_stats_from_datacube(
            l0_file_list=l0_file_list[:ndirect],
            sigma=sigma,
            cache=True,
            max_fail_fraction=max_fail_fraction,
            max_fail_number=max_fail_number,
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
            # rate in stack_frames. Reduces to survivor count for a bias stack.
            exact_stats[f"{chip}_CCD"]["exptime_sum"] = np.zeros(
                (nrow, ncol), dtype=np.float32
            )

        failure = 0
        clipping_mask = np.ones((nrow, ncol), dtype=bool)
        self._stacked_files = []

        for fn in l0_file_list:
            # The first ``ndirect`` frames are cache hits from the approximation
            # pass above; the rest are read once here and not worth caching.
            l1_obj = self._load_frame(fn, cache=False)

            if l1_obj is None:
                failure += 1
                self._check_load_failures(
                    failure, len(l0_file_list), max_fail_fraction, max_fail_number
                )
                continue

            self._stacked_files.append(fn)
            l1_obj = self._process_frame(l1_obj)

            exptime = l1_obj.headers["PRIMARY"]["EXPTIME"]

            if exptime < 0:
                raise ValueError("Exposure times cannot be negative")

            is_zero = exptime <= self._ZERO_EXPTIME_TOL_SECONDS
            if zero_exptime != is_zero:
                raise ValueError("Exposure times must be all zero or all non-zero")

            T = 1.0 if is_zero else exptime

            for chip in self.chips:
                # Clip each pixel against the CCD rate bounds (VAR = |CCD| + RN
                # adds no independent info); the CCD mask is applied to both below.
                ext = f"{chip}_CCD"
                rate = l1_obj.data[ext] / T
                lower = approx_stats[ext]["rate_lower"]
                upper = approx_stats[ext]["rate_upper"]
                clipping_mask[:] = (rate >= lower) & (rate <= upper)

                exact_stats[f"{chip}_CCD"]["exptime_sum"] += T * clipping_mask

                for suffix in ["CCD", "VAR"]:
                    ext = f"{chip}_{suffix}"
                    exact_stats[ext]["nframe"] += clipping_mask
                    exact_stats[ext]["counts_sum"] += l1_obj.data[ext] * clipping_mask

        return exact_stats, zero_exptime

    def _clean_l1_arrays(self, l1_arrays, sigma, cal_type=None):
        """
        Interpolate bad pixels and recompute the per-chip bad-pixel mask
        (True = good) on the stacked '{chip}_IMG/_SNR/_MASK' arrays.

        ``cal_type`` selects the final outlier-flagging mode, since the detector is
        illuminated only for flats (in the assembled image dispersion runs along
        axis=1, cross-dispersion along axis=0):

        - 'bias' : per-column median (axis=0) -- judges each pixel against its own
          cross-dispersion column, removing column-wise bias structure.
        - 'dark'/None : global median (no illumination/column structure to preserve).
        - 'flat' : residual deviation from the smooth illumination trend along
          dispersion (axis=1), tolerating the blaze while catching departures.
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

    def _build_ml1_obj(self, l1_arrays, *, master_type):
        """
        Assemble a KPFMasterL1 from finalized per-chip '{chip}_IMG/_SNR/_MASK'
        arrays. ``master_type`` ('bias'/'dark'/'flat') drives the WMKO filename
        token, the receipt key (``master_{master_type}``), and the BUNIT units
        (``_BUNIT_BY_TYPE``). INPUT_FILES records ``self._stacked_files``.
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

        ml1_obj.set_input_files(self._stacked_files, master_type)
        ml1_obj.receipt_add_entry(receipt_key, "", "PASS")

        return ml1_obj

    def _populate_stack_info(self, l1_arrays):
        """
        Cache per-chip master statistics ({'num_bad','pct_bad','median','rms'}) on
        ``self._stack_info`` for the subclass ``_track_info()``.
        """
        self._stack_info = {
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
        cal_type=None,
        max_fail_fraction=0.2,
        max_fail_number=2,
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
        cal_type : {'bias', 'dark', 'flat', None}, optional
            Calibration type, forwarded to ``_clean_l1_arrays`` to select the
            final outlier-flagging mode. Defaults to the conservative
            global-median mode.
        max_fail_fraction : float, optional
            Maximum fraction of frames allowed to fail loading before raising.
            Defaults to 0.2.
        max_fail_number : int, optional
            Maximum absolute number of frames allowed to fail loading before
            raising. Defaults to 2. Stacking raises when either limit is
            exceeded.

        Returns
        -------
        l1_arrays : dict
            Dictionary containing per-chip stacked products, with bad pixels
            interpolated and the bad-pixel mask recomputed (``_clean_l1_arrays``):
            - '{chip}_IMG'  : mean count rate FFI
            - '{chip}_SNR'  : signal-to-noise ratio FFI
            - '{chip}_MASK' : boolean bad pixel mask (1 = good, 0 = bad)

        Notes
        -----
        If number of frames is less than ``nstream``, statistics are computed
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

        logger.debug(
            "stacking %d frames via %s method",
            nframe,
            "datacube" if nframe < nstream else "streaming",
        )

        if nframe < nstream:
            stats, _ = self._compute_stats_from_datacube(
                l0_file_list,
                sigma=sigma,
                max_fail_fraction=max_fail_fraction,
                max_fail_number=max_fail_number,
            )
        else:
            stats, _ = self._compute_stats_from_stream(
                l0_file_list,
                nstream=nstream,
                sigma=sigma,
                max_fail_fraction=max_fail_fraction,
                max_fail_number=max_fail_number,
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

            img = np.zeros_like(counts)
            if cal_type == "flat":
                # Master flat: total electrons summed over the stack.
                img[good] = counts[good]
            else:
                # Exposure-weighted rate: counts / total exposure (bias: /count).
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

        return self._clean_l1_arrays(l1_arrays, sigma, cal_type=cal_type)

    def save_master(self, level, path, *, overwrite=False):
        """
        Write the cached master object to a FITS file at ``path``.

        Parameters
        ----------
        level : str
            Data level of the master to save; selects which cached
            object to write. One of 'L1' or 'L2', matching the
            subclass's ``make_master_l1`` / ``make_master_l2`` entry point.
        path : str
            Output FITS path. Parent directories are created as needed.
        overwrite : bool, optional
            If False (default), refuse to clobber an existing file and
            raise FileExistsError. If True, replace any existing file
            at ``path``. Defaults to False to protect against accidental
            overwrites when called directly; entry points that pass an
            output path through ``make_master_lN()`` should set True
            explicitly.

        Raises
        ------
        ValueError
            If ``level`` is not a recognized data level.
        FileExistsError
            If ``path`` already exists and ``overwrite`` is False.
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
