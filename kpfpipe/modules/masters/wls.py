"""
KPF Master Wavelength Solution construction module.
"""

import os
import warnings

import h5py
import numpy as np
import pandas as pd
from astropy.stats import mad_std
from numpy.polynomial import legendre

from kpfpipe import REPO_ROOT
from kpfpipe.data_models.masters import KPFMasterL2
from kpfpipe.modules.masters.base import BaseMasterModule
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import optimize_lsq

_ROUGH_WLS_FILE = f"{REPO_ROOT}/reference/rough_wls_fallback.csv"


class WLS(BaseMasterModule):
    """
    Construct a master wavelength solution from a stack of WLS L0 exposures.

    Each frame is processed individually through the L0→L2 pipeline. Fitted
    line positions across the stack are combined to derive per-fiber
    wavelength solutions.

    Standard reduction: like science, a WLS frame is bias- and dark-subtracted
    before extraction (`_STANDARD_CORRECTIONS = ("bias", "dark")`); flat
    division is part of the standard but stays off until it is implemented.

    Parameters
    ----------
    l0_file_list : list of str
        Sorted list of L0 FITS file paths to process.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: linelist, lineprofile,
        polyorder_x, polyorder_m, polyorder_f, chips, fibers,
        KPF_MASTERS_OUTPUT.
    """

    _DEFAULTS = {
        **BaseMasterModule._DEFAULTS,
        "linelist": f"{REPO_ROOT}/reference/thar_line_list.csv",
        "lineprofile": "gaussian",
        "polyorder_x": 6,
        "polyorder_m": 6,
        "polyorder_f": 2,
    }

    # Bias+dark subtraction is standard for WLS; `_process_frame` (run before
    # `_extract_frame`) applies whichever of these the config has enabled.
    _STANDARD_CORRECTIONS = ("bias", "dark")

    def __init__(self, l0_file_list, config=None):
        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(
                ["DATA_DIRS", "KPFPIPE", "WLS", "MODULE_IMAGE_PROCESSING"]
            )
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")
        super().__init__(l0_file_list, params)
        # WLS extraction associates a master bias for each ThAr frame. Masters
        # (including the bias the masters recipe just built) live under
        # KPF_MASTERS_OUTPUT, which is also where CalibrationAssociation reads
        # them, so there is no input/output mismatch.
        self._masters_root = params.get("KPF_MASTERS_OUTPUT")
        self.rough_wls_file = _ROUGH_WLS_FILE

        self._load_rough_wls()
        self._load_linelist()

        self._l2_obj_cache = []  # populated by process_stack_l0_to_l2()
        self._results = None  # populated by make_master_l2()
        self._coeffs_stack = (
            None  # populated by make_master_l2(); used by save_diagnostics()
        )
        self._lines_stack = (
            None  # populated by make_master_l2(); used by save_diagnostics()
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_linelist(self, linelist=None):
        """
        Return the cached line list DataFrame, loading or reloading if needed.

        Columns: CHIP, ORDER (0-indexed per chip), WAVE [Å, vacuum]. Loads from
        disk when nothing is cached, or when `linelist` (a file path) differs
        from the cached `self.linelist`; otherwise returns the cache unchanged.
        """
        needs_load = not hasattr(self, "_linelist_df")
        if linelist is not None and linelist != self.linelist:
            self.linelist = linelist
            needs_load = True
        if needs_load:
            self._linelist_df = pd.read_csv(self.linelist)
        return self._linelist_df

    def _load_rough_wls(self, rough_wls_file=None):
        """
        Return the cached rough WLS dict, loading or reloading if needed.

        Loads from disk when no rough WLS is cached yet, or when
        `rough_wls_file` (a file path) differs from the cached
        `self.rough_wls_file`. Otherwise returns the cache unchanged. The
        cache is transparently kept in sync with the most recently
        passed-in file path.
        """
        needs_load = not hasattr(self, "rough_wls")
        if rough_wls_file is not None and rough_wls_file != self.rough_wls_file:
            self.rough_wls_file = rough_wls_file
            needs_load = True
        if needs_load:
            df = pd.read_csv(self.rough_wls_file)

            ncol = self.ccd["ncol"]
            # Per-order Legendre coefficients (C0..Cn) evaluated on the
            # normalized pixel grid; see scripts/build_rough_wls_from_legacy_wls.py.
            coeff_cols = sorted(
                (c for c in df.columns if c.startswith("C") and c[1:].isdigit()),
                key=lambda c: int(c[1:]),
            )
            x = 2 * np.arange(ncol) / (ncol - 1) - 1
            self.rough_wls = {}

            for chip in self.chips:
                norder = self.norder[chip]
                for fiber in self.fibers:
                    channel_ext = f"{chip}_{fiber}_WAVE"
                    self.rough_wls[channel_ext] = np.zeros((norder, ncol))

                    for o in range(norder):
                        use = (df.CHIP == chip) & (df.ORDER == o)
                        coeffs = df.loc[use, coeff_cols].values[0]
                        self.rough_wls[channel_ext][o] = legendre.legval(x, coeffs)

        return self.rough_wls

    def _line_fit_qc(self, lines, lineprofile, window, loc):
        """
        Quality-control the per-line fits and return a boolean flag array
        (True = line failed QC), aligned with the per-line arrays in `lines`.

        `loc` is the per-line window-center pixel; a centroid more than
        `window` pixels from it is flagged as a runaway fit.
        """
        if lineprofile != "gaussian":
            raise ValueError(f"Unsupported lineprofile: {lineprofile}")

        bad = (lines["amp"] < 0) | (lines["amp"] > 1.5e6)  # 10x single-pixel saturation
        bad |= (lines["std"] < 0.5) | (lines["std"] >= window)
        bad |= np.abs(lines["pix"] - loc) > window

        return bad

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def process_stack_l0_to_l2(self, l0_file_list=None, verbose=True):
        """
        Run each L0 frame in the stack through the L0→L2 pipeline.

        Parameters
        ----------
        l0_file_list : list of str, optional
            L0 files to process. Defaults to self.l0_file_list.
        verbose : bool, optional
            If True (default), emit progress prints and per-frame warnings
            from the underlying L0 → L1 → L2 calls.

        Returns
        -------
        list of KPF2
            Extracted L2 objects for all successfully processed frames.

        Notes
        -----
        Resets self._l2_obj_cache at entry. Raises ValueError if more than
        20% of frames fail to load.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list

        if len(l0_file_list) == 0:
            raise ValueError("Empty l0_file_list; must supply at least one valid file")

        self._l2_obj_cache = []
        failure = 0

        for fn in l0_file_list:
            l1_obj, success = self._load_frame(fn, ncache=0, verbose=verbose)

            if not success:
                failure += 1
                if failure / len(l0_file_list) > 0.2:
                    raise ValueError("more than 20% of frames in stack failed to load")
                continue

            l1_obj = self._process_frame(l1_obj)
            l2_obj = self._extract_frame(l1_obj, verbose=verbose)
            self._l2_obj_cache.append(l2_obj)

        return self._l2_obj_cache

    def fit_line_positions_1d(
        self,
        flux1d,
        wave1d,
        line_waves,
        lineprofile="gaussian",
        window=5,
    ):
        """
        Fit line positions (in pixel space) along a 1D extracted order.

        Fits a 1D line model over a ±`window` pixel neighborhood around each
        reference line in `line_waves` (the lines pre-selected for this order),
        then quality-controls the fits via `_line_fit_qc`. The rough WLS span
        of `wave1d` is a guardrail: a reference line outside it raises.

        Parameters
        ----------
        flux1d : ndarray
            1D extracted flux for a single order.
        wave1d : ndarray
            1D rough wavelength grid for the same order.
        line_waves : ndarray
            Reference line wavelengths to fit in this order [Å], already
            selected for the order by the caller.
        lineprofile : str, optional
            Line profile model name. See kpfpipe.utils.stats._FUNCTIONS
            for supported values.
        window : int, optional
            Half-width of the fit window, in pixels.

        Returns
        -------
        lines : dict of ndarray
            Per-line arrays of equal length. All entries are retained
            regardless of QC status; the caller is responsible for
            filtering on 'bad' before downstream use. Lines whose fit
            window contains any non-finite flux are dropped silently
            before fitting. If the input order is entirely non-finite
            (e.g. an extraction-failed orderlet filled with NaN), all
            arrays are returned empty. Keys:
              'wav' - reference line wavelength
              'pix' - fitted pixel position
              'std' - fitted line sigma (Gaussian width)
              'amp' - fitted line amplitude
              'bad' - boolean QC flag (True = line failed QC)
        """
        # Fit in float64 (wavelength solutions are float64).
        flux1d = np.asarray(flux1d, dtype=np.float64)
        wave1d = np.asarray(wave1d, dtype=np.float64)
        line_waves = np.asarray(line_waves, dtype=np.float64)

        if len(flux1d) != len(wave1d):
            raise ValueError("length of flux and wave arrays are mismatched")
        ncol = len(flux1d)

        lines = {k: np.zeros(0, dtype="float") for k in ["wav", "pix", "std", "amp"]}
        lines["bad"] = np.zeros(0, dtype=bool)

        if not np.isfinite(flux1d).any():
            return lines

        candidate_wavs = np.sort(line_waves)

        # Guardrail: supplied lines must lie within this order's rough WLS span;
        # an out-of-range line signals a CHIP/ORDER labeling inconsistency.
        lo, hi = wave1d.min(), wave1d.max()
        outside = (candidate_wavs < lo) | (candidate_wavs > hi)
        if np.any(outside):
            raise ValueError(
                f"{int(np.sum(outside))} reference line(s) outside the rough "
                f"WLS range [{lo:.3f}, {hi:.3f}] A; line-list CHIP/ORDER labels "
                f"are inconsistent with the rough WLS"
            )

        keep = np.zeros(len(candidate_wavs), dtype=bool)
        for i, lw in enumerate(candidate_wavs):
            loc = np.argmin(np.abs(wave1d - lw))
            cols = np.arange(loc - window, loc + window + 1)
            cols = cols[(cols >= 0) & (cols < ncol)]
            keep[i] = np.isfinite(flux1d[cols]).all()

        lines["wav"] = candidate_wavs[keep]
        nlines = len(lines["wav"])

        if nlines == 0:
            return lines

        for key in ["pix", "std", "amp"]:
            lines[key] = np.zeros(nlines, dtype="float")

        locs = np.zeros(nlines, dtype="float")
        for i, lw in enumerate(lines["wav"]):
            loc = np.argmin(np.abs(wave1d - lw))
            locs[i] = loc
            cols = np.arange(loc - window, loc + window + 1)
            cols = cols[(cols >= 0) & (cols < ncol)]

            x = cols
            y = flux1d[cols]
            theta, _ = optimize_lsq(x, y, lineprofile)

            if lineprofile == "gaussian":
                # gaussian_dist theta convention: [b, a, mu, sigma]
                lines["pix"][i] = theta[2]
                lines["std"][i] = theta[3]
                lines["amp"][i] = theta[1]
            else:
                raise ValueError(f"Unsupported lineprofile: {lineprofile}")

        lines["bad"] = self._line_fit_qc(lines, lineprofile, window, locs)

        return lines

    def fit_line_positions_ffi(
        self,
        l2_obj,
        chip,
        fibers,
        linelist=None,
        lineprofile=None,
        window=5,
        verbose=True,
    ):
        """
        Fit line positions across all orders and fibers of one chip.

        Loops over the requested fibers, calling `fit_line_positions_1d`
        on each (order, fiber) extracted spectrum, and concatenates the
        surviving lines into flat arrays tagged with their order number
        and fiber name.

        Parameters
        ----------
        l2_obj : KPF2
            Extracted L2 object containing per-fiber FLUX arrays of shape
            (norder, ncol).
        chip : str
            Chip identifier ('GREEN' or 'RED').
        fibers : list of str
            Fiber identifiers (e.g. ['SCI1', 'SCI2', 'SCI3']).
        linelist : str, optional
            Path to a CSV line list. If different from the currently
            cached `self.linelist`, the file is reloaded and the cache
            is updated. Defaults to `self.linelist` (no reload).
        lineprofile : str, optional
            Line profile model name. See kpfpipe.utils.stats._FUNCTIONS.
        window : int, optional
            Half-width of the per-line fit window, in pixels.
        verbose : bool, optional
            If True, print a progress line for each fiber.

        Returns
        -------
        lines : dict of ndarray
            Flat 1D arrays, all of equal length. All lines are retained
            regardless of QC status; the caller is responsible for
            filtering on 'bad' before downstream use. All per-line keys
            produced by `fit_line_positions_1d` are carried through, plus
            'order' and 'fiber' which tag each line with its source order
            and fiber. Keys:
              'wav' - reference line wavelength
              'pix' - fitted pixel position
              'std' - fitted line sigma (Gaussian width)
              'amp' - fitted line amplitude
              'bad' - boolean QC flag (True = line failed QC)
              'order' - 1-indexed order number
              'fiber' - fiber name
        """
        linelist_df = self._load_linelist(linelist)

        if lineprofile is None:
            lineprofile = self.lineprofile

        norder = self.norder[chip]
        # keys mirror fit_line_positions_1d's output plus per-line provenance tags
        keys = ("wav", "pix", "std", "amp", "bad", "order", "fiber")
        lines = {k: [[None] * norder for _ in fibers] for k in keys}

        for i, fiber in enumerate(fibers):
            if verbose:
                print(f"fitting {chip} {fiber} line positions")

            flux_arr = l2_obj.data[f"{chip}_{fiber}_FLUX"]
            wave_arr = self.rough_wls[f"{chip}_{fiber}_WAVE"]

            if np.shape(flux_arr) != np.shape(wave_arr):
                raise ValueError("shape mismatch between flux array and rough WLS")

            for o in range(norder):
                line_waves = linelist_df.loc[
                    (linelist_df["CHIP"] == chip) & (linelist_df["ORDER"] == o), "WAVE"
                ].to_numpy(dtype=float)
                line_dict = self.fit_line_positions_1d(
                    flux_arr[o],
                    wave_arr[o],
                    line_waves,
                    lineprofile=lineprofile,
                    window=window,
                )

                nlines = len(line_dict["wav"])
                if nlines == 0 and verbose:
                    warnings.warn(
                        f"{chip} {fiber} order {o + 1}: orderlet skipped "
                        f"(no fittable lines; flux likely NaN-filled)",
                        stacklevel=2,
                    )
                line_dict["order"] = (o + 1) * np.ones(nlines, dtype=int)
                line_dict["fiber"] = np.full(nlines, fiber)

                for k in keys:
                    lines[k][i][o] = line_dict[k]

            for k in lines:
                lines[k][i] = np.hstack(lines[k][i])

            n_total = len(lines["wav"][i])
            n_good = int(np.sum(~lines["bad"][i]))
            if verbose:
                print(f"  {chip} {fiber}: {n_good}/{n_total} good lines")
            if n_good == 0 and verbose:
                warnings.warn(
                    f"{chip} {fiber}: no good lines retained "
                    f"({n_total} attempted; all rejected or NaN-filled)",
                    stacklevel=2,
                )

        for k in lines:
            lines[k] = np.hstack(lines[k])

        return lines

    def calculate_wls_coeffs(
        self,
        lines,
        norder,
        polyorder_x=None,
        polyorder_m=None,
        polyorder_f=None,
    ):
        """
        Fit a multivariate Legendre polynomial wavelength solution to
        fitted line positions.

        Treats wavelength as a smooth function of pixel position (x), order
        number (m), and (optionally) fiber index (f). All variables are
        rescaled to [-1, 1] before fitting.

        Parameters
        ----------
        lines : dict of ndarray
            Flat 1D arrays as produced by `fit_line_positions_ffi`. Lines
            with `lines['bad']` set are excluded from the fit. Required
            keys: 'wav', 'pix', 'order', 'fiber', 'bad'.
        norder : int
            Total number of spectral orders on the chip. Used to rescale
            the order axis to [-1, 1].
        polyorder_x : int, optional
            Polynomial degree along the pixel axis.
        polyorder_m : int, optional
            Polynomial degree along the order axis.
        polyorder_f : int, optional
            Polynomial degree along the fiber axis (only used for 3-fiber fits).

        Returns
        -------
        coeffs : ndarray
            Legendre coefficient array. Shape is
            (polyorder_x+1, polyorder_m+1) for a single-fiber fit, or
            (polyorder_x+1, polyorder_m+1, polyorder_f+1) for a 3-fiber fit.

        Notes
        -----
        Raises ValueError if `lines['fiber']` contains anything other than
        one fiber, all five fibers, or three SCI fibers (SCI1, SCI2, SCI3).
        """
        if polyorder_x is None:
            polyorder_x = self.polyorder_x
        if polyorder_m is None:
            polyorder_m = self.polyorder_m
        if polyorder_f is None:
            polyorder_f = self.polyorder_f

        good = ~lines["bad"]
        wav = lines["wav"][good]
        pix = lines["pix"][good]
        order_num = lines["order"][good]
        fiber_names = lines["fiber"][good]

        fibers = list(set(fiber_names))

        if (len(fibers) != 1) and (len(fibers) != 3) and (len(fibers) != 5):
            raise ValueError(f"expected 1, 3, or 5 fibers, got {len(fibers)}")

        if len(fibers) == 3:
            expected_fibers = ["SCI1", "SCI2", "SCI3"]
        elif len(fibers) == 5:
            expected_fibers = ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]

        if len(fibers) != 1 and not (
            np.all(np.isin(fibers, expected_fibers))
            and np.all(np.isin(expected_fibers, fibers))
        ):
            raise ValueError(f"unexpected fibers input: {fibers}")

        # guard against degenerate / underconstrained fits
        n_params = (polyorder_x + 1) * (polyorder_m + 1)
        if len(fibers) != 1:
            n_params *= polyorder_f + 1
        if len(wav) < n_params:
            raise ValueError(
                f"WLS fit underconstrained: {len(wav)} good lines < "
                f"{n_params} free parameters "
                f"(polyorder_x={polyorder_x}, polyorder_m={polyorder_m}, "
                f"polyorder_f={polyorder_f}, fibers={sorted(fibers)})"
            )

        ncol = self.ccd["ncol"]

        # rescale position variables to [-1,1] for Legendre fitting
        x = 2 * pix / (ncol - 1) - 1
        m = 2 * (order_num - 1) / (norder - 1) - 1

        if len(fibers) != 1:
            # map fibers to their positional rank then rescale to [-1, 1]
            canonical = sorted(expected_fibers, key=lambda fb: self.fiber_positions[fb])
            fiber_pos = {fb: i for i, fb in enumerate(canonical)}
            f = np.array([fiber_pos[fb] for fb in fiber_names], dtype=int)
            f = 2 * f / (len(canonical) - 1) - 1

        if len(fibers) == 1:
            V = legendre.legvander2d(x, m, deg=[polyorder_x, polyorder_m])

            coeffs, *_ = np.linalg.lstsq(V, wav, rcond=None)
            coeffs = coeffs.reshape(polyorder_x + 1, polyorder_m + 1)

        else:
            V = legendre.legvander3d(
                x, m, f, deg=[polyorder_x, polyorder_m, polyorder_f]
            )

            coeffs, *_ = np.linalg.lstsq(V, wav, rcond=None)
            coeffs = coeffs.reshape(polyorder_x + 1, polyorder_m + 1, polyorder_f + 1)

        return coeffs

    @staticmethod
    def evaluate_wls_coeffs(coeffs, ncol, norder, nfiber):
        """
        Evaluate a Legendre wavelength solution onto a regular grid.

        Parameters
        ----------
        coeffs : ndarray
            Legendre coefficient array from `calculate_wls_coeffs`. Either
            2D (single-fiber) or 3D (three-fiber).
        ncol : int
            Number of detector columns at which to evaluate.
        norder : int
            Number of spectral orders at which to evaluate.
        nfiber : int
            Number of fibers at which to evaluate. Ignored for 2D `coeffs`.

        Returns
        -------
        W : ndarray
            Wavelength array of shape (norder, ncol) for 2D `coeffs`, or
            (norder, ncol, nfiber) for 3D `coeffs`.
        """
        x = np.linspace(-1, 1, ncol)
        y = np.linspace(-1, 1, norder)
        z = np.linspace(-1, 1, nfiber)

        if coeffs.ndim == 2:
            X, Y = np.meshgrid(x, y)
            W = legendre.legval2d(X, Y, coeffs)

        elif coeffs.ndim == 3:
            X, Y, Z = np.meshgrid(x, y, z)
            W = legendre.legval3d(X, Y, Z, coeffs)

        else:
            raise ValueError(f"coeffs.ndim expected to be 2 or 3, got {coeffs.ndim}")

        return W

    def compute_wls_from_stack(
        self,
        chip,
        fibers,
        linelist=None,
        lineprofile=None,
        polyorder_x=None,
        polyorder_m=None,
        polyorder_f=None,
        window=5,
        qc_sigma=2.5,
        max_bad_frac=0.05,
        verbose=True,
    ):
        """
        Compute a master wavelength solution from a stack of extracted L2 frames.

        For each L2 frame in `self._l2_obj_cache` (populated by
        `process_stack_l0_to_l2`), fits line positions across the requested
        fibers, fits a Legendre WLS to those line positions, then combines
        the per-frame coefficient sets via per-coefficient outlier-rejected
        averaging. The averaged coefficients are evaluated to produce a
        master wavelength array.

        Parameters
        ----------
        chip : str
            Chip identifier, i.e. 'GREEN' or 'RED'.
        fibers : list of str
            Fiber identifiers, e.g. ['SCI1', 'SCI2', 'SCI3'] or a single fiber.
        linelist : str, optional
            Path to a CSV line list. If different from the currently
            cached `self.linelist`, the file is reloaded and the cache
            is updated. Defaults to `self.linelist` (no reload).
        lineprofile : str, optional
            Line profile model name. Defaults to self.lineprofile.
        polyorder_x : int, optional
            Polynomial degree along the pixel axis.
        polyorder_m : int, optional
            Polynomial degree along the order axis.
        polyorder_f : int, optional
            Polynomial degree along the fiber axis (only used for 3-fiber fits).
        window : int, optional
            Half-width of the per-line fit window, in pixels.
        qc_sigma : float, optional
            Outlier rejection threshold applied to the per-frame Legendre
            coefficients when combining them.
        max_bad_frac : float, optional
            Maximum fraction of per-line fits allowed to fail QC before a
            frame is rejected from the stack. Frames exceeding this are
            dropped; if more than one frame is rejected, an error is raised.
        verbose : bool, optional
            If True, print progress for each frame and fiber.

        Returns
        -------
        W : ndarray
            Master wavelength array, shape (norder, ncol) or
            (norder, ncol, nfiber).
        coeffs_mean : ndarray
            Outlier-rejected mean Legendre coefficient array.
        coeffs_stack : ndarray
            Per-frame coefficient arrays (stacked).
        lines_stack : list of dict
            Per-frame line dicts from `fit_line_positions_ffi`.

        Notes
        -----
        Raises ValueError if `self._l2_obj_cache` is empty (i.e.,
        `process_stack_l0_to_l2` has not been run), or if more than one
        frame is rejected for having > `max_bad_frac` of its line fits fail
        QC. Rejected frames are excluded from the returned `coeffs_stack`
        and `lines_stack`.
        """
        self._load_linelist(linelist)
        if lineprofile is None:
            lineprofile = self.lineprofile
        if polyorder_x is None:
            polyorder_x = self.polyorder_x
        if polyorder_m is None:
            polyorder_m = self.polyorder_m
        if polyorder_f is None:
            polyorder_f = self.polyorder_f

        if not self._l2_obj_cache:
            raise ValueError("No L2 objects found; please run process_stack_l0_to_l2")
        l2_obj_list = self._l2_obj_cache

        nobs = len(l2_obj_list)

        lines_stack = [None] * nobs
        coeffs_stack = [None] * nobs
        keep = np.ones(nobs, dtype=bool)
        rejected = 0

        for i, l2_obj in enumerate(l2_obj_list):
            if verbose:
                print(f"\n{i + 1} of {nobs}")

            lines_stack[i] = self.fit_line_positions_ffi(
                l2_obj,
                chip,
                fibers,
                linelist=linelist,
                lineprofile=lineprofile,
                window=window,
                verbose=verbose,
            )

            nlines = len(lines_stack[i]["bad"])
            bad_frac = np.sum(lines_stack[i]["bad"]) / nlines if nlines else 0.0

            if bad_frac > max_bad_frac:
                keep[i] = False
                rejected += 1
                if rejected > 1:
                    raise ValueError(
                        f"{chip}: more than one frame rejected from stack "
                        f"(> {max_bad_frac:.0%} of line fits failed QC)"
                    )
                if verbose:
                    warnings.warn(
                        f"{chip} frame {i + 1}: {bad_frac:.1%} of line fits failed QC "
                        f"(> {max_bad_frac:.0%}); frame rejected from stack",
                        stacklevel=2,
                    )
                continue

            coeffs_stack[i] = self.calculate_wls_coeffs(
                lines_stack[i],
                self.norder[chip],
                polyorder_x=polyorder_x,
                polyorder_m=polyorder_m,
                polyorder_f=polyorder_f,
            )

        lines_stack = [lines_stack[i] for i in range(nobs) if keep[i]]
        coeffs_stack = [coeffs_stack[i] for i in range(nobs) if keep[i]]

        coeffs_stack = np.array(coeffs_stack)

        # Non-finite coeffs compare False below and would silently poison the
        # master WAVE grid; fail loudly instead.
        if not np.isfinite(coeffs_stack).all():
            bad_frames = [
                i + 1
                for i in range(len(coeffs_stack))
                if not np.isfinite(coeffs_stack[i]).all()
            ]
            raise ValueError(
                f"{chip}: non-finite Legendre coefficients from frame(s) "
                f"{bad_frames}; a degenerate line fit poisoned the stack"
            )

        diff = np.abs(coeffs_stack - np.median(coeffs_stack, axis=0))
        sigma = mad_std(coeffs_stack, axis=0)
        bad = diff > qc_sigma * sigma
        denom = np.sum(~bad, axis=0)
        if np.any(denom == 0):
            raise ValueError(
                f"{chip}: all frames rejected as outliers for at least one "
                f"Legendre coefficient; cannot combine the coefficient stack"
            )
        coeffs_mean = np.sum(coeffs_stack * ~bad, axis=0) / denom

        W = self.evaluate_wls_coeffs(
            coeffs_mean, self.ccd["ncol"], self.norder[chip], len(fibers)
        )

        return W, coeffs_mean, coeffs_stack, lines_stack

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def make_master_l2(
        self,
        l0_file_list=None,
        *,
        linelist=None,
        lineprofile=None,
        polyorder_x=None,
        polyorder_m=None,
        polyorder_f=None,
        bias=None,
        dark=None,
        flat=None,
        master_path=None,
        diagnostics_path=None,
        verbose=True,
    ):
        """
        Build a master wavelength solution from a stack of L0 frames.

        Processes each input L0 frame through the L0-to-L2 pipeline, then
        computes per-chip Legendre wavelength solutions using
        `compute_wls_from_stack`. The resulting wavelength arrays are
        written to the per-fiber _WAVE extensions of a KPFMasterL2 object,
        which is returned and cached on `self.ml2_obj`; pass `master_path`
        to also persist it to disk via `save_master('L2', ...)`. Per-frame
        coefficient and line stacks are always stashed on
        `self._coeffs_stack` / `self._lines_stack`; pass `diagnostics_path`
        to also persist them to disk via `save_diagnostics()`.

        Parameters
        ----------
        l0_file_list : list of str, optional
            L0 files to process. Defaults to self.l0_file_list.
        linelist : str, optional
            Path to a CSV line list. If different from the currently
            cached `self.linelist`, the file is reloaded and the cache
            is updated. Defaults to `self.linelist` (no reload).
        lineprofile : str, optional
            Line profile model name. Defaults to self.lineprofile.
        polyorder_x : int, optional
            Polynomial degree along the pixel axis. Defaults to self.polyorder_x.
        polyorder_m : int, optional
            Polynomial degree along the order axis. Defaults to self.polyorder_m.
        polyorder_f : int, optional
            Polynomial degree along the fiber axis (used for 3- and 5-fiber fits).
            Defaults to self.polyorder_f.
        bias, dark, flat : bool, optional
            Per-call correction overrides (clamped by the WLS standard of
            bias+dark). E.g. `dark=False` extracts with bias subtraction only.
        master_path : str, optional
            If provided, calls `self.save_master('L2', master_path)` at
            the end to persist the master L2 to a FITS file at this path.
        diagnostics_path : str, optional
            If provided, calls `self.save_diagnostics(diagnostics_path)`
            at the end to persist the per-frame coefficient and line stacks
            to an HDF5 file at this path.
        verbose : bool, optional
            If True (default), emit progress prints and informational
            warnings from frame loading, spectral extraction, and the
            per-frame WLS fit. Hard failures still raise.

        Returns
        -------
        KPFMasterL2
            Master L2 object with per-fiber _WAVE extensions populated for
            every chip in self.chips, INPUT_FILES recording the stacked
            L0 files, and a 'master_wls' receipt entry.

        Notes
        -----
        Resets self._l2_obj_cache before processing so repeat calls do not
        carry stale frames forward.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if lineprofile is None:
            lineprofile = self.lineprofile
        if polyorder_x is None:
            polyorder_x = self.polyorder_x
        if polyorder_m is None:
            polyorder_m = self.polyorder_m
        if polyorder_f is None:
            polyorder_f = self.polyorder_f

        self._active_corrections = self._resolve_corrections(
            bias=bias, dark=dark, flat=flat
        )

        self._load_linelist(linelist)

        # process_stack_l0_to_l2 resets self._l2_obj_cache at entry.
        self.process_stack_l0_to_l2(l0_file_list=l0_file_list, verbose=verbose)

        self.ml2_obj = KPFMasterL2()

        self._coeffs_stack = {}
        self._lines_stack = {}
        self._results = {}

        for chip in self.chips:
            result = self.compute_wls_from_stack(
                chip=chip,
                fibers=self.fibers,
                lineprofile=lineprofile,
                polyorder_x=polyorder_x,
                polyorder_m=polyorder_m,
                polyorder_f=polyorder_f,
                verbose=verbose,
            )
            W, coeffs_mean, coeffs_stack, lines_stack = result

            self._results[chip] = {
                "n_total": sum(len(frame["wav"]) for frame in lines_stack),
                "n_fit": sum(int(np.sum(~frame["bad"])) for frame in lines_stack),
            }
            self._coeffs_stack[chip] = coeffs_stack
            self._lines_stack[chip] = lines_stack

            for i, fiber in enumerate(self.fibers):
                if W.ndim == 2:
                    self.ml2_obj.data[f"{chip}_{fiber}_WAVE"] = W
                else:
                    self.ml2_obj.data[f"{chip}_{fiber}_WAVE"] = W[:, :, i]

            coeffs_ext = f"{chip}_WLS_COEFFS"
            if coeffs_ext not in self.ml2_obj.extensions:
                self.ml2_obj.create_extension(coeffs_ext, "ImageHDU")
            self.ml2_obj.set_data(coeffs_ext, coeffs_mean)

            # (value, comment) tuples are rejected for non-PRIMARY headers
            # by rvdata's fits.Header(dict) round-trip; keep these plain.
            coeffs_hdr = self.ml2_obj.headers[coeffs_ext]
            coeffs_hdr["POLYORDX"] = polyorder_x
            coeffs_hdr["POLYORDM"] = polyorder_m
            coeffs_hdr["POLYORDF"] = polyorder_f

        self.ml2_obj.set_input_files(l0_file_list)

        primary = self.ml2_obj.headers["PRIMARY"]
        primary["ROUGHWLS"] = (self.rough_wls_file, "Rough WLS reference file")
        primary["LINELIST"] = (self.linelist, "Line list reference file")
        primary["LINEPROF"] = (lineprofile, "Line profile model used in WLS fit")
        primary["POLYORDX"] = (polyorder_x, "WLS polynomial degree, pixel axis")
        primary["POLYORDM"] = (polyorder_m, "WLS polynomial degree, order axis")
        primary["POLYORDF"] = (polyorder_f, "WLS polynomial degree, fiber axis")
        primary["CHIPS"] = (",".join(self.chips), "Chips included in master WLS")
        primary["FIBERS"] = (",".join(self.fibers), "Fibers included in master WLS")

        self.ml2_obj.receipt_add_entry("master_wls", "PASS")

        if master_path is not None:
            self.save_master("L2", master_path, overwrite=True)

        if diagnostics_path is not None:
            self.save_diagnostics(diagnostics_path)

        return self.ml2_obj

    def save_diagnostics(self, path):
        """
        Write the per-frame WLS diagnostic stacks to an HDF5 file at `path`.

        Layout: /<chip>/coeffs_stack as a dataset, /<chip>/lines_stack/
        frame_<NNN>/<key> as per-frame subgroups, with every key from
        the per-frame `lines` dict (e.g. wav, pix, order, fiber, bad, plus
        diagnostics like std, amp).

        Raises
        ------
        RuntimeError
            If make_master_l2() has not been run yet, or raised before
            populating any chip.
        """
        if not self._coeffs_stack or not self._lines_stack:
            raise RuntimeError("No diagnostics available; run make_master_l2() first")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        str_dt = h5py.string_dtype(encoding="utf-8")
        with h5py.File(path, "w") as f:
            for chip, coeffs_stack in self._coeffs_stack.items():
                chip_group = f.create_group(chip)
                chip_group.create_dataset("coeffs_stack", data=np.asarray(coeffs_stack))

                lines_group = chip_group.create_group("lines_stack")
                for i, lines in enumerate(self._lines_stack[chip]):
                    frame_group = lines_group.create_group(f"frame_{i:03d}")
                    for key, value in lines.items():
                        arr = np.asarray(value)
                        if arr.dtype.kind in ("U", "S", "O"):
                            frame_group.create_dataset(
                                key, data=arr.astype(object), dtype=str_dt
                            )
                        else:
                            frame_group.create_dataset(key, data=arr)

    def info(self):
        """Print a summary of the module configuration and WLS results."""
        print("WLS")
        print("  l0_file_list:")
        for fn in self.l0_file_list:
            print(f"    {fn}")
        print(f"  chips:           {self.chips}")
        print(f"  fibers:          {self.fibers}")
        print(f"  linelist:        {self.linelist}")
        print(f"  rough_wls_file:  {self.rough_wls_file}")
        print(f"  lineprofile:     {self.lineprofile}")
        print(
            f"  polyorder:       x={self.polyorder_x}, m={self.polyorder_m}, "
            f"f={self.polyorder_f}"
        )

        if self._results is None:
            print("  make_master_l2() has not been called")
            return

        print(f"\n  {'chip':<8s} {'n lines fit/total'}")
        print("  " + "-" * 40)
        for chip, stats in self._results.items():
            n_fit, n_total = stats["n_fit"], stats["n_total"]
            pct = 100.0 * n_fit / n_total if n_total else 0.0
            print(f"  {chip:<8s} {n_fit} / {n_total} ({pct:.1f}%)")
