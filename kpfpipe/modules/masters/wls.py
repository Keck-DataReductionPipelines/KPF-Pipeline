"""
KPF Master Wavelength Solution construction module.
"""

import logging
import os

import h5py
import numpy as np
import pandas as pd
from numpy.polynomial import legendre

from kpfpipe import FIBERS, REPO_ROOT, SCI_FIBERS
from kpfpipe.data_models.masters import KPFMasterL2
from kpfpipe.modules.masters.base import BaseMasterModule
from kpfpipe.quality_control.diagnostics import DiagL2
from kpfpipe.quality_control.qc_flags import QCL2
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import kpf_directory
from kpfpipe.utils.kpf import get_obs_id
from kpfpipe.utils.stats import optimize_lsq

logger = logging.getLogger(__name__)

_ROUGH_WLS_FILE = f"{REPO_ROOT}/reference/rough_wls_fallback.csv"

# Per-line fit diagnostics, column -> (units, description). The line fitters
# return these keys; save_diagnostics writes each to the <CHIP>_LINES group.
_LINE_FIT_COLUMNS = {
    "CHIP": ("", "CCD"),
    "FIBER": ("", "fiber"),
    "ORDER_INDEX": ("", "0-based order row on the chip"),
    "ECHELLE_ORDER": ("", "physical echelle order"),
    "WAVE": ("A, vacuum", "reference line wavelength"),
    "PIX_GUESS": ("px", "rough-WLS pixel; fit window center"),
    "NPIX": ("px", "pixels in the fit window (fewer at a detector edge)"),
    "SNR": ("", "sum(FLUX) / sqrt(sum(VAR)) over the fit window"),
    "FIT_MEAN": ("px", "fitted line center"),
    "FIT_MEAN_ERR": ("px", "1-sigma error on FIT_MEAN"),
    "FIT_STD": ("px", "fitted Gaussian sigma"),
    "FIT_AMP": ("L2 FLUX units", "fitted amplitude"),
    "FIT_BASELINE": ("L2 FLUX units", "fitted constant background"),
    "FIT_RMS": ("L2 FLUX units", "RMS of the fit residuals"),
    "FIT_REDUCED_CHI2": ("", "residual chi-square against VAR per degree of freedom"),
    "CONVERGED": ("", "least-squares fit converged"),
    "FINITE": ("", "every window FLUX and VAR pixel finite; fit attempted"),
}


class WLS(BaseMasterModule):
    """
    Construct a master wavelength solution from a stack of WLS L0 exposures.

    Each frame is processed individually through the L0→L2 pipeline. Fitted
    line positions across the stack are combined to derive per-fiber
    wavelength solutions.

    Standard reduction: like science, a WLS frame is bias- and dark-subtracted
    before extraction; flat division is part of the standard reduction but
    stays off until it is implemented.

    Parameters
    ----------
    l0_file_list : list of str
        Sorted list of L0 FITS file paths to process.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: linelist, lineprofile,
        poly_degree_x, poly_degree_m, poly_degree_f, chips, fibers,
        min_stack_size (minimum frames extracted to L2), KPF_MASTERS_OUTPUT.
    """

    _DEFAULT_CFG = {
        **BaseMasterModule._DEFAULT_CFG,
        "linelist": f"{REPO_ROOT}/reference/thar_line_list.csv",
        "lineprofile": "gaussian",
        "poly_degree_x": 6,
        "poly_degree_m": 6,
        "poly_degree_f": 2,
    }

    # Bias+dark subtraction is standard for WLS; ``_process_frame`` (run before
    # ``_extract_frame``) applies whichever of these the config has enabled.
    _STANDARD_CALIBRATIONS = ("bias", "dark")

    def __init__(self, l0_file_list, config=None):
        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(
                [
                    "DATA_DIRS",
                    "TRACES",
                    "WLS",
                    "MODULE_CALIBRATION_ASSOCIATION",
                    "MODULE_IMAGE_PROCESSING",
                ]
            )
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")
        super().__init__(l0_file_list, params)
        self.master_type = "thar"
        self.rough_wls_file = _ROUGH_WLS_FILE

        # physical echelle order per row (bluest first), cached per chip;
        # e.g. GREEN 137..103. "order" means this echelle order throughout.
        self._echelle_orders = {
            chip: np.linspace(*self.echelle_orders[chip], self.norder[chip])
            .round()
            .astype(int)
            for chip in self.echelle_orders
        }

        self._load_rough_wls()
        self._load_linelist()

        self._l2_obj_cache = []  # populated by _process_stack_l0_to_l2()
        self._info = None
        self._frame_diagnostics = (
            None  # populated by make_master_l2(); used by save_diagnostics()
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_linelist(self, linelist=None):
        """
        Return the cached line list DataFrame, loading/reloading if ``linelist``
        differs from the cached path.

        Columns: CHIP, INDEX (0-based row), ECHELLE (physical order), WAVE
        [Å, vacuum].
        """
        needs_load = not hasattr(self, "_linelist_df")
        if linelist is not None and linelist != self.linelist:
            self.linelist = linelist
            needs_load = True
        if needs_load:
            logger.info("reading ThAr line list from %s", self.linelist)
            self._linelist_df = pd.read_csv(self.linelist)
        return self._linelist_df

    def _load_rough_wls(self, rough_wls_file=None):
        """
        Return the cached rough WLS dict, loading/reloading if ``rough_wls_file``
        differs from the cached path. The cache tracks the most recent path.
        """
        needs_load = not hasattr(self, "rough_wls")
        if rough_wls_file is not None and rough_wls_file != self.rough_wls_file:
            self.rough_wls_file = rough_wls_file
            needs_load = True
        if needs_load:
            logger.info("reading rough WLS from %s", self.rough_wls_file)
            df = pd.read_csv(self.rough_wls_file)

            ncol = self.ncol
            # Per-order Legendre coefficients (C0..Cn) evaluated on the
            # normalized pixel grid; see scripts/build_rough_wls_from_legacy_wls.py.
            coeff_cols = sorted(
                (c for c in df.columns if c.startswith("C") and c[1:].isdigit()),
                key=lambda c: int(c[1:]),
            )
            x = 2 * np.arange(ncol) / (ncol - 1) - 1
            self.rough_wls = {}

            for chip in self.chips:
                orders = self._echelle_orders[chip]
                for fiber in self.fibers:
                    channel_ext = f"{chip}_{fiber}_WAVE"
                    self.rough_wls[channel_ext] = np.zeros((len(orders), ncol))

                    for j, o in enumerate(orders):
                        use = (df.CHIP == chip) & (df.ECHELLE == o)
                        coeffs = df.loc[use, coeff_cols].values[0]
                        self.rough_wls[channel_ext][j] = legendre.legval(x, coeffs)

        return self.rough_wls

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def _process_stack_l0_to_l2(
        self,
        l0_file_list,
        max_fail_fraction=0.2,
        max_fail_number=2,
        output_dir=None,
    ):
        """
        Run each frame of ``l0_file_list`` through the L0->L2 pipeline, with the
        rough WLS as its WAVE and DiagL2/QCL2 run on it, returning the
        extracted KPF2 objects (also cached on ``self._l2_obj_cache``, reset at
        entry). With ``output_dir``, each frame's L2 is written there as
        ``{obs_id}_thar_L2.fits`` as soon as it is extracted. Every frame is
        attempted; afterwards, raises if frame failures exceed
        ``max_fail_fraction`` or ``max_fail_number``, or fewer than
        ``min_stack_size`` frames reached L2.
        """
        if len(l0_file_list) == 0:
            raise ValueError("Empty l0_file_list; must supply at least one valid file")

        self._l2_obj_cache = []
        self._stacked_files = []

        for fn in l0_file_list:
            l1_obj = self._load_frame(fn, cache=False)

            if l1_obj is None:
                continue

            l1_obj = self._process_frame(l1_obj)

            if l1_obj is None:
                continue

            l2_obj = self._extract_frame(l1_obj)

            if l2_obj is None:
                continue

            # Extraction writes no WAVE; the rough WLS the line fits start from
            # stands in so DiagL2 can locate its bands. Not yet a gate.
            for ext, wave in self.rough_wls.items():
                l2_obj.set_data(ext, wave)
            DiagL2(l2_obj).run()
            QCL2(l2_obj).run()

            if output_dir is not None:
                self._write_frame(l2_obj, output_dir=output_dir, overwrite=True)
            self._stacked_files.append(fn)
            self._l2_obj_cache.append(l2_obj)

        self._check_stack_size(l0_file_list, max_fail_fraction, max_fail_number)

        return self._l2_obj_cache

    def _fit_line_positions_1d(
        self,
        flux1d,
        var1d,
        wave1d,
        line_waves,
        lineprofile="gaussian",
        window=5,
    ):
        """
        Fit line positions (pixel space) along one 1D extracted order. The fit
        is unweighted; ``var1d`` (the order's L2 VAR) only scores it.

        Fits a 1D line model over a +/-``window`` px neighborhood around each
        reference line in ``line_waves``. A reference line outside the rough WLS
        span of ``wave1d`` raises (a CHIP/ORDER labeling-inconsistency
        guardrail).

        Returns a dict of per-line arrays keyed by ``_LINE_FIT_COLUMNS`` (less
        the provenance columns), one row per line in ``line_waves`` (sorted). A
        line whose window holds any non-finite FLUX or VAR is not fit: FINITE
        and CONVERGED False, SNR and FIT_* NaN.
        """
        # Fit in float64 (wavelength solutions are float64).
        flux1d = np.asarray(flux1d, dtype=np.float64)
        var1d = np.asarray(var1d, dtype=np.float64)
        wave1d = np.asarray(wave1d, dtype=np.float64)
        line_waves = np.asarray(line_waves, dtype=np.float64)

        if not len(flux1d) == len(var1d) == len(wave1d):
            raise ValueError("length of flux, var, and wave arrays are mismatched")
        ncol = len(flux1d)

        candidate_wavs = np.sort(line_waves)
        nlines = len(candidate_wavs)

        lines = {"WAVE": candidate_wavs}
        for key in ("PIX_GUESS", "NPIX"):
            lines[key] = np.zeros(nlines, dtype=int)
        for key in ("SNR", "FIT_MEAN", "FIT_MEAN_ERR", "FIT_STD", "FIT_AMP"):
            lines[key] = np.full(nlines, np.nan)
        for key in ("FIT_BASELINE", "FIT_RMS", "FIT_REDUCED_CHI2"):
            lines[key] = np.full(nlines, np.nan)
        for key in ("CONVERGED", "FINITE"):
            lines[key] = np.zeros(nlines, dtype=bool)

        windows = []
        for i, lw in enumerate(candidate_wavs):
            loc = np.argmin(np.abs(wave1d - lw))
            cols = np.arange(loc - window, loc + window + 1)
            cols = cols[(cols >= 0) & (cols < ncol)]
            lines["PIX_GUESS"][i] = loc
            lines["NPIX"][i] = len(cols)
            windows.append(cols)

        if not np.isfinite(flux1d).any():
            return lines

        # Guardrail (see docstring).
        lo, hi = wave1d.min(), wave1d.max()
        outside = (candidate_wavs < lo) | (candidate_wavs > hi)
        if np.any(outside):
            raise ValueError(
                f"{int(np.sum(outside))} reference line(s) outside the rough "
                f"WLS range [{lo:.3f}, {hi:.3f}] A; line-list CHIP/ORDER labels "
                f"are inconsistent with the rough WLS"
            )

        for i, cols in enumerate(windows):
            y = flux1d[cols]
            var = var1d[cols]
            lines["FINITE"][i] = np.isfinite(y).all() and np.isfinite(var).all()
            if not lines["FINITE"][i]:
                continue

            lines["SNR"][i] = np.sum(y) / np.sqrt(np.sum(var))
            theta, resid, theta_err, converged = optimize_lsq(cols, y, lineprofile)

            if lineprofile == "gaussian":
                # _gaussian_dist theta convention: [b, a, mu, sigma]
                lines["FIT_BASELINE"][i] = theta[0]
                lines["FIT_AMP"][i] = theta[1]
                lines["FIT_MEAN"][i] = theta[2]
                lines["FIT_MEAN_ERR"][i] = theta_err[2]
                lines["FIT_STD"][i] = theta[3]
            else:
                raise ValueError(f"Unsupported lineprofile: {lineprofile}")
            lines["FIT_RMS"][i] = np.std(resid)
            dof = len(cols) - len(theta)
            lines["FIT_REDUCED_CHI2"][i] = np.sum(resid**2 / var) / dof
            lines["CONVERGED"][i] = converged

        return lines

    def _fit_line_positions_ffi(
        self,
        l2_obj,
        chip,
        fibers,
        linelist=None,
        lineprofile=None,
        window=5,
    ):
        """
        Fit line positions across all orders and fibers of one chip.

        Calls ``_fit_line_positions_1d`` per (fiber, order), adds the CHIP,
        FIBER, ORDER_INDEX and ECHELLE_ORDER columns, and concatenates every
        row into one dict of flat arrays.
        """
        linelist_df = self._load_linelist(linelist)

        if lineprofile is None:
            lineprofile = self.lineprofile

        orders = self._echelle_orders[chip]
        per_order = []

        for fiber in fibers:
            logger.debug("fitting %s %s line positions", chip, fiber)

            flux_arr = l2_obj.data[f"{chip}_{fiber}_FLUX"]
            var_arr = l2_obj.data[f"{chip}_{fiber}_VAR"]
            wave_arr = self.rough_wls[f"{chip}_{fiber}_WAVE"]

            if not np.shape(flux_arr) == np.shape(var_arr) == np.shape(wave_arr):
                raise ValueError("shape mismatch between flux, var, and rough WLS")

            fiber_lines = []
            for j, o in enumerate(orders):
                line_waves = linelist_df.loc[
                    (linelist_df["CHIP"] == chip) & (linelist_df["ECHELLE"] == o),
                    "WAVE",
                ].to_numpy(dtype=float)
                line_dict = self._fit_line_positions_1d(
                    flux_arr[j],
                    var_arr[j],
                    wave_arr[j],
                    line_waves,
                    lineprofile=lineprofile,
                    window=window,
                )

                if not line_dict["FINITE"].any():
                    logger.debug(
                        "%s %s order %s: orderlet skipped "
                        "(no fittable lines; flux likely NaN-filled)",
                        chip,
                        fiber,
                        o,
                    )
                nlines = len(line_dict["WAVE"])
                line_dict["CHIP"] = np.full(nlines, chip)
                line_dict["FIBER"] = np.full(nlines, fiber)
                line_dict["ORDER_INDEX"] = np.full(nlines, j)
                line_dict["ECHELLE_ORDER"] = np.full(nlines, o)
                fiber_lines.append(line_dict)

            n_total = sum(len(d["WAVE"]) for d in fiber_lines)
            n_fit = sum(int(np.sum(d["FINITE"])) for d in fiber_lines)
            logger.info("%s %s: %d/%d lines fit", chip, fiber, n_fit, n_total)
            if n_fit == 0:
                logger.warning(
                    "%s %s: no lines fit (all windows non-finite)", chip, fiber
                )
            per_order.extend(fiber_lines)

        return {k: np.concatenate([d[k] for d in per_order]) for k in per_order[0]}

    def _calculate_wls_coeffs(
        self,
        lines,
        orders,
        poly_degree_x=None,
        poly_degree_m=None,
        poly_degree_f=None,
        window=5,
    ):
        """
        Fit a multivariate Legendre wavelength solution to fitted line positions.

        Treats the grating invariant m*lambda (echelle order x wavelength) as a
        smooth function of pixel (x), order (m), and optionally fiber (f); all
        variables are rescaled to [-1, 1] before fitting. Only lines whose
        FIT_MEAN lies within ``window`` px of PIX_GUESS are used (unfit NaN rows
        are excluded too). Returns the Legendre coefficient array,
        shape (poly_degree_x+1, poly_degree_m+1) for a single-fiber fit or with a
        trailing (poly_degree_f+1) axis for a multi-fiber fit. Raises for a fiber
        set other than 1, the 3 SCI fibers, or all 5 fibers, or when the fit is
        underconstrained.
        """
        if poly_degree_x is None:
            poly_degree_x = self.poly_degree_x
        if poly_degree_m is None:
            poly_degree_m = self.poly_degree_m
        if poly_degree_f is None:
            poly_degree_f = self.poly_degree_f

        good = np.abs(lines["FIT_MEAN"] - lines["PIX_GUESS"]) <= window
        wav = lines["WAVE"][good]
        pix = lines["FIT_MEAN"][good]
        order = lines["ECHELLE_ORDER"][good]
        fiber_names = lines["FIBER"][good]

        fibers = list(set(fiber_names))

        if (len(fibers) != 1) and (len(fibers) != 3) and (len(fibers) != 5):
            raise ValueError(f"expected 1, 3, or 5 fibers, got {len(fibers)}")

        if len(fibers) == 3:
            expected_fibers = list(SCI_FIBERS)
        elif len(fibers) == 5:
            expected_fibers = list(FIBERS)

        if len(fibers) != 1 and not (
            np.all(np.isin(fibers, expected_fibers))
            and np.all(np.isin(expected_fibers, fibers))
        ):
            raise ValueError(f"unexpected fibers input: {fibers}")

        # guard against degenerate / underconstrained fits
        n_params = (poly_degree_x + 1) * (poly_degree_m + 1)
        if len(fibers) != 1:
            n_params *= poly_degree_f + 1
        if len(wav) < n_params:
            raise ValueError(
                f"WLS fit underconstrained: {len(wav)} good lines < "
                f"{n_params} free parameters "
                f"(poly_degree_x={poly_degree_x}, poly_degree_m={poly_degree_m}, "
                f"poly_degree_f={poly_degree_f}, fibers={sorted(fibers)})"
            )

        ncol = self.ncol

        # rescale position variables to [-1,1] for Legendre fitting
        blue, red = orders[0], orders[-1]
        x = 2 * pix / (ncol - 1) - 1
        m = 2 * (order - red) / (blue - red) - 1

        # unweighted fit on the grating invariant approximates velocity-space
        mlambda = order * wav

        if len(fibers) != 1:
            # map fibers to their positional rank then rescale to [-1, 1]
            canonical = sorted(expected_fibers, key=lambda fb: self.fiber_positions[fb])
            fiber_pos = {fb: i for i, fb in enumerate(canonical)}
            f = np.array([fiber_pos[fb] for fb in fiber_names], dtype=int)
            f = 2 * f / (len(canonical) - 1) - 1

        if len(fibers) == 1:
            V = legendre.legvander2d(x, m, deg=[poly_degree_x, poly_degree_m])

            coeffs, *_ = np.linalg.lstsq(V, mlambda, rcond=None)
            coeffs = coeffs.reshape(poly_degree_x + 1, poly_degree_m + 1)

        else:
            V = legendre.legvander3d(
                x, m, f, deg=[poly_degree_x, poly_degree_m, poly_degree_f]
            )

            coeffs, *_ = np.linalg.lstsq(V, mlambda, rcond=None)
            coeffs = coeffs.reshape(
                poly_degree_x + 1, poly_degree_m + 1, poly_degree_f + 1
            )

        return coeffs

    def _evaluate_wls_coeffs(self, coeffs, orders, nfiber):
        """
        Evaluate a Legendre wavelength solution over all detector columns.

        The coefficients describe the grating-invariant m*lambda surface, so the
        evaluated surface is divided by the physical order to recover wavelength.
        ``orders`` sets the [-1, 1] order rescaling (must match
        ``_calculate_wls_coeffs``). Returns W of shape (norder, ncol) for 2D
        ``coeffs`` or (norder, ncol, nfiber) for 3D ``coeffs`` (``nfiber`` ignored for
        2D).
        """
        blue, red = orders[0], orders[-1]
        x = np.linspace(-1, 1, self.ncol)
        y = 2 * (orders - red) / (blue - red) - 1
        z = np.linspace(-1, 1, nfiber)

        # L is the fitted grating-invariant surface m*lambda
        if coeffs.ndim == 2:
            X, Y = np.meshgrid(x, y)
            L = legendre.legval2d(X, Y, coeffs)

        elif coeffs.ndim == 3:
            X, Y, Z = np.meshgrid(x, y, z)
            L = legendre.legval3d(X, Y, Z, coeffs)

        else:
            raise ValueError(f"coeffs.ndim expected to be 2 or 3, got {coeffs.ndim}")

        # recover wavelength by dividing out the physical order (axis 0)
        W = L / orders.reshape((-1,) + (1,) * (L.ndim - 1))
        return W

    def _fit_lines_stack(
        self,
        chip,
        fibers,
        linelist=None,
        lineprofile=None,
        poly_degree_x=None,
        poly_degree_m=None,
        poly_degree_f=None,
        window=5,
    ):
        """
        Fit line positions and a per-frame Legendre WLS for every frame in
        ``self._l2_obj_cache``.

        Line positions are fit for every frame first, and the frame list is
        stashed on ``self._frame_diagnostics[chip]`` before any per-frame WLS
        fit, so a WLS fit failure (a ValueError naming the chip and frame)
        still leaves every line fit available to ``save_diagnostics``.

        Returns the list of per-frame dicts (original order) with keys 'obs_id',
        'lines' (the ``_fit_line_positions_ffi`` dict), 'coeffs', and the
        line-fit 'window' [px] and 'lineprofile' used.
        """
        self._load_linelist(linelist)
        if lineprofile is None:
            lineprofile = self.lineprofile
        if poly_degree_x is None:
            poly_degree_x = self.poly_degree_x
        if poly_degree_m is None:
            poly_degree_m = self.poly_degree_m
        if poly_degree_f is None:
            poly_degree_f = self.poly_degree_f

        if not self._l2_obj_cache:
            raise ValueError("No L2 objects found; please run _process_stack_l0_to_l2")
        l2_obj_list = self._l2_obj_cache

        nobs = len(l2_obj_list)
        frames = []
        for i, l2_obj in enumerate(l2_obj_list):
            logger.debug("fitting lines in observation %d of %d", i + 1, nobs)
            lines = self._fit_line_positions_ffi(
                l2_obj,
                chip,
                fibers,
                linelist=linelist,
                lineprofile=lineprofile,
                window=window,
            )
            frames.append(
                {
                    "obs_id": l2_obj.obs_id,
                    "lines": lines,
                    "coeffs": None,
                    "window": window,
                    "lineprofile": lineprofile,
                }
            )

        if self._frame_diagnostics is None:
            self._frame_diagnostics = {}
        self._frame_diagnostics[chip] = frames

        for frame in frames:
            try:
                frame["coeffs"] = self._calculate_wls_coeffs(
                    frame["lines"],
                    self._echelle_orders[chip],
                    poly_degree_x=poly_degree_x,
                    poly_degree_m=poly_degree_m,
                    poly_degree_f=poly_degree_f,
                    window=window,
                )
            except ValueError as exc:
                raise ValueError(
                    f"{chip} frame {frame['obs_id']}: WLS fit failed ({exc})"
                ) from exc

        return frames

    def _combine_coeffs_stack(self, frames, chip, nfibers):
        """
        Combine per-frame coefficients into a master WLS.

        Averages the frames' Legendre coefficients and evaluates the mean to a
        master wavelength array. Returns (W, coeffs_mean).
        """
        coeffs_mean = np.mean([fr["coeffs"] for fr in frames], axis=0)

        W = self._evaluate_wls_coeffs(coeffs_mean, self._echelle_orders[chip], nfibers)

        return W, coeffs_mean

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def make_master_l2(
        self,
        l0_file_list=None,
        *,
        linelist=None,
        lineprofile=None,
        poly_degree_x=None,
        poly_degree_m=None,
        poly_degree_f=None,
        bias=None,
        dark=None,
        flat=None,
        output_dir=None,
    ):
        """
        Build a master wavelength solution from a stack of L0 frames.

        Processes each input L0 frame through the L0-to-L2 pipeline, fits
        per-frame line positions and WLS coefficients (``_fit_lines_stack``),
        then combines the frames into per-chip Legendre wavelength
        solutions (``_combine_coeffs_stack``). The resulting wavelength arrays are
        written to the per-fiber _WAVE extensions of a KPFMasterL2 object,
        which is returned and cached on ``self.ml2_obj``. Per-frame diagnostics
        are always stashed on ``self._frame_diagnostics``; pass ``output_dir``
        to persist the master under its standard name, along with the per-frame
        L2s and the diagnostics HDF5.

        Parameters
        ----------
        l0_file_list : list of str, optional
            L0 files to process; replaces ``self.l0_file_list`` if given.
            Defaults to ``self.l0_file_list``.
        linelist : str, optional
            CSV line list path; reloaded if different from the cached path.
            Defaults to ``self.linelist`` (no reload).
        lineprofile : str, optional
            Defaults to ``self.lineprofile``.
        poly_degree_x : int, optional
            Pixel-axis polynomial degree. Defaults to ``self.poly_degree_x``.
        poly_degree_m : int, optional
            Order-axis polynomial degree. Defaults to ``self.poly_degree_m``.
        poly_degree_f : int, optional
            Fiber-axis polynomial degree (3- and 5-fiber fits). Defaults to
            ``self.poly_degree_f``.
        bias, dark, flat : bool | str | KPFMasterL1, optional
            Per-call calibration overrides (same forms as ImageProcessing.perform:
            bool, a master filepath, or a KPFMasterL1 object), clamped by the WLS
            standard of bias+dark. E.g. ``dark=False`` extracts with bias only, and
            ``dark="/path/master_dark.fits"`` uses a specific master.
        output_dir : str, optional
            If provided, write the master L2 into this directory under its
            standard DRP-RUN-05 filename via ``save_master('L2', ...)``, and
            persist each processed ThAr frame's L2 (as it is extracted) and the
            per-frame diagnostics HDF5 (``save_diagnostics()``) to the ThAr
            ``cal_stack`` directory under the configured masters output.

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

        Every frame that reaches L2 is line-fit and contributes to the master;
        none is rejected. Each per-frame L2 is written right after extraction,
        and the diagnostics HDF5 after line fitting even if a per-frame WLS fit
        raises, so both persist if the master is not built. Raises ValueError if
        load failures exceed the limits, fewer than ``self.min_stack_size``
        frames reach L2, or a per-frame WLS fit fails.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        self.l0_file_list = l0_file_list
        if lineprofile is None:
            lineprofile = self.lineprofile
        if poly_degree_x is None:
            poly_degree_x = self.poly_degree_x
        if poly_degree_m is None:
            poly_degree_m = self.poly_degree_m
        if poly_degree_f is None:
            poly_degree_f = self.poly_degree_f

        self._active_calibrations = self._resolve_calibrations(
            bias=bias, dark=dark, flat=flat
        )

        self._load_linelist(linelist)

        stack_dir = None
        if output_dir is not None:
            stack_dir = kpf_directory(
                kind="cal_stack",
                data_root=self._masters_output,
                level="L2",
                obs_id=get_obs_id(l0_file_list[0]),
                cal_type=self.master_type,
            )

        # _process_stack_l0_to_l2 resets self._l2_obj_cache at entry.
        self._process_stack_l0_to_l2(l0_file_list, output_dir=stack_dir)

        # Persist diagnostics even when a per-frame WLS fit raises, so a failed
        # stack still leaves every line fit on disk (only the master is withheld).
        self._frame_diagnostics = {}
        try:
            for chip in self.chips:
                self._fit_lines_stack(
                    chip=chip,
                    fibers=self.fibers,
                    lineprofile=lineprofile,
                    poly_degree_x=poly_degree_x,
                    poly_degree_m=poly_degree_m,
                    poly_degree_f=poly_degree_f,
                )
        finally:
            if stack_dir is not None and self._frame_diagnostics:
                self.save_diagnostics(stack_dir, overwrite=True)

        self._track_info()
        logger.info("%s", self._info)

        self.ml2_obj = KPFMasterL2(kind="wls")
        for chip in self.chips:
            W, coeffs_mean = self._combine_coeffs_stack(
                self._frame_diagnostics[chip], chip, len(self.fibers)
            )

            # W's fiber planes are ordered by physical slicer position (the fit
            # ranks fibers via fiber_positions and _evaluate_wls_coeffs emits
            # planes in that order). Assign by that same canonical order, not
            # self.fibers' (config-overridable) order, so a reordered self.fibers
            # cannot mis-route a solution (e.g. SKY's onto CAL).
            canonical = sorted(self.fibers, key=lambda fb: self.fiber_positions[fb])
            for i, fiber in enumerate(canonical):
                # set_data, not data[...]: only set_data checks the manifest's
                # declared BitDepth, which is what keeps master WAVE float64.
                self.ml2_obj.set_data(
                    f"{chip}_{fiber}_WAVE", W if W.ndim == 2 else W[:, :, i]
                )

            self.ml2_obj.set_data(f"{chip}_WLS_COEFFS", coeffs_mean)

        self.ml2_obj.set_input_files(self._stacked_files, self.master_type)

        # WLS metadata is out of EPRV scope but registered in
        # ML2-wls-PRIMARY-keywords.csv,
        # so it routes through set_keyword (-> PRIMARY, registry comments).
        self.ml2_obj.set_keyword("ROUGHWLS", self.rough_wls_file)
        self.ml2_obj.set_keyword("LINELIST", self.linelist)
        self.ml2_obj.set_keyword("LINEPROF", lineprofile)
        self.ml2_obj.set_keyword("POLYDEGX", poly_degree_x)
        self.ml2_obj.set_keyword("POLYDEGM", poly_degree_m)
        self.ml2_obj.set_keyword("POLYDEGF", poly_degree_f)

        self.ml2_obj.receipt_add_entry("master_wls", "", "PASS")

        if output_dir is not None:
            self.save_master(
                "L2",
                os.path.join(output_dir, self.ml2_obj.generate_standard_filename()),
                overwrite=True,
            )

        return self.ml2_obj

    def save_diagnostics(self, output_dir, *, overwrite=False):
        """
        Write the per-frame WLS diagnostics to an HDF5 file in ``output_dir``
        (``make_master_l2`` passes the ThAr ``cal_stack`` directory), named
        ``{obs_id}_thar_l2_diagnostics.h5`` (obs_id from the first stacked
        frame).

        Layout::

            /                            attrs LINEPROF, FITWIN [px], LINELIST, ROUGHWLS
            /<obs_id>/<CHIP>_LINES/<COLUMN>
            /<obs_id>/<CHIP>_WLS_COEFFS

        ``<CHIP>_LINES`` is one table per frame and chip, a 1-D dataset per
        ``_LINE_FIT_COLUMNS`` column with its ``units`` and ``description`` as
        attrs. ``<CHIP>_WLS_COEFFS`` holds the frame's Legendre coefficients;
        it is absent if that frame's WLS fit raised or was not reached.

        Parameters
        ----------
        output_dir : str
            Directory to write into; created if needed.
        overwrite : bool, optional
            Refuse to clobber an existing file unless True.

        Raises
        ------
        RuntimeError
            If make_master_l2() has not been run yet, or raised before
            populating any chip.
        FileExistsError
            If the file already exists and ``overwrite`` is False.
        """
        if not self._frame_diagnostics:
            raise RuntimeError("No diagnostics available; run make_master_l2() first")

        obs_id = get_obs_id(self._stacked_files[0])
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{obs_id}_thar_l2_diagnostics.h5")
        if not overwrite and os.path.exists(path):
            raise FileExistsError(
                f"{path} already exists; pass overwrite=True to replace it"
            )

        str_dt = h5py.string_dtype(encoding="utf-8")
        first = next(iter(self._frame_diagnostics.values()))[0]
        with h5py.File(path, "w") as f:
            f.attrs["LINEPROF"] = first["lineprofile"]
            f.attrs["FITWIN"] = first["window"]
            f.attrs["LINELIST"] = str(self.linelist)
            f.attrs["ROUGHWLS"] = str(self.rough_wls_file)

            for chip, frames in self._frame_diagnostics.items():
                for frame in frames:
                    grp = f.require_group(frame["obs_id"])
                    if frame["coeffs"] is not None:
                        grp.create_dataset(f"{chip}_WLS_COEFFS", data=frame["coeffs"])
                    table = grp.create_group(f"{chip}_LINES")
                    for name, (units, description) in _LINE_FIT_COLUMNS.items():
                        arr = frame["lines"][name]
                        if arr.dtype.kind in ("U", "S", "O"):
                            dset = table.create_dataset(
                                name, data=arr.astype(object), dtype=str_dt
                            )
                        else:
                            dset = table.create_dataset(name, data=arr)
                        dset.attrs["units"] = units
                        dset.attrs["description"] = description
        logger.info("wrote WLS diagnostics to %s", path)

    def _track_info(self):
        """Build and cache the info() summary text from _frame_diagnostics."""
        lines = ["WLS", "  l0_file_list:"]
        for fn in self.l0_file_list:
            lines.append(f"    {fn}")
        lines.append(f"  chips:           {self.chips}")
        lines.append(f"  fibers:          {self.fibers}")
        lines.append(f"  linelist:        {self.linelist}")
        lines.append(f"  rough_wls_file:  {self.rough_wls_file}")
        lines.append(f"  lineprofile:     {self.lineprofile}")
        lines.append(
            f"  poly_degree:     x={self.poly_degree_x}, m={self.poly_degree_m}, "
            f"f={self.poly_degree_f}"
        )
        lines.append(f"  min_stack_size:  {self.min_stack_size}")
        lines.append(f"\n  {'chip':<8s} {'frames':<8s} {'n lines fit/total'}")
        lines.append("  " + "-" * 40)
        for chip, frames in self._frame_diagnostics.items():
            n_fit = sum(int(np.sum(fr["lines"]["FINITE"])) for fr in frames)
            n_total = sum(len(fr["lines"]["FINITE"]) for fr in frames)
            pct = 100.0 * n_fit / n_total if n_total else 0.0
            lines.append(
                f"  {chip:<8s} {len(frames):<8d} {n_fit} / {n_total} ({pct:.1f}%)"
            )
        self._info = "\n\n" + "\n".join(lines) + "\n\n"

    def info(self):
        """Print a summary of the module configuration and WLS results."""
        if self._info is None:
            print(f"{type(self).__name__}: make_master_l2() has not been called")
        else:
            print(self._info)
