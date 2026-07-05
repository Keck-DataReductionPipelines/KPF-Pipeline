"""
KPF Radial Velocity module.

Computes radial velocities from a KPF4 (L4) whose per-order cross-correlation
functions were already built by CrossCorrelation. For each illuminated orderlet
it fits the per-order CCFs to radial velocities, collapses them into RVs per
order, orderlet, and CCD (weighted where appropriate), and fills the results into
the L4's per-order RV tables and headers plus the PRIMARY combined RV.

Everything the fit needs is read from the L4: the CCF cubes (CCFn) and their
per-bin variances (CCF_VARn); the velocity grid, reconstructed from each CCFn
header (VELSTART/VELSTEP/VELNSTEP); the mask width (VELMASK); and the per-order
WEIGHT / BERV / BJD_TDB / WAVE_START / WAVE_END columns of the RVn tables. The
native per-pixel velocity scale for the photon error comes from the WAVE
endpoints and the detector column count (detector.toml ncol).
"""

import logging

import numpy as np
from astropy.constants import c
from astropy.stats import mad_std

from kpfpipe import DEFAULTS
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import optimize_lsq

logger = logging.getLogger(__name__)

_DEFAULTS = {
    **DEFAULTS,
    "rv_window": [-25.0, 25.0],
}


class RadialVelocity:
    """
    Compute radial velocities from a KPF4's per-order CCFs.

    Parameters
    ----------
    l4_obj : KPF4
        L4 frame carrying the per-order CCFs (CCFn/CCF_VARn), the CCF velocity-grid
        and mask headers, and the metadata-seeded RVn tables produced by
        CrossCorrelation. RV/RV_ERR are filled in place.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: chips, fibers, rv_window.
    """

    def __init__(self, l4_obj, config=None):
        self.l4_obj = l4_obj

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(
                ["DATA_DIRS", "KPFPIPE", "MODULE_RADIAL_VELOCITY"]
            )
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in _DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        # CCF caches loaded from the L4 by _load_ccfs(), keyed by f'{chip}_{fiber}'.
        self._ccf = {}  # per-chip CCF cube
        self._ccf_var = {}  # per-bin CCF variance cube
        self._velocity_grid = {}  # reconstructed CCF velocity grid
        self._ccf_mask_width = None  # mask hole full width [km/s] behind the CCFs
        # Header-source stashes, filled by perform() and read by _set_headers.
        self._processed = []  # illuminated fibers written this run
        self._per_fiber = {}  # per-fiber rv/rv_err arrays and per-CCD RV/err
        self._sci_combined_ran = False  # whether a science combine was formed
        self._sci_ccd_rv = {}  # SCI-combined per-CCD RV, for CCD{n}RV
        self._sci_ccd_err = {}
        self._combined_rv = np.nan  # PRIMARY RV
        self._combined_rverr = np.nan  # PRIMARY RVERR
        self._primary_berv = np.nan  # PRIMARY BERV
        self._primary_bjdtdb = np.nan  # PRIMARY BJDTDB
        self._info = None

    # Science orderlets that may be summed together (shared mask and grid).
    _SCI_FIBERS = ("SCI1", "SCI2", "SCI3")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_ccfs(self, chips, fibers):
        """
        Load the CCF cubes, per-bin variances, velocity grids, and mask width from
        the (CrossCorrelation-produced) L4 into the caches the RV methods read.

        Only illuminated fibers (non-empty CCF) are loaded; the velocity grid is
        reconstructed from the CCF header (VELSTART/VELSTEP/VELNSTEP) and the mask
        width from VELMASK.
        """
        for fiber in fibers:
            if self.l4_obj.data[f"{fiber}_CCF"].size == 0:
                continue
            hdr = self.l4_obj.headers[f"{fiber}_CCF"]
            grid = hdr["VELSTART"] + hdr["VELSTEP"] * np.arange(hdr["VELNSTEP"])
            self._ccf_mask_width = float(hdr["VELMASK"])
            for chip in chips:
                key = f"{chip}_{fiber}"
                self._ccf[key] = np.asarray(
                    self.l4_obj.data[f"{key}_CCF"], dtype=np.float64
                )
                self._ccf_var[key] = np.asarray(
                    self.l4_obj.data[f"{key}_CCF_VAR"], dtype=np.float64
                )
                self._velocity_grid[key] = grid

    def _get_order_weights(self, chip, fiber):
        """
        Per-order CCF-combination weights for one orderlet, read from the WEIGHT
        column of the L4 RVn table (written by CrossCorrelation). Returns a 1D
        ndarray of length norder_chip, ordered by ORDER (green-then-red rows; the
        chip-prefixed read returns this chip's rows).
        """
        table = self.l4_obj.data[f"{chip.upper()}_{fiber.upper()}_RV"]
        return np.asarray(table["WEIGHT"], dtype=np.float64)

    @staticmethod
    def _pixel_velocity_scale(wave_start, wave_end, ncol):
        """
        Native per-pixel velocity scale [km/s] of an order, from its wavelength
        endpoints and column count: c * (dispersion per pixel) / mean wavelength.

        The RVn WAVE_START/WAVE_END and detector ncol stand in for the full WAVE
        array; this feeds the CCF-noise correlation length in _compute_rv_1d.
        """
        speed_of_light_kms = c.to("km/s").value
        return (
            speed_of_light_kms
            * (wave_end - wave_start)
            / (ncol - 1)
            / (0.5 * (wave_start + wave_end))
        )

    @staticmethod
    def _ccf_noise_corr_length(vel_span_per_pixel, mask_width):
        """
        Decorrelation length [km/s] of the CCF photon noise.

        The Boisse (2010) N_scale factor divides the CCF velocity step by the
        length over which CCF-noise bins are independent, correcting the
        diagonal information sum for the fact that a finely-stepped CCF is
        oversampled. That length is *not* the native pixel: the CCF is the
        spectrum seen through a mask hole, so its noise kernel is the
        cross-correlation of two top-hats -- the mask hole (full width
        `mask_width`) and the native pixel (`vel_span_per_pixel`) -- i.e. a
        trapezoid. The integral length of a trapezoid's autocorrelation,
        (integral)^2 / integral-of-square, is `M**2 / (M - m/3)` with M, m the
        larger/smaller of the two widths (reduces to 1.5*w for equal widths).
        """
        big, small = (
            max(vel_span_per_pixel, mask_width),
            min(vel_span_per_pixel, mask_width),
        )
        return big**2 / (big - small / 3.0)

    @staticmethod
    def _compute_rv_1d(
        vel,
        ccf,
        ccf_var,
        vel_span_per_pixel,
        mask_width,
        window,
        fit_nsigma=3.0,
        min_npts=9,
    ):
        """
        Two-pass Gaussian fit to a CCF dip, with a photon-limited error.

        The first pass fits the `window` ([min, max] km/s) about the CCF
        minimum, yielding a mean and sigma. The second pass refits a window of
        +/-`fit_nsigma` sigma about that mean (symmetric about it); those points
        also set the error estimate (Bouchy et al. 2001). Both windows use at
        least min_npts grid points.

        Parameters
        ----------
        vel : ndarray
            CCF velocity steps [km/s].
        ccf : ndarray
            CCF value at each velocity step.
        ccf_var : ndarray
            Per-velocity-bin CCF variance sum(w**2 * var) aligned with `ccf`; sets
            the photon noise in the error estimate (Bouchy 2001).
        vel_span_per_pixel : float
            Native per-pixel velocity scale [km/s] of the order (from
            _pixel_velocity_scale); with `mask_width` it sets the CCF-noise
            correlation length (see `_ccf_noise_corr_length`).
        mask_width : float
            CCF mask hole full width [km/s] used to build the CCF (each top-hat
            hole is `mask_width` wide, centered on its line).
        window : list of float
            [min, max] km/s velocity window about the dip for the first pass.
        fit_nsigma : float
            Half-width of the second-pass fit window, in units of the
            first-pass fitted sigma.
        min_npts : int
            Minimum number of grid points to use in each fit window.

        Returns
        -------
        rv : float
            Fitted radial velocity [km/s], or NaN if the fit fails.
        rv_err : float
            Photon-limited RV uncertainty [km/s], or NaN if unavailable.

        Raises
        ------
        ValueError
            If the CCF or its variance is non-physical (zero/negative flux
            counts) across the fit window, so a photon error cannot be computed.
        """
        # Fail loudly on non-finite CCF values rather than masking them out.
        if ccf.size < min_npts or not np.all(np.isfinite(ccf)) or np.ptp(ccf) == 0:
            return np.nan, np.nan

        # Locate the absorption dip; fit the inverted CCF so it presents as a peak
        # (matching optimize_lsq's peak-oriented guess); theta = [b, a, mu, sigma].
        peak = vel[np.argmin(ccf)]

        # First pass: [min, max] velocity window about the dip.
        lo_kms, hi_kms = window
        win = (vel - peak >= lo_kms) & (vel - peak <= hi_kms)
        if win.sum() < min_npts:
            return np.nan, np.nan
        try:
            theta = optimize_lsq(vel[win], -ccf[win], "gaussian")[0]
        except (RuntimeError, ValueError):
            return np.nan, np.nan
        mu1, sigma1 = theta[2], theta[3]
        if not np.isfinite(mu1) or not np.isfinite(sigma1):
            return np.nan, np.nan

        # Second pass: +/-fit_nsigma sigma about the first-pass mean, symmetric,
        # with at least min_npts points; these points also set the error estimate.
        dv = np.mean(np.diff(vel))
        half_pts = max(
            int(np.floor(fit_nsigma * sigma1 / dv)), int(np.ceil((min_npts - 1) / 2))
        )
        center = int(np.argmin(np.abs(vel - mu1)))
        idx_lo, idx_hi = center - half_pts, center + half_pts + 1
        if idx_lo < 0 or idx_hi > vel.size:
            return mu1, np.nan  # symmetric window runs off the grid
        vel_fit, ccf_fit = vel[idx_lo:idx_hi], ccf[idx_lo:idx_hi]
        rv = mu1
        try:
            mu2 = optimize_lsq(vel_fit, -ccf_fit, "gaussian")[0][2]
            if vel_fit.min() <= mu2 <= vel_fit.max():
                rv = mu2
        except (RuntimeError, ValueError):
            pass

        # Photon-limited RV uncertainty from the weighted CCF slope over the
        # second-pass window (Bouchy 2001).
        ccf_var_fit = ccf_var[idx_lo:idx_hi]
        if not np.any(ccf_fit) or np.any(ccf_fit < 0) or np.any(ccf_var_fit <= 0):
            raise ValueError(
                "non-physical CCF or CCF variance in the RV fit window "
                "(zero/negative flux counts); cannot compute a photon-limited error"
            )

        # Boisse (2010) N_scale = CCF step / CCF-noise correlation length,
        # decorrelating the oversampled CCF.
        corr_length = RadialVelocity._ccf_noise_corr_length(
            vel_span_per_pixel, mask_width
        )
        n_scale = dv / corr_length  # dv: mean velocity-grid step

        weighted_slope = np.gradient(ccf_fit, vel_fit) ** 2 / ccf_var_fit
        qccf = (np.sum(weighted_slope) ** 0.5 / np.sum(ccf_fit) ** 0.5) * n_scale**0.5
        rv_err = 1.0 / (qccf * np.sum(ccf_fit) ** 0.5)

        return rv, rv_err

    def _combine_ccfs(self, chip, fibers):
        """
        Combine the cached per-order CCFs of one chip into a weighted-average CCF
        (for the RV value) and an unweighted-sum CCF (for the photon error).

        Summing happens within a single chip: the cached order CCFs are summed
        across `fibers`, then collapsed across orders two ways -- normalized to
        unit sum and scaled by each order's CCF weight (the value CCF), and a raw
        nansum (the count-scale error CCF). Cross-CCD combination is done at the
        RV level by compute_weighted_rvs, not here.

        Parameters
        ----------
        chip : str
            Chip identifier, i.e. 'GREEN' or 'RED'.
        fibers : str or list of str
            A single fiber (e.g. 'SCI1') OR exactly the three science fibers
            (SCI1, SCI2, SCI3), which are summed before collapsing. All summed
            fibers share a mask, so the first is used for the grid, pixel scale,
            and order weights.

        Returns
        -------
        velocity_grid : ndarray
            Shared CCF velocity grid [km/s].
        weighted_ccf : ndarray
            Order-weighted collapsed CCF (the RV value); identically zero if no
            order carries weight.
        summed_ccf : ndarray
            Unweighted order-summed collapsed CCF (the photon-error signal).
        summed_ccf_var : ndarray
            Per-bin photon variance of summed_ccf.
        rep_scale : float
            Native per-pixel velocity scale of the strongest order (the
            photon-error velocity scale).

        Raises
        ------
        ValueError
            If `fibers` is neither a single fiber nor exactly the three science
            fibers.
        RuntimeError
            If the CCFs have not been loaded for any requested fiber.
        """
        chip = chip.upper()
        fibers = [fibers] if isinstance(fibers, str) else list(fibers)
        fibers = [f.upper() for f in fibers]
        if len(fibers) != 1 and set(fibers) != set(self._SCI_FIBERS):
            raise ValueError(
                f"fibers must be a single fiber or exactly the three science "
                f"fibers {list(self._SCI_FIBERS)}; got {fibers}"
            )
        for f in fibers:
            if f"{chip}_{f}" not in self._ccf:
                raise RuntimeError(
                    f"CCF for {chip}_{f} not loaded; call perform() (which loads "
                    "the L4 CCFs) first"
                )

        # All fibers share the grid/mask, so the first is representative.
        velocity_grid = self._velocity_grid[f"{chip}_{fibers[0]}"]
        table = self.l4_obj.data[f"{chip}_{fibers[0]}_RV"]
        wave_start = np.asarray(table["WAVE_START"], dtype=np.float64)
        wave_end = np.asarray(table["WAVE_END"], dtype=np.float64)
        weights = self._get_order_weights(chip, fibers[0])
        ccf = np.sum([self._ccf[f"{chip}_{f}"] for f in fibers], axis=0)
        ccf_var = np.sum([self._ccf_var[f"{chip}_{f}"] for f in fibers], axis=0)

        ccf_weighted = np.zeros(velocity_grid.size)
        for o in range(ccf.shape[0]):
            ccf_sum = np.nansum(ccf[o])
            if ccf_sum > 0 and weights[o] != 0:
                ccf_weighted += (ccf[o] / ccf_sum) * weights[o]
        ccf_summed = np.nansum(ccf, axis=0)
        ccf_summed_var = np.nansum(ccf_var, axis=0)

        # The dispersion is ~uniform across the chip; use the strongest order's
        # pixel scale for the photon-noise velocity scale.
        rep = int(np.argmax(np.nansum(ccf, axis=1)))
        rep_scale = self._pixel_velocity_scale(
            wave_start[rep], wave_end[rep], self.ccd["ncol"]
        )
        return velocity_grid, ccf_weighted, ccf_summed, ccf_summed_var, rep_scale

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def compute_order_by_order_rvs(
        self, chip, fiber, window=None, fit_nsigma=3.0, min_npts=9
    ):
        """
        Per-order radial velocities for one chip/fiber, from the loaded CCF.

        Parameters
        ----------
        chip : str
            Chip identifier, i.e. 'GREEN' or 'RED'.
        fiber : str
            Fiber identifier, e.g. 'SCI1'.
        window : list of float, optional
            [min, max] km/s window about the dip for the first-pass fit.
            Defaults to the configured value.
        fit_nsigma : float, optional
            Half-width of the second-pass fit window in units of the first-pass
            sigma. Not a configurable parameter; set in code (default 3.0).
        min_npts : int, optional
            Minimum number of grid points to use in each fit window. Not a
            configurable parameter; set in code (default 9).

        Returns
        -------
        dict
            {'rv', 'rv_err'} -- length-norder_chip ndarrays [km/s], one per order.

        Raises
        ------
        RuntimeError
            If the CCFs have not been loaded for this chip/fiber (call perform()
            or _load_ccfs first).
        """
        if window is None:
            window = self.rv_window

        chip = chip.upper()
        fiber = fiber.upper()
        ext = f"{chip}_{fiber}"

        if ext not in self._ccf:
            raise RuntimeError(
                f"CCF for {ext} not loaded; call perform() (which loads the L4 "
                "CCFs) before compute_order_by_order_rvs"
            )
        ccf = self._ccf[ext]
        ccf_var = self._ccf_var[ext]

        velocity_grid = self._velocity_grid[ext]  # the grid CrossCorrelation used
        table = self.l4_obj.data[ext + "_RV"]
        wave_start = np.asarray(table["WAVE_START"], dtype=np.float64)
        wave_end = np.asarray(table["WAVE_END"], dtype=np.float64)
        mask_width = self._ccf_mask_width
        ncol = self.ccd["ncol"]
        rv = np.full(ccf.shape[0], np.nan)
        rv_err = np.full(ccf.shape[0], np.nan)
        for o in range(ccf.shape[0]):
            if not np.any(ccf[o]):
                continue
            vel_span_per_pixel = self._pixel_velocity_scale(
                wave_start[o], wave_end[o], ncol
            )
            # Degrade a non-physical single order to NaN; the loud raise is kept
            # for the fiber-summed science CCF in compute_weighted_rvs.
            try:
                rv[o], rv_err[o] = self._compute_rv_1d(
                    velocity_grid,
                    ccf[o],
                    ccf_var[o],
                    vel_span_per_pixel,
                    mask_width,
                    window,
                    fit_nsigma,
                    min_npts,
                )
            except ValueError as e:
                logger.warning(
                    "%s order %d: non-physical CCF window (%s); RV/RV_ERR set NaN",
                    ext,
                    o,
                    e,
                )

        return {"rv": rv, "rv_err": rv_err}

    def compute_weighted_rvs(
        self,
        chips,
        fibers,
        combine_fibers,
        combine_ccds,
        window=None,
        fit_nsigma=3.0,
        min_npts=9,
    ):
        """
        Weighted-combined RVs from the loaded CCFs, collapsing orders (and
        optionally fibers and CCDs) into one RV per group.

        Orders are always weighted-collapsed (the per-order path is
        compute_order_by_order_rvs). Two flags control the rest:

          combine_fibers : sum the three science fibers' CCFs before collapsing.
                           True requires `fibers` == the three SCI fibers; False
                           requires a single fiber.
          combine_ccds   : combine the per-chip RVs across CCDs at the RV level --
                           value = order-weight-weighted mean of the per-chip RVs,
                           error = inverse-variance combination.

        Per chip, the value comes from fitting the weighted-average CCF and the
        error from fitting the unweighted-sum CCF (see _combine_ccfs).

        Parameters
        ----------
        chips : str or list of str
            Chip identifier(s), i.e. 'GREEN' and/or 'RED'.
        fibers : str or list of str
            A single fiber, or the three science fibers when combine_fibers=True.
        combine_fibers : bool
            Sum the three science fibers before collapsing orders.
        combine_ccds : bool
            Combine the per-chip RVs across CCDs (RV-level weighted mean).
        window : list of float, optional
            [min, max] km/s window about the dip for the first-pass fit.
            Defaults to the configured value.
        fit_nsigma : float, optional
            Half-width of the second-pass fit window in units of the first-pass
            sigma. Not a configurable parameter; set in code (default 3.0).
        min_npts : int, optional
            Minimum number of grid points to use in each fit window.

        Returns
        -------
        dict or tuple
            combine_ccds=False -> dict {chip: (rv, rv_err)} over `chips`.
            combine_ccds=True  -> a single (rv, rv_err) tuple [km/s].
            A non-finite fit (no order carrying weight) yields (NaN, NaN).

        Raises
        ------
        ValueError
            If the combine_fibers flag and `fibers` are inconsistent.
        RuntimeError
            If the CCFs have not been loaded for any requested chip/fiber.
        """
        if window is None:
            window = self.rv_window

        chips = [chips] if isinstance(chips, str) else list(chips)
        chips = [c.upper() for c in chips]
        fibers = [fibers] if isinstance(fibers, str) else list(fibers)
        fibers = [f.upper() for f in fibers]

        if combine_fibers and set(fibers) != set(self._SCI_FIBERS):
            raise ValueError(
                f"combine_fibers=True requires the three science fibers "
                f"{list(self._SCI_FIBERS)}; got {fibers}"
            )
        if not combine_fibers and len(fibers) != 1:
            raise ValueError(
                f"combine_fibers=False requires a single fiber; got {fibers}"
            )

        # Per-chip weighted RV: value from the weighted CCF, error from the
        # unweighted-sum CCF.
        mask_width = self._ccf_mask_width
        per_chip = {}
        for chip in chips:
            grid, ccf_w, ccf_s, ccf_s_var, scale = self._combine_ccfs(chip, fibers)
            if not np.any(ccf_w):
                per_chip[chip] = (np.nan, np.nan)
                continue
            rv = self._compute_rv_1d(
                grid,
                ccf_w,
                ccf_s_var,
                scale,
                mask_width,
                window,
                fit_nsigma,
                min_npts,
            )[0]
            rv_err = self._compute_rv_1d(
                grid,
                ccf_s,
                ccf_s_var,
                scale,
                mask_width,
                window,
                fit_nsigma,
                min_npts,
            )[1]
            per_chip[chip] = (rv, rv_err)

        if not combine_ccds:
            return per_chip

        # Cross-CCD combine: order-weight-weighted RV, inverse-variance error.
        num = den = ivar = 0.0
        for chip in chips:
            rv, rv_err = per_chip[chip]
            w = float(np.nansum(self._get_order_weights(chip, fibers[0])))
            if np.isfinite(rv) and w > 0:
                num += rv * w
                den += w
            if np.isfinite(rv_err) and rv_err > 0:
                ivar += 1.0 / rv_err**2
        rv = num / den if den > 0 else np.nan
        rv_err = ivar**-0.5 if ivar > 0 else np.nan
        return rv, rv_err

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------

    def _track_info(self, fibers):
        """Build and cache the info() summary text from instance attributes."""
        info = {}
        for fiber in fibers:
            fiber = fiber.upper()
            pf = self._per_fiber.get(fiber, {})
            entry = {"rv": pf.get("rv"), "rv_err": pf.get("rv_err")}
            if fiber in self._processed:
                entry["ccd_rv"] = pf.get("ccd_rv", {})
                entry["ccd_rv_err"] = pf.get("ccd_rv_err", {})
                entry["mask"] = self.l4_obj.headers[f"{fiber}_CCF"].get("CCFMASK")
            info[fiber] = entry

        obs_id = self.l4_obj.headers.get("RECEIPT", {}).get("ORIGID", "unknown")
        lines = [
            "RadialVelocity",
            f"  obs_id:     {obs_id}",
            f"  rv_window:  {self.rv_window} km/s",
        ]

        # Per-CCD, per-orderlet summary. MASK is the CCF mask; CCD_RV/CCD_ERV are
        # the combined per-CCD RV and its error; RV_RMS is the order-to-order
        # mad_std (a diagnostic of per-order spread), in m/s.
        fiber_order = [f for f in ("SCI1", "SCI2", "SCI3", "SKY", "CAL") if f in info]
        fiber_order += [f for f in info if f not in fiber_order]

        lines.append(
            f"\n  {'CHIP':<8s}{'FIBER':<8s}{'MASK':<14s}{'NVALID':>8s}"
            f"{'CCD_RV [km/s]':>16s}{'CCD_ERV [m/s]':>16s}{'RV_RMS [m/s]':>16s}"
        )
        lines.append("  " + "-" * 86)
        norder_green = self.norder["GREEN"]
        norder = norder_green + self.norder["RED"]
        for chip, rows in (
            ("GREEN", slice(0, norder_green)),
            ("RED", slice(norder_green, norder)),
        ):
            for fiber in fiber_order:
                res = info[fiber]
                rv = res["rv"]
                if rv is None:
                    continue
                rv = rv[rows]
                nvalid = int(np.sum(np.isfinite(rv)))
                if nvalid == 0:
                    continue
                ccd_rv = res.get("ccd_rv", {}).get(chip, np.nan)
                ccd_erv = res.get("ccd_rv_err", {}).get(chip, np.nan)
                rv_rms = mad_std(rv, ignore_nan=True) * 1e3 if nvalid >= 2 else np.nan
                lines.append(
                    f"  {chip:<8s}{fiber:<8s}{str(res.get('mask', '')):<14s}"
                    f"{nvalid:>8d}{ccd_rv:>+16.5f}{ccd_erv * 1e3:>16.3f}"
                    f"{rv_rms:>16.3f}"
                )
        self._info = "\n".join(lines)

    def _set_headers(self, l4_obj):
        """
        Write all RV keywords, the single place this module writes headers, called
        just before the receipt entry. Reads only the stashes filled by perform().

        Per illuminated orderlet: the RVn RV-processing descriptors
        (RVMETHOD/SKYRMVD/TELLRMVD) and the legacy per-fiber per-CCD RV
        CCD{n}RV{sfx}/CCD{n}ERV{sfx} (routed to the RV# table). On PRIMARY:
        RVMETHOD, and -- when a science combine ran -- the SCI-combined per-CCD
        CCD{n}RV/CCD{n}ERV and the EPRV RV/RVERR/BERV/BJDTDB. A non-finite value is
        written as None (a FITS UNDEFINED card). The RVn CTYPE cards belong to
        CrossCorrelation and are not rewritten here.
        """
        sfx = {"SCI1": "1", "SCI2": "2", "SCI3": "3", "CAL": "C", "SKY": "S"}
        for fiber in self._processed:
            rv_ext = f"{fiber}_RV"
            l4_obj.set_keyword("RVMETHOD", "CCF", ext=rv_ext)
            l4_obj.set_keyword("SKYRMVD", False, ext=rv_ext)
            l4_obj.set_keyword("TELLRMVD", False, ext=rv_ext)
            pf = self._per_fiber[fiber]
            for chip, v in pf["ccd_rv"].items():
                n = 1 if chip == "GREEN" else 2
                e = pf["ccd_rv_err"][chip]
                l4_obj.set_keyword(
                    f"CCD{n}RV{sfx[fiber]}", float(v) if np.isfinite(v) else None
                )
                l4_obj.set_keyword(
                    f"CCD{n}ERV{sfx[fiber]}", float(e) if np.isfinite(e) else None
                )

        # PRIMARY (EPRV L4): always the RV method; the combined RV only when a
        # science combine was formed (else RV/RVERR/BERV/BJDTDB stay UNDEFINED).
        l4_obj.set_keyword("RVMETHOD", "CCF")
        if not self._sci_combined_ran:
            return
        for chip, v in self._sci_ccd_rv.items():
            n = 1 if chip == "GREEN" else 2
            e = self._sci_ccd_err[chip]
            l4_obj.set_keyword(f"CCD{n}RV", float(v) if np.isfinite(v) else None)
            l4_obj.set_keyword(f"CCD{n}ERV", float(e) if np.isfinite(e) else None)
        l4_obj.set_keyword(
            "RV", float(self._combined_rv) if np.isfinite(self._combined_rv) else None
        )
        l4_obj.set_keyword(
            "RVERR",
            float(self._combined_rverr) if np.isfinite(self._combined_rverr) else None,
        )
        l4_obj.set_keyword(
            "BERV",
            float(self._primary_berv) if np.isfinite(self._primary_berv) else None,
        )
        l4_obj.set_keyword(
            "BJDTDB",
            float(self._primary_bjdtdb) if np.isfinite(self._primary_bjdtdb) else None,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(
        self,
        chips=None,
        fibers=None,
        *,
        rv_window=None,
        fit_nsigma=3.0,
        min_npts=9,
    ):
        """
        Fit the per-order CCFs to radial velocities and store them on the KPF4.

        For each illuminated orderlet the per-order RVs of both chips are filled
        into the orderlet's RV table ({fiber}_RV RV/RV_ERR columns), leaving the
        CrossCorrelation-seeded metadata columns intact.

        Parameters
        ----------
        chips : list of str, optional
            Chip identifiers, i.e. 'GREEN' or 'RED'. Defaults to the configured
            chips.
        fibers : list of str, optional
            Fiber identifiers, e.g. ['SCI1', 'SCI2']. Defaults to all configured
            fibers (SCI, CAL, and SKY).
        rv_window : list of float, optional
            [min, max] km/s window about the dip for the first-pass fit.
            Overrides the configured value.
        fit_nsigma : float, optional
            Half-width of the second-pass fit window in units of the first-pass
            sigma. Not a configurable parameter; set in code (default 3.0).
        min_npts : int, optional
            Minimum number of grid points to use in each fit window. Not a
            configurable parameter; set in code (default 9).

        Returns
        -------
        l4_obj : KPF4
            The input L4 with per-order RV/RV_ERR filled per illuminated orderlet.
            Each RV extension carries RVMETHOD/SKYRMVD/TELLRMVD and the per-fiber
            legacy CCD<n>RV<sfx>/CCD<n>ERV<sfx>. PRIMARY carries the final science
            RV: the EPRV RVMETHOD/RV/RVERR/BERV/BJDTDB plus the KPF SCI-combined
            per-CCD CCD<n>RV/CCD<n>ERV. Fibers with no CCF (unilluminated) are
            skipped.
        """
        if chips is None:
            chips = self.chips
        if fibers is None:
            fibers = self.fibers
        if rv_window is None:
            rv_window = self.rv_window

        chips = [c.upper() for c in chips]
        fibers = [f.upper() for f in fibers]

        norder_green = self.norder["GREEN"]
        norder = norder_green + self.norder["RED"]
        l4_obj = self.l4_obj

        self._load_ccfs(chips, fibers)

        self._processed = []
        self._per_fiber = {}
        self._sci_combined_ran = False
        for fiber in fibers:
            # Skip fibers CrossCorrelation did not produce a CCF for.
            if l4_obj.data[f"{fiber}_CCF"].size == 0:
                self._per_fiber[fiber] = {
                    "rv": np.full(norder, np.nan),
                    "rv_err": np.full(norder, np.nan),
                }
                logger.info("%s: no CCF; skipping (no RV)", fiber)
                continue
            self._processed.append(fiber)

            rv = np.full(norder, np.nan)
            rv_err = np.full(norder, np.nan)
            for chip in chips:
                result = self.compute_order_by_order_rvs(
                    chip, fiber, rv_window, fit_nsigma, min_npts
                )
                rows = (
                    slice(0, norder_green)
                    if chip == "GREEN"
                    else slice(norder_green, norder)
                )
                rv[rows] = result["rv"]
                rv_err[rows] = result["rv_err"]

            # Fill RV/RV_ERR into the CrossCorrelation-seeded RV table, preserving
            # its metadata columns (set_data replaces the whole table).
            table = l4_obj.data[f"{fiber}_RV"]
            table["RV"] = rv
            table["RV_ERR"] = rv_err
            l4_obj.set_data(f"{fiber}_RV", table)

            # Per-fiber per-CCD weighted RV (orders collapsed, single fiber).
            per_ccd = self.compute_weighted_rvs(
                chips,
                fiber,
                combine_fibers=False,
                combine_ccds=False,
                window=rv_window,
                fit_nsigma=fit_nsigma,
                min_npts=min_npts,
            )
            self._per_fiber[fiber] = {
                "rv": rv,
                "rv_err": rv_err,
                "ccd_rv": {chip: per_ccd[chip][0] for chip in chips},
                "ccd_rv_err": {chip: per_ccd[chip][1] for chip in chips},
            }

        # Final science RV: sum the science orderlets' CCFs per chip, fit, then
        # combine the two CCDs at the RV level (see compute_weighted_rvs). RVs are
        # already barycentric, so the reported BERV/BJDTDB are descriptive.
        sci_req = [f for f in fibers if f in self._SCI_FIBERS]
        sci = [f for f in sci_req if f in self._processed]
        if not sci_req:
            # Calibration-only run: PRIMARY RV/RVERR/BERV/BJDTDB stay UNDEFINED.
            logger.info(
                "combined RV: no science orderlet requested; PRIMARY RV left UNDEFINED"
            )
            self._set_headers(l4_obj)
            self._track_info(fibers)
            l4_obj.receipt_add_entry("radial_velocity", "", "PASS")
            return l4_obj
        if not sci:
            raise ValueError(
                f"science orderlet(s) {sci_req} requested but none illuminated; "
                "cannot form a combined RV"
            )
        rep = sci[0]
        if len(chips) == 1:
            logger.info(
                "combined RV: only chip %s present; the combined RV uses it alone",
                chips[0],
            )

        # Per CCD: bare SCI-combined RV/error (the three science fibers summed).
        per_ccd = self.compute_weighted_rvs(
            chips,
            sci,
            combine_fibers=True,
            combine_ccds=False,
            window=rv_window,
            fit_nsigma=fit_nsigma,
            min_npts=min_npts,
        )
        for chip in chips:
            v = per_ccd[chip][0]
            if not np.isfinite(v):
                logger.info(
                    "combined RV: %s science fit non-finite; "
                    "excluded from the combined RV",
                    chip,
                )
        self._sci_ccd_rv = {chip: per_ccd[chip][0] for chip in chips}
        self._sci_ccd_err = {chip: per_ccd[chip][1] for chip in chips}

        # Cross-chip weighted RV and inverse-variance error (PRIMARY RV/RVERR).
        ccfrv, ccferv = self.compute_weighted_rvs(
            chips,
            sci,
            combine_fibers=True,
            combine_ccds=True,
            window=rv_window,
            fit_nsigma=fit_nsigma,
            min_npts=min_npts,
        )
        if not np.isfinite(ccfrv):
            logger.info(
                "combined RV: no finite per-CCD science RV; PRIMARY RV UNDEFINED"
            )

        # PRIMARY BERV/BJDTDB: WEIGHT-weighted mean of the representative science
        # fiber's per-order BERV/BJD_TDB (RVn), matching DiagL4's BERVMEAN/BJDMEAN.
        table = l4_obj.data[f"{rep}_RV"]
        w = np.asarray(table["WEIGHT"], dtype=np.float64)

        def weighted_mean(col):
            x = np.asarray(col, dtype=np.float64)
            good = np.isfinite(x) & np.isfinite(w) & (w > 0)
            if not np.any(good):
                return np.nan
            return float(np.sum(x[good] * w[good]) / np.sum(w[good]))

        self._combined_rv = ccfrv
        self._combined_rverr = ccferv
        self._primary_berv = weighted_mean(table["BERV"])
        self._primary_bjdtdb = weighted_mean(table["BJD_TDB"])
        self._sci_combined_ran = True

        self._set_headers(l4_obj)
        self._track_info(fibers)
        l4_obj.receipt_add_entry("radial_velocity", "", "PASS")
        logger.info("summary:\n%s", self._info)
        return l4_obj

    def info(self):
        """Print a summary of the module configuration and RV results."""
        if self._info is None:
            print(f"{type(self).__name__}: perform() has not been called")
        else:
            print(self._info)
