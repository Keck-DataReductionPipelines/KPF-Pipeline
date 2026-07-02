"""
KPF Radial Velocity module.

Computes radial velocities from an extracted, wavelength-calibrated,
barycentric-corrected KPF L2 by cross-correlating each order's 1D spectrum
against a line mask, then collapsing the per-order CCFs into radial velocities
per order, orderlet, and CCD (weighted where appropriate). Produces a KPF4 (L4)
holding the CCFs and RVs.

Each fiber's mask, barycentric handling, and CCF grid center are dispatched from
its illumination source (SCI-OBJ/SKY-OBJ/CAL-OBJ in INSTRUMENT_HEADER):

  source   mask                 barycorr  grid center
  target   TARGTEFF-lookup      yes       TARGRADV (systemic)
  sky      G2_espresso (solar)  yes       0
  thar     ThAr list (unit wt)  no        0
  etalon   / lfc                                skipped (no CCF/RV; not implemented)
  none     not illuminated -> skipped (no CCF/RV)

All reference wavelengths (stellar line masks, ThAr line list) are in vacuum;
no air/vacuum conversion is performed.
"""

import warnings

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.constants import c
from astropy.io import fits
from astropy.stats import mad_std

from kpfpipe import DEFAULTS, REPO_ROOT
from kpfpipe.utils.astro import compute_redshift
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import optimize_lsq
from kpfpipe.utils.validation import strictly_increasing

_DEFAULTS = {
    **DEFAULTS,
    "ccf_mask_width": 0.5,
    "ccf_step_size": 0.25,
    "ccf_window": [-100.0, 100.0],
    "rv_window": [-25.0, 25.0],
}


class RadialVelocity:
    """
    Compute radial velocities from a KPF2 via cross-correlation.

    Parameters
    ----------
    l2_obj : KPF2
        Extracted L2 frame. Must have per-fiber FLUX and WAVE arrays populated
        (by SpectralExtraction and WavelengthCalibration) and the per-order
        barycentric correction extensions populated by BarycentricCorrection.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: chips, fibers, ccf_mask_width,
        ccf_step_size, ccf_window, rv_window.
    """

    def __init__(self, l2_obj, config=None):
        self.l2_obj = l2_obj

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

        # Lazily-populated caches; the per-orderlet ones are keyed by f'{chip}_{fiber}'.
        # source dict, set by _resolve_illumination_source()
        self._illumination_source = {}
        self._line_mask = {}  # line mask, set by _build_line_mask()
        self._velocity_grid = {}  # velocity grid, set by _build_velocity_grid()
        self._ccf = {}  # CCF cube, set by compute_ccfs()
        self._ccf_var = {}  # per-bin CCF variance cube, set by compute_ccfs()
        self._order_weights = (
            None  # shared order-weight table, loaded by _get_order_weights()
        )
        self._info = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # Fiber -> the INSTRUMENT_HEADER keyword giving its illumination source.
    _OBJ_KEYWORD = {
        "SCI1": "SCI-OBJ",
        "SCI2": "SCI-OBJ",
        "SCI3": "SCI-OBJ",
        "SKY": "SKY-OBJ",
        "CAL": "CAL-OBJ",
    }

    def _resolve_illumination_source(self, chip, fiber):
        """
        Resolve (and cache) the illumination source and its CCF settings for one
        orderlet, from its SCI-OBJ/SKY-OBJ/CAL-OBJ keyword in INSTRUMENT_HEADER.

        Returns a dict with keys 'object' (the normalized source), 'mask_name',
        'apply_barycorr', and 'vel_grid_center'. An unilluminated fiber ('none')
        has None mask/barycorr/center. Sources whose CCF path is not yet built
        (etalon, lfc) are skipped the same way, with a warning.
        """
        key = f"{chip.upper()}_{fiber.upper()}"
        if key in self._illumination_source:
            return self._illumination_source[key]
        try:
            keyword = self._OBJ_KEYWORD[fiber.upper()]
        except KeyError:
            raise ValueError(
                f"unknown fiber {fiber!r}; expected one of {sorted(self._OBJ_KEYWORD)}"
            ) from None
        inst = self.l2_obj.headers.get("INSTRUMENT_HEADER", {})
        if keyword not in inst:
            raise ValueError(
                f"illumination keyword {keyword!r} not in INSTRUMENT_HEADER; "
                f"cannot dispatch a mask for fiber {fiber}"
            )

        # Map the raw keyword value straight to the source object and its CCF
        # settings: mask, whether to barycentric-correct, and the velocity-grid
        # center (systemic RV for a star, 0 for sky/calibration).
        raw = inst.get(keyword)
        v = str(raw).strip().lower()
        if v == "target":
            source = {
                "object": "target",
                "mask_name": self._resolve_stellar_mask(),
                "apply_barycorr": True,
                "vel_grid_center": self._get_systemic_rv(),
            }
        elif v == "sky":
            source = {
                "object": "sky",
                "mask_name": "G2_espresso",
                "apply_barycorr": True,
                "vel_grid_center": 0.0,
            }
        elif v in ("th_gold", "th_daily"):
            source = {
                "object": "thar",
                "mask_name": "thar",
                "apply_barycorr": False,
                "vel_grid_center": 0.0,
            }
        elif v == "none":
            source = {
                "object": "none",
                "mask_name": None,
                "apply_barycorr": None,
                "vel_grid_center": None,
            }
        elif v == "lfcfiber":
            source = {
                "object": "lfc",
                "mask_name": None,
                "apply_barycorr": None,
                "vel_grid_center": None,
            }
            warnings.warn(
                f"{fiber.upper()} is lfc-illuminated; CCF is not implemented. "
                "Skipping this fiber.",
                UserWarning,
                stacklevel=2,
            )
        elif "etalon" in v:
            source = {
                "object": "etalon",
                "mask_name": None,
                "apply_barycorr": None,
                "vel_grid_center": None,
            }
            warnings.warn(
                f"{fiber.upper()} is etalon-illuminated; CCF is not implemented. "
                "Skipping this fiber.",
                UserWarning,
                stacklevel=2,
            )
        else:
            raise ValueError(f"unrecognized illumination source {raw!r}")

        self._illumination_source[key] = source
        return source

    def _resolve_stellar_mask(self):
        """Select the stellar line-mask name from TARGTEFF via line_mask_lookup.csv."""
        inst = self.l2_obj.headers.get("INSTRUMENT_HEADER", {})
        try:
            teff = float(inst.get("TARGTEFF"))
        except (TypeError, ValueError):
            teff = None
        if teff is None or not np.isfinite(teff) or teff <= 0:
            raise ValueError(
                "target effective temperature (TARGTEFF) not available in "
                "INSTRUMENT_HEADER; cannot select a stellar line mask"
            )
        line_map = pd.read_csv(f"{REPO_ROOT}/reference/line_masks/line_mask_lookup.csv")
        row = line_map[(line_map["TEFF_MIN"] <= teff) & (teff < line_map["TEFF_MAX"])]
        return row["DEFAULT_MASK"].iloc[0]

    def _get_systemic_rv(self):
        """Target systemic RV (TARGRADV) [km/s] — the stellar CCF grid center."""
        inst = self.l2_obj.headers.get("INSTRUMENT_HEADER", {})
        try:
            star_rv = float(inst.get("TARGRADV"))
        except (TypeError, ValueError):
            star_rv = None
        if star_rv is None or not np.isfinite(star_rv):
            raise ValueError(
                "target radial velocity (TARGRADV) not available in "
                "INSTRUMENT_HEADER; cannot center the CCF velocity grid"
            )
        return star_rv

    def _build_line_mask(self, chip, fiber, width=None):
        """
        Build (and cache) the orderlet's CCF line mask: vacuum line centers,
        weights, and per-line top-hat edges at velocity -/+ width about each
        center (relativistic Doppler). The mask is selected from the orderlet's
        illumination source.

        Stellar masks load from reference/line_masks/stellar_masks/; the 'thar'
        mask is built from the ThAr line list with uniform weights.

        Returns
        -------
        dict
            Mask with keys 'center', 'weight', 'start', 'end', each a 1D ndarray
            of length n_line; wavelengths are vacuum [Å].
        """
        if width is None:
            width = self.ccf_mask_width
        key = f"{chip.upper()}_{fiber.upper()}"
        if key in self._line_mask:
            return self._line_mask[key]

        mask_name = self._resolve_illumination_source(chip, fiber)["mask_name"]
        if mask_name == "thar":
            df = pd.read_csv(f"{REPO_ROOT}/reference/thar_line_list.csv")
            # Deduplicate: lines recur across overlapping orders and would
            # otherwise be double-counted.
            centers = np.unique(df["WAVE"].to_numpy(dtype=float))
            weights = np.ones(centers.size)
        else:
            mask_path = (
                f"{REPO_ROOT}/reference/line_masks/stellar_masks/{mask_name}.txt"
            )
            centers, weights = np.loadtxt(mask_path, unpack=True)  # vacuum wavelengths

        mask = {
            "center": centers,
            "weight": weights,
            "start": centers * (1.0 + compute_redshift(-width * u.km / u.s)),
            "end": centers * (1.0 + compute_redshift(+width * u.km / u.s)),
        }
        self._line_mask[key] = mask
        return mask

    def _build_velocity_grid(self, chip, fiber, step_size=None, window=None):
        """
        Build (and cache) the orderlet's CCF velocity grid: evenly spaced over
        `window` about the orderlet's grid center in `step_size` increments.

        The [min, max] `window` is converted to an integer number of `step_size`
        steps (so the step is exact), then shifted by the center — the systemic
        RV (TARGRADV) for stellar fibers, 0 for sky/cal fibers.
        """
        if step_size is None:
            step_size = self.ccf_step_size
        if window is None:
            window = self.ccf_window
        key = f"{chip.upper()}_{fiber.upper()}"
        if key in self._velocity_grid:
            return self._velocity_grid[key]

        center = self._resolve_illumination_source(chip, fiber)["vel_grid_center"]
        lo_kms, hi_kms = window
        lo = int(round(lo_kms / step_size))
        hi = int(round(hi_kms / step_size))
        grid = np.arange(lo, hi + 1) * step_size + center
        self._velocity_grid[key] = grid
        return grid

    def _get_order_weights(self, chip, fiber):
        """
        Per-order CCF-combination weights for one orderlet, from
        reference/ccf_order_weights.csv (column selected by the orderlet's mask).
        Returns a 1D ndarray of length norder_chip, ordered by ORDER.
        """
        if self._order_weights is None:
            self._order_weights = pd.read_csv(
                f"{REPO_ROOT}/reference/ccf_order_weights.csv"
            )
        df = self._order_weights
        mask_name = self._resolve_illumination_source(chip, fiber)["mask_name"]
        if mask_name not in df.columns:
            raise KeyError(
                f"no CCF order-weight column for mask {mask_name!r} in "
                f"ccf_order_weights.csv; have "
                f"{[c for c in df.columns if c not in ('CHIP', 'ORDER')]}"
            )
        rows = df[df["CHIP"] == chip.upper()].sort_values("ORDER")
        return rows[mask_name].to_numpy(dtype=float)

    @staticmethod
    def _compute_ccf_1d(wave, flux, var, line_mask, velocity_grid, barycorr_z):
        """
        Cross-correlate one order's spectrum against the mask over the velocity
        grid, folding in the order's barycentric redshift z.

        Parameters
        ----------
        wave : ndarray
            1D wavelength solution for the order [Å].
        flux : ndarray
            1D extracted flux for the order.
        var : ndarray
            1D per-pixel variance for the order (TRACE_VAR); sets the CCF photon
            variance.
        line_mask : dict
            Line mask (keys 'start', 'end', 'weight') from _build_line_mask.
        velocity_grid : ndarray
            CCF velocity steps [km/s].
        barycorr_z : float
            Barycentric redshift for the order.

        Returns
        -------
        ccf : ndarray
            CCF value at each velocity step (all zeros if the order is unusable
            or no mask lines fall fully within it).
        ccf_var : ndarray
            Per-velocity-bin photon variance sum(w**2 * var), where w is the
            per-pixel mask weight (all zeros in the same unusable cases).

        Raises
        ------
        ValueError
            If the WAVE array is descending; an ascending (blue->red) solution
            is expected, so a reversed order signals an upstream orientation
            error rather than something to silently correct.
        """
        wave = np.asarray(wave, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        var = np.asarray(var, dtype=np.float64)
        if wave[0] > wave[-1]:
            raise ValueError(
                f"WAVE array is descending (wave[0]={wave[0]:.4f} > "
                f"wave[-1]={wave[-1]:.4f}); expected ascending blue->red "
                f"orientation. This signals an upstream orientation error."
            )

        ccf = np.zeros(velocity_grid.size)
        ccf_var = np.zeros(velocity_grid.size)
        n_pix = wave.size
        if n_pix < 3 or not strictly_increasing(wave):
            return ccf, ccf_var

        # Wavelength bin edges (length n+1) and widths at the pixel midpoints.
        edges = np.empty(n_pix + 1)
        edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
        edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
        edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
        widths = np.diff(edges)
        # Relativistic mask shift per velocity step, de-redshifting the barycorr.
        shift = (1.0 + compute_redshift(velocity_grid * u.km / u.s)) / (
            1.0 + barycorr_z
        )

        # Keep only mask lines that stay fully inside the order across the whole
        # scan, so the same lines contribute at every step (flat CCF baseline).
        smin, smax = shift.min(), shift.max()
        keep = (line_mask["start"] * smin >= wave[0]) & (
            line_mask["end"] * smax <= wave[-1]
        )
        if not np.any(keep):
            return ccf, ccf_var
        l_start, l_end = line_mask["start"][keep], line_mask["end"][keep]
        l_weight = line_mask["weight"][keep]

        # NaN-clean flux once: np.nansum replaces NaNs with 0 before summing, so
        # summing flux_clean * overlap_frac (overlap weights are always finite)
        # is identical but skips the per-step NaN mask.
        flux_clean = np.nan_to_num(flux)
        var_clean = np.nan_to_num(var)

        # Shifted line edges and their covering-pixel indices for every velocity
        # step at once: searchsorted takes an array of any shape, so one batched
        # call over the (nv, nline) grid replaces a per-step call (same result).
        line_lo_all = shift[:, None] * l_start[None, :]
        line_hi_all = shift[:, None] * l_end[None, :]
        idx_lo_all = np.clip(
            np.searchsorted(edges, line_lo_all, side="right") - 1, 0, n_pix - 1
        )
        idx_hi_all = np.clip(
            np.searchsorted(edges, line_hi_all, side="right") - 1, 0, n_pix - 1
        )

        for vi in range(velocity_grid.size):
            line_lo = line_lo_all[vi]
            line_hi = line_hi_all[vi]
            idx_lo = idx_lo_all[vi]
            idx_hi = idx_hi_all[vi]

            # Fractional overlap of each (narrow) line with the pixels it covers.
            overlap_frac = np.zeros(n_pix)
            for offset in range(int((idx_hi - idx_lo).max()) + 1):
                pix = idx_lo + offset
                still_spanning = pix <= idx_hi
                pix_sel = pix[still_spanning]
                overlap = np.minimum(
                    edges[pix_sel + 1], line_hi[still_spanning]
                ) - np.maximum(edges[pix_sel], line_lo[still_spanning])
                np.maximum(overlap, 0.0, out=overlap)
                np.add.at(
                    overlap_frac,
                    pix_sel,
                    l_weight[still_spanning] * overlap / widths[pix_sel],
                )

            ccf[vi] = np.sum(flux_clean * overlap_frac)
            ccf_var[vi] = np.sum(var_clean * overlap_frac**2)

        return ccf, ccf_var

    @staticmethod
    def _compute_rv_1d(vel, ccf, ccf_var, wave, window, fit_nsigma=3.0, min_npts=9):
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
        wave : ndarray
            1D wavelength solution for the order [Å]; sets the per-pixel velocity
            scale used in the error estimate.
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
        # second-pass window; per-bin noise is the propagated CCF variance
        # sum(w**2 * var) (Bouchy 2001).
        ccf_var_fit = ccf_var[idx_lo:idx_hi]
        if not np.any(ccf_fit) or np.any(ccf_fit < 0) or np.any(ccf_var_fit <= 0):
            raise ValueError(
                "non-physical CCF or CCF variance in the RV fit window "
                "(zero/negative flux counts); cannot compute a photon-limited error"
            )

        # nan-aware: in the combined-CCF path `wave` is the full, unclipped order
        # row, which may carry NaN edge pixels; plain np.median would NaN the error.
        speed_of_light_kms = c.to("km/s").value
        vel_span_per_pixel = (
            speed_of_light_kms
            * np.nanmedian(np.abs(np.diff(wave)))
            / np.nanmedian(wave)
        )
        n_pix_per_vel_step = dv / vel_span_per_pixel  # dv: mean velocity-grid step

        weighted_slope = np.gradient(ccf_fit, vel_fit) ** 2 / ccf_var_fit
        qccf = (
            np.sum(weighted_slope) ** 0.5 / np.sum(ccf_fit) ** 0.5
        ) * n_pix_per_vel_step**0.5
        rv_err = 1.0 / (qccf * np.sum(ccf_fit) ** 0.5)

        return rv, rv_err

    # Science orderlets that may be summed together (shared mask and grid).
    _SCI_FIBERS = ("SCI1", "SCI2", "SCI3")

    def _combine_ccfs(self, chip, fibers):
        """
        Combine the cached per-order CCFs of one chip into a weighted-average CCF
        (for the RV value) and an unweighted-sum CCF (for the photon error).

        Summing happens within a single chip: the cached order CCFs are summed
        across `fibers`, then collapsed across orders two ways — normalized to
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
            fibers share a mask, so the first is used for the grid, wave scale,
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
        rep_wave : ndarray
            WAVE row of the strongest order (the photon-error velocity scale).

        Raises
        ------
        ValueError
            If `fibers` is neither a single fiber nor exactly the three science
            fibers.
        RuntimeError
            If compute_ccfs has not been called for any requested fiber.
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
                    f"CCF for {chip}_{f} not available; call compute_ccfs("
                    f"{chip!r}, {f!r}) first"
                )

        # All fibers share the grid/mask, so the first is representative.
        velocity_grid = self._velocity_grid[f"{chip}_{fibers[0]}"]
        wave = np.asarray(
            self.l2_obj.data[f"{chip}_{fibers[0]}_WAVE"], dtype=np.float64
        )
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
        # WAVE for the photon-noise velocity scale.
        rep = int(np.argmax(np.nansum(ccf, axis=1)))
        return velocity_grid, ccf_weighted, ccf_summed, ccf_summed_var, wave[rep]

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def compute_ccfs(
        self,
        chip,
        fiber,
        width=None,
        step_size=None,
        window=None,
        clip_edge_pixels=(500, 500),
    ):
        """
        Cross-correlate every order of one chip/fiber against the line mask.

        Parameters
        ----------
        chip : str
            Chip identifier, i.e. 'GREEN' or 'RED'.
        fiber : str
            Fiber identifier, e.g. 'SCI1'.
        width : float, optional
            Per-line mask top-hat width [km/s]. Defaults to the configured value.
        step_size : float, optional
            CCF velocity step size [km/s]. Defaults to the configured value.
        window : list of float, optional
            CCF velocity grid range [km/s] as [min, max] about the grid center.
            Defaults to the configured value.
        clip_edge_pixels : tuple of int, optional
            Number of pixels to drop from the (short_wavelength_end,
            long_wavelength_end) of each order before correlating, removing the
            blaze-faint, low-S/N order edges. Defaults to (500, 500).

        Returns
        -------
        dict or None
            {'velocity', 'ccf'}: the CCF velocity grid [km/s] and the CCF with
            shape (norder_chip, n_velocity_step). The CCF is also cached under
            f'{chip}_{fiber}' for a subsequent compute_order_by_order_rvs call.
            Returns None if the fiber is not illuminated (source 'none').

        Raises
        ------
        ValueError
            If BARYCORR_Z is required (astronomical source) but not populated.
        NotImplementedError
            If the fiber's illumination source has no CCF path yet (etalon, lfc).
        """
        chip = chip.upper()
        fiber = fiber.upper()
        if width is None:
            width = self.ccf_mask_width
        if step_size is None:
            step_size = self.ccf_step_size
        if window is None:
            window = self.ccf_window

        source = self._resolve_illumination_source(chip, fiber)
        if source["object"] == "none":
            return None  # fiber not illuminated; caller skips
        apply_barycorr = source["apply_barycorr"]

        line_mask = self._build_line_mask(chip, fiber, width)
        velocity_grid = self._build_velocity_grid(chip, fiber, step_size, window)

        flux = np.asarray(self.l2_obj.data[f"{chip}_{fiber}_FLUX"], dtype=np.float64)
        wave = np.asarray(self.l2_obj.data[f"{chip}_{fiber}_WAVE"], dtype=np.float64)
        var = np.asarray(self.l2_obj.data[f"{chip}_{fiber}_VAR"], dtype=np.float64)

        # Drop the blaze-faint, low-S/N pixels at each order's edges, which
        # otherwise inject noise into the CCF. clip_edge_pixels counts pixels to
        # remove from the [short_wavelength_end, long_wavelength_end]; map those
        # to the pixel axis using the measured dispersion direction.
        n_short, n_long = int(clip_edge_pixels[0]), int(clip_edge_pixels[1])
        if n_short or n_long:
            ncol = flux.shape[1]
            if n_short + n_long >= ncol:
                raise ValueError(
                    f"clip_edge_pixels {list(clip_edge_pixels)} removes all "
                    f"{ncol} pixels of {chip}_{fiber}"
                )
            if np.nanmedian(np.diff(wave, axis=1)) >= 0:  # pixel 0 = short wavelength
                cols = slice(n_short, ncol - n_long)
            else:
                cols = slice(n_long, ncol - n_short)
            flux = flux[:, cols]
            wave = wave[:, cols]
            var = var[:, cols]

        norder = flux.shape[0]

        # Per-order barycentric redshift — only for astronomical sources (target,
        # sky). Calibration sources (thar) stay in the instrument frame (z = 0).
        if apply_barycorr:
            if np.size(self.l2_obj.data.get("BARYCORR_Z", np.array([]))) == 0:
                raise ValueError(
                    "per-order barycentric redshift (BARYCORR_Z) not populated; "
                    "run BarycentricCorrection first"
                )
            barycorr_z = np.asarray(
                self.l2_obj.data[f"{chip}_BARYCORR_Z"], dtype=np.float64
            )
        else:
            barycorr_z = np.zeros(norder)

        ccf = np.zeros((norder, velocity_grid.size))
        ccf_var = np.zeros((norder, velocity_grid.size))

        # Some orders legitimately yield a zero CCF (no mask lines fall within
        # their wavelength coverage, or the order has no usable flux); those are
        # skipped, not failures. A whole-orderlet zero result, however, is a
        # failure and is caught below rather than silently passed through.
        for o in range(norder):
            if not np.all(np.isfinite(wave[o])) or not np.any(np.isfinite(flux[o])):
                continue
            ccf[o], ccf_var[o] = self._compute_ccf_1d(
                wave[o], flux[o], var[o], line_mask, velocity_grid, barycorr_z[o]
            )

        # Fail loudly: an identically-zero CCF across every order means the
        # cross-correlation produced nothing usable for this orderlet (no mask
        # line overlapped any order with finite flux). Common causes: an
        # unpopulated/garbage wavelength solution or flux, or a line mask whose
        # wavelengths (vacuum, Angstrom) do not overlap the data.
        if not np.any(ccf):
            raise RuntimeError(
                f"CCF for {chip}_{fiber} is identically zero across all {norder} "
                "orders; cross-correlation produced no usable signal. Check that "
                f"{chip}_{fiber}_WAVE and {chip}_{fiber}_FLUX are populated and "
                "finite, and that the line mask overlaps the data wavelengths."
            )

        self._ccf[f"{chip}_{fiber}"] = ccf
        self._ccf_var[f"{chip}_{fiber}"] = ccf_var

        return {"velocity": velocity_grid, "ccf": ccf}

    def compute_order_by_order_rvs(
        self, chip, fiber, window=None, fit_nsigma=3.0, min_npts=9
    ):
        """
        Per-order radial velocities for one chip/fiber, from the cached CCF.

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
            {'rv', 'rv_err'} — length-norder_chip ndarrays [km/s], one per order.

        Raises
        ------
        RuntimeError
            If compute_ccfs has not been called for this chip/fiber; the
            CCF must be computed (and cached) first.
        """
        if window is None:
            window = self.rv_window

        chip = chip.upper()
        fiber = fiber.upper()
        ext = f"{chip}_{fiber}"

        if ext not in self._ccf:
            raise RuntimeError(
                f"CCF for {ext} not available; call compute_ccfs({chip!r}, {fiber!r}) "
                "before compute_order_by_order_rvs"
            )
        ccf = self._ccf[ext]
        ccf_var = self._ccf_var[ext]

        velocity_grid = self._velocity_grid[ext]  # the grid compute_ccfs used
        wave = np.asarray(self.l2_obj.data[f"{chip}_{fiber}_WAVE"], dtype=np.float64)
        rv = np.full(ccf.shape[0], np.nan)
        rv_err = np.full(ccf.shape[0], np.nan)
        for o in range(ccf.shape[0]):
            if not np.any(ccf[o]):
                continue
            rv[o], rv_err[o] = self._compute_rv_1d(
                velocity_grid, ccf[o], ccf_var[o], wave[o], window, fit_nsigma, min_npts
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
        Weighted-combined RVs from the cached CCFs, collapsing orders (and
        optionally fibers and CCDs) into one RV per group.

        Orders are always weighted-collapsed (the per-order path is
        compute_order_by_order_rvs). Two flags control the rest:

          combine_fibers : sum the three science fibers' CCFs before collapsing.
                           True requires `fibers` == the three SCI fibers; False
                           requires a single fiber.
          combine_ccds   : combine the per-chip RVs across CCDs at the RV level —
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
            If compute_ccfs has not been called for any requested chip/fiber.
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
        per_chip = {}
        for chip in chips:
            grid, ccf_w, ccf_s, ccf_s_var, wave = self._combine_ccfs(chip, fibers)
            if not np.any(ccf_w):
                per_chip[chip] = (np.nan, np.nan)
                continue
            rv = self._compute_rv_1d(
                grid, ccf_w, ccf_s_var, wave, window, fit_nsigma, min_npts
            )[0]
            rv_err = self._compute_rv_1d(
                grid, ccf_s, ccf_s_var, wave, window, fit_nsigma, min_npts
            )[1]
            per_chip[chip] = (rv, rv_err)

        if not combine_ccds:
            return per_chip

        # Cross-CCD combine at the RV level: value = per-chip RVs weighted by each
        # chip's total order weight; error = inverse-variance combination. A
        # single chip returns that chip's (rv, rv_err) exactly.
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
    # Public entry point
    # ------------------------------------------------------------------

    def perform(
        self,
        chips=None,
        fibers=None,
        *,
        ccf_mask_width=None,
        ccf_step_size=None,
        ccf_window=None,
        rv_window=None,
        fit_nsigma=3.0,
        min_npts=9,
        clip_edge_pixels=(500, 500),
    ):
        """
        Compute per-order CCFs and radial velocities and package them in a KPF4.

        For each requested orderlet (fiber), the per-order CCFs of both chips are
        written to the orderlet's CCF cube ({fiber}_CCF, green+red concatenated)
        and the per-order RVs to the orderlet's RV table ({fiber}_RV).

        Parameters
        ----------
        chips : list of str, optional
            Chip identifiers, i.e. 'GREEN' or 'RED'. Defaults to the configured
            chips.
        fibers : list of str, optional
            Fiber identifiers, e.g. ['SCI1', 'SCI2']. Defaults to all configured
            fibers (SCI, CAL, and SKY).
        ccf_mask_width : float, optional
            Per-line mask top-hat width [km/s]. Overrides the configured value.
        ccf_step_size : float, optional
            CCF velocity step size [km/s]. Overrides the configured value.
        ccf_window : list of float, optional
            CCF velocity grid range [km/s] as [min, max] about the grid center.
            Overrides the configured value.
        rv_window : list of float, optional
            [min, max] km/s window about the dip for the first-pass fit.
            Overrides the configured value.
        fit_nsigma : float, optional
            Half-width of the second-pass fit window in units of the first-pass
            sigma. Not a configurable parameter; set in code (default 3.0).
        min_npts : int, optional
            Minimum number of grid points to use in each fit window. Not a
            configurable parameter; set in code (default 9).
        clip_edge_pixels : tuple of int, optional
            Pixels to drop from the (short_wavelength_end, long_wavelength_end)
            of each order before correlating. Defaults to (500, 500).

        Returns
        -------
        l4_obj : KPF4
            L4 with a CCF cube and per-order RV table per illuminated orderlet.
            Each CCF extension carries VELSTART/VELSTEP/VELNSTEP/CCFMASK; each RV
            extension carries RVMETHOD/SKYRMVD/TELLRMVD. The per-fiber legacy RV
            keywords CCD<n>RV<sfx>/CCD<n>ERV<sfx> are registered KPF-pipeline
            keywords routed to their RV# table by orderlet. PRIMARY carries the
            final science RV: the EPRV keywords RVMETHOD/RV/RVERR/BERV/BJDTDB, plus
            the KPF SCI-combined per-CCD CCD<n>RV/CCD<n>ERV -- KPF-registered yet
            homed on PRIMARY as a deliberate exception to the EPRV-only-PRIMARY
            rule, since these are the pipeline's final RV measurements.
            Unilluminated ('none') fibers are skipped (empty extensions).
        """
        if chips is None:
            chips = self.chips
        if fibers is None:
            fibers = self.fibers
        if ccf_mask_width is None:
            ccf_mask_width = self.ccf_mask_width
        if ccf_step_size is None:
            ccf_step_size = self.ccf_step_size
        if ccf_window is None:
            ccf_window = self.ccf_window
        if rv_window is None:
            rv_window = self.rv_window

        chips = [c.upper() for c in chips]
        fibers = [f.upper() for f in fibers]

        norder_green = self.norder["GREEN"]
        norder = norder_green + self.norder["RED"]

        l4_obj = self.l2_obj.to_kpf4()

        # Per-order barycentric metadata, shared by every orderlet's RV table.
        bjd_tdb = np.asarray(self.l2_obj.data["BJD_TDB"], dtype=np.float64)
        berv = np.asarray(self.l2_obj.data["BARYCORR_KMS"], dtype=np.float64)

        self._info = {}
        for fiber in fibers:
            rv = np.full(norder, np.nan)
            rv_err = np.full(norder, np.nan)
            weight = np.full(norder, np.nan)

            # Dispatch the mask/barycorr/grid-center from the fiber's illumination
            # source; unilluminated or not-yet-implemented sources are skipped
            # with NaN RVs and no CCF/RV extension.
            source = self._resolve_illumination_source(chips[0], fiber)
            if source["mask_name"] is None:
                print(
                    f"  {fiber}: illumination source {source['object']!r}; "
                    "skipping (no CCF/RV)"
                )
                self._info[fiber] = {
                    "rv": rv,
                    "rv_err": rv_err,
                    "source": source["object"],
                }
                continue
            mask_name = source["mask_name"]

            for chip in chips:
                ccf = self.compute_ccfs(
                    chip,
                    fiber,
                    ccf_mask_width,
                    ccf_step_size,
                    ccf_window,
                    clip_edge_pixels=clip_edge_pixels,
                )["ccf"]
                l4_obj.set_data(f"{chip}_{fiber}_CCF", ccf)
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
                weight[rows] = self._get_order_weights(chip, fiber)

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
            ccd_rv = {chip: per_ccd[chip][0] for chip in chips}
            ccd_rv_err = {chip: per_ccd[chip][1] for chip in chips}
            self._info[fiber] = {
                "rv": rv,
                "rv_err": rv_err,
                "source": source["object"],
                "ccd_rv": ccd_rv,
                "ccd_rv_err": ccd_rv_err,
            }

            # Per-orderlet RV table, one row per spectral order (green orders
            # then red). ORDER_ID is the KPF chip/fiber/order name, 1-based per
            # chip (a KPF-custom extra column). ECHELLE_ORDER is the physical
            # grating order from detector.toml echelle_orders, listed blue->red,
            # so order index 0 -- the bluest -- carries the highest echelle
            # number. WEIGHT is the per-order CCF-combination weight
            # (ccf_order_weights.csv, by mask), persisted for downstream
            # weighting (e.g. DiagL4 BJD/BCV statistics).
            order_id = np.array(
                [
                    f"{chip}_{fiber}_{order}"
                    for chip in ("GREEN", "RED")
                    for order in range(1, self.norder[chip] + 1)
                ]
            )
            echelle_order = np.concatenate(
                [
                    np.linspace(
                        self.echelle_orders[chip][0],
                        self.echelle_orders[chip][1],
                        self.norder[chip],
                    )
                    .round()
                    .astype(np.int64)
                    for chip in ("GREEN", "RED")
                ]
            )
            wave = np.asarray(self.l2_obj.data[f"{fiber}_WAVE"], dtype=np.float64)
            l4_obj.set_data(
                f"{fiber}_RV",
                pd.DataFrame(
                    {
                        "ORDER_INDEX": np.arange(norder, dtype=np.int64),
                        "ORDER_ID": order_id,
                        "ECHELLE_ORDER": echelle_order,
                        "BJD_TDB": bjd_tdb,
                        "BERV": berv,
                        "WAVE_START": np.nanmin(wave, axis=1),
                        "WAVE_END": np.nanmax(wave, axis=1),
                        "RV": rv,
                        "RV_ERR": rv_err,
                        "WEIGHT": weight,
                    }
                ),
            )

            # Per-orderlet CCF/RV extension headers (EPRV L4 standard). The
            # velocity grid is per-fiber (center varies) but shared across chips.
            grid = self._velocity_grid[f"{chips[-1]}_{fiber}"]
            ccf_hdr = fits.Header()
            # CCF cube is (norder, n_velocity_step): FITS axis 1 = velocity,
            # axis 2 = order (EXTNAME is stamped by rvdata on serialization).
            ccf_hdr["CTYPE1"] = ("Velocity", "Name of axis 1")
            ccf_hdr["CTYPE2"] = ("Order-N", "Name of axis 2")
            ccf_hdr["VELSTART"] = (float(grid[0]), "[km/s] Velocity grid start")
            ccf_hdr["VELSTEP"] = (float(ccf_step_size), "[km/s] Velocity grid step")
            ccf_hdr["VELNSTEP"] = (int(grid.size), "Number of velocity grid steps")
            ccf_hdr["CCFMASK"] = (mask_name, "Mask used to generate CCF")
            l4_obj.set_header(f"{fiber}_CCF", ccf_hdr)

            rv_hdr = fits.Header()
            # RV table has one row per spectral order: FITS axis 2 = order.
            rv_hdr["CTYPE1"] = ("Columns", "Name of axis 1")
            rv_hdr["CTYPE2"] = ("Order-N", "Name of axis 2")
            rv_hdr["RVMETHOD"] = ("CCF", "RV derivation method")
            rv_hdr["SKYRMVD"] = (False, "Sky model removed?")
            rv_hdr["TELLRMVD"] = (False, "Telluric model removed?")
            l4_obj.set_header(f"{fiber}_RV", rv_hdr)

            # Per-orderlet RV/error are KPF legacy carryovers, not EPRV RV1
            # keywords. They are registered KPF-pipeline keywords
            # (config/L4-headers.csv) under the legacy per-fiber suffix scheme
            # CCD<n>RV<sfx>/CCD<n>ERV<sfx> (n: GREEN=1, RED=2; sfx: 1/2/3=SCI1/2/3,
            # C=CAL, S=SKY), routed by set_keyword to their RV# table header (e.g.
            # CCD1RV1 -> RV2). The bare CCD<n>RV/CCD<n>ERV names stay reserved for
            # the SCI-combined RV (on PRIMARY). A non-finite value (failed fit) is
            # written as None so it becomes a FITS UNDEFINED card, not a NaN.
            sfx = {"SCI1": "1", "SCI2": "2", "SCI3": "3", "CAL": "C", "SKY": "S"}[fiber]
            for chip in ccd_rv:
                n = 1 if chip == "GREEN" else 2
                v, e = ccd_rv[chip], ccd_rv_err[chip]
                l4_obj.set_keyword(
                    f"CCD{n}RV{sfx}", float(v) if np.isfinite(v) else None
                )
                l4_obj.set_keyword(
                    f"CCD{n}ERV{sfx}", float(e) if np.isfinite(e) else None
                )

        # PRIMARY (EPRV L4): always record the RV method.
        l4_obj.set_keyword("RVMETHOD", "CCF")

        # Final science RV (PRIMARY RV/RVERR + the KPF per-CCD CCD<n>RV/CCD<n>ERV).
        # Sum the science orderlets' CCFs per chip and fit (bare CCD<n>RV), then
        # combine the two CCDs at the RV level, weighted by their summed
        # order-weights (PRIMARY RV); the error is the inverse-variance
        # combination (PRIMARY RVERR). RVs are already in the barycentric frame
        # (compute_ccfs folds barycorr into the mask shift), so the reported
        # BERV/BJDTDB are descriptive, not applied.
        sci_req = [f for f in fibers if f in ("SCI1", "SCI2", "SCI3")]
        sci = [f for f in sci_req if self._info[f]["source"] not in (None, "none")]
        if not sci_req:
            # A calibration-only run (no science orderlet requested): the combined
            # science RV is not applicable, so PRIMARY RV/RVERR/BERV/BJDTDB stay
            # UNDEFINED.
            print(
                "  combined RV: no science orderlet requested; "
                "PRIMARY RV left UNDEFINED"
            )
            l4_obj.receipt_add_entry("radial_velocity", "", "PASS")
            return l4_obj
        if not sci:
            raise ValueError(
                f"science orderlet(s) {sci_req} requested but none illuminated; "
                "cannot form a combined RV"
            )
        rep = sci[0]
        if len(chips) == 1:
            print(
                f"  combined RV: only chip {chips[0]} present; "
                "the combined RV uses it alone"
            )

        # Per CCD: bare SCI-combined RV/error (the science orderlets' CCFs summed
        # within each chip, then orders collapsed). The three science fibers are
        # always illuminated together, so they are always summed.
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
            v, e = per_ccd[chip]
            if not np.isfinite(v):
                print(
                    f"  combined RV: {chip} science fit non-finite; "
                    "excluded from the combined RV"
                )
            n = 1 if chip == "GREEN" else 2
            l4_obj.set_keyword(f"CCD{n}RV", float(v) if np.isfinite(v) else None)
            l4_obj.set_keyword(f"CCD{n}ERV", float(e) if np.isfinite(e) else None)

        # Cross-chip weighted RV and inverse-variance error (the PRIMARY RV/RVERR,
        # written below): the per-CCD science RVs combined at the RV level by their
        # summed order weights.
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
            print("  combined RV: no finite per-CCD science RV; PRIMARY RV UNDEFINED")

        # PRIMARY BERV/BJDTDB: chip-weighted mean of the per-CCD photon-weighted
        # bary summaries (CCD<n>BKMS/CCD<n>BJD from BarycentricCorrection), using
        # the same summed-order-weight chip weights as the combined-RV combine so
        # the two stay consistent. Not a CCF operation, so computed here.
        bnum = jnum = bden = 0.0
        for chip in chips:
            n = 1 if chip == "GREEN" else 2
            w = float(np.nansum(self._get_order_weights(chip, rep)))
            bkms = self.l2_obj.headers["BARYCORR_KMS"].get(f"CCD{n}BKMS")
            bjd = self.l2_obj.headers["BJD_TDB"].get(f"CCD{n}BJD")
            if (
                w > 0
                and bkms is not None
                and np.isfinite(bkms)
                and bjd is not None
                and np.isfinite(bjd)
            ):
                bnum += bkms * w
                jnum += bjd * w
                bden += w
        berv_p = bnum / bden if bden > 0 else np.nan
        bjd_p = jnum / bden if bden > 0 else np.nan

        # PRIMARY (EPRV L4): the recommended combined RV. SYSVEL is left UNDEFINED
        # (absolute barycentric RVs, nothing removed); per-fiber velocity grids
        # live on the CCF extensions. set_keyword routes each to PRIMARY with the
        # registry-owned comment; a non-finite value writes an UNDEFINED card.
        l4_obj.set_keyword("RV", float(ccfrv) if np.isfinite(ccfrv) else None)
        l4_obj.set_keyword("RVERR", float(ccferv) if np.isfinite(ccferv) else None)
        l4_obj.set_keyword("BERV", float(berv_p) if np.isfinite(berv_p) else None)
        l4_obj.set_keyword("BJDTDB", float(bjd_p) if np.isfinite(bjd_p) else None)

        l4_obj.receipt_add_entry("radial_velocity", "", "PASS")
        return l4_obj

    def info(self):
        """Print a summary of the module configuration and RV results."""
        print("RadialVelocity")
        obs_id = self.l2_obj.headers.get("RECEIPT", {}).get("ORIGID", "unknown")
        print(f"  obs_id:         {obs_id}")
        print(f"  ccf_mask_width: {self.ccf_mask_width} km/s")
        print(f"  ccf_step_size:  {self.ccf_step_size} km/s")
        print(f"  ccf_window:     {self.ccf_window} km/s")
        print(f"  rv_window:      {self.rv_window} km/s")

        if self._info is None:
            print("  perform() has not been called")
            return

        # CCF velocity grid: per-fiber center, shared step/span.
        print(
            f"\n  CCF velocity grid: {self.ccf_window[0]:+.1f} to "
            f"{self.ccf_window[1]:+.1f} km/s "
            f"about each fiber's center, step {self.ccf_step_size} km/s"
        )

        # Per-CCD, per-orderlet summary. SOURCE is the illumination source;
        # CCD_RV/CCD_ERV are the combined per-CCD RV and its error; RV_RMS is the
        # order-to-order mad_std (a diagnostic of per-order spread), in m/s.
        fiber_order = [
            f for f in ("SCI1", "SCI2", "SCI3", "SKY", "CAL") if f in self._info
        ]
        fiber_order += [f for f in self._info if f not in fiber_order]

        print(
            f"\n  {'CHIP':<8s}{'FIBER':<8s}{'SOURCE':<10s}{'NVALID':>8s}"
            f"{'CCD_RV [km/s]':>16s}{'CCD_ERV [m/s]':>16s}{'RV_RMS [m/s]':>16s}"
        )
        print("  " + "-" * 82)
        norder_green = self.norder["GREEN"]
        norder = norder_green + self.norder["RED"]
        for chip, rows in (
            ("GREEN", slice(0, norder_green)),
            ("RED", slice(norder_green, norder)),
        ):
            for fiber in fiber_order:
                res = self._info[fiber]
                rv = res["rv"][rows]
                nvalid = int(np.sum(np.isfinite(rv)))
                if nvalid == 0:
                    continue
                ccd_rv = res.get("ccd_rv", {}).get(chip, np.nan)
                ccd_erv = res.get("ccd_rv_err", {}).get(chip, np.nan)
                rv_rms = mad_std(rv, ignore_nan=True) * 1e3 if nvalid >= 2 else np.nan
                print(
                    f"  {chip:<8s}{fiber:<8s}{res.get('source', ''):<10s}{nvalid:>8d}"
                    f"{ccd_rv:>+16.5f}{ccd_erv * 1e3:>16.3f}{rv_rms:>16.3f}"
                )
