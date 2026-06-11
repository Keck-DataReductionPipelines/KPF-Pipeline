"""
KPF Radial Velocity module.

Computes radial velocities from an extracted, wavelength-calibrated,
barycentric-corrected KPF L2 by cross-correlating each order's 1D spectrum
against a line mask, then collapsing the per-order CCFs into radial velocities
per order, orderlet, and CCD (weighted where appropriate). Produces a KPF4 (L4)
holding the CCFs and RVs.

ESPRESSO line masks give in-air wavelengths
Calibration line masks give vacuum wavelengths
"""
from astropy.constants import c
from astropy.stats import mad_std
import numpy as np
import pandas as pd

from kpfpipe import REPO_ROOT, DEFAULTS, DETECTOR
from kpfpipe.utils.astro import air_to_vac
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import optimize_lsq
from kpfpipe.utils.validation import strictly_increasing

SPEED_OF_LIGHT_KMS = np.float64(c.to('km/s').value)  # km/s

NORDER_GREEN = DETECTOR['norder']['GREEN']
NORDER_RED   = DETECTOR['norder']['RED']
NORDER       = NORDER_GREEN + NORDER_RED

DEFAULTS.update({
    'ccf_mask_width': 0.5,
    'ccf_step_size': 0.25,
    'ccf_window': [-100.0, 100.0],
    'rv_window': [-25.0, 25.0]
})


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
        Module configuration. Recognized keys: chips, fibers.
    """

    def __init__(self, l2_obj, config=None):
        self.l2_obj = l2_obj

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS", "KPFPIPE", "MODULE_RADIAL_VELOCITY"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self._ccf_line_mask = None     # cached by _build_ccf_line_mask()
        self._ccf_velocity_grid = None  # cached by _build_ccf_velocity_grid()
        self._ccf = {}              # cached by compute_ccf(), keyed by f'{chip}_{fiber}'
        self._results = None        # populated by perform()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_ccf_line_mask(self, width=None):
        """
        Build (and cache) the CCF line mask for the TARGTEFF-selected stellar
        mask: vacuum line centers, weights, and per-line top-hat edges
        (center ± center * width / c).

        Returns
        -------
        dict
            Mask with keys 'center', 'weight', 'start', 'end', each a 1D ndarray
            of length n_line; wavelengths are vacuum [Å].

        Raises
        ------
        ValueError
            If TARGTEFF is not available in INSTRUMENT_HEADER.
        """
        if self._ccf_line_mask is not None:
            return self._ccf_line_mask
        
        if width is None:
            width = self.ccf_mask_width

        inst = self.l2_obj.headers.get('INSTRUMENT_HEADER', {})
        try:
            teff = float(inst.get('TARGTEFF'))
        except (TypeError, ValueError):
            teff = None
        if teff is None or not np.isfinite(teff) or teff <= 0:
            raise ValueError(
                "target effective temperature (TARGTEFF) not available in "
                "INSTRUMENT_HEADER; cannot select a stellar line mask"
            )

        line_map = pd.read_csv(f'{REPO_ROOT}/reference/line_masks/line_mask_lookup.csv')
        row = line_map[(line_map['TEFF_MIN'] <= teff) & (teff < line_map['TEFF_MAX'])]
        mask_name = row['DEFAULT_MASK'].iloc[0]
        mask_path = f'{REPO_ROOT}/reference/line_masks/stellar_masks/{mask_name}.txt'

        centers, weights = np.loadtxt(mask_path, unpack=True)
        centers = air_to_vac(centers)
        half_width = centers * (width / SPEED_OF_LIGHT_KMS)
        self._ccf_line_mask = {
            'center': centers,
            'weight': weights,
            'start': centers - half_width,
            'end': centers + half_width,
        }
        return self._ccf_line_mask


    def _build_ccf_velocity_grid(self, step_size=None, window=None):
        """
        Build (and cache) the evenly-spaced CCF velocity grid, centered on the
        target's systemic RV (TARGRADV).

        The [min, max] `window` is converted to an integer number of `step_size`
        steps about TARGRADV, so the grid step is exact.

        Returns
        -------
        ndarray
            Velocity steps [km/s], spanning `window` about TARGRADV in
            `step_size` increments.

        Raises
        ------
        ValueError
            If TARGRADV is not available in INSTRUMENT_HEADER.
        """
        if self._ccf_velocity_grid is not None:
            return self._ccf_velocity_grid

        if step_size is None:
            step_size = self.ccf_step_size
        if window is None:
            window = self.ccf_window

        inst = self.l2_obj.headers.get('INSTRUMENT_HEADER', {})
        try:
            star_rv = float(inst.get('TARGRADV'))
        except (TypeError, ValueError):
            star_rv = None
        if star_rv is None or not np.isfinite(star_rv):
            raise ValueError(
                "target radial velocity (TARGRADV) not available in "
                "INSTRUMENT_HEADER; cannot center the CCF velocity grid"
            )

        # Convert the [min, max] km/s window to integer step counts.
        lo_kms, hi_kms = window
        lo = int(round(lo_kms / step_size))
        hi = int(round(hi_kms / step_size))
        self._ccf_velocity_grid = np.arange(lo, hi + 1) * step_size + star_rv
        return self._ccf_velocity_grid


    @staticmethod
    def _compute_ccf(wave, flux, line_mask, velocity_grid, barycorr_z):
        """
        Cross-correlate one order's spectrum against the mask over the velocity
        grid, folding in the order's barycentric redshift z.

        Parameters
        ----------
        wave : ndarray
            1D wavelength solution for the order [Å].
        flux : ndarray
            1D extracted flux for the order.
        line_mask : dict
            Line mask (keys 'start', 'end', 'weight') from _build_ccf_line_mask.
        velocity_grid : ndarray
            CCF velocity steps [km/s].
        barycorr_z : float
            Barycentric redshift for the order.

        Returns
        -------
        ndarray
            CCF value at each velocity step (all zeros if the order is unusable
            or no mask lines fall fully within it).

        Raises
        ------
        ValueError
            If the WAVE array is descending; an ascending (blue->red) solution
            is expected, so a reversed order signals an upstream orientation
            error rather than something to silently correct.
        """
        wave = np.asarray(wave, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        if wave[0] > wave[-1]:
            raise ValueError(
                f"WAVE array is descending (wave[0]={wave[0]:.4f} > "
                f"wave[-1]={wave[-1]:.4f}); expected ascending blue->red "
                f"orientation. This signals an upstream orientation error."
            )

        ccf = np.zeros(velocity_grid.size)
        n_pix = wave.size
        if n_pix < 3 or not strictly_increasing(wave):
            return ccf

        # Wavelength bin edges (length n+1) and widths at the pixel midpoints.
        edges = np.empty(n_pix + 1)
        edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
        edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
        edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
        widths = np.diff(edges)
        shift = (1.0 + velocity_grid / SPEED_OF_LIGHT_KMS) / (1.0 + barycorr_z)  # mask shift per step

        # Keep only mask lines that stay fully inside the order across the whole
        # scan, so the same lines contribute at every step (flat CCF baseline).
        smin, smax = shift.min(), shift.max()
        keep = (line_mask['start'] * smin >= wave[0]) & (line_mask['end'] * smax <= wave[-1])
        if not np.any(keep):
            return ccf
        l_start, l_end = line_mask['start'][keep], line_mask['end'][keep]
        l_weight = line_mask['weight'][keep]

        for v in range(velocity_grid.size):
            a = l_start * shift[v]
            b = l_end * shift[v]
            ia = np.clip(np.searchsorted(edges, a, side='right') - 1, 0, n_pix - 1)
            ib = np.clip(np.searchsorted(edges, b, side='right') - 1, 0, n_pix - 1)

            # Fractional overlap of each (narrow) line with the pixels it covers.
            frac = np.zeros(n_pix)
            for d in range(int((ib - ia).max()) + 1):
                n = ia + d
                sel = n <= ib
                nn = n[sel]
                overlap = np.minimum(edges[nn + 1], b[sel]) - np.maximum(edges[nn], a[sel])
                np.clip(overlap, 0.0, None, out=overlap)
                np.add.at(frac, nn, l_weight[sel] * overlap / widths[nn])

            ccf[v] = np.nansum(flux * frac)

        return ccf

    @staticmethod
    def _compute_rv(vel, ccf, wave, window, min_npts=9):
        """
        Two-pass Gaussian fit to a CCF dip, with a photon-limited error.

        The first pass fits the `window` ([min, max] km/s) about the CCF
        minimum, yielding a mean and sigma. The second pass refits a window of
        +/-3 sigma about that mean (symmetric about it); those points also set
        the error estimate (Bouchy et al. 2001). Both windows use at least
        min_npts grid points.

        Parameters
        ----------
        vel : ndarray
            CCF velocity steps [km/s].
        ccf : ndarray
            CCF value at each velocity step.
        wave : ndarray
            1D wavelength solution for the order [Å]; sets the per-pixel velocity
            scale used in the error estimate.
        window : list of float
            [min, max] km/s velocity window about the dip for the first pass.
        min_npts : int
            Minimum number of grid points to use in each fit window.

        Returns
        -------
        rv : float
            Fitted radial velocity [km/s], or NaN if the fit fails.
        rv_err : float
            Photon-limited RV uncertainty [km/s], or NaN if unavailable.
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
            theta = optimize_lsq(vel[win], -ccf[win], 'gaussian')[0]
        except (RuntimeError, ValueError):
            return np.nan, np.nan
        mu1, sigma1 = theta[2], theta[3]
        if not np.isfinite(mu1) or not np.isfinite(sigma1):
            return np.nan, np.nan

        # Second pass: +/-3 sigma about the first-pass mean, symmetric, with at
        # least min_npts points; these points also set the error estimate.
        dv = np.mean(np.diff(vel))
        half_pts = max(int(np.floor(3.0 * sigma1 / dv)), int(np.ceil((min_npts - 1) / 2)))
        center = int(np.argmin(np.abs(vel - mu1)))
        lo, hi = center - half_pts, center + half_pts + 1
        if lo < 0 or hi > vel.size:
            return mu1, np.nan          # symmetric window runs off the grid
        vel_fit, ccf_fit = vel[lo:hi], ccf[lo:hi]
        rv = mu1
        try:
            mu2 = optimize_lsq(vel_fit, -ccf_fit, 'gaussian')[0][2]
            if vel_fit.min() <= mu2 <= vel_fit.max():
                rv = mu2
        except (RuntimeError, ValueError):
            pass

        # Photon-limited RV uncertainty from the weighted CCF slope over the
        # second-pass window.
        if not np.any(ccf_fit) or np.any(ccf_fit < 0):
            return rv, np.nan

        vel_step = np.mean(np.diff(vel))
        vel_span_per_pixel = SPEED_OF_LIGHT_KMS * np.median(np.abs(np.diff(wave))) / np.median(wave)
        n_pix_per_vel_step = vel_step / vel_span_per_pixel

        weighted_slope = np.gradient(ccf_fit, vel_fit) ** 2 / ccf_fit
        qccf = (np.sum(weighted_slope) ** 0.5 / np.sum(ccf_fit) ** 0.5) * n_pix_per_vel_step ** 0.5
        rv_err = 1.0 / (qccf * np.sum(ccf_fit) ** 0.5)

        return rv, rv_err

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def compute_ccf(self, chip, fiber, width=None, step_size=None,
                    window=None, clip_edge_pixels=[500, 500]):
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
            CCF velocity grid range [km/s] as [min, max] about the systemic RV.
            Defaults to the configured value.
        clip_edge_pixels : list of int, optional
            Number of pixels to drop from the [short_wavelength_end,
            long_wavelength_end] of each order before correlating, removing the
            blaze-faint, low-S/N order edges. Defaults to [500, 500].

        Returns
        -------
        dict
            {'velocity', 'ccf'}: the CCF velocity grid [km/s] and the CCF with
            shape (norder_chip, n_velocity_step). The CCF is also cached under
            f'{chip}_{fiber}' for a subsequent compute_rv call.

        Raises
        ------
        ValueError
            If BARYCORR_Z is not populated on the L2.
        """
        chip = chip.upper()
        fiber = fiber.upper()
        if width is None:
            width = self.ccf_mask_width
        if step_size is None:
            step_size = self.ccf_step_size
        if window is None:
            window = self.ccf_window

        if np.size(self.l2_obj.data.get('BARYCORR_Z', np.array([]))) == 0:
            raise ValueError(
                "per-order barycentric redshift (BARYCORR_Z) not populated; run BarycentricCorrection first"
            )

        line_mask = self._build_ccf_line_mask(width)
        velocity_grid = self._build_ccf_velocity_grid(step_size, window)
        barycorr_z = np.asarray(self.l2_obj.data[f'{chip}_BARYCORR_Z'], dtype=np.float64)

        flux = np.asarray(self.l2_obj.data[f'{chip}_{fiber}_FLUX'], dtype=np.float64)
        wave = np.asarray(self.l2_obj.data[f'{chip}_{fiber}_WAVE'], dtype=np.float64)

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
            if np.nanmedian(np.diff(wave, axis=1)) >= 0:   # pixel 0 = short wavelength
                cols = slice(n_short, ncol - n_long)
            else:
                cols = slice(n_long, ncol - n_short)
            flux = flux[:, cols]
            wave = wave[:, cols]

        norder = flux.shape[0]
        ccf = np.zeros((norder, velocity_grid.size))

        # Some orders legitimately yield a zero CCF (no mask lines fall within
        # their wavelength coverage, or the order has no usable flux); those are
        # skipped, not failures. A whole-orderlet zero result, however, is a
        # failure and is caught below rather than silently passed through.
        for o in range(norder):
            if not np.all(np.isfinite(wave[o])) or not np.any(np.isfinite(flux[o])):
                continue
            ccf[o] = self._compute_ccf(wave[o], flux[o], line_mask, velocity_grid, barycorr_z[o])

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

        self._ccf[f'{chip}_{fiber}'] = ccf

        return {'velocity': velocity_grid, 'ccf': ccf}

    def compute_rv(self, chip, fiber, window=None, min_npts=9):
        """
        Compute per-order radial velocities for one chip/fiber.

        Parameters
        ----------
        chip : str
            Chip identifier, i.e. 'GREEN' or 'RED'.
        fiber : str
            Fiber identifier, e.g. 'SCI1'.
        window : list of float, optional
            [min, max] km/s window about the dip for the first-pass fit.
            Defaults to the configured value.
        min_npts : int, optional
            Minimum number of grid points to use in each fit window. Not a
            configurable parameter; set in code (default 9).

        Returns
        -------
        dict
            {'rv', 'rv_err'}, each a length-norder_chip ndarray [km/s].

        Raises
        ------
        RuntimeError
            If compute_ccf has not been called for this chip/fiber; the CCF must
            be computed (and cached) first.
        """
        if window is None:
            window = self.rv_window

        chip = chip.upper()
        fiber = fiber.upper()
        ext = f'{chip}_{fiber}'

        if ext not in self._ccf:
            raise RuntimeError(
                f"CCF for {ext} not available; call compute_ccf({chip!r}, {fiber!r}) "
                "before compute_rv"
            )
        ccf = self._ccf[ext]

        velocity_grid = self._build_ccf_velocity_grid()
        wave = np.asarray(self.l2_obj.data[f'{chip}_{fiber}_WAVE'], dtype=np.float64)
        rv = np.full(ccf.shape[0], np.nan)
        rv_err = np.full(ccf.shape[0], np.nan)
        for o in range(ccf.shape[0]):
            if not np.any(ccf[o]):
                continue
            rv[o], rv_err[o] = self._compute_rv(
                velocity_grid, ccf[o], wave[o], window, min_npts)

        return {'rv': rv, 'rv_err': rv_err}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, chips=None, fibers=None, ccf_mask_width=None, ccf_step_size=None, ccf_window=None,
                rv_window=None, min_npts=9, clip_edge_pixels=[500, 500]):
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
            CCF velocity grid range [km/s] as [min, max] about the systemic RV.
            Overrides the configured value.
        rv_window : list of float, optional
            [min, max] km/s window about the dip for the first-pass fit.
            Overrides the configured value.
        min_npts : int, optional
            Minimum number of grid points to use in each fit window. Not a
            configurable parameter; set in code (default 9).
        clip_edge_pixels : list of int, optional
            Pixels to drop from the [short_wavelength_end, long_wavelength_end]
            of each order before correlating. Defaults to [500, 500].

        Returns
        -------
        l4_obj : KPF4
            L4 data product with one CCF cube and one per-order RV table per
            orderlet (e.g. SCI2_CCF, SCI2_RV), the CCF velocity grid and
            RVMETHOD written to PRIMARY, and a 'radial_velocity' receipt entry.
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

        l4_obj = self.l2_obj.to_kpf4()

        # Per-order barycentric metadata, shared by every orderlet's RV table.
        bjd_tdb = np.asarray(self.l2_obj.data['BJD_TDB'], dtype=np.float64)
        berv = np.asarray(self.l2_obj.data['BARYCORR_KMS'], dtype=np.float64)

        self._results = {}
        for fiber in fibers:
            rv = np.full(NORDER, np.nan)
            rv_err = np.full(NORDER, np.nan)
            for chip in chips:
                ccf = self.compute_ccf(chip, fiber, ccf_mask_width, ccf_step_size, ccf_window,
                                       clip_edge_pixels=clip_edge_pixels)['ccf']
                l4_obj.set_data(f'{chip}_{fiber}_CCF', ccf)
                result = self.compute_rv(chip, fiber, rv_window, min_npts)
                rows = slice(0, NORDER_GREEN) if chip == 'GREEN' else slice(NORDER_GREEN, NORDER)
                rv[rows] = result['rv']
                rv_err[rows] = result['rv_err']
            self._results[fiber] = {'rv': rv, 'rv_err': rv_err}

            # Per-orderlet RV table, one row per spectral order.
            wave = np.asarray(self.l2_obj.data[f'{fiber}_WAVE'], dtype=np.float64)
            l4_obj.set_data(f'{fiber}_RV', pd.DataFrame({
                'ORDER_INDEX': np.arange(NORDER, dtype=np.int64),
                'BJD_TDB':     bjd_tdb,
                'BERV':        berv,
                'WAVE_START':  np.nanmin(wave, axis=1),
                'WAVE_END':    np.nanmax(wave, axis=1),
                'RV':          rv,
                'RV_ERR':      rv_err,
            }))

        # Record the CCF velocity grid (linear, shared by all CCFs) and method.
        primary = l4_obj.headers['PRIMARY']
        primary['RVMETHOD'] = ('CCF', 'RV derivation method')
        primary['CCFVSTRT'] = (float(self._ccf_velocity_grid[0]), '[km/s] CCF velocity grid start')
        primary['CCFVSTEP'] = (float(ccf_step_size), '[km/s] CCF velocity grid step')
        primary['CCFNVEL'] = (int(self._ccf_velocity_grid.size), 'CCF velocity grid length')

        l4_obj.receipt_add_entry('radial_velocity', 'PASS')
        return l4_obj

    def info(self):
        """Print a summary of the module configuration and RV results."""
        print("RadialVelocity")
        obs_id = self.l2_obj.headers.get('PRIMARY', {}).get('ORIGID', 'unknown')
        if isinstance(obs_id, tuple):
            obs_id = obs_id[0]
        print(f"  obs_id:         {obs_id}")
        print(f"  ccf_mask_width: {self.ccf_mask_width} km/s")
        print(f"  ccf_step_size:  {self.ccf_step_size} km/s")
        print(f"  ccf_window:     {self.ccf_window} km/s")
        print(f"  rv_window:      {self.rv_window} km/s")

        if self._results is None:
            print("  perform() has not been called")
            return

        # CCF velocity grid (linear, shared by all orderlets).
        grid = self._ccf_velocity_grid
        print(f"\n  CCF velocity grid: {grid[0]:+.2f} to {grid[-1]:+.2f} km/s, "
              f"{grid.size} steps of {self.ccf_step_size} km/s")

        # Per-CCD, per-orderlet RV summary (median over valid orders). RV_ERR is
        # the median formal photon error; RV_RMS is the order-to-order mad_std,
        # both in m/s so the scatter can be compared to the formal error.
        fiber_order = [f for f in ('SCI1', 'SCI2', 'SCI3', 'SKY', 'CAL')
                       if f in self._results]
        fiber_order += [f for f in self._results if f not in fiber_order]

        print(f"\n  {'CHIP':<8s}{'FIBER':<8s}{'NVALID':>8s}"
              f"{'RV [km/s]':>16s}{'RV_ERR [m/s]':>16s}{'RV_RMS [m/s]':>16s}")
        print("  " + "-" * 72)
        for chip, rows in (('GREEN', slice(0, NORDER_GREEN)),
                           ('RED', slice(NORDER_GREEN, NORDER))):
            for fiber in fiber_order:
                rv, rv_err = self._results[fiber]['rv'][rows], self._results[fiber]['rv_err'][rows]
                nvalid = int(np.sum(np.isfinite(rv)))
                if nvalid == 0:
                    continue
                rv_rms = mad_std(rv, ignore_nan=True) * 1e3 if nvalid >= 2 else np.nan
                print(f"  {chip:<8s}{fiber:<8s}{nvalid:>8d}"
                      f"{np.nanmedian(rv):>+16.5f}{np.nanmedian(rv_err) * 1e3:>16.3f}"
                      f"{rv_rms:>16.3f}")
