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
import numpy as np
import pandas as pd

from kpfpipe import REPO_ROOT, DEFAULTS
from kpfpipe.utils.astro import air_to_vac
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import optimize_lsq
from kpfpipe.utils.validation import strictly_increasing

SPEED_OF_LIGHT_KMS = np.float64(c.to('km/s').value)  # km/s

DEFAULTS.update({
    'mask_width_kms': 0.5,
    'ccf_step_size': 0.25,
    'ccf_step_range': [-402, 402],
    'rv_window_kms': 50.0,
    'rv_window_pts': 11,
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

    def _build_ccf_line_mask(self, mask_width_kms=None):
        """
        Build (and cache) the CCF line mask for the TARGTEFF-selected stellar
        mask: vacuum line centers, weights, and per-line top-hat edges
        (center ± center * mask_width_kms / c).

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
        
        if mask_width_kms is None:
            mask_width_kms = self.mask_width_kms

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
        half_width = centers * (mask_width_kms / SPEED_OF_LIGHT_KMS)
        self._ccf_line_mask = {
            'center': centers,
            'weight': weights,
            'start': centers - half_width,
            'end': centers + half_width,
        }
        return self._ccf_line_mask


    def _build_ccf_velocity_grid(self, ccf_step_size=None, ccf_step_range=None):
        """
        Build (and cache) the evenly-spaced CCF velocity grid, centered on the
        target's systemic RV (TARGRADV).

        Returns
        -------
        ndarray
            Velocity steps [km/s], spanning ccf_step_range * ccf_step_size about
            TARGRADV.

        Raises
        ------
        ValueError
            If TARGRADV is not available in INSTRUMENT_HEADER.
        """
        if self._ccf_velocity_grid is not None:
            return self._ccf_velocity_grid
        
        if ccf_step_size is None:
            ccf_step_size = self.ccf_step_size
        if ccf_step_range is None:
            ccf_step_range = self.ccf_step_range

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

        lo, hi = ccf_step_range
        self._ccf_velocity_grid = np.arange(lo, hi + 1) * ccf_step_size + star_rv
        return self._ccf_velocity_grid


    @staticmethod
    def _compute_ccf(wave, flux, ccf_line_mask, ccf_velocity_grid, barycorr_z):
        """
        Cross-correlate one order's spectrum against the mask over the velocity
        grid, folding in the order's barycentric redshift z.

        Parameters
        ----------
        wave : ndarray
            1D wavelength solution for the order [Å].
        flux : ndarray
            1D extracted flux for the order.
        ccf_line_mask : dict
            Line mask (keys 'start', 'end', 'weight') from _build_ccf_line_mask.
        ccf_velocity_grid : ndarray
            CCF velocity steps [km/s].
        barycorr_z : float
            Barycentric redshift for the order.

        Returns
        -------
        ndarray
            CCF value at each velocity step (all zeros if the order is unusable
            or no mask lines fall fully within it).
        """
        wave = np.asarray(wave, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        if wave[0] > wave[-1]:          # reversed order -> flip to ascending
            wave, flux = wave[::-1], flux[::-1]

        ccf = np.zeros(ccf_velocity_grid.size)
        n_pix = wave.size
        if n_pix < 3 or not strictly_increasing(wave):
            return ccf

        # Wavelength bin edges (length n+1) and widths at the pixel midpoints.
        edges = np.empty(n_pix + 1)
        edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
        edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
        edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
        widths = np.diff(edges)
        shift = (1.0 + ccf_velocity_grid / SPEED_OF_LIGHT_KMS) / (1.0 + barycorr_z)  # mask shift per step

        # Keep only mask lines that stay fully inside the order across the whole
        # scan, so the same lines contribute at every step (flat CCF baseline).
        smin, smax = shift.min(), shift.max()
        keep = (ccf_line_mask['start'] * smin >= wave[0]) & (ccf_line_mask['end'] * smax <= wave[-1])
        if not np.any(keep):
            return ccf
        l_start, l_end = ccf_line_mask['start'][keep], ccf_line_mask['end'][keep]
        l_weight = ccf_line_mask['weight'][keep]

        for v in range(ccf_velocity_grid.size):
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
    def _compute_rv(vel, ccf, wave, rv_window_kms, rv_window_pts):
        """
        Two-pass Gaussian fit to a CCF dip, with a photon-limited error.

        The first pass locates the CCF minimum and fits a +/-rv_window_kms/2
        velocity window around it. The second pass refits the rv_window_pts grid
        points centered on the first-pass RV (snapped to the grid); those same
        points set the error estimate (Bouchy et al. 2001).

        Parameters
        ----------
        vel : ndarray
            CCF velocity steps [km/s].
        ccf : ndarray
            CCF value at each velocity step.
        wave : ndarray
            1D wavelength solution for the order [Å]; sets the per-pixel velocity
            scale used in the error estimate.
        rv_window_kms : float
            Full width [km/s] of the first-pass velocity window.
        rv_window_pts : int
            Number of grid points in the second-pass fit/error window; must be
            odd so the window is symmetric about the peak.

        Returns
        -------
        rv : float
            Fitted radial velocity [km/s], or NaN if the fit fails.
        rv_err : float
            Photon-limited RV uncertainty [km/s], or NaN if unavailable.

        Raises
        ------
        ValueError
            If rv_window_pts is even.
        """
        if rv_window_pts % 2 == 0:
            raise ValueError("rv_window_pts must be odd for a symmetric fitting window")

        finite = np.isfinite(ccf)
        if finite.sum() < 4 or np.ptp(ccf[finite]) == 0:
            return np.nan, np.nan

        # Locate the absorption dip; fit the inverted CCF so it presents as a peak
        # (matching optimize_lsq's peak-oriented guess); theta = [b, a, mu, sigma].
        peak = vel[np.nanargmin(ccf)]

        # First pass: +/-ccf_window_kms/2 velocity window around the dip.
        win = finite & (np.abs(vel - peak) <= rv_window_kms / 2)
        if win.sum() < 4:
            return np.nan, np.nan
        try:
            rv = optimize_lsq(vel[win], -ccf[win], 'gaussian')[0][2]
        except (RuntimeError, ValueError):
            return np.nan, np.nan
        if not np.isfinite(rv):
            return np.nan, np.nan

        # Second pass: ccf_window_pts points centered on the first-pass RV snapped
        # to the nearest grid point; these points also set the error estimate.
        center = int(np.argmin(np.abs(vel - rv)))
        half = rv_window_pts // 2
        lo, hi = center - half, center + half + 1
        if lo < 0 or hi > vel.size:
            return rv, np.nan          # symmetric window runs off the grid
        vel_fit, ccf_fit = vel[lo:hi], ccf[lo:hi]
        try:
            mu = optimize_lsq(vel_fit, -ccf_fit, 'gaussian')[0][2]
            if vel_fit.min() <= mu <= vel_fit.max():
                rv = mu
        except (RuntimeError, ValueError):
            pass

        # Photon-limited RV uncertainty from the weighted CCF slope over the
        # same ccf_window_pts points.
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

    def compute_ccf(self, chip, fiber, mask_width_kms=None, ccf_step_size=None,
                    ccf_step_range=None):
        """
        Cross-correlate every order of one chip/fiber against the line mask.

        Parameters
        ----------
        chip : str
            Chip identifier, i.e. 'GREEN' or 'RED'.
        fiber : str
            Fiber identifier, e.g. 'SCI1'.
        mask_width_kms : float, optional
            Per-line mask top-hat width [km/s]. Defaults to the configured value.
        ccf_step_size : float, optional
            CCF velocity step size [km/s]. Defaults to the configured value.
        ccf_step_range : list of int, optional
            CCF velocity grid range, in steps. Defaults to the configured value.

        Returns
        -------
        ndarray
            CCF with shape (norder_chip, n_velocity_step). Also cached under
            f'{chip}_{fiber}' for a subsequent compute_rv call.

        Raises
        ------
        ValueError
            If BARYCORR_Z is not populated on the L2.
        """
        chip = chip.upper()
        fiber = fiber.upper()
        if mask_width_kms is None:
            mask_width_kms = self.mask_width_kms
        if ccf_step_size is None:
            ccf_step_size = self.ccf_step_size
        if ccf_step_range is None:
            ccf_step_range = self.ccf_step_range

        if np.size(self.l2_obj.data.get('BARYCORR_Z', np.array([]))) == 0:
            raise ValueError(
                "per-order barycentric redshift (BARYCORR_Z) not populated; run BarycentricCorrection first"
            )

        ccf_line_mask = self._build_ccf_line_mask(mask_width_kms)
        ccf_velocity_grid = self._build_ccf_velocity_grid(ccf_step_size, ccf_step_range)
        barycorr_z = np.asarray(self.l2_obj.data[f'{chip}_BARYCORR_Z'], dtype=np.float64)

        flux = np.asarray(self.l2_obj.data[f'{chip}_{fiber}_FLUX'], dtype=np.float64)
        wave = np.asarray(self.l2_obj.data[f'{chip}_{fiber}_WAVE'], dtype=np.float64)
        ccf = np.zeros((flux.shape[0], ccf_velocity_grid.size))
        
        for o in range(flux.shape[0]):
            if not np.all(np.isfinite(wave[o])) or not np.any(np.isfinite(flux[o])):
                continue
            ccf[o] = self._compute_ccf(wave[o], flux[o], ccf_line_mask, ccf_velocity_grid, barycorr_z[o])

        self._ccf[f'{chip}_{fiber}'] = ccf
        return ccf

    def compute_rv(self, chip, fiber, rv_window_kms=None, rv_window_pts=None):
        """
        Compute per-order radial velocities for one chip/fiber.

        Parameters
        ----------
        chip : str
            Chip identifier, i.e. 'GREEN' or 'RED'.
        fiber : str
            Fiber identifier, e.g. 'SCI1'.
        rv_window_kms : float, optional
            Full width [km/s] of the first-pass fit window. Defaults to the
            configured value.
        rv_window_pts : int, optional
            Number of grid points in the second-pass fit/error window; must be
            odd. Defaults to the configured value.

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
        if rv_window_kms is None:
            rv_window_kms = self.rv_window_kms
        if rv_window_pts is None:
            rv_window_pts = self.rv_window_pts

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
                velocity_grid, ccf[o], wave[o], rv_window_kms, rv_window_pts)

        return {'rv': rv, 'rv_err': rv_err}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, chips=None, fibers=None, mask_width_kms=None, ccf_step_size=None, ccf_step_range=None,
                rv_window_kms=None, rv_window_pts=None):
        """
        Compute per-order CCFs and radial velocities for each requested chip and
        fiber.

        Parameters
        ----------
        chips : list of str, optional
            Chip identifiers, i.e. 'GREEN' or 'RED'. Defaults to the configured
            chips.
        fibers : list of str, optional
            Fiber identifiers, e.g. ['SCI1', 'SCI2']. Defaults to the configured
            science fibers.
        mask_width_kms : float, optional
            Per-line mask top-hat width [km/s]. Overrides the configured value.
        ccf_step_size : float, optional
            CCF velocity step size [km/s]. Overrides the configured value.
        ccf_step_range : list of int, optional
            CCF velocity grid range, in steps. Overrides the configured value.
        rv_window_kms : float, optional
            Full width [km/s] of the first-pass fit window. Overrides the
            configured value.
        rv_window_pts : int, optional
            Number of grid points in the second-pass fit/error window; must be
            odd. Overrides the configured value.

        Returns
        -------
        ccf_arrays : dict
            {f'{chip}_{fiber}': ccf}, ccf shape (norder_chip, n_velocity_step).
        rv_arrays : dict
            {f'{chip}_{fiber}': {'rv', 'rv_err'}}, each length norder_chip [km/s].
        """
        if chips is None:
            chips = self.chips
        if fibers is None:
            fibers = [f for f in self.fibers if f.startswith('SCI')]
        if mask_width_kms is None:
            mask_width_kms = self.mask_width_kms
        if ccf_step_size is None:
            ccf_step_size = self.ccf_step_size
        if ccf_step_range is None:
            ccf_step_range = self.ccf_step_range
        if rv_window_kms is None:
            rv_window_kms = self.rv_window_kms
        if rv_window_pts is None:
            rv_window_pts = self.rv_window_pts

        ccf_arrays = {}
        rv_arrays = {}

        for chip in chips:
            for fiber in fibers:
                ext = f'{chip}_{fiber}'
                ccf_arrays[ext] = self.compute_ccf(chip, fiber, mask_width_kms, ccf_step_size, ccf_step_range)
                rv_arrays[ext] = self.compute_rv(chip, fiber, rv_window_kms, rv_window_pts)

        return ccf_arrays, rv_arrays
