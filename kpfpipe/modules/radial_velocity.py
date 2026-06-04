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
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import optimize_lsq
from kpfpipe.utils.validation import strictly_increasing

SPEED_OF_LIGHT_KMS = np.float64(c.to('km/s').value)  # km/s

DEFAULTS.update({
    'mask_width_kms': 0.5,
    'ccf_step_size': 0.25,
    'ccf_step_range': [-402, 402],
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

        self._results = None  # populated by perform()

    # ------------------------------------------------------------------
    # Private helpers — CCF inputs
    # ------------------------------------------------------------------

    @staticmethod
    def _air_to_vac(wave_air):
        """
        Convert air wavelengths [Å] to vacuum via the Edlén 1953 formula
        (two iterations, only wavelengths > 2000 Å are modified).
        """
        wave_vac = np.asarray(wave_air, dtype=np.float64).copy()
        modify = wave_vac > 2000.0
        if np.any(modify):
            wave_new = wave_vac[modify]
            for _ in range(2):
                sigma2 = (1e4 / wave_vac[modify]) ** 2
                fact = (1.0 + 5.792105e-2 / (238.0185 - sigma2)
                        + 1.67917e-3 / (57.362 - sigma2))
                wave_vac[modify] = wave_new * fact
        return wave_vac

    def _build_line_mask(self):
        """
        Build the CCF line mask for the TARGTEFF-selected stellar mask: vacuum
        line centers, weights, and per-line top-hat edges (center ± center *
        mask_width_kms / c). Raises ValueError if TARGTEFF is unavailable.
        """
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
        centers = self._air_to_vac(centers)
        half_width = centers * (self.mask_width_kms / SPEED_OF_LIGHT_KMS)
        return {
            'center': centers,
            'weight': weights,
            'start': centers - half_width,
            'end': centers + half_width,
        }

    def _build_ccf_velocity_grid(self):
        """
        Evenly-spaced CCF velocity grid [km/s], centered on the target's
        systemic RV (TARGRADV). Raises ValueError if TARGRADV is unavailable.
        """
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

        lo, hi = self.ccf_step_range
        return np.arange(lo, hi + 1) * self.ccf_step_size + star_rv

    # ------------------------------------------------------------------
    # Private helpers — CCF & RV math
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ccf(wave, flux, mask, velocity_grid, z):
        """
        Cross-correlate one order's 1D spectrum against the mask over the
        velocity grid, folding in the order's barycentric redshift z. Returns
        the CCF (one value per velocity step).
        """
        wave = np.asarray(wave, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        if wave[0] > wave[-1]:          # reversed order -> flip to ascending
            wave, flux = wave[::-1], flux[::-1]

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

        shift = (1.0 + velocity_grid / SPEED_OF_LIGHT_KMS) / (1.0 + z)  # mask shift per step

        # Keep only mask lines that stay fully inside the order across the whole
        # scan, so the same lines contribute at every step (flat CCF baseline).
        smin, smax = shift.min(), shift.max()
        keep = (mask['start'] * smin >= wave[0]) & (mask['end'] * smax <= wave[-1])
        if not np.any(keep):
            return ccf
        l_start, l_end = mask['start'][keep], mask['end'][keep]
        l_weight = mask['weight'][keep]

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

    def _fit_order_rv(self, vel, ccf, wave, ccf_window_kms=50.0, ccf_window_pts=9):
        """
        Two-pass Gaussian fit to a CCF dip. Returns the radial velocity [km/s]
        (the fitted line center) and its photon-limited uncertainty [km/s]
        (Bouchy et al. 2001), or (NaN, NaN) if the fit fails.

        The first pass locates the CCF minimum and fits a +/-ccf_window_kms/2
        velocity window around it. The second pass refits the ccf_window_pts grid
        points centered on the first-pass RV (snapped to the grid); those same
        points set the error estimate. ccf_window_pts must be odd so the window
        is symmetric about the peak.
        """
        if ccf_window_pts % 2 == 0:
            raise ValueError("ccf_window_pts must be odd for a symmetric fitting window")

        finite = np.isfinite(ccf)
        if finite.sum() < 4 or np.ptp(ccf[finite]) == 0:
            return np.nan, np.nan

        # Locate the absorption dip; fit the inverted CCF so it presents as a peak
        # (matching optimize_lsq's peak-oriented guess); theta = [b, a, mu, sigma].
        peak = vel[np.nanargmin(ccf)]

        # First pass: +/-ccf_window_kms/2 velocity window around the dip.
        win = finite & (np.abs(vel - peak) <= ccf_window_kms / 2)
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
        half = ccf_window_pts // 2
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

    def compute_ccf(self, chip, fibers=None, **kwargs):
        """
        Cross-correlate every order of one chip against the line mask for the
        requested fibers. Returns {fiber: ccf}, where ccf has shape
        (norder_chip, n_velocity_step).
        """
        if fibers is None:
            fibers = [f for f in self.fibers if f.startswith('SCI')]
        chip = chip.upper()
        fibers = [f.upper() for f in fibers]

        if np.size(self.l2_obj.data.get('BARYCORR_Z', np.array([]))) == 0:
            raise ValueError(
                "per-order barycentric redshift (BARYCORR_Z) not populated on "
                "the L2; run BarycentricCorrection first"
            )

        mask = self._build_line_mask()
        velocity_grid = self._build_ccf_velocity_grid()
        z = np.asarray(self.l2_obj.data[f'{chip}_BARYCORR_Z'], dtype=np.float64)

        ccf = {}
        for fiber in fibers:
            flux = np.asarray(self.l2_obj.data[f'{chip}_{fiber}_FLUX'], dtype=np.float64)
            wave = np.asarray(self.l2_obj.data[f'{chip}_{fiber}_WAVE'], dtype=np.float64)
            cube = np.zeros((flux.shape[0], velocity_grid.size))
            for o in range(flux.shape[0]):
                if not np.all(np.isfinite(wave[o])) or not np.any(np.isfinite(flux[o])):
                    continue
                cube[o] = self._compute_ccf(wave[o], flux[o], mask, velocity_grid, z[o])
            ccf[fiber] = cube
        return ccf

    def compute_rv(self, chip, fibers=None, **kwargs):
        """
        Compute per-order CCFs and radial velocities for one chip and the
        requested fibers. Returns {fiber: {'ccf', 'rv', 'rv_err'}}, with rv and
        rv_err in km/s. Extra keyword args (e.g. ccf_window_kms, ccf_window_pts)
        are forwarded to the per-order fit.
        """
        if fibers is None:
            fibers = [f for f in self.fibers if f.startswith('SCI')]
        chip = chip.upper()
        fibers = [f.upper() for f in fibers]

        velocity_grid = self._build_ccf_velocity_grid()
        ccf = self.compute_ccf(chip, fibers)

        results = {}
        for fiber in fibers:
            cube = ccf[fiber]
            wave = np.asarray(self.l2_obj.data[f'{chip}_{fiber}_WAVE'], dtype=np.float64)
            rv = np.full(cube.shape[0], np.nan)
            rv_err = np.full(cube.shape[0], np.nan)
            for o in range(cube.shape[0]):
                if not np.any(cube[o]):
                    continue
                rv[o], rv_err[o] = self._fit_order_rv(velocity_grid, cube[o], wave[o], **kwargs)
            results[fiber] = {'ccf': cube, 'rv': rv, 'rv_err': rv_err}
        return results

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, chips=None, fibers=None, **kwargs):
        """
        Compute per-order CCFs and radial velocities for each requested chip and
        science fiber. Returns a dict with the velocity grid and, per chip and
        fiber, the CCF cube (norder, n_step), per-order RVs [km/s], and per-order
        RV errors [km/s].
        """
        if chips is None:
            chips = self.chips
        if fibers is None:
            fibers = [f for f in self.fibers if f.startswith('SCI')]

        velocity_grid = self._build_ccf_velocity_grid()

        orderlets = {}
        for chip in chips:
            orderlets[chip.upper()] = self.compute_rv(chip, fibers, **kwargs)

        self._results = {'velocity_grid': velocity_grid, 'orderlets': orderlets}
        return self._results
