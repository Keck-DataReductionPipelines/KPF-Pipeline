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
import numpy as np
import pandas as pd

from kpfpipe import REPO_ROOT, DEFAULTS
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import optimize_lsq

LIGHT_SPEED = 299792.458  # speed of light, km/s

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
        wave_vac = np.asarray(wave_air, dtype=float).copy()
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
        half_width = centers * (self.mask_width_kms / LIGHT_SPEED)
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

    def _get_per_order_barycorr_z(self):
        """
        Per-order barycentric redshift (BARYCORR_Z) from the L2, aligned with
        the FLUX/WAVE trace orders. Raises ValueError if not populated.
        """
        z = self.l2_obj.data.get('BARYCORR_Z')
        if z is None or np.size(z) == 0:
            raise ValueError(
                "per-order barycentric redshift (BARYCORR_Z) not populated on "
                "the L2; run BarycentricCorrection first"
            )
        return np.asarray(z, dtype=float)

    # ------------------------------------------------------------------
    # Private helpers — CCF & RV math
    # ------------------------------------------------------------------

    @staticmethod
    def _pixel_edges(wave):
        """Wavelength bin edges (length n+1) at the pixel midpoints."""
        edges = np.empty(wave.size + 1)
        edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
        edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
        edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
        return edges

    @staticmethod
    def _ccf_rv_error(velocity_grid, ccf, rv, vel_span_pixel, fit_width=50.0):
        """
        Photon-limited RV uncertainty [km/s] from the weighted CCF slope
        (Bouchy et al. 2001), evaluated within +/-fit_width/2 of rv.
        """
        sel = (velocity_grid >= rv - fit_width / 2.0) & (velocity_grid <= rv + fit_width / 2.0)
        c, v = ccf[sel], velocity_grid[sel]
        if not np.any(c) or np.any(c < 0):
            return np.nan

        vel_step = np.mean(np.diff(velocity_grid))
        n_scale_pix = vel_step / vel_span_pixel       # CCD pixels per velocity step
        weighted_slope = np.gradient(c, v) ** 2 / c   # noise variance ~ c (photon)
        qccf = (np.sum(weighted_slope) ** 0.5 / np.sum(c) ** 0.5) * n_scale_pix ** 0.5
        return 1.0 / (qccf * np.sum(c) ** 0.5)

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def compute_ccf(self, wave, flux, mask, velocity_grid, z):
        """
        Cross-correlate one order's 1D spectrum against the mask over the
        velocity grid, folding in the order's barycentric redshift z. Returns
        the CCF (one value per velocity step).
        """
        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        if wave[0] > wave[-1]:          # ensure ascending wavelength
            wave, flux = wave[::-1], flux[::-1]

        ccf = np.zeros(velocity_grid.size)
        n_pix = wave.size
        if n_pix < 3 or not np.all(np.isfinite(wave)):
            return ccf

        edges = self._pixel_edges(wave)
        pix_width = np.diff(edges)
        shift = (1.0 + velocity_grid / LIGHT_SPEED) / (1.0 + z)  # mask shift per step

        # Keep only mask lines that stay fully inside the order across the whole
        # scan, so the same lines contribute at every step (flat CCF baseline).
        smin, smax = shift.min(), shift.max()
        keep = (mask['start'] * smin >= wave[0]) & (mask['end'] * smax <= wave[-1])
        if not np.any(keep):
            return ccf
        l_start, l_end = mask['start'][keep], mask['end'][keep]
        l_weight = mask['weight'][keep]

        for c in range(velocity_grid.size):
            a = l_start * shift[c]
            b = l_end * shift[c]
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
                np.add.at(frac, nn, l_weight[sel] * overlap / pix_width[nn])

            ccf[c] = np.nansum(flux * frac)

        return ccf

    def compute_rv(self, velocity_grid, ccf):
        """
        Two-pass Gaussian fit to a CCF dip; returns the radial velocity [km/s]
        (the fitted line center), or NaN if the fit fails.
        """
        finite = np.isfinite(ccf)
        if finite.sum() < 4 or np.ptp(ccf[finite]) == 0:
            return np.nan

        # Fit the inverted CCF so the absorption dip is a peak (matching the
        # peak-oriented initial guess in optimize_lsq); theta = [b, a, mu, sigma].
        try:
            rv = optimize_lsq(velocity_grid[finite], -ccf[finite], 'gaussian')[0][2]
        except (RuntimeError, ValueError):
            return np.nan

        # Second pass refined within +/-25 km/s of the first fit.
        near = finite & (np.abs(velocity_grid - rv) <= 25.0)
        if near.sum() >= 4:
            try:
                mu = optimize_lsq(velocity_grid[near], -ccf[near], 'gaussian')[0][2]
                if velocity_grid[near].min() <= mu <= velocity_grid[near].max():
                    rv = mu
            except (RuntimeError, ValueError):
                pass
        return rv

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self):
        """
        Compute per-order CCFs and radial velocities for each science orderlet.
        Returns a dict with the velocity grid and, per orderlet, the CCF cube
        (n_order, n_step), per-order RVs [km/s], and per-order RV errors [km/s].
        """
        line_mask = self._build_line_mask()
        velocity_grid = self._build_ccf_velocity_grid()
        barycorr_z = self._get_per_order_barycorr_z()

        results = {'velocity_grid': velocity_grid, 'orderlets': {}}
        for fiber in (f for f in self.fibers if f.startswith('SCI')):
            flux = np.asarray(self.l2_obj.data[f'{fiber}_FLUX'], dtype=float)
            wave = np.asarray(self.l2_obj.data[f'{fiber}_WAVE'], dtype=float)
            n_order = flux.shape[0]

            ccf = np.zeros((n_order, velocity_grid.size))
            rv = np.full(n_order, np.nan)
            rv_err = np.full(n_order, np.nan)
            for o in range(n_order):
                w, f = wave[o], flux[o]
                if not np.all(np.isfinite(w)) or not np.any(np.isfinite(f)):
                    continue
                ccf[o] = self.compute_ccf(w, f, line_mask, velocity_grid, barycorr_z[o])
                if not np.any(ccf[o]):
                    continue
                rv[o] = self.compute_rv(velocity_grid, ccf[o])
                vel_span_pixel = LIGHT_SPEED * np.median(np.abs(np.diff(w))) / np.median(w)
                rv_err[o] = self._ccf_rv_error(velocity_grid, ccf[o], rv[o], vel_span_pixel)

            results['orderlets'][fiber] = {'ccf': ccf, 'rv': rv, 'rv_err': rv_err}

        self._results = results
        return results
