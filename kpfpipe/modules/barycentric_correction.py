"""
KPF Barycentric Correction module.
"""
import re

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.stats import mad_std
from astropy.time import Time
import astropy.units as u
from astroquery.gaia import Gaia
from barycorrpy import get_BC_vel
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter
from scipy.special import erfcinv

from kpfpipe import DEFAULTS
from kpfpipe.utils.astro import compute_doppler_shift
from kpfpipe.utils.config import ConfigHandler


class BarycentricCorrection:
    """
    Compute and apply barycentric correction to KPF2 wavelength arrays.

    Reads EXPMETER_SCI from the input KPF2 object to compute a
    flux-weighted midpoint observation time, queries Gaia DR3 for
    the target's astrometric solution, and applies a Doppler wavelength
    shift to all fiber wavelength arrays in-place.

    Notes
    -----
    Follows the barycentric correction approach described in:
    - Wright & Eastman (2014) for the velocity calculation (barycorrpy)
    - Flux-weighted midpoint time following Butler et al. (1996)
    """

    def __init__(self, kpf2_obj, config=None):
        self.kpf2_obj = kpf2_obj

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(
                ["DATA_DIRS", "KPFPIPE", "MODULE_BARYCENTRIC_CORRECTION"]
            )
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self._results = None  # populated by perform()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_timestamps(self):
        """
        Read start and end timestamps from EXPMETER_SCI.

        Returns
        -------
        t_beg : Time
            Array of exposure start times.
        t_mid : Time
            Array of exposure midpoint times.
        t_end : Time
            Array of exposure end times.

        Notes
        -----
        Prefers corrected timestamps (Date-Beg-Corr / Date-End-Corr) when
        available, falling back to uncorrected values.
        """
        expmeter = self.kpf2_obj.data['EXPMETER_SCI']

        try:
            t_beg = Time(np.array(expmeter['Date-Beg-Corr']).astype(str), format='isot', scale='utc')
            t_end = Time(np.array(expmeter['Date-End-Corr']).astype(str), format='isot', scale='utc')
        except KeyError:
            t_beg = Time(np.array(expmeter['Date-Beg']).astype(str), format='isot', scale='utc')
            t_end = Time(np.array(expmeter['Date-End']).astype(str), format='isot', scale='utc')

        if not self._strictly_increasing(t_beg):
            raise ValueError("EXPMETER_SCI Date-Beg timestamps are not strictly increasing")
        if not self._strictly_increasing(t_end):
            raise ValueError("EXPMETER_SCI Date-End timestamps are not strictly increasing")

        t_mid = Time((t_beg.jd + t_end.jd) / 2, format='jd', scale='utc')
        return t_beg, t_mid, t_end


    def _get_normalized_flux(self):
        """
        Read EXPMETER_SCI flux and normalize by gain and wavelength dispersion.

        Returns
        -------
        f : ndarray, shape (ntime, nwave)
            Flux in units of photons per Angstrom, normalized by dispersion.

        Notes
        -----
        Columns with numeric names are treated as wavelength channels.
        Non-numeric columns (e.g., timestamps) are skipped.
        """
        expmeter = self.kpf2_obj.data['EXPMETER_SCI']

        wave_cols = []
        for col in expmeter.colnames:
            try:
                float(col)
                wave_cols.append(col)
            except (ValueError, TypeError):
                pass

        w = np.array([float(col) for col in wave_cols])
        f = np.column_stack([np.array(expmeter[col], dtype=float) for col in wave_cols])

        dispersion = np.abs(np.gradient(w))
        f = f * 1.48424 / dispersion  # 1.48424 e-/ADU: exposure meter detector gain

        return f


    @staticmethod
    def _strictly_increasing(t):
        """Return True if Time array is strictly increasing."""
        return bool(np.all(t[:-1].jd < t[1:].jd))


    @staticmethod
    def _fix_bad_exposures(f, kernel_size=5):
        """
        Detect and interpolate outlier pixels in the exposure meter array.

        Parameters
        ----------
        f : ndarray, shape (ntime, nwave)
            Exposure meter flux array.
        kernel_size : int
            Kernel size for median and Gaussian smoothing filters.

        Returns
        -------
        f_fixed : ndarray, shape (ntime, nwave)
            Flux array with outliers replaced by interpolated values.

        Notes
        -----
        Outlier threshold is set adaptively using the expected maximum
        deviation for the array size (Chauvenet-like criterion).
        """
        f_smooth = gaussian_filter(median_filter(f, size=kernel_size), sigma=kernel_size)

        eta = 3 * np.sqrt(2) * erfcinv(1 / np.min(np.shape(f)))
        bad = np.abs(f - f_smooth) / mad_std(f - f_smooth) > eta

        if np.sum(bad) == 0:
            return f.copy()

        ny, nx = f.shape
        y, x = np.indices((ny, nx))

        points = np.column_stack((x[~bad], y[~bad]))
        values = f[~bad]
        points_bad = np.column_stack((x[bad], y[bad]))

        f_fixed = f.copy()
        f_fixed[bad] = griddata(points, values, points_bad, method='linear')

        nan_mask = np.isnan(f_fixed)
        if np.any(nan_mask):
            f_fixed[nan_mask] = griddata(
                points, values,
                np.column_stack((x[nan_mask], y[nan_mask])),
                method='nearest'
            )

        return f_fixed


    @staticmethod
    def _interpolate(t_beg, t_end, f):
        """
        Estimate flux during gaps between consecutive exposure meter readings.

        Parameters
        ----------
        t_beg : Time, shape (ntime,)
            Exposure start times.
        t_end : Time, shape (ntime,)
            Exposure end times.
        f : ndarray, shape (ntime, nwave)
            Flux during each exposure.

        Returns
        -------
        t_gap : Time, shape (ngap,)
            Midpoint time of each inter-exposure gap.
        f_gap : ndarray, shape (ngap, nwave)
            Estimated flux during each gap.

        Notes
        -----
        Gap flux is estimated as the average count rate of adjacent
        exposures multiplied by the gap duration.
        """
        dt_exp = (t_end - t_beg).jd[:, None]     # (ntime, 1)
        dt_gap = (t_beg[1:] - t_end[:-1]).jd     # (ngap,)

        rate = f / dt_exp                         # flux per day
        rate_gap = 0.5 * (rate[1:] + rate[:-1])  # (ngap, nwave)

        t_gap = Time(t_end[:-1].jd + dt_gap / 2, format='jd', scale='utc')
        f_gap = rate_gap * dt_gap[:, None]

        return t_gap, f_gap


    @staticmethod
    def _extrapolate(t0, t_beg, t_end, f):
        """
        Estimate flux for a gap before the first or after the last expmeter reading.

        Parameters
        ----------
        t0 : Time
            True start or end of the exposure (shutter open / close).
        t_beg : Time
            Start of the first or last exposure meter reading.
        t_end : Time
            End of the first or last exposure meter reading.
        f : ndarray, shape (nwave,)
            Flux during the reference exposure meter reading.

        Returns
        -------
        t_ext : Time
            Midpoint of the extrapolated gap.
        f_ext : ndarray, shape (nwave,)
            Estimated flux during the gap.
        """
        dt_exp = (t_end - t_beg).jd
        rate = f / dt_exp

        if t0 < t_beg:
            dt_gap = (t_beg - t0).jd
            t_ext = Time(t0.jd + dt_gap / 2, format='jd', scale='utc')
        elif t0 > t_end:
            dt_gap = (t0 - t_end).jd
            t_ext = Time(t_end.jd + dt_gap / 2, format='jd', scale='utc')
        else:
            raise ValueError("t0 must be before t_beg or after t_end")

        f_ext = rate * dt_gap
        return t_ext, f_ext


    @staticmethod
    def _query_gaia(gaia_id):
        """
        Query Gaia DR3 for the astrometric solution of a target.

        Parameters
        ----------
        gaia_id : int or str
            Gaia DR3 source ID (numeric).

        Returns
        -------
        skycoord : SkyCoord
            Target coordinates including proper motion and distance.
        """
        query = f"""
        SELECT ra, dec, pmra, pmdec, parallax, ref_epoch
        FROM gaiadr3.gaia_source
        WHERE source_id = {gaia_id}
        """
        job = Gaia.launch_job(query)
        result = job.get_results()[0]

        skycoord = SkyCoord(
            ra=result['ra'] * u.deg,
            dec=result['dec'] * u.deg,
            pm_ra_cosdec=result['pmra'] * u.mas / u.yr,
            pm_dec=result['pmdec'] * u.mas / u.yr,
            distance=(1e3 / result['parallax']) * u.pc,
            obstime=Time(result['ref_epoch'], format='jyear'),
            frame='icrs',
        )
        return skycoord


    @staticmethod
    def _compute_barycorr_rv(skycoord, obs_time, location):
        """
        Compute barycentric radial velocity correction using barycorrpy.

        Parameters
        ----------
        skycoord : SkyCoord
            Target coordinates with proper motion and distance.
        obs_time : Time
            Observation time (scalar).
        location : EarthLocation
            Observatory location.

        Returns
        -------
        delta_rv : Quantity
            Barycentric velocity correction (m/s).
        """
        icrs = skycoord.icrs
        ra  = icrs.ra.to(u.deg).value
        dec = icrs.dec.to(u.deg).value
        pmra  = icrs.pm_ra_cosdec.to(u.mas / u.yr).value if icrs.pm_ra_cosdec is not None else 0.0
        pmdec = icrs.pm_dec.to(u.mas / u.yr).value if icrs.pm_dec is not None else 0.0
        px    = (1 / icrs.distance.to(u.pc).value * 1e3) if icrs.distance is not None else 0.0
        epoch = icrs.obstime.jyear

        lat = location.lat.to(u.deg).value
        lon = location.lon.to(u.deg).value
        alt = location.height.to(u.m).value

        bc_vel, *_ = get_BC_vel(
            JDUTC=obs_time.utc.jd,
            ra=ra, dec=dec,
            lat=lat, longi=lon, alt=alt,
            pmra=pmra, pmdec=pmdec,
            px=px, epoch=epoch,
        )

        return bc_vel[0] * u.m / u.s


    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def flux_weighted_midpoint(self, interpolate=True, extrapolate=True, fix_bad_exposures=True):
        """
        Compute the flux-weighted midpoint observation time from EXPMETER_SCI.

        Parameters
        ----------
        interpolate : bool, optional
            If True, estimate flux during gaps between expmeter readings.
        extrapolate : bool, optional
            If True, estimate flux during gaps before the first or after the
            last expmeter reading, using DATE-BEG / DATE-END from the header.
        fix_bad_exposures : bool, optional
            If True, detect and replace outlier expmeter readings before
            computing the flux-weighted midpoint. Defaults to config value.

        Returns
        -------
        t_fwm : Time, shape (nwave,)
            Flux-weighted midpoint time for each wavelength channel.

        Notes
        -----
        Returns one midpoint time per wavelength channel because the flux
        distribution (and therefore the flux-weighted time) varies across
        the exposure meter bandpass.
        """
        t_beg, t_mid, t_end = self._get_timestamps()
        f = self._get_normalized_flux()

        if np.any(f < 0):
            raise ValueError("negative exposure meter flux values detected")

        if fix_bad_exposures:
            f = self._fix_bad_exposures(f)

        t = t_mid.copy()

        if interpolate:
            t_gap, f_gap = self._interpolate(t_beg, t_end, f)
            t = Time(np.concatenate([t.jd, t_gap.jd]), format='jd', scale='utc')
            f = np.vstack([f, f_gap])

        if extrapolate:
            hdr = self.kpf2_obj.headers['PRIMARY']
            obs_beg = Time(hdr['DATE-BEG'], format='isot', scale='utc')
            obs_end = Time(hdr['DATE-END'], format='isot', scale='utc')

            if obs_beg < t_beg[0]:
                t_ext, f_ext = self._extrapolate(obs_beg, t_beg[0], t_end[0], f[0])
                t = Time(np.concatenate([t.jd, [t_ext.jd]]), format='jd', scale='utc')
                f = np.vstack([f, f_ext])

            if obs_end > t_end[-1]:
                t_ext, f_ext = self._extrapolate(obs_end, t_beg[-1], t_end[-1], f[-1])
                t = Time(np.concatenate([t.jd, [t_ext.jd]]), format='jd', scale='utc')
                f = np.vstack([f, f_ext])

        t_fwm = Time(
            np.sum(t.jd[:, None] * f, axis=0) / np.sum(f, axis=0),
            format='jd', scale='utc'
        )
        return t_fwm


    def barycentric_correction(self, obs_time):
        """
        Compute the barycentric Doppler shift factor for a given observation time.

        Parameters
        ----------
        obs_time : Time
            Observation time (scalar).

        Returns
        -------
        z : float
            Multiplicative Doppler shift factor to apply to wavelength arrays.
        delta_rv : Quantity
            Barycentric velocity correction in m/s.

        Notes
        -----
        Reads GAIAID from the PRIMARY header to identify the target.
        Expects the format 'DR3 <numeric_id>' or plain '<numeric_id>'.
        """
        gaia_id_raw = self.kpf2_obj.headers['PRIMARY']['GAIAID']
        gaia_id = re.split(r'\s+', str(gaia_id_raw).strip())[-1]

        location = EarthLocation.of_site('Keck Observatory')
        skycoord = self._query_gaia(gaia_id)
        delta_rv = self._compute_barycorr_rv(skycoord, obs_time, location)
        z = float(compute_doppler_shift(delta_rv))

        return z, delta_rv

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, chips=None, fibers=None):
        """
        Compute and apply barycentric correction to all wavelength arrays.

        Parameters
        ----------
        chips : list of str, optional
            Chip identifiers, i.e. 'GREEN' or 'RED'.
        fibers : list of str, optional
            Fiber identifiers, e.g. 'SCI2'.

        Returns
        -------
        kpf2_obj : KPF2
            Input KPF2 object with wavelength arrays corrected in-place.

        Notes
        -----
        Flux-weighted midpoint time is computed per wavelength channel;
        the mean across channels is used as the scalar observation time
        passed to barycorrpy.
        """
        if chips is None:
            chips = self.chips
        if fibers is None:
            fibers = self.fibers

        t_fwm = self.flux_weighted_midpoint()
        obs_time = Time(np.mean(t_fwm.jd), format='jd', scale='utc')

        z, delta_rv = self.barycentric_correction(obs_time)

        for chip in chips:
            for fiber in fibers:
                wave_ext = f'{chip}_{fiber}_WAVE'
                if wave_ext in self.kpf2_obj.data:
                    self.kpf2_obj.data[wave_ext] = self.kpf2_obj.data[wave_ext] * z

        self.kpf2_obj.receipt_add_entry('barycentric_correction', 'PASS')

        self._results = {
            'obs_time': obs_time,
            'delta_rv': delta_rv,
            'z': z,
        }

        return self.kpf2_obj


    def info(self):
        """Print a summary of the module configuration and correction results."""
        print("BarycentricCorrection")
        print(f"  fix_bad_exposures: {self.fix_bad_exposures}")

        if self._results is None:
            print("  perform() has not been called")
            return

        delta_rv_ms = round(float(self._results['delta_rv'].to(u.m / u.s).value), 4)
        obs_time_bjd = round(float(self._results['obs_time'].tdb.jd), 6)
        z = round(float(self._results['z']), 10)

        print(f"\n  obs_time (BJD):    {obs_time_bjd}")
        print(f"  delta_rv (m/s):    {delta_rv_ms}")
        print(f"  z:                 {z}")
