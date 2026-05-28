"""
KPF Barycentric Correction module.

Computes per-order barycentric corrections from the EXPMETER_SCI flux-weighted
midpoint times. Populates the rvdata-standard L2 extensions and applies the
per-order redshift to all WAVE arrays in place.

Outputs (rvdata-standard ImageHDUs, shape (NORDER,)):
  - BJD_TDB       photon-weighted midpoint in BJD_TDB per spectral order
  - BARYCORR_KMS  barycentric velocity per spectral order [km/s]
  - BARYCORR_Z    barycentric redshift per spectral order

Per-CCD scalar summaries (mean across that chip's orders) written to PRIMARY:
  - CCD1BJD       GREEN photon-weighted mid-time (BJD_TDB)
  - CCD1BKMS      GREEN mean barycentric velocity [km/s]
  - CCD1BZ        GREEN mean barycentric redshift
  - CCD2BJD       RED   photon-weighted mid-time (BJD_TDB)
  - CCD2BKMS      RED   mean barycentric velocity [km/s]
  - CCD2BZ        RED   mean barycentric redshift

CCD1BJD/CCD2BJD match the legacy keyword names and semantics; the *BVEL/*BZ
companions follow the same CCD{n} naming pattern.

Notes
-----
Follows the barycentric correction approach described in:
- Wright & Eastman (2014) — velocity calculation (barycorrpy)
- Butler et al. (1996)    — flux-weighted midpoint time
"""
import re

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.stats import mad_std
from astropy.time import Time
import astropy.units as u
from astroquery.gaia import Gaia
from barycorrpy import get_BC_vel, utc_tdb
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter
from scipy.special import erfcinv

from kpfpipe import DEFAULTS, DETECTOR
from kpfpipe.utils.astro import compute_doppler_shift
from kpfpipe.utils.config import ConfigHandler

DEFAULTS.update({
    'fix_bad_exposures': True,
})

NORDER_GREEN = DETECTOR['norder']['GREEN']
NORDER_RED   = DETECTOR['norder']['RED']
NORDER       = NORDER_GREEN + NORDER_RED


class BarycentricCorrection:
    """
    Compute and apply per-order barycentric correction to KPF2 wavelength arrays.

    Reads EXPMETER_SCI to derive the flux-weighted midpoint time per expmeter
    wavelength channel, interpolates onto each spectral order's central
    wavelength (using SCI2_WAVE as the reference), queries Gaia DR3 for the
    target's astrometric solution, and runs barycorrpy per order to populate
    the rvdata-standard BJD_TDB, BARYCORR_KMS, and BARYCORR_Z extensions.

    Parameters
    ----------
    kpf2_obj : KPF2
        Extracted L2 frame. Must have EXPMETER_SCI populated and SCI2_WAVE
        populated by WavelengthCalibration. INSTRUMENT_HEADER (the preserved
        L1 PRIMARY) must contain GAIAID and DATE-BEG/DATE-END.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: chips, fibers, fix_bad_exposures.
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
    # Private helpers — exposure-meter handling
    # ------------------------------------------------------------------

    def _get_timestamps(self):
        """
        Read start and end timestamps from EXPMETER_SCI.

        Returns
        -------
        t_beg : Time
            Array of exposure start times.
        t_mid : Time
            Array of exposure midpoint times (unweighted).
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
        w : ndarray, shape (nwave,)
            Wavelength of each expmeter channel (label units; typically
            Angstroms for KPF).
        f : ndarray, shape (ntime, nwave)
            Flux in units of photons per (Å|nm), normalized by dispersion.

        Notes
        -----
        Columns with numeric names are treated as wavelength channels;
        non-numeric columns (e.g., timestamps) are skipped.
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

        return w, f

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
        rate_gap = 0.5 * (rate[1:] + rate[:-1])   # (ngap, nwave)

        t_gap = Time(t_end[:-1].jd + dt_gap / 2, format='jd', scale='utc')
        f_gap = rate_gap * dt_gap[:, None]

        return t_gap, f_gap

    @staticmethod
    def _extrapolate(t0, t_beg, t_end, f):
        """
        Estimate flux for a gap before the first or after the last expmeter reading.

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

    # ------------------------------------------------------------------
    # Private helpers — barycorr handoffs
    # ------------------------------------------------------------------

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
    def _compute_barycorr(skycoord, obs_times, location):
        """
        Compute barycentric velocity and BJD_TDB for an array of obs_times.

        Parameters
        ----------
        skycoord : SkyCoord
            Target coordinates with proper motion and distance.
        obs_times : Time
            Observation times (array).
        location : EarthLocation
            Observatory location.

        Returns
        -------
        bc_vel_mps : ndarray, shape (n,)
            Barycentric velocity correction per time [m/s].
        bjd_tdb : ndarray, shape (n,)
            Barycentric Julian date in TDB per time.
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

        JDUTC = np.atleast_1d(obs_times.utc.jd)

        bc_vel, *_ = get_BC_vel(
            JDUTC=JDUTC,
            ra=ra, dec=dec,
            lat=lat, longi=lon, alt=alt,
            pmra=pmra, pmdec=pmdec,
            px=px, epoch=epoch,
        )
        bjd_tdb, *_ = utc_tdb.JDUTC_to_BJDTDB(
            JDUTC=JDUTC,
            ra=ra, dec=dec,
            lat=lat, longi=lon, alt=alt,
            pmra=pmra, pmdec=pmdec,
            px=px, epoch=epoch,
        )
        return np.asarray(bc_vel), np.asarray(bjd_tdb)

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def flux_weighted_midpoint(self, interpolate=True, extrapolate=True,
                                fix_bad_exposures=None):
        """
        Compute the flux-weighted midpoint observation time per wavelength channel.

        Parameters
        ----------
        interpolate : bool, optional
            If True, estimate flux during gaps between expmeter readings.
        extrapolate : bool, optional
            If True, estimate flux during gaps before the first or after the
            last expmeter reading, using DATE-BEG / DATE-END from the header.
        fix_bad_exposures : bool, optional
            If True, detect and replace outlier expmeter readings. Defaults
            to the config value.

        Returns
        -------
        w : ndarray, shape (nwave,)
            Wavelength of each expmeter channel (label units; typically Å).
        t_fwm : Time, shape (nwave,)
            Flux-weighted midpoint time (JD-UTC) for each wavelength channel.
        """
        if fix_bad_exposures is None:
            fix_bad_exposures = self.fix_bad_exposures

        t_beg, t_mid, t_end = self._get_timestamps()
        w, f = self._get_normalized_flux()

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
            hdr = self.kpf2_obj.headers['INSTRUMENT_HEADER']
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
        return w, t_fwm

    def map_to_orders(self, w_em, t_fwm):
        """
        Interpolate per-channel PHOTON_JD onto each spectral order's wavelength.

        Uses SCI2_WAVE (TRACE3_WAVE) as the reference wavelength solution.
        Per-order central wavelength is the median across columns. Orders
        outside the expmeter range get the nearest endpoint value (np.interp
        default behavior).

        Parameters
        ----------
        w_em : ndarray, shape (nwave,)
            Expmeter channel wavelengths (label units).
        t_fwm : Time, shape (nwave,)
            Per-channel flux-weighted midpoint times.

        Returns
        -------
        t_per_order : Time, shape (NORDER,)
            Per-order flux-weighted midpoint time (JD-UTC).
        """
        wave = self.kpf2_obj.data['SCI2_WAVE']
        if wave is None or np.size(wave) == 0:
            raise KeyError(
                "SCI2_WAVE missing or empty; run WavelengthCalibration first"
            )
        order_centers = np.median(np.asarray(wave), axis=1)  # (NORDER,)

        sort_idx = np.argsort(w_em)
        jd_per_order = np.interp(order_centers, w_em[sort_idx], t_fwm.jd[sort_idx])
        return Time(jd_per_order, format='jd', scale='utc')

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, chips=None, fibers=None, interpolate=True, extrapolate=True,
                fix_bad_exposures=None):
        """
        Compute per-order barycentric correction and apply it to wavelength arrays.

        Parameters
        ----------
        chips : list of str, optional
            Chip identifiers, e.g. ['GREEN', 'RED']. Defaults to self.chips.
        fibers : list of str, optional
            Fiber identifiers, e.g. ['SCI1', 'SCI2', ...]. Defaults to self.fibers.
        interpolate, extrapolate, fix_bad_exposures : bool, optional
            Forwarded to flux_weighted_midpoint(). See that method for semantics.

        Returns
        -------
        kpf2_obj : KPF2
            Input KPF2 with BJD_TDB, BARYCORR_KMS, BARYCORR_Z populated;
            WAVE arrays scaled in-place by per-order redshift; PRIMARY
            BJDTDB/BARYKMS/BARYZ summaries written; 'barycentric_correction'
            receipt entry added.
        """
        if chips is None:
            chips = self.chips
        if fibers is None:
            fibers = self.fibers

        # Per-channel flux-weighted midpoint times → per-order via interpolation
        w_em, t_fwm = self.flux_weighted_midpoint(
            interpolate=interpolate, extrapolate=extrapolate,
            fix_bad_exposures=fix_bad_exposures,
        )
        t_per_order = self.map_to_orders(w_em, t_fwm)

        # Astrometric solution + per-order barycorr
        # GAIAID is preserved from L1 PRIMARY by KPF1.to_kpf2() into L2 INSTRUMENT_HEADER.
        gaia_id_raw = self.kpf2_obj.headers['INSTRUMENT_HEADER']['GAIAID']
        gaia_id = re.split(r'\s+', str(gaia_id_raw).strip())[-1]
        location = EarthLocation.of_site('Keck Observatory')
        skycoord = self._query_gaia(gaia_id)

        bc_vel_mps, bjd_tdb = self._compute_barycorr(skycoord, t_per_order, location)
        bary_kms = bc_vel_mps / 1000.0
        bary_z = np.array([
            float(compute_doppler_shift(v * u.m / u.s)) for v in bc_vel_mps
        ])

        # Write rvdata standard extensions (per-order arrays)
        self.kpf2_obj.set_data('BJD_TDB',      np.asarray(bjd_tdb,  dtype=np.float64))
        self.kpf2_obj.set_data('BARYCORR_KMS', np.asarray(bary_kms, dtype=np.float64))
        self.kpf2_obj.set_data('BARYCORR_Z',   np.asarray(bary_z,   dtype=np.float64))

        # Apply per-order redshift to all WAVE arrays (in-place, per-order multiply)
        for fiber in fibers:
            for chip in chips:
                wave_ext = f'{chip}_{fiber}_WAVE'
                if wave_ext not in self.kpf2_obj.data:
                    continue
                arr = self.kpf2_obj.data[wave_ext]
                if arr is None or np.size(arr) == 0:
                    continue
                z = bary_z[:NORDER_GREEN] if chip.upper() == 'GREEN' else bary_z[NORDER_GREEN:]
                self.kpf2_obj.data[wave_ext] = (arr * z[:, None]).astype(arr.dtype)

        # Per-CCD scalar summaries on PRIMARY.
        # CCD1 = GREEN (orders [:NORDER_GREEN]), CCD2 = RED (orders [NORDER_GREEN:]).
        primary = self.kpf2_obj.headers['PRIMARY']
        green = slice(0, NORDER_GREEN)
        red   = slice(NORDER_GREEN, NORDER)
        primary['CCD1BJD']  = (float(np.mean(bjd_tdb[green])),  'GREEN photon-weighted mid-time (BJD_TDB)')
        primary['CCD1BKMS'] = (float(np.mean(bary_kms[green])), 'GREEN mean barycentric velocity [km/s]')
        primary['CCD1BZ']   = (float(np.mean(bary_z[green])),   'GREEN mean barycentric redshift')
        primary['CCD2BJD']  = (float(np.mean(bjd_tdb[red])),    'RED photon-weighted mid-time (BJD_TDB)')
        primary['CCD2BKMS'] = (float(np.mean(bary_kms[red])),   'RED mean barycentric velocity [km/s]')
        primary['CCD2BZ']   = (float(np.mean(bary_z[red])),     'RED mean barycentric redshift')

        self.kpf2_obj.receipt_add_entry('barycentric_correction', 'PASS')

        self._results = {
            'bjd_tdb':  np.asarray(bjd_tdb),
            'bary_kms': np.asarray(bary_kms),
            'bary_z':   np.asarray(bary_z),
        }
        return self.kpf2_obj

    def info(self):
        """Print a summary of the module configuration and correction results."""
        print("BarycentricCorrection")
        print(f"  obs_id:            {self.kpf2_obj.obs_id}")
        print(f"  fix_bad_exposures: {self.fix_bad_exposures}")

        if self._results is None:
            print("  perform() has not been called")
            return

        bjd = self._results['bjd_tdb']
        kms = self._results['bary_kms']
        z   = self._results['bary_z']
        green = slice(0, NORDER_GREEN)
        red   = slice(NORDER_GREEN, NORDER)

        print(f"\n  {'':<8s}{'BJD_TDB':>18s}{'BARYCORR_KMS':>18s}{'BARYCORR_Z':>18s}")
        print("  " + "-" * 62)
        print(f"  {'GREEN':<8s}{np.mean(bjd[green]):>18.6f}{np.mean(kms[green]):>+18.4f}{np.mean(z[green]):>18.10f}")
        print(f"  {'RED':<8s}{np.mean(bjd[red]):>18.6f}{np.mean(kms[red]):>+18.4f}{np.mean(z[red]):>18.10f}")
        print(f"\n  per-order spread:   BJD {np.ptp(bjd) * 86400:.3f} sec,"
              f" BARY {np.ptp(kms) * 1000:.3f} m/s")
