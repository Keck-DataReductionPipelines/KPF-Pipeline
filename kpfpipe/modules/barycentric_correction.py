"""
KPF Barycentric Correction module.

Computes per-order barycentric corrections from the EXPMETER_SCI flux-weighted
midpoint times and stores them on the L2. WAVE arrays are not modified.

Outputs (rvdata-standard ImageHDUs, shape (NORDER,)):
  - BJD_TDB       photon-weighted midpoint in BJD_TDB per spectral order
  - BARYCORR_KMS  barycentric velocity per spectral order [km/s]
  - BARYCORR_Z    barycentric redshift per spectral order

Per-CCD scalar summaries (at each chip's flux-weighted photon-midpoint time)
written to PRIMARY as registered KPF-pipeline keywords
(config/L2-headers.csv):
  - CCD1BJD       GREEN photon-weighted mid-time (BJD_TDB)
  - CCD1BKMS      GREEN barycentric velocity [km/s]
  - CCD1BZ        GREEN barycentric redshift
  - CCD2BJD       RED   photon-weighted mid-time (BJD_TDB)
  - CCD2BKMS      RED   barycentric velocity [km/s]
  - CCD2BZ        RED   barycentric redshift

CCD1BJD/CCD2BJD match the legacy keyword names and semantics; the *BKMS/*BZ
companions follow the same CCD{n} naming pattern. INSTRUMENT_HEADER is an
immutable pure pass-through of the raw instrument header and is never written.

Notes
-----
Follows the barycentric correction approach described in:
- Wright & Eastman (2014) — velocity calculation (barycorrpy)
- Butler et al. (1996)    — flux-weighted midpoint time
"""

import re
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.stats import mad_std
from astropy.time import Time
from astroquery.gaia import Gaia
from barycorrpy import get_BC_vel, utc_tdb
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter
from scipy.special import erfcinv

from kpfpipe import DEFAULTS
from kpfpipe.utils.astro import compute_redshift
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.validation import strictly_increasing

_DEFAULTS = {
    **DEFAULTS,
    "use_gaia_astrometry": True,
    "use_wmko_fallback": False,
}


class BarycentricCorrection:
    """
    Compute and store per-order barycentric correction on a KPF2.

    Derives the flux-weighted midpoint time per expmeter channel (EXPMETER_SCI),
    averages it over the channels within each spectral order's wavelength range
    (SCI2_WAVE), resolves the target's astrometry (Gaia DR3, with a WMKO header
    fallback), and calls barycorrpy to populate BJD_TDB / BARYCORR_KMS /
    BARYCORR_Z. WAVE arrays are not modified.

    Parameters
    ----------
    l2_obj : KPF2
        Extracted L2 frame. Must have EXPMETER_SCI populated and SCI2_WAVE
        populated by WavelengthCalibration. INSTRUMENT_HEADER (the preserved
        L1 PRIMARY) must contain GAIAID, and DATE-BEG/DATE-END when extrapolating.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: use_gaia_astrometry,
        use_wmko_fallback.
    """

    # WMKO site coordinates
    KECK_LOCATION = EarthLocation(
        lat=19.8260 * u.deg,
        lon=-155.474719 * u.deg,
        height=4145.0 * u.m,
    )

    def __init__(self, l2_obj, config=None):
        self.l2_obj = l2_obj

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

        for k, v in _DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self._info = None
        self._ccd_bjd = None  # Per-CCD [GREEN, RED] arrays for _set_headers
        self._ccd_kms = None
        self._ccd_z = None
        self._exposure_meter = None  # (toggle_key, w_em, t_em)
        self._skycoord = None  # cached Gaia DR3 SkyCoord
        self._astrometry_source = (
            None  # 'Gaia DR3' | 'WMKO header', set by _get_skycoord
        )

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
        expmeter = self.l2_obj.data["EXPMETER_SCI"]

        try:
            t_beg = Time(
                np.array(expmeter["Date-Beg-Corr"]).astype(str),
                format="isot",
                scale="utc",
            )
            t_end = Time(
                np.array(expmeter["Date-End-Corr"]).astype(str),
                format="isot",
                scale="utc",
            )
        except KeyError:
            t_beg = Time(
                np.array(expmeter["Date-Beg"]).astype(str), format="isot", scale="utc"
            )
            t_end = Time(
                np.array(expmeter["Date-End"]).astype(str), format="isot", scale="utc"
            )

        if not strictly_increasing(t_beg.jd):
            raise ValueError(
                "EXPMETER_SCI Date-Beg timestamps are not strictly increasing"
            )
        if not strictly_increasing(t_end.jd):
            raise ValueError(
                "EXPMETER_SCI Date-End timestamps are not strictly increasing"
            )

        t_mid = Time((t_beg.jd + t_end.jd) / 2, format="jd", scale="utc")
        return t_beg, t_mid, t_end

    def _get_normalized_flux(self):
        """
        Read EXPMETER_SCI flux and normalize by gain and wavelength dispersion.

        Columns with numeric names are wavelength channels; non-numeric
        columns (e.g. timestamps) are skipped.

        EXPMETER_SCI column labels are in Å on L1+ data (converted from the
        native L0 nm by ImageAssembly), so `w` is returned in Å.

        Returns
        -------
        w : ndarray, shape (nwave,)
            Wavelength of each expmeter channel [Å].
        f : ndarray, shape (ntime, nwave)
            Dispersion-normalized flux [e- / Å].
        """
        expmeter = self.l2_obj.data["EXPMETER_SCI"]

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
    def _fix_expmeter_outliers(f, kernel_size=5, k=3.0):
        """
        Detect and interpolate outlier pixels in an expmeter flux array.

        Outlier threshold is adaptive (Chauvenet-like criterion based on array
        size), scaled by the coefficient `k` (larger rejects fewer points).
        Replacement uses scipy.griddata linear interpolation, falling back to
        nearest for any remaining NaNs.

        Returns a copy of f with outliers replaced; shape unchanged.
        """
        f_smooth = gaussian_filter(
            median_filter(f, size=kernel_size), sigma=kernel_size
        )

        eta = k * np.sqrt(2) * erfcinv(1 / np.min(np.shape(f)))
        bad = np.abs(f - f_smooth) / mad_std(f - f_smooth) > eta

        if np.sum(bad) == 0:
            return f.copy()

        ny, nx = f.shape
        y, x = np.indices((ny, nx))

        points = np.column_stack((x[~bad], y[~bad]))
        values = f[~bad]
        points_bad = np.column_stack((x[bad], y[bad]))

        f_fixed = f.copy()
        f_fixed[bad] = griddata(points, values, points_bad, method="linear")

        nan_mask = np.isnan(f_fixed)
        if np.any(nan_mask):
            f_fixed[nan_mask] = griddata(
                points,
                values,
                np.column_stack((x[nan_mask], y[nan_mask])),
                method="nearest",
            )

        return f_fixed

    @staticmethod
    def _interpolate(t_beg, t_end, f):
        """
        Estimate flux during gaps between consecutive exposure meter readings.

        Gap flux is the average count rate of adjacent exposures times
        the gap duration. Returns (t_gap, f_gap), shapes (ngap,) and
        (ngap, nwave).
        """
        dt_exp = (t_end - t_beg).jd[:, None]
        dt_gap = (t_beg[1:] - t_end[:-1]).jd

        rate = f / dt_exp
        rate_gap = 0.5 * (rate[1:] + rate[:-1])

        t_gap = Time(t_end[:-1].jd + dt_gap / 2, format="jd", scale="utc")
        f_gap = rate_gap * dt_gap[:, None]

        return t_gap, f_gap

    @staticmethod
    def _extrapolate(t0, t_beg, t_end, f):
        """
        Estimate flux for a gap before the first or after the last reading.

        Same rate-based model as `_interpolate`. Returns (t_ext, f_ext)
        for the single gap defined by t0 vs t_beg/t_end.
        """
        dt_exp = (t_end - t_beg).jd
        rate = f / dt_exp

        if t0 < t_beg:
            dt_gap = (t_beg - t0).jd
            t_ext = Time(t0.jd + dt_gap / 2, format="jd", scale="utc")
        elif t0 > t_end:
            dt_gap = (t0 - t_end).jd
            t_ext = Time(t_end.jd + dt_gap / 2, format="jd", scale="utc")
        else:
            raise ValueError("t0 must be before t_beg or after t_end")

        f_ext = rate * dt_gap
        return t_ext, f_ext

    def _compute_per_chanel_flux_weighted_midpoint_time(
        self, interpolate=True, extrapolate=True, fix_expmeter_outliers=True
    ):
        """
        Compute the flux-weighted midpoint time for each expmeter channel.

        Reads EXPMETER_SCI, optionally fills gaps between readings
        (`interpolate`), gaps before/after the shutter window (`extrapolate`,
        using DATE-BEG/DATE-END), and replaces outlier readings
        (`fix_expmeter_outliers`), then collapses the time axis to a single
        flux-weighted mean time per channel.

        Returns
        -------
        w_em : ndarray, shape (nwave,)
            Wavelength of each expmeter channel [Å].
        t_em : Time, shape (nwave,)
            Flux-weighted midpoint time (JD-UTC) per channel.
        """
        t_beg, t_mid, t_end = self._get_timestamps()
        w_em, f = self._get_normalized_flux()

        if np.any(f < 0):
            raise ValueError("negative exposure meter flux values detected")

        if fix_expmeter_outliers:
            f = self._fix_expmeter_outliers(f)

        # Cache boundary readings; f[0] and f[-1] get clobbered by later vstacks.
        f_first_reading = f[0].copy()
        f_last_reading = f[-1].copy()

        t = t_mid.copy()

        if interpolate:
            t_gap, f_gap = self._interpolate(t_beg, t_end, f)
            t = Time(np.concatenate([t.jd, t_gap.jd]), format="jd", scale="utc")
            f = np.vstack([f, f_gap])

        if extrapolate:
            hdr = self.l2_obj.headers["INSTRUMENT_HEADER"]
            obs_beg = Time(hdr["DATE-BEG"], format="isot", scale="utc")
            obs_end = Time(hdr["DATE-END"], format="isot", scale="utc")

            if obs_beg < t_beg[0]:
                t_ext, f_ext = self._extrapolate(
                    obs_beg, t_beg[0], t_end[0], f_first_reading
                )
                t = Time(np.concatenate([t.jd, [t_ext.jd]]), format="jd", scale="utc")
                f = np.vstack([f, f_ext])

            if obs_end > t_end[-1]:
                t_ext, f_ext = self._extrapolate(
                    obs_end, t_beg[-1], t_end[-1], f_last_reading
                )
                t = Time(np.concatenate([t.jd, [t_ext.jd]]), format="jd", scale="utc")
                f = np.vstack([f, f_ext])

        flux_sum = np.sum(f, axis=0)
        if np.any(flux_sum <= 0):
            raise ValueError(
                "exposure meter channel with non-positive total flux; "
                "cannot compute flux-weighted midpoint"
            )
        t_em = Time(
            np.sum(t.jd[:, None] * f, axis=0) / flux_sum,
            format="jd",
            scale="utc",
        )
        return w_em, t_em

    # ------------------------------------------------------------------
    # Private helpers — barycorr handoffs
    # ------------------------------------------------------------------

    def _gaia_astrometry(self):
        """
        Query Gaia DR3 for the target's ICRS astrometry (cached).

        The source_id is read from GAIAID in INSTRUMENT_HEADER (preserved from
        L1 PRIMARY by KPF1.to_kpf2()). Returns a SkyCoord with proper motion
        and distance (from parallax) attached, at the Gaia ref_epoch. The
        network query runs once and is reused across calls (one target/frame).
        """
        if self._skycoord is None:
            gaia_id_raw = self.l2_obj.headers["INSTRUMENT_HEADER"]["GAIAID"]
            gaia_id = re.split(r"\s+", str(gaia_id_raw).strip())[-1]
            if not gaia_id.isdigit():
                raise ValueError(f"Gaia source_id must be all digits; got {gaia_id!r}")
            query = f"""
            SELECT ra, dec, pmra, pmdec, parallax, ref_epoch
            FROM gaiadr3.gaia_source
            WHERE source_id = {gaia_id}
            """
            job = Gaia.launch_job(query)
            result = job.get_results()[0]

            self._skycoord = SkyCoord(
                ra=result["ra"] * u.deg,
                dec=result["dec"] * u.deg,
                pm_ra_cosdec=result["pmra"] * u.mas / u.yr,
                pm_dec=result["pmdec"] * u.mas / u.yr,
                distance=(1e3 / result["parallax"]) * u.pc,
                obstime=Time(result["ref_epoch"], format="jyear"),
                frame="icrs",
            )
        return self._skycoord

    def _wmko_astrometry(self):
        """
        Build a SkyCoord from WMKO/DCS astrometry in INSTRUMENT_HEADER.

        Uses TARGRA/TARGDEC (TARGFRAM, FK5-default equinox J2000 = TARGEPOC)
        with proper motion and parallax. TARGPMRA is in time-seconds/yr
        (-> mas/yr via x15 cos(dec)); TARGPLAX is in mas (-> distance via
        1e3/plax), matching the Gaia path.
        """
        inst = self.l2_obj.headers["INSTRUMENT_HEADER"]
        pos = SkyCoord(inst["TARGRA"], inst["TARGDEC"], unit=(u.hourangle, u.deg))
        pm_ra_cosdec = float(inst["TARGPMRA"]) * 15.0 * np.cos(pos.dec.rad) * 1e3
        return SkyCoord(
            ra=pos.ra,
            dec=pos.dec,
            pm_ra_cosdec=pm_ra_cosdec * u.mas / u.yr,
            pm_dec=float(inst["TARGPMDC"]) * 1e3 * u.mas / u.yr,
            distance=(1e3 / float(inst["TARGPLAX"])) * u.pc,
            frame=str(inst["TARGFRAM"]).lower(),
            obstime=Time(float(inst["TARGEPOC"]), format="jyear"),
        )

    def _get_skycoord(self):
        """
        Resolve target astrometry: Gaia DR3 first, then WMKO header fallback.

        Each source is tried only if its config toggle is set. If neither
        yields a SkyCoord, raises, surfacing the captured Gaia error so the
        failure (our-side vs Gaia-server-side) is distinguishable.
        """
        gaia_error = None
        if self.use_gaia_astrometry:
            try:
                skycoord = self._gaia_astrometry()
                self._astrometry_source = "Gaia DR3"
                return skycoord
            except Exception as e:
                gaia_error = e
        if self.use_wmko_fallback:
            if gaia_error is not None:
                warnings.warn(
                    f"Gaia astrometry unavailable ({type(gaia_error).__name__}: "
                    f"{gaia_error}); using WMKO header astrometry",
                    stacklevel=2,
                )
            self._astrometry_source = "WMKO header"
            return self._wmko_astrometry()
        raise ValueError(
            "no target astrometry: Gaia "
            + (
                f"failed ({type(gaia_error).__name__}: {gaia_error})"
                if gaia_error
                else "disabled"
            )
            + ", WMKO fallback disabled"
        )

    @staticmethod
    def _compute_barycorr(skycoord, obs_times, location, rv_mps=0.0):
        """
        Compute barycentric velocity (m/s) and BJD_TDB for an array of obs_times.

        `rv_mps` is the target's systemic RV (m/s), passed to barycorrpy so
        the light-travel correction in BJD_TDB accounts for stellar motion.

        Returns (bc_vel_mps, bjd_tdb), each shape (n,).
        """
        icrs = skycoord.icrs
        ra = icrs.ra.to(u.deg).value
        dec = icrs.dec.to(u.deg).value
        pmra = (
            icrs.pm_ra_cosdec.to(u.mas / u.yr).value
            if icrs.pm_ra_cosdec is not None
            else 0.0
        )
        pmdec = icrs.pm_dec.to(u.mas / u.yr).value if icrs.pm_dec is not None else 0.0
        px = (
            (1 / icrs.distance.to(u.pc).value * 1e3)
            if icrs.distance is not None
            else 0.0
        )
        epoch = icrs.obstime.jd

        lat = location.lat.to(u.deg).value
        lon = location.lon.to(u.deg).value
        alt = location.height.to(u.m).value

        JDUTC = np.atleast_1d(obs_times.utc.jd)

        bc_vel, *_ = get_BC_vel(
            JDUTC=JDUTC,
            ra=ra,
            dec=dec,
            lat=lat,
            longi=lon,
            alt=alt,
            pmra=pmra,
            pmdec=pmdec,
            px=px,
            epoch=epoch,
            rv=rv_mps,
        )
        bjd_tdb, *_ = utc_tdb.JDUTC_to_BJDTDB(
            JDUTC=JDUTC,
            ra=ra,
            dec=dec,
            lat=lat,
            longi=lon,
            alt=alt,
            pmra=pmra,
            pmdec=pmdec,
            px=px,
            epoch=epoch,
            rv=rv_mps,
        )
        return np.asarray(bc_vel), np.asarray(bjd_tdb)

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def compute_flux_weighted_midpoint_times(
        self,
        output="orders",
        interpolate=True,
        extrapolate=True,
        fix_expmeter_outliers=True,
        weight_percentile=90,
    ):
        """
        Compute the flux-weighted midpoint observation time.

        The per-channel midpoint is always derived internally; `output`
        controls how it is projected.

        Parameters
        ----------
        output : {'expmeter', 'orders', 'ccds'}
            'expmeter' — per-channel times, shape (nwave,).
            'orders'   — per spectral order: mean of the per-channel times for
                         the expmeter channels within the order's wavelength
                         range, shape (NORDER,). Orders outside expmeter
                         coverage fall back to the nearest channel.
            'ccds'     — per chip: the SCI2-flux-weighted mean of the per-order
                         values, shape (2,); [GREEN, RED].
        interpolate : bool, optional
            If True, estimate flux during gaps between expmeter readings.
        extrapolate : bool, optional
            If True, estimate flux during gaps before the first or after the
            last expmeter reading, using DATE-BEG / DATE-END from
            INSTRUMENT_HEADER.
        fix_expmeter_outliers : bool, optional
            If True (default), detect and replace outlier expmeter readings.
        weight_percentile : float, optional
            Per-order SCI2 brightness percentile used to weight orders for the
            'ccds' output (robust to cosmics). Defaults to 90.

        Returns
        -------
        w : ndarray
            Wavelengths corresponding to each output bin [Å].
        t_fwm : Time
            Flux-weighted midpoint time (JD-UTC) per output bin.
        """
        if output not in ("expmeter", "orders", "ccds"):
            raise ValueError(
                f"output must be 'expmeter', 'orders', or 'ccds'; got {output!r}"
            )

        # Reuse the cached integration when the toggles match (perform() asks
        # for 'orders' then 'ccds').
        key = (interpolate, extrapolate, fix_expmeter_outliers)
        if self._exposure_meter is None or self._exposure_meter[0] != key:
            w_em, t_em = self._compute_per_chanel_flux_weighted_midpoint_time(
                interpolate=interpolate,
                extrapolate=extrapolate,
                fix_expmeter_outliers=fix_expmeter_outliers,
            )
            self._exposure_meter = (key, w_em, t_em)
        _, w_em, t_em = self._exposure_meter

        if output == "expmeter":
            return w_em, t_em

        wave = self.l2_obj.data["SCI2_WAVE"]
        if wave is None or np.size(wave) == 0:
            raise KeyError(
                "SCI2_WAVE missing or empty; run WavelengthCalibration first"
            )
        wave = np.asarray(wave)
        wave_min = wave.min(axis=1)
        wave_max = wave.max(axis=1)
        w_orders = 0.5 * (wave_min + wave_max)

        t_em_jd = t_em.jd
        jd_per_order = np.empty(len(w_orders))
        for i in range(len(w_orders)):
            in_order = (w_em >= wave_min[i]) & (w_em <= wave_max[i])
            if np.any(in_order):
                jd_per_order[i] = np.mean(t_em_jd[in_order])
            else:
                jd_per_order[i] = t_em_jd[np.argmin(np.abs(w_em - w_orders[i]))]
        t_orders = Time(jd_per_order, format="jd", scale="utc")

        if output == "orders":
            return w_orders, t_orders

        # Weight each order by its SCI2 brightness (`weight_percentile`, robust
        # to cosmics); NaN/failed orders get zero weight, uniform if SCI2_FLUX
        # absent.
        norder_green = self.norder["GREEN"]
        norder = norder_green + self.norder["RED"]
        flux = self.l2_obj.data["SCI2_FLUX"]
        if flux is None or np.size(flux) == 0:
            weights = np.ones(norder)
        else:
            weights = np.nanpercentile(
                np.asarray(flux, dtype=float), weight_percentile, axis=1
            )
        weights = np.nan_to_num(weights, nan=0.0)

        green = slice(0, norder_green)
        red = slice(norder_green, norder)
        if weights[green].sum() <= 0 or weights[red].sum() <= 0:
            raise ValueError(
                "all SCI2 order weights are zero for a CCD; "
                "cannot compute per-CCD midpoint"
            )
        w_ccds = np.array(
            [
                np.average(w_orders[green], weights=weights[green]),
                np.average(w_orders[red], weights=weights[red]),
            ]
        )
        t_ccds = Time(
            [
                np.average(jd_per_order[green], weights=weights[green]),
                np.average(jd_per_order[red], weights=weights[red]),
            ],
            format="jd",
            scale="utc",
        )
        return w_ccds, t_ccds

    def compute_barycentric_correction(
        self,
        output="orders",
        interpolate=True,
        extrapolate=True,
        fix_expmeter_outliers=True,
    ):
        """
        Compute the barycentric correction at the flux-weighted photon-midpoint
        time for each output bin.

        Mirrors compute_flux_weighted_midpoint_times: `output` selects the
        binning. The per-channel integration and the Gaia astrometry are both
        cached, so calling this for 'orders' then 'ccds' does the heavy work
        once.

        Parameters
        ----------
        output : {'expmeter', 'orders', 'ccds'}
            Binning level; forwarded to compute_flux_weighted_midpoint_times().
        interpolate, extrapolate, fix_expmeter_outliers : bool, optional
            Forwarded to compute_flux_weighted_midpoint_times().

        Returns
        -------
        bjd_tdb : ndarray
            Photon-weighted midpoint in BJD_TDB per output bin.
        bary_kms : ndarray
            Barycentric velocity per output bin [km/s].
        bary_z : ndarray
            Barycentric redshift per output bin.
        """
        _, t_fwm = self.compute_flux_weighted_midpoint_times(
            output=output,
            interpolate=interpolate,
            extrapolate=extrapolate,
            fix_expmeter_outliers=fix_expmeter_outliers,
        )
        skycoord = self._get_skycoord()

        # TARGRADV is in km/s; barycorrpy expects rv in m/s. Missing → 0.
        inst = self.l2_obj.headers["INSTRUMENT_HEADER"]
        rv_mps = float(inst.get("TARGRADV", 0.0) or 0.0) * 1000.0

        bc_vel_mps, bjd_tdb = self._compute_barycorr(
            skycoord,
            t_fwm,
            self.KECK_LOCATION,
            rv_mps=rv_mps,
        )
        bary_kms = bc_vel_mps / 1000.0
        bary_z = np.asarray(compute_redshift(bc_vel_mps * u.m / u.s))
        return bjd_tdb, bary_kms, bary_z

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------

    def _track_info(self):
        """Populate _info (the info() summary) from instance attributes."""
        self._info = {
            "bjd_tdb": np.asarray(self.l2_obj.data["BJD_TDB"]),
            "bary_kms": np.asarray(self.l2_obj.data["BARYCORR_KMS"]),
            "ccd_bjd": np.asarray(self._ccd_bjd),
            "ccd_kms": np.asarray(self._ccd_kms),
            "ccd_z": np.asarray(self._ccd_z),
            "astrometry_source": self._astrometry_source,
        }

    def _set_headers(self, l2_obj):
        """Write all summary header keywords for barycentric correction.

        Reads self._ccd_bjd/_ccd_kms/_ccd_z and self._astrometry_source
        (populated by perform()); the single place this module writes summary
        keywords, called just before the receipt entry. Per-CCD keywords are
        registered KPF-pipeline keywords (config/L2-headers.csv); set_keyword
        routes them to their registry homes (BJD_TDB / BARYCORR_KMS / BARYCORR_Z,
        and RECEIPT for ASTRSRC). CCD1=GREEN, CCD2=RED.
        """
        l2_obj.set_keyword("CCD1BJD", float(self._ccd_bjd[0]))
        l2_obj.set_keyword("CCD1BKMS", float(self._ccd_kms[0]))
        l2_obj.set_keyword("CCD1BZ", float(self._ccd_z[0]))
        l2_obj.set_keyword("CCD2BJD", float(self._ccd_bjd[1]))
        l2_obj.set_keyword("CCD2BKMS", float(self._ccd_kms[1]))
        l2_obj.set_keyword("CCD2BZ", float(self._ccd_z[1]))
        l2_obj.set_keyword("ASTRSRC", self._astrometry_source)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(
        self,
        *,
        use_gaia_astrometry=None,
        use_wmko_fallback=None,
        interpolate=True,
        extrapolate=True,
        fix_expmeter_outliers=True,
    ):
        """
        Compute per-order barycentric correction and store it on the KPF2.

        Parameters
        ----------
        use_gaia_astrometry, use_wmko_fallback : bool, optional
            Override the configured astrometry-source toggles for this call.
        interpolate, extrapolate, fix_expmeter_outliers : bool, optional
            Forwarded to compute_flux_weighted_midpoint_times(). See that
            method for semantics.

        Returns
        -------
        l2_obj : KPF2
            Input KPF2 with BJD_TDB / BARYCORR_KMS / BARYCORR_Z populated,
            per-CCD CCD{1,2}BJD/BKMS/BZ summaries and the astrometry source
            (ASTRSRC) written to PRIMARY, and a
            'barycentric_correction' receipt entry.
        """
        if use_gaia_astrometry is not None:
            self.use_gaia_astrometry = use_gaia_astrometry
        if use_wmko_fallback is not None:
            self.use_wmko_fallback = use_wmko_fallback

        kwargs = dict(
            interpolate=interpolate,
            extrapolate=extrapolate,
            fix_expmeter_outliers=fix_expmeter_outliers,
        )
        bjd_tdb, bary_kms, bary_z = self.compute_barycentric_correction(
            output="orders", **kwargs
        )
        # Per-CCD summaries [GREEN, RED], consumed by _set_headers.
        self._ccd_bjd, self._ccd_kms, self._ccd_z = self.compute_barycentric_correction(
            output="ccds", **kwargs
        )

        # Per-order extensions; WAVE arrays are left untouched
        self.l2_obj.set_data("BJD_TDB", np.asarray(bjd_tdb, dtype=np.float64))
        self.l2_obj.set_data("BARYCORR_KMS", np.asarray(bary_kms, dtype=np.float64))
        self.l2_obj.set_data("BARYCORR_Z", np.asarray(bary_z, dtype=np.float64))

        self._set_headers(self.l2_obj)
        self._track_info()
        self.l2_obj.receipt_add_entry("barycentric_correction", "PASS")

        return self.l2_obj

    def info(self):
        """Print a summary of the barycentric correction results."""
        print("BarycentricCorrection")
        obs_id = self.l2_obj.headers.get("PRIMARY", {}).get("ORIGID", "unknown")
        print(f"  obs_id:  {obs_id}")

        if self._info is None:
            print("  perform() has not been called")
            return

        r = self._info
        print(f"  astrometry:  {r['astrometry_source']}")
        ccd_bjd, ccd_kms, ccd_z = r["ccd_bjd"], r["ccd_kms"], r["ccd_z"]

        # Per-CCD summaries (match CCD1*/CCD2* on PRIMARY).
        print(f"\n  {'':<8s}{'BJD_TDB':>18s}{'BARYCORR_KMS':>18s}{'BARYCORR_Z':>18s}")
        print("  " + "-" * 62)
        print(
            f"  {'GREEN':<8s}{ccd_bjd[0]:>18.6f}{ccd_kms[0]:>+18.4f}{ccd_z[0]:>18.10f}"
        )
        print(f"  {'RED':<8s}{ccd_bjd[1]:>18.6f}{ccd_kms[1]:>+18.4f}{ccd_z[1]:>18.10f}")

        bjd, kms = r["bjd_tdb"], r["bary_kms"]
        print(
            f"\n  per-order spread:   BJD {np.ptp(bjd) * 86400:.3f} sec,"
            f" BARY {np.ptp(kms) * 1000:.3f} m/s"
        )
