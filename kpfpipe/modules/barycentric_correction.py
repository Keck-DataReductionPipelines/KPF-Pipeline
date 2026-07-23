"""
KPF Barycentric Correction module.

Computes per-order barycentric corrections from the EXPMETER_SCI flux-weighted
midpoint times and stores them on the L2 as BJD_TDB, BARYCORR_KMS, and
BARYCORR_Z (per spectral order), plus per-CCD scalar summaries in the
barycentric extension headers. Wavelength arrays are not modified.

Target astrometry is read from the EPRV PRIMARY C*# catalog keywords, not queried:
AstroQuery is the pipeline's sole external-catalog client, and KPF0.to_kpf1 overlays
its merged canonical record onto those cards.

Follows the barycentric-correction approach of Wright & Eastman (2014,
barycorrpy) with the flux-weighted midpoint time of Butler et al. (1996).
"""

import logging

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle, EarthLocation
from astropy.stats import mad_std
from astropy.time import Time
from barycorrpy import get_BC_vel, utc_tdb
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter
from scipy.special import erfcinv

from kpfpipe import DEFAULTS
from kpfpipe.utils.astro import compute_redshift
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import strictly_increasing

logger = logging.getLogger(__name__)

_DEFAULTS = {**DEFAULTS}

# The position block the correction cannot proceed without; an absent card means
# AstroQuery never ran. The trailing 3 (here and on CPLX3/CSRC3/CRV3 below) selects
# the SCI2 trace: the C*# cards are written identically to every science fiber
# (SCI1-3 = traces 2-4), and SCI2 is the fiber this module uses for SCI2_WAVE/FLUX.
_REQUIRED_CARDS = ("CRA3", "CDEC3", "CPMR3", "CPMD3", "CEPCH3")


class BarycentricCorrection:
    """
    Compute and store per-order barycentric correction on a KPF2.

    Derives the flux-weighted midpoint time per expmeter channel (EXPMETER_SCI),
    averages it over the channels within each spectral order's wavelength range
    (SCI2_WAVE), reads the target's astrometry from the PRIMARY C*# catalog
    keywords, and calls barycorrpy to populate BJD_TDB / BARYCORR_KMS /
    BARYCORR_Z. WAVE arrays are not modified.

    Parameters
    ----------
    l2_obj : KPF2
        Extracted L2 frame. Must have EXPMETER_SCI populated and SCI2_WAVE
        populated by WavelengthCalibration. PRIMARY must carry the SCI2 catalog
        cards (CRA3/CDEC3/CPMR3/CPMD3/CEPCH3, written by AstroQuery via
        KPF0.to_kpf1), unless perform() is given a ``skycoord`` override.
        INSTRUMENT_HEADER (the preserved L1 PRIMARY) must contain
        DATE-BEG/DATE-END when extrapolating.
    config : None | dict | ConfigHandler
        Module configuration. Recognizes no module-specific keys.
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
                ["DATA_DIRS", "TRACES", "MODULE_BARYCENTRIC_CORRECTION"]
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
        self._astrometry = None  # cached barycorrpy kwargs, set by _get_astrometry
        self._astrometry_source = None  # CSRC3 ('gaia'|'simbad'|'wmko'), same setter

    # ------------------------------------------------------------------
    # Private helpers -- exposure-meter handling
    # ------------------------------------------------------------------

    def _get_timestamps(self):
        """
        Read EXPMETER_SCI start/mid/end timestamps (t_beg, t_mid, t_end).

        Prefers the corrected columns (Date-Beg-Corr / Date-End-Corr), falling
        back to uncorrected; t_mid is the unweighted (t_beg + t_end) / 2.
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
        Read EXPMETER_SCI flux, gain- and dispersion-normalized -> (w [Å], f [e-/Å]).

        Numeric-named columns are wavelength channels; non-numeric columns (e.g.
        timestamps) are skipped. Column labels are in Å on L1+ (converted from
        native L0 nm by ImageAssembly), so ``w`` is returned in Å.
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
        Return a copy of ``f`` with outlier pixels interpolated (shape unchanged).

        Threshold is adaptive (Chauvenet-like criterion on array size), scaled
        by ``k`` (larger rejects fewer). Replacement is griddata linear, falling
        back to nearest for residual NaNs.
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

        Same rate-based model as ``_interpolate``. Returns (t_ext, f_ext)
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

    def _compute_per_channel_flux_weighted_midpoint_time(
        self, interpolate=True, extrapolate=True, fix_outliers=True
    ):
        """
        Flux-weighted midpoint time per expmeter channel -> (w_em [Å], t_em [JD-UTC]).

        Optionally fills gaps between readings (``interpolate``), before/after the
        shutter window (``extrapolate``, using DATE-BEG/DATE-END), and replaces
        outlier readings (``fix_outliers``) before collapsing the time axis to a
        flux-weighted mean per channel.
        """
        t_beg, t_mid, t_end = self._get_timestamps()
        w_em, f = self._get_normalized_flux()

        if fix_outliers:
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
    # Private helpers -- barycorr handoffs
    # ------------------------------------------------------------------

    @staticmethod
    def _astrometry_from_skycoord(skycoord):
        """
        Convert a caller-supplied SkyCoord into barycorrpy's argument set.

        Backs ``perform(skycoord=...)``: same output as ``_get_astrometry``, sourced
        from the object instead of the header. Unvalidated -- the SkyCoord is rotated to
        ICRS and read, and astropy raises if it lacks the proper motion, distance, or
        obstime the correction needs.
        """
        icrs = skycoord.icrs
        return {
            "ra": icrs.ra.to_value(u.deg),
            "dec": icrs.dec.to_value(u.deg),
            "pmra": icrs.pm_ra_cosdec.to_value(u.mas / u.yr),
            "pmdec": icrs.pm_dec.to_value(u.mas / u.yr),
            "px": 1e3 / icrs.distance.to_value(u.pc),
            "epoch": icrs.obstime.jd,
        }

    def _get_astrometry(self):
        """
        Read the target astrometry off the PRIMARY C*# cards (cached).

        The cards carry AstroQuery's merged canonical record (see
        ``KPF0._catalog_primary_cards``), already sanitized to one schema, so this is a
        unit conversion into barycorrpy's argument set -- ra/dec [deg], pmra/pmdec
        [mas/yr], px [mas], epoch [JD] -- and nothing else. The frame is ICRS: not
        persisted as a card, but fixed by AstroQuery's construction, which rotates every
        source into ICRS before writing.

        AstroQuery persists catalog values faithfully, unphysical ones included, so the
        physical sanitation belongs here: a missing or non-positive parallax (routine
        for faint Gaia sources) becomes ``px=0``, barycorrpy's own no-parallax value.

        Raises
        ------
        ValueError
            If any of ``_REQUIRED_CARDS`` is absent or blank -- without a position
            there is nothing to correct.
        """
        if self._astrometry is None:
            primary = self.l2_obj.headers["PRIMARY"]
            cards = {name: primary.get(name) for name in _REQUIRED_CARDS}
            missing = [
                name
                for name, value in cards.items()
                if value is None or str(value).strip() == ""
            ]
            if missing:
                raise ValueError(
                    f"no target astrometry on PRIMARY: {', '.join(missing)} "
                    f"missing or blank; run AstroQuery on the L0 so the canonical "
                    f"catalog record reaches the C*# cards"
                )

            # The parallax card is the catalog's own value; QC flags an unphysical one
            # via CATLOGOK but does not repair it, so sanitize here.
            parallax = primary.get("CPLX3")
            try:
                px = float(parallax)
            except (TypeError, ValueError):
                px = np.nan
            if not np.isfinite(px) or px <= 0:
                logger.warning(
                    "CPLX3=%s is missing or unusable; using px=0 (no parallax)",
                    parallax,
                )
                px = 0.0

            self._astrometry = {
                "ra": Angle(cards["CRA3"], unit=u.hourangle).deg,
                "dec": Angle(cards["CDEC3"], unit=u.deg).deg,
                # C*# proper motion is arcsec/yr (RA incl. cos Dec); barycorrpy mas/yr.
                "pmra": float(cards["CPMR3"]) * 1e3,
                "pmdec": float(cards["CPMD3"]) * 1e3,
                "px": px,
                "epoch": Time(float(cards["CEPCH3"]), format="jyear").jd,
            }
            self._astrometry_source = primary.get("CSRC3") or "unknown"
        return self._astrometry

    @staticmethod
    def _compute_barycorr(astrometry, obs_times, location, rv_mps=0.0):
        """
        barycorrpy handoff -> (bc_vel_mps, bjd_tdb), each shape (n,).

        ``astrometry`` is the ICRS argument set from ``_get_astrometry``. ``rv_mps``
        is the target's systemic RV so the BJD_TDB light-travel correction accounts
        for stellar motion.
        """
        lat = location.lat.to(u.deg).value
        lon = location.lon.to(u.deg).value
        alt = location.height.to(u.m).value

        JDUTC = np.atleast_1d(obs_times.utc.jd)

        bc_vel, *_ = get_BC_vel(
            JDUTC=JDUTC,
            lat=lat,
            longi=lon,
            alt=alt,
            rv=rv_mps,
            **astrometry,
        )
        bjd_tdb, *_ = utc_tdb.JDUTC_to_BJDTDB(
            JDUTC=JDUTC,
            lat=lat,
            longi=lon,
            alt=alt,
            rv=rv_mps,
            **astrometry,
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
        fix_outliers=True,
        weight_percentile=90,
    ):
        """
        Compute the flux-weighted midpoint observation time.

        The per-channel midpoint is always derived internally; ``output``
        controls how it is projected.

        Parameters
        ----------
        output : {'expmeter', 'orders', 'ccds'}
            'expmeter' -- per-channel times, shape (nwave,).
            'orders'   -- per spectral order: mean of the per-channel times for
                         the expmeter channels within the order's wavelength
                         range, shape (NORDER,). Orders outside expmeter
                         coverage fall back to the nearest channel.
            'ccds'     -- per chip: the SCI2-flux-weighted mean of the per-order
                         values, shape (2,); [GREEN, RED].
        interpolate : bool, optional
            If True, estimate flux during gaps between expmeter readings.
        extrapolate : bool, optional
            If True, estimate flux during gaps before the first or after the
            last expmeter reading, using DATE-BEG / DATE-END from
            INSTRUMENT_HEADER.
        fix_outliers : bool, optional
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

        Raises
        ------
        ValueError
            If ``output`` is not one of the allowed values; if the EXPMETER_SCI
            timestamps are not strictly increasing; if an exposure meter channel
            has non-positive total flux; or (for ``output='ccds'``) if all SCI2
            order weights are zero for a CCD.
        KeyError
            If SCI2_WAVE is missing or empty (run WavelengthCalibration first).
        """
        if output not in ("expmeter", "orders", "ccds"):
            raise ValueError(
                f"output must be 'expmeter', 'orders', or 'ccds'; got {output!r}"
            )

        # Reuse the cached integration when the toggles match (perform() asks
        # for 'orders' then 'ccds').
        key = (interpolate, extrapolate, fix_outliers)
        if self._exposure_meter is None or self._exposure_meter[0] != key:
            w_em, t_em = self._compute_per_channel_flux_weighted_midpoint_time(
                interpolate=interpolate,
                extrapolate=extrapolate,
                fix_outliers=fix_outliers,
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

        # Weight each order by its SCI2 brightness (``weight_percentile``, robust
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
        interpolate_expmeter_flux=True,
        extrapolate_expmeter_flux=True,
        fix_expmeter_outliers=True,
        skycoord=False,
    ):
        """
        Compute the barycentric correction at the flux-weighted photon-midpoint
        time for each output bin.

        Mirrors compute_flux_weighted_midpoint_times: ``output`` selects the
        binning. The per-channel integration and the resolved astrometry are both
        cached, so calling this for 'orders' then 'ccds' does the heavy work
        once.

        Parameters
        ----------
        output : {'expmeter', 'orders', 'ccds'}
            Binning level; forwarded to compute_flux_weighted_midpoint_times().
        interpolate_expmeter_flux, extrapolate_expmeter_flux, fix_expmeter_outliers
                : bool, optional
            Forwarded to compute_flux_weighted_midpoint_times() as its
            ``interpolate`` / ``extrapolate`` / ``fix_outliers``.
        skycoord : False | SkyCoord, optional
            Use this astrometry instead of the PRIMARY C*# cards. See perform().

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
            interpolate=interpolate_expmeter_flux,
            extrapolate=extrapolate_expmeter_flux,
            fix_outliers=fix_expmeter_outliers,
        )
        if skycoord:
            # Caller-supplied astrometry: not cached onto self._astrometry, so a later
            # call without skycoord still reads the header rather than this override.
            astrometry = self._astrometry_from_skycoord(skycoord)
            self._astrometry_source = "user SkyCoord"
        else:
            astrometry = self._get_astrometry()

        # CRV3 (SCI2 trace) is in km/s; barycorrpy expects rv in m/s.
        primary = self.l2_obj.headers["PRIMARY"]
        systemic_rv = primary.get("CRV3")
        try:
            rv_kms = float(systemic_rv)
        except (TypeError, ValueError):
            rv_kms = np.nan
        if not np.isfinite(rv_kms):
            logger.warning(
                "CRV3=%s is missing or unusable; using rv=0 (no systemic RV)",
                systemic_rv,
            )
            rv_kms = 0.0
        rv_mps = rv_kms * 1000.0

        bc_vel_mps, bjd_tdb = self._compute_barycorr(
            astrometry,
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
        """Build and cache the info() summary text from instance attributes."""
        obs_id = self.l2_obj.obs_id or "unknown"
        ccd_bjd = np.asarray(self._ccd_bjd)
        ccd_kms = np.asarray(self._ccd_kms)
        ccd_z = np.asarray(self._ccd_z)
        lines = [
            "BarycentricCorrection",
            f"  obs_id:  {obs_id}",
            f"  astrometry:  {self._astrometry_source}",
            # Per-CCD summaries (CCD1*/CCD2* on the BJD_TDB / BARYCORR_KMS /
            # BARYCORR_Z extension headers).
            f"\n  {'':<8s}{'BJD_TDB':>18s}{'BARYCORR_KMS':>18s}{'BARYCORR_Z':>18s}",
            "  " + "-" * 62,
            f"  {'GREEN':<8s}{ccd_bjd[0]:>18.6f}{ccd_kms[0]:>+18.4f}{ccd_z[0]:>18.10f}",
            f"  {'RED':<8s}{ccd_bjd[1]:>18.6f}{ccd_kms[1]:>+18.4f}{ccd_z[1]:>18.10f}",
        ]
        bjd = np.asarray(self.l2_obj.data["BJD_TDB"])
        kms = np.asarray(self.l2_obj.data["BARYCORR_KMS"])
        lines.append(
            f"\n  per-order spread:   BJD {np.ptp(bjd) * 86400:.3f} sec,"
            f" BARY {np.ptp(kms) * 1000:.3f} m/s"
        )
        self._info = "\n\n" + "\n".join(lines) + "\n\n"

    def _set_headers(self, l2_obj):
        """Write the per-CCD summary keywords.

        Reads self._ccd_bjd/_ccd_kms/_ccd_z (populated by perform()); set_keyword
        routes each to its registry home. CCD1=GREEN, CCD2=RED.
        """
        l2_obj.set_keyword("CCD1BJD", float(self._ccd_bjd[0]))
        l2_obj.set_keyword("CCD1BKMS", float(self._ccd_kms[0]))
        l2_obj.set_keyword("CCD1BZ", float(self._ccd_z[0]))
        l2_obj.set_keyword("CCD2BJD", float(self._ccd_bjd[1]))
        l2_obj.set_keyword("CCD2BKMS", float(self._ccd_kms[1]))
        l2_obj.set_keyword("CCD2BZ", float(self._ccd_z[1]))
        # CTYPE1 names the single (spectral-order) axis of these 1-D per-order
        # arrays -- registered content, multi-homed across the three barycorr
        # extensions, so stamped directly (set_keyword can't route a multi-home
        # keyword). CTYPE2 is N/A: the arrays have no second axis.
        for ext in ("BJD_TDB", "BARYCORR_KMS", "BARYCORR_Z"):
            l2_obj.headers[ext]["CTYPE1"] = ("Order-N", "Name of axis 1")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(
        self,
        *,
        interpolate_expmeter_flux=True,
        extrapolate_expmeter_flux=True,
        fix_expmeter_outliers=True,
        skycoord=False,
    ):
        """
        Compute per-order barycentric correction and store it on the KPF2.

        Parameters
        ----------
        interpolate_expmeter_flux, extrapolate_expmeter_flux, fix_expmeter_outliers
                : bool, optional
            Forwarded to compute_flux_weighted_midpoint_times() as its
            ``interpolate`` / ``extrapolate`` / ``fix_outliers``. See that method
            for semantics.
        skycoord : False | SkyCoord, optional
            Astrometry override for interactive use: when given, the correction uses
            this SkyCoord instead of the PRIMARY C*# cards, which stay untouched --
            lets a user retry with different astrometry on an L2 in hand without
            re-running AstroQuery from L0. Used as-is: it must carry proper motion,
            distance, and obstime, and astropy raises if it does not.

        Returns
        -------
        l2_obj : KPF2
            Input KPF2 with BJD_TDB / BARYCORR_KMS / BARYCORR_Z populated, the
            per-CCD CCD{1,2}BJD/BKMS/BZ summaries written to those same bary
            extension headers, and a 'barycentric_correction' receipt entry.
        """
        kwargs = dict(
            interpolate_expmeter_flux=interpolate_expmeter_flux,
            extrapolate_expmeter_flux=extrapolate_expmeter_flux,
            fix_expmeter_outliers=fix_expmeter_outliers,
            skycoord=skycoord,
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
        self.l2_obj.receipt_add_entry("barycentric_correction", "", "PASS")

        logger.info("%s", self._info)
        return self.l2_obj

    def info(self):
        """Print a summary of the barycentric correction results."""
        if self._info is None:
            print(f"{type(self).__name__}: perform() has not been called")
        else:
            print(self._info)
