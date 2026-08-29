"""Diagnostics for KPF Level 0 (raw CCD) data products."""

import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord, get_body, get_sun
from astropy.time import Time
from scipy.optimize import curve_fit

from kpfpipe.quality_control.diagnostics.base import Diagnostics
from kpfpipe.utils.astro import KECK_LOCATION

logger = logging.getLogger(__name__)


class DiagL0(Diagnostics):
    """Diagnostics for KPF Level 0 raw data products.

    The pointing-offset metrics compare the telescope pointing against the target
    astrometry AstroQuery resolves into the L0 ``CATALOG_RECORD`` extension, so
    AstroQuery must have run. The DCS target offset is required; the Gaia and
    SIMBAD offsets are emitted only when their catalog lookup matched.
    """

    LEVEL = "L0"

    # QUALITY_CONTROL metric -> the EPRV PRIMARY keyword it also answers. These
    # three map from a diagnostic rather than a native card
    # (``EPRV-header-map.csv`` gives them ``KPF_EXT=QUALITY_CONTROL``), and
    # QUALITY_CONTROL is still empty when StandardizeDataFormat runs, so DiagL0
    # is their PRIMARY writer too.
    _PRIMARY_EQUIVALENTS = {
        "GDRSEEV": "SEEING",
        "TCSSUN": "SUNEL",
        "TCSMOON": "MOONANG",
    }

    def run(self):
        """Run the L0 diagnostics, then mirror three of them onto EPRV PRIMARY."""
        results = super().run()
        for metric, eprv_keyword in self._PRIMARY_EQUIVALENTS.items():
            if metric in results:
                self.kpf_obj.set_keyword(eprv_keyword, results[metric][0])
        return results

    def _record_skycoord(self, rec):
        """ICRS SkyCoord from a CATALOG_RECORD record.

        The canonical schema all three sources share: RA/Dec sexagesimal strings
        (RA hour-angle, Dec deg), proper motion arcsec/yr (RA incl. cos Dec),
        parallax mas, epoch in Julian years. Proper motion and parallax fall back to
        zero (no motion, no distance) when the record omits them -- for this offset
        only; the CATALOG_RECORD values are left untouched.
        """
        pmra, pmdec = float(rec["pmra"]), float(rec["pmdec"])
        parallax = float(rec["parallax"])
        pm_missing = np.isnan(pmra) or np.isnan(pmdec)
        plx_missing = np.isnan(parallax) or parallax <= 0
        if pm_missing or plx_missing:
            logger.debug(
                "%s record missing %s; using PM=0, parallax=0 for the offset",
                rec["source"],
                " and ".join(
                    n for n, m in (("PM", pm_missing), ("parallax", plx_missing)) if m
                ),
            )
        kwargs = {
            "ra": rec["ra"],
            "dec": rec["dec"],
            "unit": (u.hourangle, u.deg),
            "pm_ra_cosdec": (0.0 if pm_missing else pmra) * u.arcsec / u.yr,
            "pm_dec": (0.0 if pm_missing else pmdec) * u.arcsec / u.yr,
            "obstime": Time(float(rec["epoch"]), format="jyear"),
            "frame": str(rec["frame"]),
        }
        if not plx_missing:
            kwargs["distance"] = (1e3 / parallax) * u.pc
        return SkyCoord(**kwargs)

    def _offset(self, source):
        """Arcsec separation of the pointing from a catalog source at obs epoch.

        The catalog position is proper-motion propagated to the observation epoch
        before the comparison.
        """
        table = self.kpf_obj.data["CATALOG_RECORD"]
        rec = table[table["source"] == source][0]
        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        pointing = SkyCoord(hdr["RA"], hdr["DEC"], unit=(u.hourangle, u.deg))
        obs_time = Time(float(hdr["MJD-OBS"]), format="mjd")
        coord = self._record_skycoord(rec).apply_space_motion(new_obstime=obs_time)
        return round(float(pointing.separation(coord).arcsec), 4)

    def gaia_ra_dec_offset(self):
        """GAIAOFF: arcsec, RA/DEC pointing vs Gaia catalog position at obs epoch.

        Gaia is an optional source; an unmatched lookup emits no keyword.
        """
        if "gaia" not in self.kpf_obj.data["CATALOG_RECORD"]["source"]:
            return {}
        return self._tag(GAIAOFF=self._offset("gaia"))

    gaia_ra_dec_offset._diag_name = "gaia_ra_dec_offset"

    def target_ra_dec_offset(self):
        """TCSOFF: arcsec, RA/DEC pointing vs DCS target position at obs epoch."""
        return self._tag(TCSOFF=self._offset("wmko"))

    target_ra_dec_offset._diag_name = "target_ra_dec_offset"

    def object_ra_dec_offset(self):
        """OBJOFF: arcsec, RA/DEC pointing vs SIMBAD(OBJECT) position at obs epoch.

        SIMBAD is an optional source; an unmatched lookup emits no keyword.
        """
        if "simbad" not in self.kpf_obj.data["CATALOG_RECORD"]["source"]:
            return {}
        return self._tag(OBJOFF=self._offset("simbad"))

    object_ra_dec_offset._diag_name = "object_ra_dec_offset"

    def solar_lunar_geometry(self):
        """TCSSUN, TCSMOON: deg, Sun altitude and target-Moon separation.

        Both are evaluated at mid-exposure from the WMKO site. TCSSUN is negative
        with the Sun below the horizon.
        """
        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        obs_time = Time(str(hdr["DATE-MID"]), scale="utc")
        sun = get_sun(obs_time).transform_to(
            AltAz(obstime=obs_time, location=KECK_LOCATION)
        )
        moon = get_body("moon", obs_time, KECK_LOCATION).transform_to("icrs")
        pointing = SkyCoord(hdr["RA"], hdr["DEC"], unit=(u.hourangle, u.deg))
        return self._tag(
            TCSSUN=round(float(sun.alt.deg), 5),
            TCSMOON=round(float(pointing.separation(moon).deg), 2),
        )

    solar_lunar_geometry._diag_name = "solar_lunar_geometry"

    def _present_amps(self, chip):
        """Yield ``(i, array)`` for each present, non-empty ``{chip}_AMP{i}``.

        Only the amps a readout actually used carry data, so 2-amp and 4-amp
        frames both work.
        """
        for i in range(1, 5):
            arr = self.kpf_obj.data.get(f"{chip}_AMP{i}")
            # KPF0 stores None-data as array(None, dtype=object); skip absent.
            if (
                arr is None
                or getattr(arr, "dtype", None) == np.dtype(object)
                or np.size(arr) == 0
            ):
                continue
            yield i, arr

    def _amp_pixel_fraction(self, chip, compare, level):
        """Largest fraction of any present amp on ``chip`` satisfying ``compare``.

        Raw D.N., before ImageAssembly applies gain or subtracts overscan; the
        worst amp decides, mirroring v2.12's per-amp infobits.
        """
        fractions = [
            np.count_nonzero(compare(arr, level)) / arr.size
            for _, arr in self._present_amps(chip)
        ]
        return round(float(max(fractions)), 6)

    def dead_pixel_fractions(self):
        """DEADPXFG/DEADPXFR: worst-amp fraction of GREEN/RED pixels under 1.0e4 D.N."""
        return self._tag(
            DEADPXFG=self._amp_pixel_fraction("GREEN", np.less, 1.0e4),
            DEADPXFR=self._amp_pixel_fraction("RED", np.less, 1.0e4),
        )

    dead_pixel_fractions._diag_name = "dead_pixel_fractions"

    def saturated_pixel_fractions(self):
        """SATPXFG/SATPXFR: worst-amp fraction of GREEN/RED pixels over 5.0e8 D.N."""
        return self._tag(
            SATPXFG=self._amp_pixel_fraction("GREEN", np.greater, 5.0e8),
            SATPXFR=self._amp_pixel_fraction("RED", np.greater, 5.0e8),
        )

    saturated_pixel_fractions._diag_name = "saturated_pixel_fractions"

    def amp_percentiles(self):
        """P{16,50,84}{G,R}AMP{1-4}: raw D.N. percentiles of each amplifier image.

        Ports v2.12's per-amp MEDGRN*/P16*/P84* statistics: the whole raw amp
        image, prescan and overscan included, NaNs excluded. Absent amps emit no
        keyword, so a 2-amp readout writes only the amps it has.
        """
        values = {}
        for chip, letter in (("GREEN", "G"), ("RED", "R")):
            for i, arr in self._present_amps(chip):
                for pct in (16, 50, 84):
                    values[f"P{pct}{letter}AMP{i}"] = round(
                        float(np.nanpercentile(arr, pct)), 6
                    )
        return self._tag(**values)

    amp_percentiles._diag_name = "amp_percentiles"

    def _telemetry_average(self, keyword):
        """One TELEMETRY keyword's exposure-average reading."""
        table = self.kpf_obj.data["TELEMETRY"]
        return float(table[table["keyword"] == keyword]["average"][0])

    def ccd_temperature_offsets(self):
        """GTEMPOFF/RTEMPOFF: signed GREEN/RED CCD offset from setpoint [mK].

        Ports the measurement half of v2.12 ``CCD_not_at_temp``: the
        exposure-average kpf{green,red}.STA_CCD_T telemetry against the -100 C
        setpoint, signed so the direction of the drift is visible.
        """
        return self._tag(
            GTEMPOFF=round(
                (self._telemetry_average("kpfgreen.STA_CCD_T") + 100.0) * 1e3, 6
            ),
            RTEMPOFF=round(
                (self._telemetry_average("kpfred.STA_CCD_T") + 100.0) * 1e3, 6
            ),
        )

    ccd_temperature_offsets._diag_name = "ccd_temperature_offsets"

    def etalon_temperature_offset(self):
        """ETATOFF: signed etalon offset from setpoint [mK], worst chamber.

        Ports the measurement half of v2.12 ``etalon_set_temp``: the inner bottom
        lid (ETAV1C3T) and the outer chamber (ETAV1C4T), each against its own
        setpoint keyword, falling back to the design value when the setpoint is
        not recorded. One keyword covers both, so the chamber furthest from its
        setpoint is the one reported.
        """
        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        offsets = []
        for temp_key, set_key, design in (
            ("ETAV1C3T", "ETAV1C3S", 23.6),
            ("ETAV1C4T", "ETAV1C4S", 23.9),
        ):
            setpoint = float(hdr[set_key]) if set_key in hdr else design
            offsets.append((float(hdr[temp_key]) - setpoint) * 1e3)
        return self._tag(ETATOFF=round(max(offsets, key=abs), 6))

    etalon_temperature_offset._diag_name = "etalon_temperature_offset"

    def _guider_frames(self):
        """GUIDER_CUBE_ORIGINS rows, the unwritten ones dropped.

        A cube that recorded flux carries trailing rows with a zero timestamp and
        zero flux; v2.12 drops them, but keeps an all-zero-flux cube whole so the
        emptiness stays visible to the metrics.
        """
        table = self.kpf_obj.data["GUIDER_CUBE_ORIGINS"]
        flux = np.asarray(table["object1_flux"], dtype=float)
        if len(table) > 1 and np.any(flux != 0.0):
            return table[
                (np.asarray(table["timestamp"], dtype=float) != 0.0) & (flux != 0.0)
            ]
        return table

    def guider_errors(self):
        """GDR{X,Y,R}RMS/GDR{X,Y}BIAS: guiding error RMS and bias [mas].

        Ports v2.12 ``AnalyzeGuider.measure_guider_errors``: per frame, the
        target position minus the measured centroid, scaled by the 0.056 arcsec
        CRED-2 pixel; R is the radial combination of the two axes. Fewer than 11
        distinct centroid positions means the guide camera was not tracking, so
        no keyword is emitted and GUIDEROK fails on their absence.
        """
        table = self._guider_frames()
        x_mas = (table["target_x"] - table["object1_x"]) * 56.0
        y_mas = (table["target_y"] - table["object1_y"]) * 56.0
        unique = min(
            np.unique(table["object1_x"]).size, np.unique(table["object1_y"]).size
        )
        if unique <= 10:
            return {}
        return self._tag(
            GDRXRMS=round(float(np.nanmean(x_mas**2) ** 0.5), 6),
            GDRYRMS=round(float(np.nanmean(y_mas**2) ** 0.5), 6),
            GDRRRMS=round(float(np.nanmean(x_mas**2 + y_mas**2) ** 0.5), 6),
            GDRXBIAS=round(float(np.nanmean(x_mas)), 6),
            GDRYBIAS=round(float(np.nanmean(y_mas)), 6),
        )

    guider_errors._diag_name = "guider_errors"

    def guider_image_stats(self):
        """GDR{FW,FX,PK}{MD,STD}: per-frame guider FWHM [mas] and flux [ADU].

        Ports v2.12 ``AnalyzeGuider``: the median and standard deviation across
        frames of the fitted stellar FWHM, the object flux and its peak. FWHM
        combines the two Gaussian axes the guide camera fits, in pixels, at the
        0.056 arcsec CRED-2 pixel; v2.12 divided by the pixel scale rather than
        multiplying, so its values were not the mas it labeled them.
        """
        table = self._guider_frames()
        fwhm = (
            (
                np.asarray(table["object1_a"], dtype=float) ** 2
                + np.asarray(table["object1_b"], dtype=float) ** 2
            )
            ** 0.5
            * (2 * (2 * np.log(2)) ** 0.5)
            * 56.0
        )
        values = {}
        for prefix, column in (
            ("GDRFW", fwhm),
            ("GDRFX", table["object1_flux"]),
            ("GDRPK", table["object1_peak"]),
        ):
            column = np.asarray(column, dtype=float)
            values[f"{prefix}MD"] = round(float(np.median(column)), 6)
            values[f"{prefix}STD"] = round(float(np.std(column)), 6)
        return self._tag(**values)

    guider_image_stats._diag_name = "guider_image_stats"

    def guider_seeing(self):
        """GDRSEEJZ, GDRSEEV: seeing [arcsec] from a Moffat fit to GUIDER_AVG.

        Ports v2.12 ``AnalyzeGuider.measure_seeing``: a 2D Moffat profile fit to
        the median-subtracted co-added guider image, whose alpha is the seeing at
        the guide camera's 950-1200 nm band. The fit is seeded at three widths
        spanning 0.4-2.5 arcsec, centred on the guider reference pixel, and the
        smallest-residual seed wins; a fit that never converges emits no keyword.
        GDRSEEV rescales that alpha from the band midpoint to V by the Kolmogorov
        lambda^(1/5) law, both cards deriving from the unrounded fit.
        """
        image = self.kpf_obj.data["GUIDER_AVG"]
        flat = np.asarray(image, dtype=float).ravel()
        flat = flat - np.median(flat)
        y, x = np.indices(image.shape)
        xy = (x.ravel(), y.ravel())

        def moffat(coords, amplitude, x0, y0, alpha, beta):
            px, py = coords
            return (
                amplitude * (1 + ((px - x0) ** 2 + (py - y0) ** 2) / alpha**2) ** -beta
            )

        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        center = (float(hdr.get("GCCRPIX1", 343.1)), float(hdr.get("GCCRPIX2", 264.7)))
        best, smallest = None, np.inf
        for alpha in (0.4 / 0.056, 1.0 / 0.056, 2.5 / 0.056):
            try:
                popt, _ = curve_fit(moffat, xy, flat, p0=[1, *center, alpha, 2.5])
            except (RuntimeError, ValueError) as e:
                logger.debug("guider seeing fit failed at alpha=%.1f px: %s", alpha, e)
                continue
            residuals = float(np.sum((flat - moffat(xy, *popt)) ** 2))
            if residuals < smallest:
                best, smallest = popt, residuals
        if best is None:
            return {}
        seeing = abs(float(best[3])) * 0.056
        return self._tag(
            GDRSEEJZ=round(seeing, 6),
            GDRSEEV=round(seeing * ((1200 + 950) / 2 / 550) ** 0.2, 6),
        )

    guider_seeing._diag_name = "guider_seeing"

    def guider_saturation(self):
        """GDRNSAT/GDRFRSAT: saturated guider pixels and saturated-frame fraction.

        Ports v2.12 ``AnalyzeGuider``: the CRED-2 saturates at 15830 ADU and both
        metrics are taken at 90% of it. GDRNSAT counts pixels in the central
        100x100 box of the co-added GUIDER_AVG, where the target sits; GDRFRSAT
        is the fraction of frames whose brightest object pixel is saturated.
        """
        level = 0.9 * 15830
        image = self.kpf_obj.data["GUIDER_AVG"]
        peak = np.asarray(self._guider_frames()["object1_peak"], dtype=float)
        return self._tag(
            GDRNSAT=int(np.count_nonzero(image[205:305, 270:370] > level)),
            GDRFRSAT=round(float(np.count_nonzero(peak > level) / len(peak)), 6),
        )

    guider_saturation._diag_name = "guider_saturation"

    def _expmeter_flux(self, ext):
        """One EM fiber's channel wavelengths [nm] and raw flux, readings x channels.

        The numeric column labels are the wavelength channels, in nm at L0 --
        ImageAssembly renames them to Angstroms only at the L0 -> L1 boundary. The
        Date* columns are not channels.
        """
        table = self.kpf_obj.data[ext]
        waves, channels = [], []
        for name in table.colnames:
            try:
                wave = float(name)
            except ValueError:
                continue
            waves.append(wave)
            channels.append(np.asarray(table[name], dtype=float))
        return np.array(waves), np.column_stack(channels)

    @staticmethod
    def _longest_run(mask):
        """Longest run of adjacent True values in a 1D channel mask."""
        longest = run = 0
        for flagged in mask:
            run = run + 1 if flagged else 0
            longest = max(longest, run)
        return longest

    def expmeter_channel_metrics(self):
        """EM{SCI,SKY}{SAT,NEG,INF}: per-fiber exposure meter channel metrics.

        Ports v2.12 ``EM_not_saturated`` and ``EM_flux_not_negative``, which judge
        each fiber on its own. SAT is saturated elements per reading -- elements
        above 90% of the 1.93e6 reduced-spectrum saturation level, over the
        interior readings (the first and last are partial and are dropped when
        there are 3+) -- the form v2.12 gates at 1.5. NEG is the longest run of
        adjacent channels whose time-summed flux is negative, the signature of
        bias over-subtraction in the raw EM images; INF is the same run length for
        channels holding a non-finite reading.
        """
        values = {}
        for ext, fiber in (("EXPMETER_SCI", "SCI"), ("EXPMETER_SKY", "SKY")):
            _, flux = self._expmeter_flux(ext)
            interior = flux[1:-1] if len(flux) >= 3 else flux
            values[f"EM{fiber}SAT"] = round(
                float(np.count_nonzero(interior > 0.9 * 1.93e6) / len(interior)), 6
            )
            values[f"EM{fiber}NEG"] = self._longest_run(flux.sum(axis=0) < 0)
            values[f"EM{fiber}INF"] = self._longest_run(~np.isfinite(flux).all(axis=0))
        return self._tag(**values)

    expmeter_channel_metrics._diag_name = "expmeter_channel_metrics"

    def expmeter_counts(self):
        """EM{SC,SK}CT{48,45,56,67,78}: cumulative EM counts [ADU] per band.

        Ports v2.12 ``AnalyzeEM``: raw counts summed over every reading and over
        the channels of each band, per fiber. The 445-870 nm total spans the EM's
        full range and the four sub-bands partition it at v2.12's 551.25, 657.50
        and 763.75 nm edges, so the sub-bands always add up to the total.
        """
        values = {}
        for ext, fiber in (("EXPMETER_SCI", "SC"), ("EXPMETER_SKY", "SK")):
            waves, flux = self._expmeter_flux(ext)
            per_channel = np.nansum(flux, axis=0)
            for band, mask in (
                ("48", (waves >= 445.0) & (waves < 870.0)),
                ("45", (waves >= 445.0) & (waves < 551.25)),
                ("56", (waves >= 551.25) & (waves < 657.50)),
                ("67", (waves >= 657.50) & (waves < 763.75)),
                ("78", (waves >= 763.75) & (waves < 870.0)),
            ):
                values[f"EM{fiber}CT{band}"] = int(np.nansum(per_channel[mask]))
        return self._tag(**values)

    expmeter_counts._diag_name = "expmeter_counts"

    def sky_sci_flux_ratio(self):
        """SKYSCIMS: SKY/SCI flux ratio in the main spectrometer, scaled from EM.

        Ports v2.12 ``AnalyzeEM.SKY_SCI_main_spectrometer``: total SKY counts over
        total SCI counts, the SKY side divided by the 14.1 SKY-to-SCI flux ratio
        measured on bright twilight observations.
        """
        sci = np.nansum(self._expmeter_flux("EXPMETER_SCI")[1])
        sky = np.nansum(self._expmeter_flux("EXPMETER_SKY")[1])
        return self._tag(SKYSCIMS=round(float(sky / 14.1 / sci), 6))

    sky_sci_flux_ratio._diag_name = "sky_sci_flux_ratio"
