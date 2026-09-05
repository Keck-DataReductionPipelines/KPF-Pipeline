"""Diagnostics for the KPF Level 0 guide camera extensions."""

import numpy as np
from scipy.optimize import curve_fit

from kpfpipe.quality_control.diagnostics.base import Diagnostics
from kpfpipe.utils.stats import flag_outliers, interpolate_bad_pixels


class Guider(Diagnostics):
    """Diagnostics from the GUIDER_AVG image and the GUIDER_CUBE_ORIGINS table.

    Also carries the two observing conditions the guiding is judged against:
    SEEING, which is this class's own V-band measurement, and AIRMASS.
    """

    LEVEL = "L0"

    def _guider_frames(self):
        """GUIDER_CUBE_ORIGINS rows, the unwritten ones dropped.

        A cube that recorded flux carries trailing rows with a zero timestamp and
        zero flux; those are dropped, but an all-zero-flux cube is kept whole so
        the emptiness stays visible to the metrics.
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

        Per frame, the target position minus the measured centroid, scaled by
        the 0.056 arcsec CRED-2 pixel; R is the radial combination of the two
        axes. Fewer than 11 distinct centroid positions means the guide camera
        was not tracking, so no keyword is emitted and GUIDEROK fails on their
        absence.
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

        Median and standard deviation across frames of the fitted stellar FWHM,
        the object flux and its peak. FWHM combines the two Gaussian axes the
        guide camera fits, in pixels, at the 0.056 arcsec CRED-2 pixel.
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
        """GDRSEEJZ, GDRSEEV, SEEING: seeing [arcsec] from a Moffat fit to GUIDER_AVG.

        The fitted alpha is the seeing in the guide camera's 950-1200 nm band;
        GDRSEEV rescales it to V by the Kolmogorov lambda^(1/5) law, and SEEING
        is that same V-band value under its EPRV name, routed to PRIMARY.
        """
        beta = 2.5
        image = np.asarray(self.kpf_obj.data["GUIDER_AVG"], dtype=float)
        outliers = flag_outliers(image, 5.0, kernel_size=3, method="trend")
        image = interpolate_bad_pixels(image, ~outliers)
        flat = image.ravel() - np.median(image)
        y, x = np.indices(image.shape)
        xy = (x.ravel(), y.ravel())

        def moffat(coords, amplitude, x0, y0, alpha, beta):
            px, py = coords
            return (
                amplitude * (1 + ((px - x0) ** 2 + (py - y0) ** 2) / alpha**2) ** -beta
            )

        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        center = (float(hdr.get("GCCRPIX1", 343.1)), float(hdr.get("GCCRPIX2", 264.7)))
        # Seed alpha from the area above half the peak, the Moffat half-max radius.
        peak = float(np.percentile(flat, 99.9))
        half_width = (np.count_nonzero(flat > 0.5 * peak) / np.pi) ** 0.5
        alpha = half_width / (2 ** (1 / beta) - 1) ** 0.5

        popt, _ = curve_fit(moffat, xy, flat, p0=[peak, *center, alpha, beta])
        seeing = abs(float(popt[3])) * 0.056
        v_band = round(seeing * ((1200 + 950) / 2 / 550) ** 0.2, 6)
        return self._tag(GDRSEEJZ=round(seeing, 6), GDRSEEV=v_band, SEEING=v_band)

    guider_seeing._diag_name = "guider_seeing"

    def guider_saturation(self):
        """GDRNSAT/GDRFRSAT: saturated guider pixels and saturated-frame fraction.

        The CRED-2 saturates at 15830 ADU and both metrics are taken at 90% of
        it. GDRNSAT counts pixels in the central 100x100 box of the co-added
        GUIDER_AVG, where the target sits; GDRFRSAT is the fraction of frames
        whose brightest object pixel is saturated.
        """
        level = 0.9 * 15830
        image = self.kpf_obj.data["GUIDER_AVG"]
        peak = np.asarray(self._guider_frames()["object1_peak"], dtype=float)
        return self._tag(
            GDRNSAT=int(np.count_nonzero(image[205:305, 270:370] > level)),
            GDRFRSAT=round(float(np.count_nonzero(peak > level) / len(peak)), 6),
        )

    guider_saturation._diag_name = "guider_saturation"

    def airmass(self):
        """AIRMASS: the airmass the DCS recorded at mid-exposure [secZ]."""
        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        return self._tag(AIRMASS=round(float(hdr["AIRMASS"]), 6))

    airmass._diag_name = "airmass"
