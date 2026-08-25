"""Diagnostics for KPF Level 0 (raw CCD) data products."""

import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from kpfpipe.quality_control.diagnostics.base import Diagnostics

logger = logging.getLogger(__name__)


class DiagL0(Diagnostics):
    """Diagnostics for KPF Level 0 raw data products.

    The pointing-offset metrics compare the telescope pointing against the target
    astrometry AstroQuery resolves into the L0 ``CATALOG_RECORD`` extension, so
    AstroQuery must have run. The DCS target offset is required; the Gaia and
    SIMBAD offsets are emitted only when their catalog lookup matched.
    """

    LEVEL = "L0"

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
        hdr = self.kpf_obj.headers["PRIMARY"]
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

    def _expmeter_flux(self, ext):
        """One EM fiber's raw flux, readings x wavelength channels.

        Numeric column labels are the wavelength channels; the Date* columns are
        not.
        """
        table = self.kpf_obj.data[ext]
        channels = []
        for name in table.colnames:
            try:
                float(name)
            except ValueError:
                continue
            channels.append(np.asarray(table[name], dtype=float))
        return np.column_stack(channels)

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
            flux = self._expmeter_flux(ext)
            interior = flux[1:-1] if len(flux) >= 3 else flux
            values[f"EM{fiber}SAT"] = round(
                float(np.count_nonzero(interior > 0.9 * 1.93e6) / len(interior)), 6
            )
            values[f"EM{fiber}NEG"] = self._longest_run(flux.sum(axis=0) < 0)
            values[f"EM{fiber}INF"] = self._longest_run(~np.isfinite(flux).all(axis=0))
        return self._tag(**values)

    expmeter_channel_metrics._diag_name = "expmeter_channel_metrics"
