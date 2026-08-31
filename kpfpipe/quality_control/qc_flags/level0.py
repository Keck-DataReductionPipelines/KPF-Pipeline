"""QC checks for KPF Level 0 (raw CCD) data products."""

import os
import re
from datetime import datetime

import numpy as np

from kpfpipe import DETECTOR
from kpfpipe.quality_control.qc_flags.base import QC
from kpfpipe.utils.io import load_junk_obs_ids

_CHIPS = ["GREEN", "RED"]
_SUPPORTED_NAMP = (2, 4)  # valid KPF readout modes (see ImageAssembly.count_amplifiers)


class QCL0(QC):
    """QC checks for KPF Level 0 raw data products."""

    LEVEL = "L0"

    def data_l0_red_green(self):
        """Raw CCD data present: each of GREEN/RED is a supported amp readout.

        KPF reads out 2 or 4 amplifiers per chip (``_SUPPORTED_NAMP``), mirroring
        ``ImageAssembly.count_amplifiers``. Each present amp must match the shape
        ``ImageAssembly.dims`` implies for that readout mode (prescan/overscan
        included) and hold at least one finite value, so a truncated readout or
        an all-NaN placeholder is not mistaken for good data.
        """
        ccd = DETECTOR["ccd"]
        for chip in _CHIPS:
            amps = []
            for i in range(1, 5):  # GREEN_AMP1..4 / RED_AMP1..4
                arr = self.kpf_obj.data.get(f"{chip}_AMP{i}")
                # KPF0 stores None-data as array(None, dtype=object); skip absent.
                if (
                    arr is None
                    or getattr(arr, "dtype", None) == np.dtype(object)
                    or np.size(arr) == 0
                ):
                    continue
                amps.append(arr)
            if len(amps) not in _SUPPORTED_NAMP:
                return False
            nrow = ccd["nrow"] // (2 if len(amps) == 4 else 1) + ccd["oscan_prl"]
            ncol = ccd["ncol"] // 2 + ccd["prescan"] + ccd["oscan_srl"]
            if any(
                arr.shape != (nrow, ncol) or not np.any(np.isfinite(arr))
                for arr in amps
            ):
                return False
        return True

    data_l0_red_green._qc_key = "DATAPRL0"

    def header_keywords_present(self):
        """Required PRIMARY keywords exist -- not yet implemented; see ``QC``."""
        raise NotImplementedError(
            "KWRDPRL0 is pending a KPF-owned definition of a required keyword"
        )

    header_keywords_present._qc_key = "KWRDPRL0"

    def telemetry_present(self):
        """TELEMETRY extension present and populated.

        Populated to the DATAPRL0 standard: rows carrying at least one finite
        average, so an all-NaN placeholder is not mistaken for a recording. The
        bench thermal and vacuum state during the exposure is recorded nowhere
        else and cannot be recovered once the frame is written.
        """
        table = self.kpf_obj.data.get("TELEMETRY", [])
        if len(table) == 0:
            return False
        return bool(np.any(np.isfinite(np.asarray(table["average"], dtype=float))))

    telemetry_present._qc_key = "TELEPR"

    def cahk_present(self):
        """CA_HK extension present and populated.

        Populated to the DATAPRL0 standard: a non-empty image carrying at least
        one finite value, so an all-NaN placeholder is not mistaken for a
        readout. The Ca H&K image is the only record of the chromospheric
        activity indicator for the exposure and cannot be recovered once the
        frame is written.
        """
        image = self.kpf_obj.data.get("CA_HK", [])
        return np.size(image) > 0 and bool(np.any(np.isfinite(image)))

    cahk_present._qc_key = "CAHKPR"

    def _expmeter_populated(self, ext):
        """One EM fiber table present, carrying readings and finite flux.

        The wavelength channels are the numerically-labeled columns, so a table of
        Date* columns alone, one with no readings, or one whose flux is entirely
        non-finite is not a readout.
        """
        table = self.kpf_obj.data.get(ext)
        if table is None or len(table) == 0:
            return False
        for name in table.colnames:
            try:
                float(name)
            except ValueError:
                continue
            if np.any(np.isfinite(np.asarray(table[name], dtype=float))):
                return True
        return False

    def expmeter_sci_present(self):
        """EXPMETER_SCI present and populated."""
        return self._expmeter_populated("EXPMETER_SCI")

    expmeter_sci_present._qc_key = "EMSCIPR"

    def expmeter_sky_present(self):
        """EXPMETER_SKY present and populated."""
        return self._expmeter_populated("EXPMETER_SKY")

    expmeter_sky_present._qc_key = "EMSKYPR"

    def times_consistent(self):
        """DATE-BEG <= DATE-MID <= DATE-END, matching ELAPSED and the shutters.

        The exposure-meter half of this check is EMTIMEOK. The shutter window
        must agree with ELAPSED to 0.1 s, and each per-chip shutter time must
        fall within 0.1 s of the window edge it bounds; mismatched chips have
        different photon-weighted midpoints, so one barycentric correction
        cannot serve both.

        At L0 the raw instrument times live on the WMKO-native PRIMARY (the
        header later snapshotted verbatim into INSTRUMENT_HEADER at to_kpf1); the
        0/1 flag then propagates downstream on QUALITY_CONTROL.
        """
        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        beg, mid, end = (
            datetime.fromisoformat(str(hdr[k]))
            for k in ("DATE-BEG", "DATE-MID", "DATE-END")
        )
        if not beg <= mid <= end:
            return False
        if abs((end - beg).total_seconds() - float(hdr["ELAPSED"])) > 0.1:
            return False
        for key, edge in (
            ("GRDATE-B", beg),
            ("GRDATE-E", end),
            ("RDDATE-B", beg),
            ("RDDATE-E", end),
        ):
            shutter = datetime.fromisoformat(str(hdr[key]))
            if abs((edge - shutter).total_seconds()) > 0.1:
                return False
        return True

    times_consistent._qc_key = "DATTIMOK"

    def ntp_timing(self):
        """NTP reports the host clock correct to better than 100 ms.

        DATTIMOK and EXPTIMOK check only that the exposure timestamps are
        self-consistent, which a uniformly offset clock still satisfies. TIMEERR
        is free text ("NTP time correct to within 12.3 ms"); text that does not
        report an error, or one at/above the limit, fails.
        """
        timeerr = str(self.kpf_obj.headers["INSTRUMENT_HEADER"]["TIMEERR"])
        match = re.search(r"NTP time correct to within ([\d.]+) ms", timeerr)
        return match is not None and float(match.group(1)) < 100.0

    ntp_timing._qc_key = "NTPOK"

    def exptime_sane(self):
        """EXPTIME present, finite, non-negative, and consistent with ELAPSED.

        Bias frames legitimately have EXPTIME=0, so we don't require strictly
        positive. The raw ELAPSED readout time must not fall short of the
        requested EXPTIME (premature readout) or exceed it by more than 0.1 s
        (the elapsed-vs-requested check formerly done in the masters frame
        loader).
        """
        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        exptime = float(hdr["EXPTIME"])
        elapsed = float(hdr["ELAPSED"])
        if not (np.isfinite(exptime) and exptime >= 0):
            return False
        return 0 <= elapsed - exptime <= 0.1

    exptime_sane._qc_key = "EXPTIMOK"

    def good_readout(self):
        """The CCD read out cleanly rather than smearing.

        A readout that aborts partway leaves ELAPSED between 6.0 and 6.7 s
        regardless of the requested EXPTIME, smearing the frame; it happens a few
        times a day on both cals and stars. Requests shorter than 7 s
        legitimately land in that window, so only longer ones are judged.
        """
        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        return not (
            float(hdr["EXPTIME"]) >= 7.0 and 6.0 <= float(hdr["ELAPSED"]) <= 6.7
        )

    good_readout._qc_key = "READOK"

    def _ccd_temp_ok(self, key):
        """One CCD held within 10 mK of its temperature setpoint.

        DiagL0 measures the signed offset; this applies the limit to its
        magnitude. Detector temperature drift moves the spectrum on the chip, so
        it bears directly on RV stability.
        """
        return abs(float(self.kpf_obj.headers["QUALITY_CONTROL"][key])) < 10.0

    def green_ccd_temp_ok(self):
        """GREEN CCD at its temperature setpoint."""
        return self._ccd_temp_ok("GTEMPOFF")

    green_ccd_temp_ok._qc_key = "GTEMPOK"

    def red_ccd_temp_ok(self):
        """RED CCD at its temperature setpoint."""
        return self._ccd_temp_ok("RTEMPOFF")

    red_ccd_temp_ok._qc_key = "RTEMPOK"

    def guiding_ok(self):
        """Guiding tracked to spec and the guide camera was not saturated.

        The guiding error must hold to 50 mas in both X/Y RMS and bias, at most 3
        pixels of the co-added image saturated, and at most 10% of frames
        carrying a saturated peak. DiagL0 measures all six; this applies the
        limits. Bias is judged on magnitude, not signed value.
        """
        hdr = self.kpf_obj.headers["QUALITY_CONTROL"]
        if any(float(hdr[key]) > 50.0 for key in ("GDRXRMS", "GDRYRMS")):
            return False
        if any(abs(float(hdr[key])) > 50.0 for key in ("GDRXBIAS", "GDRYBIAS")):
            return False
        return int(hdr["GDRNSAT"]) <= 3 and float(hdr["GDRFRSAT"]) <= 0.1

    guiding_ok._qc_key = "GUIDEROK"

    def seeing_ok(self):
        """V-band seeing under 1 arcsec.

        A frame whose guider Moffat fit never converged carries no GDRSEEV and
        fails.
        """
        return float(self.kpf_obj.headers["QUALITY_CONTROL"]["GDRSEEV"]) < 1.0

    seeing_ok._qc_key = "SEEINGOK"

    def elevation_ok(self):
        """Telescope above 30 deg, the atmospheric dispersion corrector's range.

        Below 30 deg the ADC runs out of travel, so the fiber samples a
        wavelength-dependent position on the sky and the measured RV is biased.
        """
        return float(self.kpf_obj.headers["INSTRUMENT_HEADER"]["EL"]) >= 30.0

    elevation_ok._qc_key = "ELEVOK"

    def etalon_at_temp(self):
        """Etalon chambers within 0.5 mK of their setpoints.

        DiagL0 measures the signed offset of the chamber furthest from its
        setpoint; this applies the limit to its magnitude. The etalon line
        positions shift with temperature, so an off-setpoint chamber corrupts the
        drift reference.
        """
        return abs(float(self.kpf_obj.headers["QUALITY_CONTROL"]["ETATOFF"])) <= 0.5

    etalon_at_temp._qc_key = "ETATMPOK"

    def agitator_operating(self):
        """Agitator running above its minimum speed.

        AGITSTA must report Running and the exposure-average kpfmot.AGITSPD must
        exceed 1000 counts/s. The agitator scrambles the fiber's modal noise; a
        stalled one leaves that noise in the spectrum.
        """
        if str(self.kpf_obj.headers["INSTRUMENT_HEADER"]["AGITSTA"]) != "Running":
            return False
        table = self.kpf_obj.data["TELEMETRY"]
        speed = table[table["keyword"] == "kpfmot.AGITSPD"]["average"][0]
        return abs(float(speed)) > 1000.0

    agitator_operating._qc_key = "AGITOK"

    def not_junk(self):
        """obs_id not on the observer junk list for this frame's data tree.

        "Junk" is a manual observer flag (e.g. the wrong telescope settings) that
        no automated QC can catch. The list lives at
        ``{KPF_DATA_INPUT}/vNext/reference/junk_obs.csv``;
        KPF_DATA_INPUT is recovered from the frame's own source directory, which
        rvdata records as ``self.dirname`` (``{KPF_DATA_INPUT}/L0/{datecode}``)
        when the L0 is read. An absent list yields not-junk
        (``load_junk_obs_ids`` returns the empty set); ``dirname`` is set on every
        L0 read, so a missing one signals a broken upstream invariant.
        """
        data_input = os.path.dirname(os.path.dirname(self.kpf_obj.dirname))
        return self.kpf_obj.obs_id not in load_junk_obs_ids(data_input)

    not_junk._qc_key = "NOTJUNK"

    def radec_consistent(self):
        """Pointing agrees with the target and catalog positions.

        TCSOFF (pointing vs the DCS target) is internal telescope-pointing
        consistency and is required. OBJOFF/GAIAOFF are external catalog
        cross-matches with a looser 5" bound; Gaia and SIMBAD are optional
        sources, so DiagL0 emits their offsets only when the lookup ran and
        matched, and each is checked only where present.
        """
        hdr = self.kpf_obj.headers["QUALITY_CONTROL"]
        if float(hdr["TCSOFF"]) >= 1.0:
            return False
        return all(float(hdr[key]) < 5.0 for key in ("OBJOFF", "GAIAOFF") if key in hdr)

    radec_consistent._qc_key = "TARGETOK"

    @staticmethod
    def _row_float(row, field):
        """A CATALOG_RECORD numeric cell as float, or None for an absent (NaN) one."""
        value = float(row[field])
        return None if np.isnan(value) else value

    def catalog_astrometry_sane(self):
        """Canonical CATALOG_RECORD astrometry values are physically plausible.

        Range checks apply to the merged ``kpf-drp`` row AstroQuery resolves (the
        astrometry feeding the barycentric correction), not the raw WMKO TARG*
        keywords. Each field is checked only when present: epoch/equinox in
        (1950, 2050] Julian years, |rv| <= 350 km/s, parallax in (0, 1000) mas,
        |pmra|/|pmdec| <= 15 arcsec/yr.

        AstroQuery must have resolved the frame: an absent ``kpf-drp`` row raises.
        """
        table = self.kpf_obj.data["CATALOG_RECORD"]
        row = table[table["source"] == "kpf-drp"][0]
        for field in ("epoch", "equinox"):
            val = self._row_float(row, field)
            if val is not None and not (1950.0 < val <= 2050.0):
                return False
        # |rv| bound from Chubak et al. 2012 (arXiv:1207.6212, Fig. 8).
        rv = self._row_float(row, "rv")
        if rv is not None and abs(rv) > 350.0:
            return False
        plax = self._row_float(row, "parallax")
        if plax is not None and not (0.0 < plax < 1000.0):
            return False
        for field in ("pmra", "pmdec"):
            pm = self._row_float(row, field)
            if pm is not None and abs(pm) > 15.0:
                return False
        return True

    catalog_astrometry_sane._qc_key = "ASTROMOK"

    def catalog_color_sane(self):
        """The canonical CATALOG_RECORD color is present and on the stellar sequence.

        CrossCorrelation picks the stellar line mask by turning this color into an
        effective temperature, so an absent, unlabeled, or off-sequence color is
        caught here rather than at L4. Both ``color`` and ``color_name`` must be
        present, the label must be one AstroQuery emits, and the value must lie in
        the range that index spans across the Pecaut & Mamajek (2013) dwarf sequence
        -- O3V through Y4V, so a color outside it is not a stellar color.

        AstroQuery must have resolved the frame, as ``catalog_astrometry_sane``
        requires: an absent ``kpf-drp`` row raises.
        """
        # (bluest, reddest) [mag] each index spans across the sequence, keyed by the
        # labels AstroQuery writes to color_name.
        limits = {
            "B-V": (-0.33, 2.17),
            "Gaia BP-RP": (-0.12, 5.10),
            "G-J": (-0.36, 5.36),
        }
        table = self.kpf_obj.data["CATALOG_RECORD"]
        row = table[table["source"] == "kpf-drp"][0]
        color = self._row_float(row, "color")
        bounds = limits.get(str(row["color_name"]))
        if color is None or bounds is None:
            return False
        return bounds[0] <= color <= bounds[1]

    catalog_color_sane._qc_key = "COLOROK"

    def expmeter_times_consistent(self):
        """EXPMETER_SCI brackets the shutter window to within 1 second.

        The header-date half of this check is DATTIMOK/EXPTIMOK. The first
        Date-Beg and last Date-End of the EM table must match PRIMARY
        DATE-BEG/DATE-END, preferring the corrected columns as
        BarycentricCorrection does. The 1 s tolerance absorbs EM dead time and
        catches only gross errors: the flux-weighted midpoint feeds the
        barycentric correction, where a 2.6 s timing error costs 10 cm/s.
        """
        table = self.kpf_obj.data["EXPMETER_SCI"]
        hdr = self.kpf_obj.headers["INSTRUMENT_HEADER"]
        beg, end = (
            datetime.fromisoformat(str(hdr[k])) for k in ("DATE-BEG", "DATE-END")
        )
        suffix = "-Corr" if "Date-Beg-Corr" in table.colnames else ""
        em_beg = datetime.fromisoformat(str(table[f"Date-Beg{suffix}"][0]))
        em_end = datetime.fromisoformat(str(table[f"Date-End{suffix}"][-1]))
        return (
            abs((em_beg - beg).total_seconds()) <= 1.0
            and abs((em_end - end).total_seconds()) <= 1.0
        )

    expmeter_times_consistent._qc_key = "EMTIMEOK"

    def expmeter_flux_sane(self):
        """EXPMETER flux is neither saturated, negative, nor non-finite.

        Fails on either fiber. DiagL0 measures each fiber; this applies the
        limits: at most 1.5 saturated elements per reading, and no run of 20
        adjacent negative channels. Non-finite channels are gated the same way as
        negative.
        """
        hdr = self.kpf_obj.headers["QUALITY_CONTROL"]
        for fiber in ("SCI", "SKY"):
            if float(hdr[f"EM{fiber}SAT"]) > 1.5:
                return False
            if int(hdr[f"EM{fiber}NEG"]) >= 20 or int(hdr[f"EM{fiber}INF"]) >= 20:
                return False
        return True

    expmeter_flux_sane._qc_key = "EMFLUXOK"

    def _chip_pixels_ok(self, dead_key, sat_key):
        """Both raw pixel-quality fractions of a chip within their limits.

        At most 5% of any amp below 1.0e4 D.N. (dead) and at most 15% above
        5.0e8 D.N. (saturated). DiagL0 measures the fractions; this only applies
        the limits.
        """
        hdr = self.kpf_obj.headers["QUALITY_CONTROL"]
        return float(hdr[dead_key]) <= 0.05 and float(hdr[sat_key]) <= 0.15

    def green_pixels_ok(self):
        """GREEN raw pixel quality: neither dead nor saturated beyond the limits."""
        return self._chip_pixels_ok("DEADPXFG", "SATPXFG")

    green_pixels_ok._qc_key = "GREENL0"

    def red_pixels_ok(self):
        """RED raw pixel quality: neither dead nor saturated beyond the limits."""
        return self._chip_pixels_ok("DEADPXFR", "SATPXFR")

    red_pixels_ok._qc_key = "REDL0"
