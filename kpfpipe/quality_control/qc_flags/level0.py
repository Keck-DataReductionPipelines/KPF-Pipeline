"""QC checks for KPF Level 0 (raw CCD) data products."""

import os
from datetime import datetime

import numpy as np

from kpfpipe.quality_control.qc_flags.base import QC
from kpfpipe.utils.io import load_junk_obs_ids

_L0_REQUIRED_KEYS = ["DATE-OBS", "EXPTIME", "OBJECT", "OFNAME", "IMTYPE"]

_CHIPS = ("GREEN", "RED")
_SUPPORTED_NAMP = (2, 4)  # valid KPF readout modes (see ImageAssembly.count_amplifiers)


def _parse_iso(value):
    """Parse an ISO-8601 datetime string, or None if missing/unparseable."""
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


class QCL0(QC):
    """QC checks for KPF Level 0 raw data products."""

    LEVEL = "L0"

    def data_l0_red_green(self):
        """Raw CCD data present: each of GREEN/RED is a supported amp readout.

        KPF reads out either 2 or 4 amplifiers per chip, so the amp count is
        inferred from the data: a chip passes when its number of present,
        non-empty amplifier extensions is a supported readout mode
        (``_SUPPORTED_NAMP``), mirroring ``ImageAssembly.count_amplifiers``. A
        chip with no data or a partial/invalid amp set (1 or 3) fails.
        """
        for chip in _CHIPS:
            namp = 0
            for i in range(1, 5):  # GREEN_AMP1..4 / RED_AMP1..4
                arr = self.kpf_obj.data.get(f"{chip}_AMP{i}")
                # KPF0 stores None-data as array(None, dtype=object); skip absent.
                if (
                    arr is None
                    or getattr(arr, "dtype", None) == np.dtype(object)
                    or np.size(arr) == 0
                ):
                    continue
                namp += 1
            if namp not in _SUPPORTED_NAMP:
                return False
        return True

    data_l0_red_green._qc_key = "DATAPRL0"

    def header_keywords_present(self):
        """Required PRIMARY keywords exist."""
        hdr = self.kpf_obj.headers["PRIMARY"]
        return all(k in hdr for k in _L0_REQUIRED_KEYS)

    header_keywords_present._qc_key = "KWRDPRL0"

    def times_consistent(self):
        """DATE-BEG <= DATE-MID <= DATE-END.

        Ports v2.12 ``L2_datetime``. At L0 the raw instrument times live on the
        WMKO-native PRIMARY (the header later snapshotted verbatim into
        INSTRUMENT_HEADER at to_kpf1); the 0/1 flag then propagates downstream
        on QUALITY_CONTROL.
        """
        hdr = self.kpf_obj.headers["PRIMARY"]
        beg, mid, end = (
            _parse_iso(hdr.get(k)) for k in ("DATE-BEG", "DATE-MID", "DATE-END")
        )
        if beg is None or mid is None or end is None:
            return False
        return beg <= mid <= end

    times_consistent._qc_key = "DATTIMOK"

    def exptime_sane(self):
        """EXPTIME present, finite, non-negative, and consistent with ELAPSED.

        Bias frames legitimately have EXPTIME=0, so we don't require strictly
        positive. When the raw ELAPSED readout time is present, it must not fall
        short of the requested EXPTIME (premature readout) or exceed it by more
        than 0.1 s (the elapsed-vs-requested check formerly done in the masters
        frame loader); an absent ELAPSED skips only that comparison.
        """
        hdr = self.kpf_obj.headers["PRIMARY"]
        if "EXPTIME" not in hdr:
            return False
        try:
            exptime = float(hdr.get("EXPTIME"))
        except (TypeError, ValueError):
            return False
        if not (np.isfinite(exptime) and exptime >= 0):
            return False

        elapsed = self._hdr_float(hdr, "ELAPSED")
        if elapsed is not None and not (0 <= elapsed - exptime <= 0.1):
            return False
        return True

    exptime_sane._qc_key = "EXPTIMOK"

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

        TARGOFF (pointing vs the DCS target) is internal telescope-pointing
        consistency and is required: an empty value (astrometry unavailable) or
        one >= 1" fails. OBJOFF/GAIAOFF are external catalog cross-matches with a
        looser 5" bound, checked only when present-and-valued, so a disabled or
        failed Gaia/SIMBAD lookup passes.
        """
        hdr = self.kpf_obj.headers["QUALITY_CONTROL"]
        targoff = self._hdr_float(hdr, "TARGOFF")
        if targoff is None or targoff >= 1.0:
            return False
        for key, limit in (("OBJOFF", 5.0), ("GAIAOFF", 5.0)):
            val = self._hdr_float(hdr, key)
            if val is not None and val >= limit:
                return False
        return True

    radec_consistent._qc_key = "RADECOK"

    @staticmethod
    def _row_float(row, field):
        """A CATALOG_RECORD numeric cell as float, or None for an absent (NaN) one."""
        value = float(row[field])
        return None if np.isnan(value) else value

    def catalog_astrometry_sane(self):
        """Canonical CATALOG_RECORD astrometry values are physically plausible.

        Ports the v2.12 good_TARG_headers range checks onto the merged ``kpf-drp``
        row AstroQuery resolves (the astrometry feeding the barycentric correction),
        not the raw WMKO TARG* keywords. Each field is checked only when present:
        epoch/equinox in (1950, 2050] Julian years, |rv| <= 350 km/s, parallax in
        (0, 1000) mas, |pmra|/|pmdec| <= 15 arcsec/yr.

        Passes when there is no ``kpf-drp`` row: this is value sanity, not a presence
        check.
        """
        table = self.kpf_obj.data["CATALOG_RECORD"]
        match = table[table["source"] == "kpf-drp"] if table.colnames else table
        if not len(match):
            return True
        row = match[0]
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

        Passes when there is no ``kpf-drp`` row, as ``catalog_astrometry_sane`` does:
        a calibration frame has no target to have a color.
        """
        # (bluest, reddest) [mag] each index spans across the sequence, keyed by the
        # labels AstroQuery writes to color_name.
        limits = {
            "B-V": (-0.33, 2.17),
            "Gaia BP-RP": (-0.12, 5.10),
            "G-J": (-0.36, 5.36),
        }
        table = self.kpf_obj.data["CATALOG_RECORD"]
        match = table[table["source"] == "kpf-drp"] if table.colnames else table
        if not len(match):
            return True
        row = match[0]
        color = self._row_float(row, "color")
        bounds = limits.get(str(row["color_name"]))
        if color is None or bounds is None:
            return False
        return bounds[0] <= color <= bounds[1]

    catalog_color_sane._qc_key = "COLOROK"

    def _em_table(self, ext):
        """The exposure-meter table for ``ext``, or None when the frame has none.

        Only science frames carry EM extensions; on a calibration ``data.get``
        yields None.
        """
        table = self.kpf_obj.data.get(ext)
        return table if table is not None and len(table) else None

    def expmeter_times_consistent(self):
        """EXPMETER_SCI brackets the shutter window to within 1 second.

        Ports the exposure-meter half of v2.12 ``L0_datetime``; the header-date
        half is DATTIMOK/EXPTIMOK. The first Date-Beg and last Date-End of the EM
        table must match PRIMARY DATE-BEG/DATE-END, preferring the corrected
        columns as BarycentricCorrection does. The 1 s tolerance absorbs EM dead
        time and catches only gross errors: the flux-weighted midpoint feeds the
        barycentric correction, where a 2.6 s timing error costs 10 cm/s.

        Frames without EM data pass; a frame with EM data but no shutter window
        to compare against fails.
        """
        table = self._em_table("EXPMETER_SCI")
        if table is None:
            return True
        hdr = self.kpf_obj.headers["PRIMARY"]
        beg, end = (_parse_iso(hdr.get(k)) for k in ("DATE-BEG", "DATE-END"))
        if beg is None or end is None:
            return False
        suffix = "-Corr" if "Date-Beg-Corr" in table.colnames else ""
        em_beg = _parse_iso(str(table[f"Date-Beg{suffix}"][0]))
        em_end = _parse_iso(str(table[f"Date-End{suffix}"][-1]))
        if em_beg is None or em_end is None:
            return False
        return (
            abs((em_beg - beg).total_seconds()) <= 1.0
            and abs((em_end - end).total_seconds()) <= 1.0
        )

    expmeter_times_consistent._qc_key = "EMTIMEOK"

    def expmeter_flux_sane(self):
        """EXPMETER_SCI/SKY flux is neither saturated nor significantly negative.

        Merges v2.12 ``EM_not_saturated`` and ``EM_flux_not_negative``, applied to
        each fiber present. Saturation: more than 1.5 channels per reading above
        90% of the 1.93e6 reduced-spectrum saturation level, with the first and
        last readings dropped when there are 3+ (they are partial). Negative flux:
        20 consecutive channels whose time-summed flux is negative, the signature
        of bias over-subtraction in the raw EM images.

        Frames without EM data pass; an EM table with no wavelength channel fails.
        """
        for ext in ("EXPMETER_SCI", "EXPMETER_SKY"):
            table = self._em_table(ext)
            if table is None:
                continue
            # Numeric column labels are the wavelength channels; Date* are not.
            channels = []
            for name in table.colnames:
                try:
                    float(name)
                except ValueError:
                    continue
                channels.append(np.asarray(table[name], dtype=float))
            if not channels:
                return False
            flux = np.column_stack(channels)

            readings = flux[1:-1] if len(flux) >= 3 else flux
            saturated = np.count_nonzero(readings > 0.9 * 1.93e6)
            if saturated > 1.5 * len(readings):
                return False

            negative = (flux.sum(axis=0) < 0).astype(int)
            if np.any(np.convolve(negative, np.ones(20, dtype=int), "valid") == 20):
                return False
        return True

    expmeter_flux_sane._qc_key = "EMFLUXOK"
