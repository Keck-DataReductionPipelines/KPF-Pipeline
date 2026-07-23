"""QC checks for KPF Level 0 (raw CCD) data products."""

import os
from datetime import datetime

import numpy as np

from kpfpipe.quality_control.qc_flags.base import QC
from kpfpipe.utils.io import load_junk_obs_ids

_L0_REQUIRED_KEYS = ["DATE-OBS", "EXPTIME", "OBJECT", "OFNAME", "IMTYPE"]

_CHIPS = ("GREEN", "RED")
_AMPS_PER_CHIP = 4  # GREEN_AMP1..4 / RED_AMP1..4 (only a subset is read out)
_SUPPORTED_NAMP = (2, 4)  # valid KPF readout modes (see ImageAssembly.count_amplifiers)
_TIME_TOL_S = 0.1  # DATE-END - DATE-BEG vs ELAPSED tolerance (v2.12 quality_control.py)

# Physical-range bounds for the canonical CATALOG_RECORD astrometry, ported from the
# v2.12 quality_control.py good_TARG_headers L0 checks. The epoch/equinox window is
# exclusive-low / inclusive-high (1950 < x <= 2050), matching legacy. Two are vNext
# additions justified by our Gaia source (both feed the barycentric correction): the
# parallax LOWER bound (a negative Gaia parallax is routine, and gives a negative
# distance), and the proper-motion bound (canonical arcsec/yr here; highest real PM
# is ~10.4"/yr).
_EPOCH_RANGE = (1950.0, 2050.0)
_MAX_ABS_RV = 350.0  # km/s (Chubak et al. 2012, arXiv:1207.6212, Fig. 8)
_PARALLAX_RANGE = (0.0, 1000.0)  # mas; 0 < plx < 1000 (> 0 and < 1 arcsec)
_MAX_ABS_PM = 15.0  # arcsec/yr, per component


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
            for i in range(1, _AMPS_PER_CHIP + 1):
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
        on QUALITY_CONTROL. ELAPSED consistency is validated in exptime_sane
        (EXPTIMOK), so it is not repeated here.
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
        than ``_TIME_TOL_S`` (the elapsed-vs-requested check formerly done in the
        masters frame loader); an absent ELAPSED skips only that comparison.
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
        if elapsed is not None and not (0 <= elapsed - exptime <= _TIME_TOL_S):
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

    def catalog_values_sane(self):
        """Canonical CATALOG_RECORD astrometry values are physically plausible.

        Ports the v2.12 good_TARG_headers range checks onto the merged ``kpf-drp``
        row AstroQuery resolves (the astrometry feeding the barycentric correction),
        not the raw WMKO TARG* keywords. Each field is checked only when present:
        epoch/equinox in (1950, 2050] Julian years, |rv| <= 350 km/s, parallax in
        (0, 1000) mas, |pmra|/|pmdec| <= 15 arcsec/yr. The parallax lower bound and
        the PM bound are vNext additions (see the module bounds comment).

        Passes when there is no ``kpf-drp`` row: this is value sanity, not a presence
        check -- a science frame's missing astrometry is caught upstream.
        """
        table = self.kpf_obj.data["CATALOG_RECORD"]
        match = table[table["source"] == "kpf-drp"] if table.colnames else table
        if not len(match):
            return True
        row = match[0]
        for field in ("epoch", "equinox"):
            val = self._row_float(row, field)
            if val is not None and not (_EPOCH_RANGE[0] < val <= _EPOCH_RANGE[1]):
                return False
        rv = self._row_float(row, "rv")
        if rv is not None and abs(rv) > _MAX_ABS_RV:
            return False
        plax = self._row_float(row, "parallax")
        if plax is not None and not (_PARALLAX_RANGE[0] < plax < _PARALLAX_RANGE[1]):
            return False
        for field in ("pmra", "pmdec"):
            pm = self._row_float(row, field)
            if pm is not None and abs(pm) > _MAX_ABS_PM:
                return False
        return True

    catalog_values_sane._qc_key = "CATLOGOK"
