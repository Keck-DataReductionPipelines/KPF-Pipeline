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


def _hdr_float(hdr, key):
    """Return float value for a header key, or None if absent."""
    val = hdr.get(key)
    return None if val is None else float(val)


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

        KPF reads out either 2 or 4 amplifiers per chip, so the expected amp
        count is inferred from the data rather than fixed: a chip passes when the
        number of present, non-empty amplifier extensions is a supported readout
        mode (``_SUPPORTED_NAMP``), mirroring ``ImageAssembly.count_amplifiers``.
        This accepts both 2-amp and 4-amp frames and rejects a chip with no data
        or a partial/invalid amp set (1 or 3). Absent amps are stored as
        ``array(None, dtype=object)`` and skipped -- the same present-amp scan
        QCL1 uses for read noise.
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
        """DATE-BEG < DATE-MID < DATE-END and |(END-BEG) - ELAPSED| < 0.1 s.

        Ports v2.12 ``L2_datetime``. At L0 the raw instrument times live on the
        WMKO-native PRIMARY (the header later snapshotted verbatim into
        INSTRUMENT_HEADER at to_kpf1); the 0/1 flag then propagates downstream
        on QUALITY_CONTROL. Passes the duration check when ELAPSED is absent --
        only the present cards are checked.
        """
        hdr = self.kpf_obj.headers["PRIMARY"]
        beg, mid, end = (
            _parse_iso(hdr.get(k)) for k in ("DATE-BEG", "DATE-MID", "DATE-END")
        )
        if beg is None or mid is None or end is None:
            return False
        if not (beg <= mid <= end):
            return False
        elapsed = _hdr_float(hdr, "ELAPSED")
        if (
            elapsed is not None
            and abs((end - beg).total_seconds() - elapsed) > _TIME_TOL_S
        ):
            return False
        return True

    times_consistent._qc_key = "DATTIMOK"

    def exptime_sane(self):
        """EXPTIME is present, finite, and non-negative.

        Bias frames legitimately have EXPTIME=0, so we don't require strictly
        positive. Tightening this requires frame-type-aware filtering, which
        is deferred until QC gains spectrum-type gating.
        """
        hdr = self.kpf_obj.headers["PRIMARY"]
        if "EXPTIME" not in hdr:
            return False
        try:
            f = float(hdr.get("EXPTIME"))
        except (TypeError, ValueError):
            return False
        return np.isfinite(f) and f >= 0

    exptime_sane._qc_key = "EXPTIMOK"

    def not_junk(self):
        """obs_id not on the observer junk list for this frame's data tree.

        "Junk" is a manual observer flag (e.g. the wrong telescope settings) that
        no automated QC can catch. The list lives at
        ``{KPF_DATA_INPUT}/reference/Junk_Observations_for_KPF.csv``;
        KPF_DATA_INPUT is recovered from the frame's own source directory, which
        rvdata records as ``self.dirname`` (``{KPF_DATA_INPUT}/L0/{datecode}``)
        when the L0 is read. An absent list or unknown source dir yields
        not-junk.
        """
        obs_id = self.kpf_obj.obs_id
        dirname = getattr(self.kpf_obj, "dirname", None)
        if not obs_id or not dirname:
            return True
        data_input = os.path.dirname(os.path.dirname(dirname))
        return obs_id not in load_junk_obs_ids(data_input)

    not_junk._qc_key = "NOTJUNK"
