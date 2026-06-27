"""QC checks for KPF Level 0 (raw CCD) data products."""

import numpy as np
import pandas as pd

from kpfpipe import REPO_ROOT
from kpfpipe.quality_control.qc_flags.base import QC

_JUNK_CSV = REPO_ROOT / "reference" / "junk_observations.csv"

_L0_REQUIRED_KEYS = ["DATE-OBS", "EXPTIME", "OBJECT", "OFNAME", "IMTYPE"]

_CHIPS = ("GREEN", "RED")
_AMPS_PER_CHIP = 4  # GREEN_AMP1..4 / RED_AMP1..4 (only a subset is read out)


class QCL0(QC):
    """QC checks for KPF Level 0 raw data products."""

    LEVEL = "L0"

    def data_l0_red_green(self):
        """Raw CCD data present: GREEN and RED each have >=1 non-empty amplifier.

        KPF reads out either 2 or 4 amplifiers per chip, so only the read-out
        amps are stored; absent amps are kept as ``array(None, dtype=object)``.
        Both 2-amp and 4-amp readouts must pass, so we require each chip to have
        at least one real amp array rather than all four.
        """
        for chip in _CHIPS:
            found = False
            for i in range(1, _AMPS_PER_CHIP + 1):
                arr = self.kpf_obj.data.get(f"{chip}_AMP{i}")
                # KPF0 stores None-data as array(None, dtype=object); skip absent.
                if (
                    arr is None
                    or getattr(arr, "dtype", None) == np.dtype(object)
                    or np.size(arr) == 0
                ):
                    continue
                found = True
            if not found:
                return False
        return True

    data_l0_red_green._qc_key = "DATAPRL0"

    def header_keywords_present(self):
        """Required PRIMARY keywords exist."""
        hdr = self.kpf_obj.headers["PRIMARY"]
        return all(k in hdr for k in _L0_REQUIRED_KEYS)

    header_keywords_present._qc_key = "KWRDPRL0"

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
        """obs_id not in reference/junk_observations.csv."""
        if not _JUNK_CSV.exists():
            return True
        obs_id = self.kpf_obj.obs_id
        if not obs_id:
            return True
        df = pd.read_csv(_JUNK_CSV)
        if "obs_id" not in df.columns:
            raise ValueError(
                f"junk_observations.csv missing 'obs_id' column: {_JUNK_CSV}"
            )
        return obs_id not in df["obs_id"].values

    not_junk._qc_key = "NOTJUNK"
