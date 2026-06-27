"""QC checks for KPF Level 0 (raw CCD) data products."""

import numpy as np
import pandas as pd

from kpfpipe import REPO_ROOT
from kpfpipe.quality_control.qc_booleans.base import QC

_JUNK_CSV = REPO_ROOT / "reference" / "junk_observations.csv"

_L0_REQUIRED_KEYS = ["DATE-OBS", "EXPTIME", "OBJECT", "OFNAME", "IMTYPE"]

_AMP_EXTENSIONS = [
    "GREEN_AMP1",
    "GREEN_AMP2",
    "GREEN_AMP3",
    "GREEN_AMP4",
    "RED_AMP1",
    "RED_AMP2",
    "RED_AMP3",
    "RED_AMP4",
]


class QCL0(QC):
    """QC checks for KPF Level 0 raw data products."""

    LEVEL = "L0"

    def data_l0_red_green(self):
        """GREEN_AMP1..4 and RED_AMP1..4 exist and are non-empty."""
        for ext in _AMP_EXTENSIONS:
            arr = self.kpf.data.get(ext)
            # KPF0 stores None-data as array(None, dtype=object); treat as absent.
            if (
                arr is None
                or getattr(arr, "dtype", None) == np.dtype(object)
                or np.size(arr) == 0
            ):
                return False
        return True

    data_l0_red_green._qc_key = "DATAPRL0"

    def header_keywords_present(self):
        """Required PRIMARY keywords exist."""
        hdr = self.kpf.headers["PRIMARY"]
        return all(k in hdr for k in _L0_REQUIRED_KEYS)

    header_keywords_present._qc_key = "KWRDPRL0"

    def exptime_sane(self):
        """EXPTIME is present, finite, and non-negative.

        Bias frames legitimately have EXPTIME=0, so we don't require strictly
        positive. Tightening this requires frame-type-aware filtering, which
        is deferred until QC gains spectrum-type gating.
        """
        hdr = self.kpf.headers["PRIMARY"]
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
        obs_id = self.kpf.obs_id
        if not obs_id:
            return True
        df = pd.read_csv(_JUNK_CSV)
        if "obs_id" not in df.columns:
            raise ValueError(
                f"junk_observations.csv missing 'obs_id' column: {_JUNK_CSV}"
            )
        return obs_id not in df["obs_id"].values

    not_junk._qc_key = "NOTJUNK"
