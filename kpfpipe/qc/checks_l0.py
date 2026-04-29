"""QC checks for KPF Level 0 (raw CCD) data products."""

import numpy as np
import pandas as pd

from kpfpipe import REPO_ROOT
from kpfpipe.qc.base import QC

_JUNK_CSV = REPO_ROOT / "reference" / "junk_observations.csv"

_L0_REQUIRED_KEYS = ["DATE-OBS", "EXPTIME", "OBJECT", "OFNAME", "IMTYPE"]

_AMP_EXTENSIONS = [
    "GREEN_AMP1", "GREEN_AMP2", "GREEN_AMP3", "GREEN_AMP4",
    "RED_AMP1",   "RED_AMP2",   "RED_AMP3",   "RED_AMP4",
]


class QCL0(QC):
    """QC checks for KPF Level 0 raw data products."""

    LEVEL = "L0"

    def data_l0_red_green(self):
        """GREEN_AMP1..4 and RED_AMP1..4 exist and are non-empty."""
        for ext in _AMP_EXTENSIONS:
            arr = self.kpf.data.get(ext)
            # KPF0 stores None-data as array(None, dtype=object); treat as absent.
            if arr is None or getattr(arr, "dtype", None) == object or np.size(arr) == 0:
                return False
        return True

    data_l0_red_green._qc_key = "DATAPRL0"
    data_l0_red_green._qc_comment = "QC: GREEN/RED amp extensions present and non-empty"

    def header_keywords_present(self):
        """Required PRIMARY keywords exist."""
        hdr = self.kpf.headers["PRIMARY"]
        return all(k in hdr for k in _L0_REQUIRED_KEYS)

    header_keywords_present._qc_key = "KWRDPRL0"
    header_keywords_present._qc_comment = "QC: required L0 PRIMARY keywords present"

    def exptime_positive(self):
        """EXPTIME > 0."""
        hdr = self.kpf.headers["PRIMARY"]
        if "EXPTIME" not in hdr:
            return False
        val = hdr["EXPTIME"]
        if isinstance(val, tuple):
            val = val[0]
        return float(val) > 0

    exptime_positive._qc_key = "EXPTIMOK"
    exptime_positive._qc_comment = "QC: EXPTIME > 0"

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
    not_junk._qc_comment = "QC: obs_id not in junk list"
