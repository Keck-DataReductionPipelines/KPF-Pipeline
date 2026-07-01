"""Diagnostics for KPF Level 1 (assembled FFI) data products.

The L1 metrics consumed by QCL1 that depend on intermediate processing
state (read noise from raw overscan, the BIASSUB flag) are still written by
the modules that produce them — ImageAssembly, ImageProcessing. Metrics that
can be recomputed from the finished L1 product alone live here: the master
calibration ages, derived from the master paths CalibrationAssociation wrote
to RECEIPT plus the observation timestamp (DATE-OBS) on PRIMARY.
"""

from datetime import datetime

from kpfpipe.quality_control.diagnostics.base import Diagnostics
from kpfpipe.utils.kpf import get_timestamp, kpf_timestamp_to_datetime

# Master-calibration age metrics: the RECEIPT path keyword (written by
# CalibrationAssociation) -> the age keyword whose signed (master - obs) value
# this diagnostic computes. The FITS comment comes from the registry (via the
# _tag helper / set_keyword), so it is not duplicated here.
_CAL_AGE_KEYS = {
    "BIASFILE": "BIASAGE",
    "DARKFILE": "DARKAGE",
    "FLATFILE": "FLATAGE",
    "WLSFILE": "WLSAGE",
}


class DiagL1(Diagnostics):
    LEVEL = "L1"

    def calibration_ages(self):
        """Signed fractional-day age (master - obs) for each associated master.

        Recomputed from the finished L1 product: the master path is read from
        RECEIPT (``{PREFIX}FILE``, written by CalibrationAssociation) and the
        observation timestamp from PRIMARY (DATE-OBS). The master
        timestamp is parsed from its filename. A cal type is skipped when its
        path is absent; the whole metric is skipped when DATE-OBS is missing.

        Returns
        -------
        dict
            Maps each present ``{PREFIX}AGE`` keyword to its ``(age, comment)``.
        """
        receipt = self.kpf_obj.headers.get("RECEIPT", {})
        primary = self.kpf_obj.headers.get("PRIMARY", {})
        date_obs = primary.get("DATE-OBS")
        if not date_obs:
            return {}
        obs_dt = datetime.fromisoformat(date_obs)

        results = {}
        for file_kw, age_kw in _CAL_AGE_KEYS.items():
            path = receipt.get(file_kw)
            if not path:
                continue
            master_dt = kpf_timestamp_to_datetime(get_timestamp(path))
            age_days = (master_dt - obs_dt).total_seconds() / 86400.0
            results[age_kw] = age_days
        return self._tag(**results)

    calibration_ages._diag_name = "calibration_ages"
