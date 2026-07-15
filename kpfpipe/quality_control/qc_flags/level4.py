"""QC checks for KPF Level 4 (RVs and CCFs) data products.

Checks the presence of the RV/CCF data and the per-order barycentric-RV percent
change, plus the framework required-PRIMARY-keyword presence check. Each result
is a 0/1 flag written to QUALITY_CONTROL.
"""

import numpy as np

from kpfpipe.quality_control.qc_flags.base import QC

_SCI_FIBERS = ("SCI1", "SCI2", "SCI3")
# Max |per-order BERV % deviation| from the mean (from v2.12 quality_control.py).
_BERV_PCT_TOL = 1.0


def _hdr_float(hdr, key):
    """Return float value for a header key, or None if absent."""
    val = hdr.get(key)
    return None if val is None else float(val)


class QCL4(QC):
    """QC checks for KPF Level 4 RV/CCF products."""

    LEVEL = "L4"

    def ccf_rv_present(self):
        """Each science orderlet has a non-empty CCF cube and computed RVs.

        Both stages of the split must have run: CrossCorrelation writes the CCF
        cube and seeds the RV table (with NaN RV/RV_ERR), and RadialVelocity fills
        the RV column. A seeded-but-unfilled table (CrossCorrelation without a
        following RadialVelocity) fails here, since the RVs are the L4 product.
        """
        for fiber in _SCI_FIBERS:
            ccf = self.kpf_obj.data.get(f"{fiber}_CCF")
            rv = self.kpf_obj.data.get(f"{fiber}_RV")
            if ccf is None or np.size(ccf) == 0:
                return False
            if rv is None or len(rv) == 0:
                return False
            if "RV" not in getattr(rv, "colnames", []):
                return False
            if not np.any(np.isfinite(np.asarray(rv["RV"], dtype=float))):
                return False
        return True

    ccf_rv_present._qc_key = "DATAPRL4"

    def required_keywords_present(self):
        """Every registry-required PRIMARY keyword for L4 is present (presence only)."""
        return self._required_primary_keywords() <= set(self.kpf_obj.headers["PRIMARY"])

    required_keywords_present._qc_key = "KWRDPRL4"

    def berv_within_tolerance(self):
        """Per-order BERV deviation within +/-1% of the weighted mean.

        Ports v2.12 ``L2_barycentric_rv_percent_change``, reading BERVMAXP /
        BERVMINP written by DiagL4 (run DiagL4 before QCL4). Passes when the
        metrics are absent (e.g. a calibration frame with no science RV) -- there
        is nothing to flag.
        """
        hdr = self.kpf_obj.headers["QUALITY_CONTROL"]
        mx = _hdr_float(hdr, "BERVMAXP")
        mn = _hdr_float(hdr, "BERVMINP")
        if mx is None or mn is None:
            return True
        return mx <= _BERV_PCT_TOL and mn >= -_BERV_PCT_TOL

    berv_within_tolerance._qc_key = "BERVOK"
