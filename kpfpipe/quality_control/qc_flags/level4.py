"""QC checks for KPF Level 4 (RVs and CCFs) data products.

Ports the v2.12 RV/CCF QC checks (``data_L2`` presence,
``L2_barycentric_rv_percent_change``) to vNext L4, plus the framework
required-PRIMARY-keyword presence check. Every result is a 0/1 flag written to
QUALITY_CONTROL via ``set_keyword``. (The timing-consistency check ported from
v2.12 ``L2_datetime`` now lives in QCL0 as ``DATTIMOK`` -- it is a raw-frame
property and propagates downstream via the QUALITY_CONTROL history.)
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
        """Each science orderlet has a non-empty CCF cube and RV table."""
        for fiber in _SCI_FIBERS:
            ccf = self.kpf_obj.data.get(f"{fiber}_CCF")
            rv = self.kpf_obj.data.get(f"{fiber}_RV")
            if ccf is None or np.size(ccf) == 0:
                return False
            if rv is None or len(rv) == 0:
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
