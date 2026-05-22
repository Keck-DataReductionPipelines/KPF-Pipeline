"""QC checks for KPF Level 1 (assembled FFI) data products."""

import numpy as np

from kpfpipe.quality_control.qc_binaries.base import QC

_RN_KEYS    = ["RNGREEN1", "RNGREEN2", "RNGREEN3", "RNGREEN4",
               "RNRED1",   "RNRED2",   "RNRED3",   "RNRED4"]
_RNNG_KEYS  = ["RNNGGR1",  "RNNGGR2",  "RNNGGR3",  "RNNGGR4",
               "RNNGRD1",  "RNNGRD2",  "RNNGRD3",  "RNNGRD4"]

_RN_LO,  _RN_HI   = 2.0,  6.0
_RNNG_LO, _RNNG_HI = 0.8, 1.5


def _hdr_float(hdr, key):
    """Return float value for a header key, or None if absent."""
    if key not in hdr:
        return None
    val = hdr[key]
    return float(val[0] if isinstance(val, tuple) else val)


class QCL1(QC):
    """QC checks for KPF Level 1 assembled FFI products."""

    LEVEL = "L1"

    def read_noise_in_range(self):
        """All 8 per-amp RN values in [2.0, 6.0] e-."""
        hdr = self.kpf.headers["PRIMARY"]
        for k in _RN_KEYS:
            v = _hdr_float(hdr, k)
            if v is None or not (_RN_LO <= v <= _RN_HI):
                return False
        return True

    read_noise_in_range._qc_key = "RNINRNG"
    read_noise_in_range._qc_comment = "QC: per-amp RN within 2.0-6.0 e-"

    def read_noise_nongauss(self):
        """All 8 RNNG values in [0.8, 1.5]."""
        hdr = self.kpf.headers["PRIMARY"]
        for k in _RNNG_KEYS:
            v = _hdr_float(hdr, k)
            if v is None or not (_RNNG_LO <= v <= _RNNG_HI):
                return False
        return True

    read_noise_nongauss._qc_key = "RNGAUSS"
    read_noise_nongauss._qc_comment = "QC: non-Gaussian RN within 0.8-1.5"

    def bias_subtracted(self):
        """BIASUB == True."""
        hdr = self.kpf.headers["PRIMARY"]
        if "BIASUB" not in hdr:
            return False
        val = hdr["BIASUB"]
        if isinstance(val, tuple):
            val = val[0]
        return bool(val)

    bias_subtracted._qc_key = "BIASOK"
    bias_subtracted._qc_comment = "QC: bias subtraction applied"

    def bias_age_ok(self):
        """abs(AGEBIAS) <= 7 days."""
        v = _hdr_float(self.kpf.headers["PRIMARY"], "AGEBIAS")
        return v is not None and abs(v) <= 7

    bias_age_ok._qc_key = "BIASAGE"
    bias_age_ok._qc_comment = "QC: bias master age <= 7 days"

    def dark_age_ok(self):
        """abs(AGEDARK) <= 14 days."""
        v = _hdr_float(self.kpf.headers["PRIMARY"], "AGEDARK")
        return v is not None and abs(v) <= 14

    dark_age_ok._qc_key = "DARKAGE"
    dark_age_ok._qc_comment = "QC: dark master age <= 14 days"

    def flat_age_ok(self):
        """abs(AGEFLAT) <= 30 days."""
        v = _hdr_float(self.kpf.headers["PRIMARY"], "AGEFLAT")
        return v is not None and abs(v) <= 30

    flat_age_ok._qc_key = "FLATAGE"
    flat_age_ok._qc_comment = "QC: flat master age <= 30 days"

    def ffi_finite(self):
        """All values in GREEN_CCD and RED_CCD are finite."""
        for ext in ("GREEN_CCD", "RED_CCD"):
            arr = self.kpf.data.get(ext)
            if arr is None or np.size(arr) == 0:
                return False
            if not np.all(np.isfinite(arr)):
                return False
        return True

    ffi_finite._qc_key = "FFIFIN"
    ffi_finite._qc_comment = "QC: GREEN/RED CCDs all finite"
