"""QC checks for KPF Level 1 (assembled FFI) data products."""

import numpy as np

from kpfpipe.modules.image_assembly import RN_KEYS
from kpfpipe.quality_control.qc_booleans.base import QC

_RN_LO, _RN_HI = 2.0, 6.0
_RNNG_LO, _RNNG_HI = 0.8, 1.5


def _hdr_float(hdr, key):
    """Return float value for a header key, or None if absent."""
    val = hdr.get(key)
    return None if val is None else float(val)


def _hdr_flag(hdr, key):
    """Return bool value for a header key, or False if absent."""
    return bool(hdr.get(key, False))


class QCL1(QC):
    """QC checks for KPF Level 1 assembled FFI products."""

    LEVEL = "L1"

    def _present_rn_in_range(self, idx, lo, hi):
        """Validate a read-noise keyword across every amplifier present.

        Checks the ``idx``-th RN keyword for every amplifier whose keyword is
        present, so 2-amp and 4-amp readouts both pass. Absent amps are
        skipped.

        Parameters
        ----------
        idx : int
            Index into each amp's RN keyword pair (0 = RN, 1 = non-Gaussian RN).
        lo : float
            Lower bound of the accepted range, inclusive.
        hi : float
            Upper bound of the accepted range, inclusive.

        Returns
        -------
        bool
            True if all present values fall in ``[lo, hi]``. False if any is
            out of range, or if no RN keyword is present at all (read noise
            should always be recorded).
        """
        hdr = self.kpf.headers["QUALITY_CONTROL"]
        found = False
        for keys in RN_KEYS.values():
            v = _hdr_float(hdr, keys[idx])
            if v is None:
                continue
            found = True
            if not (lo <= v <= hi):
                return False
        return found

    def read_noise_in_range(self):
        """Every per-amp RN value present in the header is in [2.0, 6.0] e-."""
        return self._present_rn_in_range(0, _RN_LO, _RN_HI)

    read_noise_in_range._qc_key = "RNINRNG"
    read_noise_in_range._qc_comment = "QC: per-amp RN within 2.0-6.0 e-"

    def read_noise_nongauss(self):
        """Every non-Gaussian RN value present in the header is in [0.8, 1.5]."""
        return self._present_rn_in_range(1, _RNNG_LO, _RNNG_HI)

    read_noise_nongauss._qc_key = "RNGAUSS"
    read_noise_nongauss._qc_comment = "QC: non-Gaussian RN within 0.8-1.5"

    def overscan_subtracted(self):
        """OSCANSUB == True."""
        return _hdr_flag(self.kpf.headers["RECEIPT"], "OSCANSUB")

    overscan_subtracted._qc_key = "OSCANSUB"
    overscan_subtracted._qc_comment = "QC: overscan subtraction applied"

    def bias_subtracted(self):
        """BIASSUB == True."""
        return _hdr_flag(self.kpf.headers["RECEIPT"], "BIASSUB")

    bias_subtracted._qc_key = "BIASSUB"
    bias_subtracted._qc_comment = "QC: bias subtraction applied"

    def bias_age_ok(self):
        """abs(BIASAGE) <= 7 days."""
        v = _hdr_float(self.kpf.headers["QUALITY_CONTROL"], "BIASAGE")
        return v is not None and abs(v) <= 7

    bias_age_ok._qc_key = "BIASAGEQ"
    bias_age_ok._qc_comment = "QC: bias master age <= 7 days"

    def dark_subtracted(self):
        """DARKSUB == True."""
        return _hdr_flag(self.kpf.headers["RECEIPT"], "DARKSUB")

    dark_subtracted._qc_key = "DARKSUB"
    dark_subtracted._qc_comment = "QC: dark subtraction applied"

    def dark_age_ok(self):
        """abs(DARKAGE) <= 14 days."""
        v = _hdr_float(self.kpf.headers["QUALITY_CONTROL"], "DARKAGE")
        return v is not None and abs(v) <= 14

    dark_age_ok._qc_key = "DARKAGEQ"
    dark_age_ok._qc_comment = "QC: dark master age <= 14 days"

    def flat_divided(self):
        """FLATDIV == True."""
        return _hdr_flag(self.kpf.headers["RECEIPT"], "FLATDIV")

    flat_divided._qc_key = "FLATDIV"
    flat_divided._qc_comment = "QC: flat division applied"

    def flat_age_ok(self):
        """abs(FLATAGE) <= 30 days."""
        v = _hdr_float(self.kpf.headers["QUALITY_CONTROL"], "FLATAGE")
        return v is not None and abs(v) <= 30

    flat_age_ok._qc_key = "FLATAGEQ"
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
