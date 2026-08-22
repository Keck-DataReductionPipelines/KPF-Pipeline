"""QC checks for KPF Level 1 (assembled FFI) data products."""

import numpy as np

from kpfpipe.modules.image_assembly import RN_KEYS
from kpfpipe.quality_control.qc_flags.base import QC

_RN_LO, _RN_HI = 2.0, 6.0
_RNNG_LO, _RNNG_HI = 0.8, 1.5


class QCL1(QC):
    """QC checks for KPF Level 1 assembled FFI products."""

    LEVEL = "L1"

    def data_present(self):
        """GREEN_CCD and RED_CCD exist and are non-empty."""
        for ext in ("GREEN_CCD", "RED_CCD"):
            arr = self.kpf_obj.data.get(ext)
            # A None-data extension is stored as array(None, dtype=object); absent.
            if (
                arr is None
                or getattr(arr, "dtype", None) == np.dtype(object)
                or np.size(arr) == 0
            ):
                return False
        return True

    data_present._qc_key = "DATAPRL1"

    def required_keywords_present(self):
        """Every registry-required PRIMARY keyword for L1 is present (presence only)."""
        return self._required_primary_keywords() <= set(self.kpf_obj.headers["PRIMARY"])

    required_keywords_present._qc_key = "KWRDPRL1"

    def _present_rn_in_range(self, idx, lo, hi):
        """True iff every present amp's ``idx``-th RN keyword is in ``[lo, hi]``.

        ``idx`` selects the RN keyword pair (0 = RN, 1 = non-Gaussian RN); absent
        amps are skipped so 2-amp and 4-amp readouts both pass. False if any is
        out of range, or if no RN keyword is present at all (read noise should
        always be recorded).
        """
        hdr = self.kpf_obj.headers["QUALITY_CONTROL"]
        found = False
        for keys in RN_KEYS.values():
            v = self._hdr_float(hdr, keys[idx])
            if v is None:
                continue
            found = True
            if not (lo <= v <= hi):
                return False
        return found

    def read_noise_ok(self):
        """Every per-amp read noise present is in [2.0, 6.0] e-."""
        return self._present_rn_in_range(0, _RN_LO, _RN_HI)

    read_noise_ok._qc_key = "RNOK"

    def read_noise_nongauss_ok(self):
        """Every per-amp non-Gaussian read noise present is in [0.8, 1.5]."""
        return self._present_rn_in_range(1, _RNNG_LO, _RNNG_HI)

    read_noise_nongauss_ok._qc_key = "RNNGOK"

    def bias_ok(self):
        """Bias subtracted (RECEIPT BIASSUB) and master bias age <= 7 days."""
        if not self._hdr_bool(self.kpf_obj.headers["RECEIPT"], "BIASSUB"):
            return False
        v = self._hdr_float(self.kpf_obj.headers["QUALITY_CONTROL"], "BIASAGE")
        return v is not None and abs(v) <= 7

    bias_ok._qc_key = "BIASOK"

    def dark_ok(self):
        """Dark subtracted (RECEIPT DARKSUB) and master dark age <= 14 days."""
        if not self._hdr_bool(self.kpf_obj.headers["RECEIPT"], "DARKSUB"):
            return False
        v = self._hdr_float(self.kpf_obj.headers["QUALITY_CONTROL"], "DARKAGE")
        return v is not None and abs(v) <= 14

    dark_ok._qc_key = "DARKOK"

    def flat_ok(self):
        """Flat divided (RECEIPT FLATDIV) and master flat age <= 30 days."""
        if not self._hdr_bool(self.kpf_obj.headers["RECEIPT"], "FLATDIV"):
            return False
        v = self._hdr_float(self.kpf_obj.headers["QUALITY_CONTROL"], "FLATAGE")
        return v is not None and abs(v) <= 30

    flat_ok._qc_key = "FLATOK"

    def ffi_finite(self):
        """All values in GREEN_CCD and RED_CCD are finite."""
        for ext in ("GREEN_CCD", "RED_CCD"):
            arr = self.kpf_obj.data.get(ext)
            if arr is None or np.size(arr) == 0:
                return False
            if not np.all(np.isfinite(arr)):
                return False
        return True

    ffi_finite._qc_key = "L1NANOK"

    def nonzero_flux(self):
        """Zero-flux fraction of the assembled CCD frames < 0.5."""
        raise NotImplementedError

    nonzero_flux._qc_key = "L1FLXOK"

    def variance_positive(self):
        """No negative GREEN_VAR/RED_VAR where the flux is finite."""
        raise NotImplementedError

    variance_positive._qc_key = "L1VAROK"

    def negative_snr_fraction(self):
        """Pixels below -5 sigma <= 1%, catching bias/dark over-subtraction."""
        raise NotImplementedError

    negative_snr_fraction._qc_key = "L1SNROK"

    def saturated_fraction(self):
        """Saturated / non-linear science pixel fraction within limit."""
        raise NotImplementedError

    saturated_fraction._qc_key = "L1SATOK"
