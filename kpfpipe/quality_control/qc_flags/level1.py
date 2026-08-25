"""QC checks for KPF Level 1 (assembled FFI) data products."""

import numpy as np

from kpfpipe import DETECTOR
from kpfpipe.modules.image_assembly import RN_KEYS
from kpfpipe.quality_control.qc_flags.base import QC


class QCL1(QC):
    """QC checks for KPF Level 1 assembled FFI products."""

    LEVEL = "L1"

    def data_present(self):
        """Both chips carry a full-frame CCD and its paired variance.

        Assembly writes the two together (``ImageAssembly.stitch_ffi``), so a
        variance absent or shaped unlike its flux is a malformed product. The
        expected shape is the detector's own, which subsumes the non-empty test;
        an all-NaN frame is present but not populated, and fails too.
        """
        shape = (DETECTOR["ccd"]["nrow"], DETECTOR["ccd"]["ncol"])
        for chip in ("GREEN", "RED"):
            for suffix in ("CCD", "VAR"):
                arr = self.kpf_obj.data.get(f"{chip}_{suffix}")
                # A None-data extension is stored as array(None, dtype=object).
                if arr is None or getattr(arr, "dtype", None) == np.dtype(object):
                    return False
                if arr.shape != shape or not np.any(np.isfinite(arr)):
                    return False
        return True

    data_present._qc_key = "DATAPRL1"

    def required_keywords_present(self):
        """Every registry-required PRIMARY keyword for L1 is present (presence only)."""
        return self._required_primary_keywords() <= set(self.kpf_obj.headers["PRIMARY"])

    required_keywords_present._qc_key = "KWRDPRL1"

    def _present_rn_in_range(self, idx, lo, hi):
        """True iff every present amp's ``idx``-th RN keyword is in ``[lo, hi]``.

        ``idx`` selects the RN keyword pair (0 = RN, 1 = non-Gaussian RN). Only
        the amps a readout actually used carry an RN keyword, so 2-amp and 4-amp
        readouts both pass; a frame carrying none at all fails, since read noise
        should always be recorded.
        """
        hdr = self.kpf_obj.headers["QUALITY_CONTROL"]
        present = [keys[idx] for keys in RN_KEYS.values() if keys[idx] in hdr]
        return bool(present) and all(lo <= float(hdr[k]) <= hi for k in present)

    def read_noise_ok(self):
        """Every per-amp read noise present is in [2.0, 6.0] e-."""
        return self._present_rn_in_range(0, 2.0, 6.0)

    read_noise_ok._qc_key = "RNOK"

    def read_noise_nongauss_ok(self):
        """Every per-amp non-Gaussian read noise present is in [0.8, 1.5]."""
        return self._present_rn_in_range(1, 0.8, 1.5)

    read_noise_nongauss_ok._qc_key = "RNNGOK"

    def bias_ok(self):
        """Bias subtracted (RECEIPT BIASSUB) and master bias age <= 7 days."""
        if not self.kpf_obj.headers["RECEIPT"]["BIASSUB"]:
            return False
        age = float(self.kpf_obj.headers["QUALITY_CONTROL"]["BIASAGE"])
        return abs(age) <= 7

    bias_ok._qc_key = "BIASOK"

    def dark_ok(self):
        """Dark subtracted (RECEIPT DARKSUB) and master dark age <= 14 days."""
        if not self.kpf_obj.headers["RECEIPT"]["DARKSUB"]:
            return False
        age = float(self.kpf_obj.headers["QUALITY_CONTROL"]["DARKAGE"])
        return abs(age) <= 14

    dark_ok._qc_key = "DARKOK"

    def flat_ok(self):
        """Flat divided (RECEIPT FLATDIV) and master flat age <= 30 days."""
        if not self.kpf_obj.headers["RECEIPT"]["FLATDIV"]:
            return False
        age = float(self.kpf_obj.headers["QUALITY_CONTROL"]["FLATAGE"])
        return abs(age) <= 30

    flat_ok._qc_key = "FLATOK"

    def ffi_finite(self):
        """All values in GREEN_CCD and RED_CCD are finite."""
        return all(
            np.all(np.isfinite(self.kpf_obj.data[ext]))
            for ext in ("GREEN_CCD", "RED_CCD")
        )

    ffi_finite._qc_key = "L1NANOK"

    def variance_positive(self):
        """No negative GREEN_VAR/RED_VAR where the flux is finite."""
        for chip in ("GREEN", "RED"):
            flux = np.asarray(self.kpf_obj.data[f"{chip}_CCD"])
            var = np.asarray(self.kpf_obj.data[f"{chip}_VAR"])
            if np.any(np.isfinite(flux) & np.isfinite(var) & (var < 0)):
                return False
        return True

    variance_positive._qc_key = "L1VAROK"

    def negative_snr_fraction(self):
        """Pixels below -5 sigma at most 1% on each chip (v2.12 POS2DSNR)."""
        for chip in ("GREEN", "RED"):
            ccd = self.kpf_obj.data[f"{chip}_CCD"]
            var = self.kpf_obj.data[f"{chip}_VAR"]
            snr = ccd / np.sqrt(var)
            if np.count_nonzero(snr < -5) / snr.size > 0.01:
                return False
        return True

    negative_snr_fraction._qc_key = "L1SNROK"
