"""QC checks for KPF Level 2 (extracted spectra) data products."""

import numpy as np

from kpfpipe import DETECTOR
from kpfpipe.quality_control.qc_flags.base import QC

_CHIPS = ["GREEN", "RED"]
_FIBERS = ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]


class QCL2(QC):
    """QC checks for KPF Level 2 extracted spectra products."""

    LEVEL = "L2"

    def extraction_present(self):
        """Every extension the science chain writes at L2, at its expected shape.

        Extraction writes FLUX and VAR, WavelengthCalibration WAVE, and
        BarycentricCorrection the three per-order ancillaries; all four modules
        run before CheckpointL2, so any one missing is an incomplete product.
        ``np.shape(None)`` is ``()``, so an absent extension fails the shape
        comparison without a separate presence test. Each array must also hold at
        least one finite value: an orderlet that never reached the detector is
        NaN-filled by extraction, which is present but not populated.

        BLAZE and ORDER_TABLE are EPRV-required but have no producer in the
        science chain, so they are out of scope until one exists.
        """
        norder = DETECTOR["norder"]
        ncol = DETECTOR["ccd"]["ncol"]
        for chip in _CHIPS:
            for fiber in _FIBERS:
                for suffix in ("FLUX", "VAR", "WAVE"):
                    arr = self.kpf_obj.data.get(f"{chip}_{fiber}_{suffix}")
                    if np.shape(arr) != (norder[chip], ncol):
                        return False
                    if not np.any(np.isfinite(arr)):
                        return False
        per_order = (norder["GREEN"] + norder["RED"],)
        for ext in ("BJD_TDB", "BARYCORR_KMS", "BARYCORR_Z"):
            arr = self.kpf_obj.data.get(ext)
            if np.shape(arr) != per_order or not np.any(np.isfinite(arr)):
                return False
        return True

    extraction_present._qc_key = "DATAPRL2"

    def required_keywords_present(self):
        """Every registry-required PRIMARY keyword for L2 is present (presence only)."""
        return self._required_primary_keywords() <= set(self.kpf_obj.headers["PRIMARY"])

    required_keywords_present._qc_key = "KWRDPRL2"

    def flux_finite_fraction(self):
        """NaN count from headers <= 1% of total L2 flux pixels."""
        hdr = self.kpf_obj.headers["QUALITY_CONTROL"]
        total_pixels = sum(
            np.size(self.kpf_obj.data[f"{chip}_{fiber}_FLUX"])
            for chip in _CHIPS
            for fiber in _FIBERS
        )
        nan_total = sum(float(hdr[f"NAN{fiber}"]) for fiber in _FIBERS)
        return (nan_total / total_pixels) <= 0.01

    flux_finite_fraction._qc_key = "L2NANOK"

    def nonzero_flux(self):
        """ZEROFRAC < 0.5."""
        return float(self.kpf_obj.headers["QUALITY_CONTROL"]["ZEROFRAC"]) < 0.5

    nonzero_flux._qc_key = "L2FLXOK"

    def variance_positive(self):
        """No strictly-negative variance where the flux is finite.

        A negative extracted variance is unphysical (box/optimal variance is a
        sum of non-negative terms) and would corrupt downstream RV weighting.
        Zero variance at off-detector / fully-masked columns is tolerated. A VAR
        whose shape disagrees with its FLUX is a malformed product, not a state to
        skip: the comparison below then raises (fail loud).
        """
        for chip in _CHIPS:
            for fiber in _FIBERS:
                flux = np.asarray(self.kpf_obj.data[f"{chip}_{fiber}_FLUX"])
                var = np.asarray(self.kpf_obj.data[f"{chip}_{fiber}_VAR"])
                if np.any(np.isfinite(flux) & np.isfinite(var) & (var < 0)):
                    return False
        return True

    variance_positive._qc_key = "L2VAROK"

    def science_snr(self):
        """Science SNR is finite and greater than 1.

        Reads the GSNRSCI/RSNRSCI metrics written by ``DiagL2.snr`` (run DiagL2
        before QCL2, the same Diagnostics -> QC ordering every metric-backed
        check relies on). Guards against a silently failed extraction.
        """
        hdr = self.kpf_obj.headers["QUALITY_CONTROL"]
        values = [float(hdr[k]) for k in ("GSNRSCI", "RSNRSCI")]
        return all(np.isfinite(v) and v > 1.0 for v in values)

    science_snr._qc_key = "L2SNROK"
