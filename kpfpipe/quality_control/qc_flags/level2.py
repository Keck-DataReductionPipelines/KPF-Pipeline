"""QC checks for KPF Level 2 (extracted spectra) data products."""

import numpy as np

from kpfpipe.quality_control.qc_flags.base import QC

_CHIPS = ["GREEN", "RED"]
_FIBERS = ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]


class QCL2(QC):
    """QC checks for KPF Level 2 extracted spectra products."""

    LEVEL = "L2"

    def extraction_present(self):
        """All expected {CHIP}_{FIBER}_FLUX extensions exist and are non-empty."""
        for chip in _CHIPS:
            for fiber in _FIBERS:
                ext = f"{chip}_{fiber}_FLUX"
                arr = self.kpf_obj.data.get(ext)
                if arr is None or np.size(arr) == 0:
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

    def saturated_fraction(self):
        """Saturated / non-linear extracted pixel fraction within limit."""
        raise NotImplementedError

    saturated_fraction._qc_key = "L2SATOK"
