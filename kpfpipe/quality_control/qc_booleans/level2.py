"""QC checks for KPF Level 2 (extracted spectra) data products."""

import numpy as np

from kpfpipe.quality_control.qc_booleans.base import QC

_CHIPS = ["GREEN", "RED"]
_FIBERS = ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]

_NAN_KEYS = ["NANSCI1", "NANSCI2", "NANSCI3", "NANSKY", "NANCAL"]

# Minimum plausible science SNR; a fully failed extraction yields ~0.
_MIN_SCI_SNR = 1.0


def _hdr_float(hdr, key):
    """Return float value for a header key, or None if absent."""
    val = hdr.get(key)
    return None if val is None else float(val)


class QCL2(QC):
    """QC checks for KPF Level 2 extracted spectra products."""

    LEVEL = "L2"

    def extraction_present(self):
        """All expected {CHIP}_{FIBER}_FLUX extensions exist and are non-empty."""
        for chip in _CHIPS:
            for fiber in _FIBERS:
                ext = f"{chip}_{fiber}_FLUX"
                arr = self.kpf.data.get(ext)
                if arr is None or np.size(arr) == 0:
                    return False
        return True

    extraction_present._qc_key = "DATAPRL2"
    extraction_present._qc_comment = "QC: L2 FLUX extensions present and non-empty"

    def required_keywords_present(self):
        """Every registry-required PRIMARY keyword for L2 is present (presence only)."""
        return self._required_primary_keywords() <= set(self.kpf.headers["PRIMARY"])

    required_keywords_present._qc_key = "KWRDPRL2"
    required_keywords_present._qc_comment = "QC: required L2 PRIMARY keywords present"

    def flux_finite_fraction(self):
        """NaN count from headers <= 1% of total L2 flux pixels."""
        hdr = self.kpf.headers["QUALITY_CONTROL"]
        total_pixels = 0
        for chip in _CHIPS:
            for fiber in _FIBERS:
                ext = f"{chip}_{fiber}_FLUX"
                arr = self.kpf.data.get(ext)
                if arr is not None:
                    total_pixels += np.size(arr)

        if total_pixels == 0:
            return False

        nan_total = 0
        for k in _NAN_KEYS:
            v = _hdr_float(hdr, k)
            if v is None:
                return False
            nan_total += v

        return (nan_total / total_pixels) <= 0.01

    flux_finite_fraction._qc_key = "L2NANOK"
    flux_finite_fraction._qc_comment = "QC: L2 NaN fraction <= 1%"

    def nonzero_flux(self):
        """ZEROFRAC < 0.5."""
        v = _hdr_float(self.kpf.headers["QUALITY_CONTROL"], "ZEROFRAC")
        return v is not None and v < 0.5

    nonzero_flux._qc_key = "L2FLXOK"
    nonzero_flux._qc_comment = "QC: L2 zero-flux fraction < 0.5"

    def variance_positive(self):
        """No strictly-negative variance where the flux is finite.

        A negative extracted variance is unphysical (box/optimal variance is a
        sum of non-negative terms) and would corrupt downstream RV weighting.
        Zero variance at off-detector / fully-masked columns is tolerated.
        """
        saw_data = False
        for chip in _CHIPS:
            for fiber in _FIBERS:
                flux = self.kpf.data.get(f"{chip}_{fiber}_FLUX")
                var = self.kpf.data.get(f"{chip}_{fiber}_VAR")
                if flux is None or var is None:
                    continue
                flux = np.asarray(flux)
                var = np.asarray(var)
                if flux.size == 0 or var.shape != flux.shape:
                    continue
                saw_data = True
                if np.any(np.isfinite(flux) & np.isfinite(var) & (var < 0)):
                    return False
        return saw_data

    variance_positive._qc_key = "L2VARPOS"
    variance_positive._qc_comment = "QC: no negative L2 variance where flux finite"

    def science_snr(self):
        """Science SNR is finite and above a minimum floor.

        Reads the GSNRSCI/RSNRSCI metrics written by DiagL2.snr (run DiagL2
        before QCL2, mirroring the flux_finite_fraction -> nan_counts
        dependency). Guards against a silently failed extraction.
        """
        hdr = self.kpf.headers["QUALITY_CONTROL"]
        values = [_hdr_float(hdr, k) for k in ("GSNRSCI", "RSNRSCI")]
        values = [v for v in values if v is not None]
        if not values:
            return False
        return all(np.isfinite(v) and v > _MIN_SCI_SNR for v in values)

    science_snr._qc_key = "L2SNROK"
    science_snr._qc_comment = "QC: science SNR finite and above floor"
