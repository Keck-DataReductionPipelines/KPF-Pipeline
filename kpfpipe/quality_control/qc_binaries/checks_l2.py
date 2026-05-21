"""QC checks for KPF Level 2 (extracted spectra) data products."""

import numpy as np

from kpfpipe.quality_control.qc_binaries.base import QC

_CHIPS  = ["GREEN", "RED"]
_FIBERS = ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]

_NAN_KEYS = ["NANSCI1", "NANSCI2", "NANSCI3", "NANSKY", "NANCAL"]


def _hdr_float(hdr, key):
    """Return float value for a header key, or None if absent."""
    if key not in hdr:
        return None
    val = hdr[key]
    return float(val[0] if isinstance(val, tuple) else val)


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

    def flux_finite_fraction(self):
        """NaN count from headers <= 1% of total L2 flux pixels."""
        hdr = self.kpf.headers["PRIMARY"]
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
        v = _hdr_float(self.kpf.headers["PRIMARY"], "ZEROFRAC")
        return v is not None and v < 0.5

    nonzero_flux._qc_key = "L2FLXOK"
    nonzero_flux._qc_comment = "QC: L2 zero-flux fraction < 0.5"
