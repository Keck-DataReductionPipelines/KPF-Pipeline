"""Diagnostics for KPF Level 2 (extracted spectra) data products."""

import numpy as np

from kpfpipe.quality_control.diagnostics.base import Diagnostics

_FIBERS = ("SCI1", "SCI2", "SCI3", "SKY", "CAL")
_CHIPS = ("GREEN", "RED")

_NAN_KEYS = {
    "SCI1": ("NANSCI1", "NaN pixel count, SCI1 (green+red)"),
    "SCI2": ("NANSCI2", "NaN pixel count, SCI2 (green+red)"),
    "SCI3": ("NANSCI3", "NaN pixel count, SCI3 (green+red)"),
    "SKY": ("NANSKY", "NaN pixel count, SKY (green+red)"),
    "CAL": ("NANCAL", "NaN pixel count, CAL (green+red)"),
}


class DiagL2(Diagnostics):
    LEVEL = "L2"

    def nan_counts(self):
        """Per-fiber NaN counts in {CHIP}_{FIBER}_FLUX, summed across chips.

        Always emits all five keys; fibers with no extracted data report 0.
        """
        results = {}
        for fiber, (kw, comment) in _NAN_KEYS.items():
            count = 0
            for chip in _CHIPS:
                arr = self.kpf.data.get(f"{chip}_{fiber}_FLUX")
                if arr is not None and np.size(arr) > 0:
                    count += int(np.sum(np.isnan(arr)))
            results[kw] = (count, comment)
        return results

    nan_counts._diag_name = "nan_counts"

    def zero_flux_fraction(self):
        """Fraction of L2 flux pixels exactly equal to zero across all FLUX exts.

        Skipped (no key written) when no L2 data is present — QCL2.DATAPRL2
        will have already failed in that case.
        """
        total_zero = 0
        total_pix = 0
        for chip in _CHIPS:
            for fiber in _FIBERS:
                arr = self.kpf.data.get(f"{chip}_{fiber}_FLUX")
                if arr is None or np.size(arr) == 0:
                    continue
                total_zero += int(np.sum(arr == 0))
                total_pix += int(np.size(arr))

        if total_pix == 0:
            return {}

        frac = round(float(total_zero / total_pix), 6)
        return {"ZEROFRAC": (frac, "Fraction of L2 flux pixels equal to zero")}

    zero_flux_fraction._diag_name = "zero_flux_fraction"
