"""Diagnostics for KPF Level 1 (assembled FFI) data products."""

import numpy as np

from kpfpipe.quality_control.diagnostics.base import Diagnostics


class DiagL1(Diagnostics):
    """Diagnostics for KPF Level 1 assembled FFI products."""

    LEVEL = "L1"

    def flux_percentiles(self):
        """Flux percentiles (99/90/50/10) of each assembled CCD frame, in e-.

        Returns
        -------
        dict
            Maps each ``FFI{G,R}{pct}P`` keyword to its ``(value, comment)``.
        """
        results = {}
        for chip, prefix in (("GREEN", "FFIG"), ("RED", "FFIR")):
            arr = self.kpf_obj.data[f"{chip}_CCD"]
            percentiles = np.nanpercentile(arr, [99, 90, 50, 10])
            for pct, value in zip([99, 90, 50, 10], percentiles, strict=True):
                results[f"{prefix}{pct}P"] = round(float(value), 3)
        return self._tag(**results)

    flux_percentiles._diag_name = "flux_percentiles"
