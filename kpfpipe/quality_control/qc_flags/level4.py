"""QC checks for KPF Level 4 (RVs and CCFs) data products.

Checks the presence of the RV/CCF data and the per-order BJD/BERV range
tolerances, plus the framework required-PRIMARY-keyword presence check. Each
result is a 0/1 flag written to QUALITY_CONTROL.
"""

import numpy as np

from kpfpipe import DETECTOR
from kpfpipe.quality_control.qc_flags.base import QC

_REQUIRED_RV_COLUMNS = frozenset(
    {"RV", "RV_ERR", "BJD_TDB", "BERV", "WAVE_START", "WAVE_END", "WEIGHT"}
)


class QCL4(QC):
    """QC checks for KPF Level 4 RV/CCF products."""

    LEVEL = "L4"

    def ccf_rv_present(self):
        """Each science orderlet has a populated CCF cube, variance, and RVs.

        Both stages of the split must have run: CrossCorrelation writes the CCF
        cube, its paired variance, and the seeded RV table (with NaN RV/RV_ERR),
        and RadialVelocity fills the RV column. A seeded-but-unfilled table
        (CrossCorrelation without a following RadialVelocity) fails here, since
        the RVs are the L4 product; so does an all-NaN CCF cube.

        Everything is per order, so CCF and table alike must span both chips.
        The velocity axis is not detector-derived and is pinned only by CCF_VAR
        agreeing with its CCF. The required columns are the EPRV set the chain
        writes, including the BJD_TDB/BERV/WEIGHT the DiagL4 dispersion metrics
        consume.
        """
        norder = DETECTOR["numorder"]
        for fiber in DETECTOR["sci_fibers"]:
            ccf = self.kpf_obj.data.get(f"{fiber}_CCF")
            rv = self.kpf_obj.data.get(f"{fiber}_RV")
            if np.shape(ccf)[:1] != (norder,):
                return False
            if np.shape(self.kpf_obj.data.get(f"{fiber}_CCF_VAR")) != np.shape(ccf):
                return False
            if not np.any(np.isfinite(np.asarray(ccf, dtype=float))):
                return False
            if rv is None or len(rv) != norder:
                return False
            if not _REQUIRED_RV_COLUMNS <= set(getattr(rv, "colnames", [])):
                return False
            if not np.any(np.isfinite(np.asarray(rv["RV"], dtype=float))):
                return False
        return True

    ccf_rv_present._qc_key = "DATAPRL4"

    def required_keywords_present(self):
        """Required PRIMARY keywords present -- not yet implemented; see ``QC``."""
        raise NotImplementedError(
            "KWRDPRL4 is pending a KPF-owned definition of a required keyword"
        )

    required_keywords_present._qc_key = "KWRDPRL4"

    def berv_within_tolerance(self):
        """BERVRNG (from DiagL4) within tolerance (4 cm/s)."""
        rng = float(self.kpf_obj.headers["QUALITY_CONTROL"]["BERVRNG"])
        return rng <= 0.04

    berv_within_tolerance._qc_key = "BERVOK"

    def bjd_within_tolerance(self):
        """BJDRNG (from DiagL4) within tolerance (1 sec)."""
        rng = float(self.kpf_obj.headers["QUALITY_CONTROL"]["BJDRNG"])
        return rng <= 1.0

    bjd_within_tolerance._qc_key = "BJDOK"
