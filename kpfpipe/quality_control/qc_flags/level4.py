"""QC checks for KPF Level 4 (RVs and CCFs) data products.

Checks the presence of the RV/CCF data and the per-order BJD/BERV range
tolerances, plus the framework required-PRIMARY-keyword presence check. Each
result is a 0/1 flag written to QUALITY_CONTROL.
"""

import numpy as np

from kpfpipe.quality_control.qc_flags.base import QC

_SCI_FIBERS = ["SCI1", "SCI2", "SCI3"]
# Columns every science RV table must carry (RV product + per-order BJD/BERV/WEIGHT).
_REQUIRED_RV_COLUMNS = frozenset({"RV", "BJD_TDB", "BERV", "WEIGHT"})
# Max per-order peak-to-peak range tolerated across orders (SCI2).
_BERV_RNG_TOL_MS = 0.1
_BJD_RNG_TOL_S = 1.0


class QCL4(QC):
    """QC checks for KPF Level 4 RV/CCF products."""

    LEVEL = "L4"

    def ccf_rv_present(self):
        """Each science orderlet has a non-empty CCF cube and computed RVs.

        Both stages of the split must have run: CrossCorrelation writes the CCF
        cube and seeds the RV table (with NaN RV/RV_ERR), and RadialVelocity fills
        the RV column. A seeded-but-unfilled table (CrossCorrelation without a
        following RadialVelocity) fails here, since the RVs are the L4 product.
        The table must also carry the per-order BJD_TDB/BERV/WEIGHT columns the
        DiagL4 dispersion metrics consume, so a table missing them fails here.
        """
        for fiber in _SCI_FIBERS:
            ccf = self.kpf_obj.data.get(f"{fiber}_CCF")
            rv = self.kpf_obj.data.get(f"{fiber}_RV")
            if ccf is None or np.size(ccf) == 0:
                return False
            if rv is None or len(rv) == 0:
                return False
            if not _REQUIRED_RV_COLUMNS <= set(getattr(rv, "colnames", [])):
                return False
            if not np.any(np.isfinite(np.asarray(rv["RV"], dtype=float))):
                return False
        return True

    ccf_rv_present._qc_key = "DATAPRL4"

    def required_keywords_present(self):
        """Every registry-required PRIMARY keyword for L4 is present (presence only)."""
        return self._required_primary_keywords() <= set(self.kpf_obj.headers["PRIMARY"])

    required_keywords_present._qc_key = "KWRDPRL4"

    def _sci2_is_target(self):
        """Whether SCI2 is star-illuminated, from INSTRUMENT_HEADER's SCI-OBJ.

        Every L4 frame carries SCI-OBJ (CrossCorrelation requires it upstream and
        fails loud otherwise), so an absent keyword is a malformed frame, not a
        non-target source; raise rather than silently defaulting to not-target.
        """
        inst = self.kpf_obj.headers.get("INSTRUMENT_HEADER", {})
        if "SCI-OBJ" not in inst:
            raise ValueError(
                "SCI-OBJ not in INSTRUMENT_HEADER; cannot determine the SCI2 "
                "illumination source for the BERV/BJD tolerance gates"
            )
        return str(inst["SCI-OBJ"]).strip().lower() == "target"

    def berv_within_tolerance(self):
        """BERVRNG (from DiagL4) within tolerance.

        Only applies when SCI2 is star-illuminated (SCI-OBJ == 'target'); other
        sources pass (no meaningful barycentric dispersion). On a target frame an
        absent BERVRNG (non-finite barycorr / non-positive weight) fails; an
        absent SCI-OBJ raises (see _sci2_is_target).
        """
        if not self._sci2_is_target():
            return True
        rng = self._hdr_float(self.kpf_obj.headers["QUALITY_CONTROL"], "BERVRNG")
        return rng is not None and rng <= _BERV_RNG_TOL_MS

    berv_within_tolerance._qc_key = "BERVOK"

    def bjd_within_tolerance(self):
        """BJDRNG (from DiagL4) within tolerance.

        Only applies when SCI2 is star-illuminated (SCI-OBJ == 'target'); other
        sources pass. On a target frame an absent BJDRNG fails; an absent SCI-OBJ
        raises (see _sci2_is_target).
        """
        if not self._sci2_is_target():
            return True
        rng = self._hdr_float(self.kpf_obj.headers["QUALITY_CONTROL"], "BJDRNG")
        return rng is not None and rng <= _BJD_RNG_TOL_S

    bjd_within_tolerance._qc_key = "BJDOK"
