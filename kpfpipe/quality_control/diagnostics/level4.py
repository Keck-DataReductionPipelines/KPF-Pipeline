"""Diagnostics for KPF Level 4 (RVs and CCFs) data products.

Per-order BJD and barycentric-RV dispersion statistics: photon-weighted means
(``BJDMEAN``/``BERVMEAN``) and the spread of the per-order values about them
(``BJDSTD``/``BJDRNG`` in seconds, ``BERVSTD``/``BERVRNG`` in m/s), weighted by the
per-order CCF-combination ``WEIGHT`` and computed on the primary science orderlet
(SCI2).
"""

import numpy as np

from kpfpipe.quality_control.diagnostics.base import Diagnostics

# Primary science orderlet the v2.12 single-table statistics map onto.
_SCI_REF = "SCI2"
_SEC_PER_DAY = 86400.0
_KMS_TO_MS = 1000.0


class DiagL4(Diagnostics):
    """Diagnostics for KPF Level 4 RV/CCF products."""

    LEVEL = "L4"

    @staticmethod
    def _weighted_dispersion(x, w):
        """Return (weighted mean, weighted std, peak-to-peak range) of ``x``.

        The mean uses all weights; the std and range use only nonzero-weight
        entries, matching v2.12. Non-finite samples are dropped.
        """
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        good = np.isfinite(x) & np.isfinite(w)
        x, w = x[good], w[good]
        wmean = float(np.sum(w * x) / w.sum())
        std = float(np.sqrt(np.sum(w * (x - wmean) ** 2) / w.sum()))
        nz = w != 0
        rng = float(x[nz].max() - x[nz].min())
        return wmean, std, rng

    def bjd_dispersion(self):
        """Photon-weighted mean BJD_TDB and its per-order spread (SCI2)."""
        tab = self.kpf_obj.data[f"{_SCI_REF}_RV"]
        mean, std, rng = self._weighted_dispersion(tab["BJD_TDB"], tab["WEIGHT"])
        return self._tag(
            BJDMEAN=round(mean, 6),
            BJDSTD=round(std * _SEC_PER_DAY, 4),
            BJDRNG=round(rng * _SEC_PER_DAY, 4),
        )

    bjd_dispersion._diag_name = "bjd_dispersion"

    def berv_dispersion(self):
        """Weighted-mean barycentric RV correction and per-order spread (SCI2)."""
        tab = self.kpf_obj.data[f"{_SCI_REF}_RV"]
        mean, std, rng = self._weighted_dispersion(tab["BERV"], tab["WEIGHT"])
        return self._tag(
            BERVMEAN=round(mean, 6),
            BERVSTD=round(std * _KMS_TO_MS, 4),
            BERVRNG=round(rng * _KMS_TO_MS, 4),
        )

    berv_dispersion._diag_name = "berv_dispersion"
