"""Diagnostics for KPF Level 4 (RVs and CCFs) data products.

Ports the per-order BJD and barycentric-RV dispersion statistics from the
v2.12 ``AnalyzeL2.compute_statistics``: photon-weighted means (``CCFBJD`` /
``CCFBCV``) and the spread of the per-order values about them (``BJDSTD`` /
``BJDRNG`` in seconds, ``BCVSTD`` / ``BCVRNG`` in m/s) plus the per-order BERV
percent deviation (``MAXPCBCV`` / ``MINPCBCV``). They are weighted by the
per-order CCF-combination ``WEIGHT`` column and computed on the primary science
orderlet (SCI2), matching the old single-table behavior.

The v2.12 ``AGES*``/``AGEU*`` wave-cal-age metrics needed the time-series
database (calibration history) and are out of scope for the DB-free vNext L4.
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

    def _sci_rv_table(self):
        """Return the SCI2 per-order RV table if it carries BJD_TDB/BERV/WEIGHT.

        Returns None when the orderlet has no RV table (e.g. unilluminated) or
        predates the WEIGHT column, so the dispersion metrics are simply skipped
        rather than guessed.
        """
        tab = self.kpf_obj.data.get(f"{_SCI_REF}_RV")
        if tab is None or len(tab) == 0:
            return None
        if not {"BJD_TDB", "BERV", "WEIGHT"} <= set(getattr(tab, "colnames", [])):
            return None
        return tab

    @staticmethod
    def _weighted_dispersion(x, w):
        """Return (weighted mean, weighted std, peak-to-peak range) of ``x``.

        The mean uses all weights; the std and range use only nonzero-weight
        entries, matching v2.12. Non-finite samples are dropped. Returns None
        when no positive total weight remains.
        """
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        good = np.isfinite(x) & np.isfinite(w)
        x, w = x[good], w[good]
        if w.sum() <= 0:
            return None
        wmean = float(np.sum(w * x) / w.sum())
        std = float(np.sqrt(np.sum(w * (x - wmean) ** 2) / w.sum()))
        nz = w != 0
        rng = float(x[nz].max() - x[nz].min())
        return wmean, std, rng

    def bjd_dispersion(self):
        """Photon-weighted mean BJD_TDB and its per-order spread (SCI2)."""
        tab = self._sci_rv_table()
        if tab is None:
            return {}
        stats = self._weighted_dispersion(tab["BJD_TDB"], tab["WEIGHT"])
        if stats is None:
            return {}
        mean, std, rng = stats
        return {
            "CCFBJD": (round(mean, 6), "Weighted-mean photon BJD_TDB (SCI2)"),
            "BJDSTD": (
                round(std * _SEC_PER_DAY, 4),
                "Weighted std of per-order BJD [s] (SCI2)",
            ),
            "BJDRNG": (
                round(rng * _SEC_PER_DAY, 4),
                "Per-order BJD range [s] (SCI2)",
            ),
        }

    bjd_dispersion._diag_name = "bjd_dispersion"

    def bcv_dispersion(self):
        """Weighted-mean barycentric RV correction and per-order spread (SCI2)."""
        tab = self._sci_rv_table()
        if tab is None:
            return {}
        stats = self._weighted_dispersion(tab["BERV"], tab["WEIGHT"])
        if stats is None:
            return {}
        mean, std, rng = stats
        out = {
            "CCFBCV": (
                round(mean, 6),
                "Weighted-mean barycentric RV correction [km/s] (SCI2)",
            ),
            "BCVSTD": (
                round(std * _KMS_TO_MS, 4),
                "Weighted std of per-order BERV [m/s] (SCI2)",
            ),
            "BCVRNG": (
                round(rng * _KMS_TO_MS, 4),
                "Per-order BERV range [m/s] (SCI2)",
            ),
        }
        # Per-order BERV percent deviation from the weighted mean, over
        # nonzero-weight orders (v2.12 QCPCBCV feeds on these).
        berv = np.asarray(tab["BERV"], dtype=float)
        w = np.asarray(tab["WEIGHT"], dtype=float)
        nz = np.isfinite(berv) & np.isfinite(w) & (w != 0)
        if mean != 0 and np.any(nz):
            perc = (berv[nz] - mean) / mean * 100.0
            out["MAXPCBCV"] = (
                round(float(perc.max()), 4),
                "Max per-order BERV deviation from mean [%] (SCI2)",
            )
            out["MINPCBCV"] = (
                round(float(perc.min()), 4),
                "Min per-order BERV deviation from mean [%] (SCI2)",
            )
        return out

    bcv_dispersion._diag_name = "bcv_dispersion"
