"""Diagnostics for KPF Level 2 (extracted spectra) data products."""

import numpy as np

from kpfpipe.quality_control.diagnostics.base import Diagnostics

_CHIPS = ["GREEN", "RED"]
_FIBERS = ["SCI1", "SCI2", "SCI3", "SKY", "CAL"]
_SCI_FIBERS = ["SCI1", "SCI2", "SCI3"]
_CHIP_PREFIX = {"GREEN": "G", "RED": "R"}


class DiagL2(Diagnostics):
    """Diagnostics for KPF Level 2 extracted spectra products."""

    LEVEL = "L2"

    def nan_counts(self):
        """Count NaN pixels per fiber in ``{CHIP}_{FIBER}_FLUX``, summed across chips.

        Returns
        -------
        dict
            Maps each per-fiber NaN-count keyword to its ``(value, comment)``.
        """
        results = {}
        for fiber in _FIBERS:
            results[f"NAN{fiber}"] = sum(
                int(np.sum(np.isnan(self.kpf_obj.data[f"{chip}_{fiber}_FLUX"])))
                for chip in _CHIPS
            )
        return self._tag(**results)

    nan_counts._diag_name = "nan_counts"

    def zero_flux_fraction(self):
        """Compute the fraction of L2 flux pixels exactly equal to zero.

        Counts across all ``{CHIP}_{FIBER}_FLUX`` extensions.

        Returns
        -------
        dict
            ``{"ZEROFRAC": (fraction, comment)}``.
        """
        total_zero = 0
        total_pix = 0
        for chip in _CHIPS:
            for fiber in _FIBERS:
                arr = self.kpf_obj.data[f"{chip}_{fiber}_FLUX"]
                total_zero += int(np.sum(arr == 0))
                total_pix += int(np.size(arr))

        frac = round(float(total_zero / total_pix), 6)
        return self._tag(ZEROFRAC=frac)

    zero_flux_fraction._diag_name = "zero_flux_fraction"

    @staticmethod
    def _median_snr(flux, var):
        """Median over orders of the per-order 95th-percentile SNR.

        SNR = flux / sqrt(|var|); non-finite values are treated as 0 so a
        single bad pixel does not poison the percentile.
        """
        snr = flux / np.sqrt(np.abs(var))
        snr = np.where(np.isfinite(snr), snr, 0.0)
        per_order = np.nanpercentile(snr, 95, axis=1)
        return round(float(np.nanmedian(per_order)), 2)

    def snr(self):
        """Representative SNR per chip for the summed-SCI, SKY, and CAL fibers.

        A compact RV-stability indicator: per order take the 95th-percentile
        of flux/sqrt(|var|), then the median across orders. Summed SCI uses
        SCI1+SCI2+SCI3 flux and variance.
        """
        out = {}
        for chip, p in _CHIP_PREFIX.items():
            sci_f = sum(self.kpf_obj.data[f"{chip}_{f}_FLUX"] for f in _SCI_FIBERS)
            sci_v = sum(self.kpf_obj.data[f"{chip}_{f}_VAR"] for f in _SCI_FIBERS)
            out[f"{p}SNRSCI"] = self._median_snr(sci_f, sci_v)
            for fiber in ("SKY", "CAL"):
                out[f"{p}SNR{fiber}"] = self._median_snr(
                    self.kpf_obj.data[f"{chip}_{fiber}_FLUX"],
                    self.kpf_obj.data[f"{chip}_{fiber}_VAR"],
                )
        return self._tag(**out)

    snr._diag_name = "snr"

    def orderlet_flux_ratios(self):
        """Median inter-fiber flux ratios per chip (fiber-throughput stability).

        Ratio is the per-order median flux of fiber A over fiber B, then the
        median across orders. All fibers share the order/pixel grid, so no
        wavelength resampling is needed.
        """
        pairs = [
            ("SCI1", "SCI2", "FR12"),
            ("SCI3", "SCI2", "FR32"),
            ("SKY", "SCI2", "FRS2"),
            ("CAL", "SCI2", "FRC2"),
        ]
        out = {}
        for chip, p in _CHIP_PREFIX.items():
            for a, b, tag in pairs:
                fa = self.kpf_obj.data[f"{chip}_{a}_FLUX"]
                fb = self.kpf_obj.data[f"{chip}_{b}_FLUX"]
                ratio = np.nanmedian(fa, axis=1) / np.nanmedian(fb, axis=1)
                ratio = ratio[np.isfinite(ratio)]
                out[f"{p}{tag}"] = round(float(np.nanmedian(ratio)), 4)
        return self._tag(**out)

    orderlet_flux_ratios._diag_name = "orderlet_flux_ratios"
