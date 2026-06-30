"""Diagnostics for KPF Level 2 (extracted spectra) data products."""

import numpy as np

from kpfpipe.quality_control.diagnostics.base import Diagnostics

_FIBERS = ("SCI1", "SCI2", "SCI3", "SKY", "CAL")
_CHIPS = ("GREEN", "RED")
_SCI_FIBERS = ("SCI1", "SCI2", "SCI3")
_SNR_PERCENTILE = 95
_CHIP_PREFIX = {"GREEN": "G", "RED": "R"}

# Fiber -> NaN-count keyword. The FITS comment comes from the registry (via
# the _tag helper / set_keyword), so it is not duplicated here.
_NAN_KEYS = {
    "SCI1": "NANSCI1",
    "SCI2": "NANSCI2",
    "SCI3": "NANSCI3",
    "SKY": "NANSKY",
    "CAL": "NANCAL",
}


class DiagL2(Diagnostics):
    """Diagnostics for KPF Level 2 extracted spectra products."""

    LEVEL = "L2"

    def nan_counts(self):
        """Count NaN pixels per fiber in `{CHIP}_{FIBER}_FLUX`, summed across chips.

        Always emits all five keys; fibers with no extracted data report 0.

        Returns
        -------
        dict
            Maps each per-fiber NaN-count keyword to its ``(value, comment)``.
        """
        results = {}
        for fiber, kw in _NAN_KEYS.items():
            count = 0
            for chip in _CHIPS:
                arr = self.kpf_obj.data.get(f"{chip}_{fiber}_FLUX")
                if arr is not None and np.size(arr) > 0:
                    count += int(np.sum(np.isnan(arr)))
            results[kw] = count
        return self._tag(**results)

    nan_counts._diag_name = "nan_counts"

    def zero_flux_fraction(self):
        """Compute the fraction of L2 flux pixels exactly equal to zero.

        Counts across all `{CHIP}_{FIBER}_FLUX` extensions. Skipped (no key
        written) when no L2 data is present — `QCL2.DATAPRL2` will have
        already failed in that case.

        Returns
        -------
        dict
            ``{"ZEROFRAC": (fraction, comment)}``, or empty when no L2 data
            is present.
        """
        total_zero = 0
        total_pix = 0
        for chip in _CHIPS:
            for fiber in _FIBERS:
                arr = self.kpf_obj.data.get(f"{chip}_{fiber}_FLUX")
                if arr is None or np.size(arr) == 0:
                    continue
                total_zero += int(np.sum(arr == 0))
                total_pix += int(np.size(arr))

        if total_pix == 0:
            return {}

        frac = round(float(total_zero / total_pix), 6)
        return self._tag(ZEROFRAC=frac)

    zero_flux_fraction._diag_name = "zero_flux_fraction"

    def _arr(self, chip, fiber, kind):
        """Return the (norder, ncol) FLUX or VAR array for a fiber, or None."""
        arr = self.kpf_obj.data.get(f"{chip}_{fiber}_{kind}")
        if arr is None:
            return None
        arr = np.asarray(arr)
        return arr if (arr.ndim == 2 and arr.size > 0) else None

    @staticmethod
    def _median_snr(flux, var):
        """Median over orders of the per-order 95th-percentile SNR.

        SNR = flux / sqrt(|var|); non-finite values are treated as 0 so a
        single bad pixel does not poison the percentile.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = flux / np.sqrt(np.abs(var))
        snr = np.where(np.isfinite(snr), snr, 0.0)
        per_order = np.nanpercentile(snr, _SNR_PERCENTILE, axis=1)
        return round(float(np.nanmedian(per_order)), 2)

    def snr(self):
        """Representative SNR per chip for the summed-SCI, SKY, and CAL fibers.

        A compact RV-stability indicator: per order take the 95th-percentile
        of flux/sqrt(|var|), then the median across orders. Summed SCI uses
        SCI1+SCI2+SCI3 flux and variance. Skipped per (chip, fiber) when that
        data is absent.
        """
        out = {}
        for chip, p in _CHIP_PREFIX.items():
            sci_f = [self._arr(chip, f, "FLUX") for f in _SCI_FIBERS]
            sci_v = [self._arr(chip, f, "VAR") for f in _SCI_FIBERS]
            if all(a is not None for a in sci_f + sci_v):
                out[f"{p}SNRSCI"] = self._median_snr(sum(sci_f), sum(sci_v))
            for fiber in ("SKY", "CAL"):
                f = self._arr(chip, fiber, "FLUX")
                v = self._arr(chip, fiber, "VAR")
                if f is not None and v is not None:
                    out[f"{p}SNR{fiber}"] = self._median_snr(f, v)
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
                fa = self._arr(chip, a, "FLUX")
                fb = self._arr(chip, b, "FLUX")
                if fa is None or fb is None:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.nanmedian(fa, axis=1) / np.nanmedian(fb, axis=1)
                ratio = ratio[np.isfinite(ratio)]
                if ratio.size == 0:
                    continue
                out[f"{p}{tag}"] = round(float(np.nanmedian(ratio)), 4)
        return self._tag(**out)

    orderlet_flux_ratios._diag_name = "orderlet_flux_ratios"
