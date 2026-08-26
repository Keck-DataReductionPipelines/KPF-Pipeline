"""Diagnostics for KPF Level 2 (extracted spectra) data products."""

import numpy as np

from kpfpipe.quality_control.diagnostics.base import Diagnostics

_CHIPS = ["GREEN", "RED"]
_FIBERS = ["SCI1", "SCI2", "SCI3", "SKY", "CAL"]


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

    def zero_counts(self):
        """Count non-positive pixels per fiber in ``{CHIP}_{FIBER}_FLUX``, summed
        across chips.

        Returns
        -------
        dict
            Maps each per-fiber non-positive-count keyword to its ``(value, comment)``.
        """
        results = {}
        for fiber in _FIBERS:
            results[f"ZERO{fiber}"] = sum(
                int(np.sum(self.kpf_obj.data[f"{chip}_{fiber}_FLUX"] <= 0))
                for chip in _CHIPS
            )
        return self._tag(**results)

    zero_counts._diag_name = "zero_counts"

    def _order_at(self, wavelength_nm):
        """``(chip, order)`` whose SCI2 wavelengths span ``wavelength_nm``.

        v2.12 hardcoded the order index per wavelength, which only holds for one
        instrument era; the WAVE arrays (Angstroms) carry the mapping directly.
        """
        for chip in _CHIPS:
            wave = np.asarray(self.kpf_obj.data[f"{chip}_SCI2_WAVE"])
            for order in range(wave.shape[0]):
                if (
                    np.nanmin(wave[order])
                    <= wavelength_nm * 10
                    <= np.nanmax(wave[order])
                ):
                    return chip, order
        raise LookupError(f"no SCI2 order covers {wavelength_nm} nm")

    def snr(self):
        """95th-percentile SNR of the summed-SCI, SKY and CAL spectra at five
        wavelengths.

        SNR = flux / sqrt(|var|) over the order carrying each wavelength;
        non-finite values are treated as 0 so a single bad pixel does not poison
        the percentile. The summed-SCI spectrum is SCI1+SCI2+SCI3 (v2.12 summed
        SCI1+SCI3+SCI3, dropping SCI2).
        """
        out = {}
        for wavelength in (452, 548, 652, 747, 852):
            chip, order = self._order_at(wavelength)
            for code, fibers in (
                ("SC", ("SCI1", "SCI2", "SCI3")),
                ("SK", ("SKY",)),
                ("CL", ("CAL",)),
            ):
                flux = sum(
                    self.kpf_obj.data[f"{chip}_{fiber}_FLUX"][order] for fiber in fibers
                )
                var = sum(
                    np.abs(self.kpf_obj.data[f"{chip}_{fiber}_VAR"][order])
                    for fiber in fibers
                )
                snr = flux / np.sqrt(var)
                out[f"SNR{code}{wavelength}"] = round(
                    float(np.nanpercentile(np.where(np.isfinite(snr), snr, 0.0), 95)), 3
                )
        return self._tag(**out)

    snr._diag_name = "snr"

    def order_flux_ratios(self):
        """Peak SCI2 flux at four wavelengths over the peak at 652 nm.

        Peak flux is the 95th percentile of the order's counts, as in v2.12. The
        ratios track the spectrum's colour, which moves with the target and with
        anything vignetting or defocusing part of the band.
        """
        peak = {}
        for wavelength in (452, 548, 652, 747, 852):
            chip, order = self._order_at(wavelength)
            flux = self.kpf_obj.data[f"{chip}_SCI2_FLUX"][order]
            peak[wavelength] = float(np.nanpercentile(flux, 95))
        return self._tag(
            **{
                f"FR{wavelength}652": round(peak[wavelength] / peak[652], 6)
                for wavelength in (452, 548, 747, 852)
            }
        )

    order_flux_ratios._diag_name = "order_flux_ratios"

    def orderlet_flux_ratios(self):
        """Median flux ratio of each orderlet to SCI2 near five wavelengths.

        Each orderlet is interpolated onto SCI2's wavelength grid before the
        ratio is formed (the orderlets do not share a wavelength solution), and
        the median is taken over the central 500 pixels of the order. The paired
        ``U`` keyword is the uncertainty on that median, 1.2533 * sigma /
        sqrt(n); v2.12 bootstrapped it from an unseeded RNG, which no
        deterministic pipeline can carry.
        """
        out = {}
        for wavelength in (452, 548, 652, 747, 852):
            chip, order = self._order_at(wavelength)
            sci2_wave = self.kpf_obj.data[f"{chip}_SCI2_WAVE"][order]
            sci2_flux = self.kpf_obj.data[f"{chip}_SCI2_FLUX"][order]
            center = np.size(sci2_flux) // 2
            window = slice(max(0, center - 250), center + 250)
            for fiber, code in (
                ("SCI1", "FR12"),
                ("SCI3", "FR32"),
                ("SKY", "FRS2"),
                ("CAL", "FRC2"),
            ):
                wave = self.kpf_obj.data[f"{chip}_{fiber}_WAVE"][order]
                flux = self.kpf_obj.data[f"{chip}_{fiber}_FLUX"][order]
                rising = np.argsort(wave)
                interpolated = np.interp(sci2_wave, wave[rising], flux[rising])
                ratio = (interpolated / sci2_flux)[window]
                ratio = ratio[np.isfinite(ratio)]
                out[f"{code}M{wavelength}"] = round(float(np.median(ratio)), 6)
                out[f"{code}U{wavelength}"] = round(
                    float(1.2533 * np.std(ratio) / np.sqrt(np.size(ratio))), 6
                )
        return self._tag(**out)

    orderlet_flux_ratios._diag_name = "orderlet_flux_ratios"
