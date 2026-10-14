"""Diagnostics for KPF Level 2 (extracted spectra) data products."""

import numpy as np

from kpfpipe import CHIPS, FIBERS, SCI_FIBERS
from kpfpipe.quality_control.applicability import applicability
from kpfpipe.quality_control.diagnostics.base import Diagnostics


class DiagL2(Diagnostics):
    """Diagnostics for KPF Level 2 extracted spectra products."""

    LEVEL = "L2"

    def nan_counts(self):
        """NAN{FIBER}: NaN pixels per fiber in ``{CHIP}_{FIBER}_FLUX``, both chips."""
        results = {}
        for fiber in FIBERS:
            results[f"NAN{fiber}"] = sum(
                int(np.sum(np.isnan(self.kpf_obj.data[f"{chip}_{fiber}_FLUX"])))
                for chip in CHIPS
            )
        return results

    nan_counts._diag_name = "nan_counts"

    def zero_counts(self):
        """ZERO{FIBER}: non-positive pixels per fiber in ``{CHIP}_{FIBER}_FLUX``."""
        results = {}
        for fiber in FIBERS:
            results[f"ZERO{fiber}"] = sum(
                int(np.sum(self.kpf_obj.data[f"{chip}_{fiber}_FLUX"] <= 0))
                for chip in CHIPS
            )
        return results

    zero_counts._diag_name = "zero_counts"

    def _order_at(self, wavelength_nm):
        """``(chip, order)`` whose SCI2 wavelengths span ``wavelength_nm``.

        Looked up from the WAVE arrays (Angstroms) rather than a hardcoded order
        index, since the order-to-wavelength mapping shifts across instrument
        eras.
        """
        for chip in CHIPS:
            wave = np.asarray(self.kpf_obj.data[f"{chip}_SCI2_WAVE"])
            for order in range(wave.shape[0]):
                if (
                    np.nanmin(wave[order])
                    <= wavelength_nm * 10
                    <= np.nanmax(wave[order])
                ):
                    return chip, order
        raise LookupError(f"no SCI2 order covers {wavelength_nm} nm")

    @staticmethod
    def _clipped(flux):
        """``flux`` with NaN as zero, and the mask of its 1st-99th percentile pixels.

        The top 1% is cut because the strongest lines are often saturated.
        """
        flux = np.nan_to_num(flux, nan=0.0)
        low, high = np.percentile(flux, (1, 99))
        return flux, (flux >= low) & (flux <= high)

    def band_snr(self):
        """95th-percentile SNR of the summed-SCI, SKY and CAL spectra at five
        wavelengths.

        SNR = flux / sqrt(|var|) over the order carrying each wavelength;
        non-finite values are treated as 0 so a single bad pixel does not poison
        the percentile. The summed-SCI spectrum is SCI1+SCI2+SCI3.
        """
        out = {}
        for index, wavelength in enumerate((452, 548, 652, 747, 852), start=1):
            chip, order = self._order_at(wavelength)
            for code, fibers in (
                ("SC", SCI_FIBERS),
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
            out[f"EXSNR{index}"] = out[f"SNRSC{wavelength}"]
            out[f"EXSNRW{index}"] = wavelength * 10.0
        return out

    band_snr._diag_name = "band_snr"

    def band_throughput(self):
        """Integrated flux (throughput) for summed-SCI, SKY, and CAL spectra at five
        wavelengths; summed flux between 1st - 99th percentile (strongest lines are
        often saturated). NaN pixels are treated as zero. The summed-SCI spectrum is
        SCI1+SCI2+SCI3.
        """
        out = {}
        for wavelength in (452, 548, 652, 747, 852):
            chip, order = self._order_at(wavelength)
            for code, fibers in (
                ("SC", SCI_FIBERS),
                ("SK", ("SKY",)),
                ("CL", ("CAL",)),
            ):
                flux = sum(
                    np.nan_to_num(
                        self.kpf_obj.data[f"{chip}_{fiber}_FLUX"][order], nan=0.0
                    )
                    for fiber in fibers
                )
                flux, keep = self._clipped(flux)
                out[f"IFX{code}{wavelength}"] = round(float(np.sum(flux[keep])), 3)
        return out

    band_throughput._diag_name = "band_throughput"

    def band_continuum(self):
        """Continuum flux for summed-SCI, SKY, and CAL emission spectra at five
        wavelengths; estimated using a simple median. NaN pixels are treated
        as zero. The summed-SCI spectrum is SCI1+SCI2+SCI3.
        """
        out = {}
        for wavelength in (452, 548, 652, 747, 852):
            chip, order = self._order_at(wavelength)
            for code, fibers in (
                ("SC", SCI_FIBERS),
                ("SK", ("SKY",)),
                ("CL", ("CAL",)),
            ):
                flux = sum(
                    np.nan_to_num(
                        self.kpf_obj.data[f"{chip}_{fiber}_FLUX"][order], nan=0.0
                    )
                    for fiber in fibers
                )
                out[f"CFX{code}{wavelength}"] = round(float(np.median(flux)), 3)
        return out

    band_continuum._diag_name = "band_continuum"

    def order_flux_ratios(self):
        """Peak SCI2 flux at four wavelengths over the peak at 652 nm.

        Peak flux is the 95th percentile of the order's counts. The ratios track
        the spectrum's colour, which moves with the target and with anything
        vignetting or defocusing part of the band. On an emission source the
        95th percentile lands on the near-zero continuum, so the order's
        integrated flux (as in ``band_throughput``) stands in for the peak.
        """
        emission = applicability.is_emission(self.kpf_obj)
        peak = {}
        for wavelength in (452, 548, 652, 747, 852):
            chip, order = self._order_at(wavelength)
            flux = self.kpf_obj.data[f"{chip}_SCI2_FLUX"][order]
            if emission:
                flux, keep = self._clipped(flux)
                peak[wavelength] = float(np.sum(flux[keep]))
            else:
                peak[wavelength] = float(np.nanpercentile(flux, 95))
        return {
            f"FR{wavelength}652": round(peak[wavelength] / peak[652], 6)
            for wavelength in (452, 548, 747, 852)
        }

    order_flux_ratios._diag_name = "order_flux_ratios"

    def fiber_flux_ratios(self):
        """Median flux ratio of each orderlet to SCI2 near five wavelengths.

        Each orderlet is interpolated onto SCI2's wavelength grid before the
        ratio is formed (the orderlets do not share a wavelength solution), and
        the median is taken over the central 500 pixels of the order. The paired
        ``U`` keyword is the uncertainty on that median, 1.2533 * sigma /
        sqrt(n) -- closed-form rather than bootstrapped, since a bootstrap needs
        an RNG no deterministic pipeline can carry.

        On an emission source a pixel ratio is noise off the lines, so the ratio
        is instead of the integrated fluxes (as in ``band_throughput``), with
        ``U`` propagated from VAR over the same pixels.
        """
        if applicability.is_emission(self.kpf_obj):
            return self._integrated_fiber_flux_ratios()
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
        return out

    fiber_flux_ratios._diag_name = "fiber_flux_ratios"

    def _integrated_fiber_flux_ratios(self):
        """``fiber_flux_ratios`` for an emission source: ratios of integrated flux."""

        def integrated(chip, fiber, order):
            flux, keep = self._clipped(self.kpf_obj.data[f"{chip}_{fiber}_FLUX"][order])
            var = np.abs(self.kpf_obj.data[f"{chip}_{fiber}_VAR"][order])
            return np.sum(flux[keep]), np.nansum(var[keep])

        out = {}
        for wavelength in (452, 548, 652, 747, 852):
            chip, order = self._order_at(wavelength)
            sci2, sci2_var = integrated(chip, "SCI2", order)
            for fiber, code in (
                ("SCI1", "FR12"),
                ("SCI3", "FR32"),
                ("SKY", "FRS2"),
                ("CAL", "FRC2"),
            ):
                flux, var = integrated(chip, fiber, order)
                ratio = flux / sci2
                out[f"{code}M{wavelength}"] = round(float(ratio), 6)
                out[f"{code}U{wavelength}"] = round(
                    float(abs(ratio) * np.sqrt(var / flux**2 + sci2_var / sci2**2)), 6
                )
        return out
