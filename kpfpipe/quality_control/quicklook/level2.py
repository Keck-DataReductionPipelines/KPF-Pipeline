"""L2 quicklook plots for extracted KPF 1D spectra."""

import matplotlib.pyplot as plt
import numpy as np

from kpfpipe import DETECTOR
from kpfpipe.quality_control.quicklook.base import Plot

_FIBERS = ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]
_SNR_PERCENTILE = 95
_FLUX_PERCENTILE = 95


class PlotL2(Plot):
    """Quicklook plots for KPF L2 (extracted 1D spectra) data.

    Parameters
    ----------
    l2_obj : KPF2
        Extracted L2 frame (post-SpectralExtraction; requires
        WavelengthCalibration to have populated the per-fiber WAVE arrays).
    output_dir : str or None
        Directory to save PNG files. None returns the Figure only.
    obs_id : str or None
        Observation ID for titles/filenames. If None, falls back to
        ``l2_obj.obs_id`` (populated on every construction path).
    """

    LEVEL = "L2"
    _PLOT_METHODS = (
        "snr_per_order",
        "peak_flux",
        "spectrum_single_order",
        "spectrum_one_row",
        "orderlet_flux_ratios",
    )

    def __init__(self, l2_obj, output_dir=None, obs_id=None):
        super().__init__(l2_obj, output_dir, obs_id)
        self.fibers = _FIBERS

    # Data access helpers

    def _flux(self, chip, fiber):
        """Return the (norder, ncol) flux array for one fiber, or None."""
        arr = self.kpf_obj.data.get(f"{chip.upper()}_{fiber.upper()}_FLUX")
        arr = np.asarray(arr) if arr is not None else None
        if arr is None or arr.ndim != 2 or arr.size == 0:
            return None
        return arr

    def _var(self, chip, fiber):
        arr = self.kpf_obj.data.get(f"{chip.upper()}_{fiber.upper()}_VAR")
        arr = np.asarray(arr) if arr is not None else None
        if arr is None or arr.ndim != 2 or arr.size == 0:
            return None
        return arr

    def _wave(self, chip, fiber):
        """Return the (norder, ncol) wavelength array, or None if not populated."""
        arr = self.kpf_obj.data.get(f"{chip.upper()}_{fiber.upper()}_WAVE")
        arr = np.asarray(arr) if arr is not None else None
        if arr is None or arr.ndim != 2 or arr.size == 0:
            return None
        return arr

    def _has_chip(self, chip):
        return self._flux(chip, "SCI2") is not None

    def _require_wave(self, chip, fiber="SCI2"):
        """Return a fiber's (norder, ncol) wavelength array, or raise if absent.

        Fails loudly so a wavelength-less plot can't pass for a calibrated one;
        run WavelengthCalibration before PlotL2.
        """
        wave = self._wave(chip, fiber)
        if wave is None:
            raise ValueError(
                f"{chip.upper()}_{fiber.upper()}_WAVE is not populated; "
                "run WavelengthCalibration before PlotL2."
            )
        return wave

    @staticmethod
    def _snr(flux, var):
        """SNR = signal / sqrt(|variance|), elementwise (may be negative)."""
        snr = flux / np.sqrt(np.abs(var))
        return np.where(np.isfinite(snr), snr, 0.0)

    # Plots

    def snr_per_order(self, chip):
        """Per-order SNR (95th pctile of flux/sqrt(|var|)) for each fiber plus
        the summed-SCI orderlet, vs order-center wavelength."""
        chip = chip.upper()
        if not self._has_chip(chip):
            return None

        wave = self._require_wave(chip)
        x = wave[:, wave.shape[1] // 2]
        xlabel = "Wavelength (Ang) at order center"

        fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True)

        for fiber in self.fibers:
            flux, var = self._flux(chip, fiber), self._var(chip, fiber)
            if flux is None or var is None:
                continue
            snr = self._snr(flux, var)
            per_order = np.nanpercentile(snr, _SNR_PERCENTILE, axis=1)
            ax.plot(x, per_order, marker=".", linewidth=1, label=fiber, alpha=0.8)

        sci_flux = [self._flux(chip, f) for f in DETECTOR["sci_fibers"]]
        sci_var = [self._var(chip, f) for f in DETECTOR["sci_fibers"]]
        if all(a is not None for a in sci_flux + sci_var):
            snr_sum = self._snr(sum(sci_flux), sum(sci_var))
            per_order = np.nanpercentile(snr_sum, _SNR_PERCENTILE, axis=1)
            ax.plot(
                x,
                per_order,
                marker=".",
                linewidth=1.5,
                color="k",
                label="SCI1+SCI2+SCI3",
            )

        ax.set_title(
            f"L2 SNR - {chip.capitalize()} CCD: {self.obs_id} - {self.name}",
            fontsize=14,
        )
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(f"SNR ({_SNR_PERCENTILE}th pctile per order)", fontsize=14)
        ax.legend(fontsize=9, ncol=3)
        ax.grid(alpha=0.3)
        return self._decorate_and_save(fig, "snr_per_order", chip)

    def peak_flux(self, chip):
        """Per-order peak flux (95th pctile counts) for each fiber and the
        summed-SCI orderlet."""
        chip = chip.upper()
        if not self._has_chip(chip):
            return None

        wave = self._require_wave(chip)
        x = wave[:, wave.shape[1] // 2]
        xlabel = "Wavelength (Ang) at order center"

        fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True)
        for fiber in self.fibers:
            flux = self._flux(chip, fiber)
            if flux is None:
                continue
            per_order = np.nanpercentile(flux, _FLUX_PERCENTILE, axis=1)
            ax.plot(x, per_order, marker=".", linewidth=1, label=fiber, alpha=0.8)

        sci_flux = [self._flux(chip, f) for f in DETECTOR["sci_fibers"]]
        if all(a is not None for a in sci_flux):
            per_order = np.nanpercentile(sum(sci_flux), _FLUX_PERCENTILE, axis=1)
            ax.plot(
                x,
                per_order,
                marker=".",
                linewidth=1.5,
                color="k",
                label="SCI1+SCI2+SCI3",
            )

        ax.set_yscale("log")
        ax.set_title(
            f"L2 Peak Flux - {chip.capitalize()} CCD: {self.obs_id} - {self.name}",
            fontsize=14,
        )
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(
            f"Peak flux (e-, {_FLUX_PERCENTILE}th pctile per order)", fontsize=14
        )
        ax.legend(fontsize=9, ncol=3)
        ax.grid(alpha=0.3)
        return self._decorate_and_save(fig, "peak_flux", chip)

    def spectrum_single_order(self, chip, order=None):
        """Overplot the science orderlets (and SKY/CAL) for a single spectral
        order vs wavelength.

        Parameters
        ----------
        chip : str
            'GREEN' or 'RED'.
        order : int or None
            1-indexed spectral order. Defaults to a representative order
            near the middle of the chip.
        """
        chip = chip.upper()
        if not self._has_chip(chip):
            return None
        norder = self._flux(chip, "SCI2").shape[0]
        if order is None:
            order = norder // 2
        if not (1 <= order <= norder):
            raise ValueError(f"order {order} out of range 1..{norder} for {chip}")
        o = order - 1

        x = self._require_wave(chip)[o]
        xlabel = "Wavelength (Ang)"

        fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True)
        for fiber in self.fibers:
            flux = self._flux(chip, fiber)
            if flux is None:
                continue
            ax.plot(x, flux[o], linewidth=0.7, label=fiber, alpha=0.8)

        ax.set_title(
            f"L2 Spectrum - {chip.capitalize()} order {order}: "
            f"{self.obs_id} - {self.name}",
            fontsize=14,
        )
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Flux (e-)", fontsize=14)
        ax.legend(fontsize=9, ncol=3)
        ax.grid(alpha=0.3)
        return self._decorate_and_save(fig, "spectrum_single_order", chip)

    def spectrum_one_row(self, chip, fibers=None):
        """Stacked per-fiber spectrum: one panel per fiber, all orders of that
        fiber concatenated, x-axis is concatenated wavelength."""
        chip = chip.upper()
        if not self._has_chip(chip):
            return None
        if fibers is None:
            fibers = self.fibers

        self._require_wave(chip)  # fail loudly before building the figure
        xlabel = "Wavelength (Ang)"

        fig, axes = plt.subplots(
            len(fibers),
            1,
            figsize=(12, 2.0 * len(fibers)),
            sharex=True,
            tight_layout=True,
        )
        if len(fibers) == 1:
            axes = [axes]

        for ax, fiber in zip(axes, fibers, strict=True):
            flux = self._flux(chip, fiber)
            if flux is None:
                ax.set_visible(False)
                continue
            wave = self._require_wave(chip, fiber)
            order = np.argsort(wave.ravel())
            ax.plot(wave.ravel()[order], flux.ravel()[order], linewidth=0.4, color="C0")
            # robust y-limits to suppress cosmic-ray / edge outliers
            finite = flux[np.isfinite(flux)]
            if finite.size:
                ax.set_ylim(
                    np.nanpercentile(finite, 0.5), np.nanpercentile(finite, 99.8)
                )
            ax.set_ylabel(fiber, fontsize=11)
            ax.grid(alpha=0.3)

        axes[0].set_title(
            f"L2 Spectrum - {chip.capitalize()} CCD: {self.obs_id} - {self.name}",
            fontsize=14,
        )
        last_visible = axes[0]
        for ax in axes:
            if ax.get_visible():
                last_visible = ax
        last_visible.set_xlabel(xlabel, fontsize=14)
        return self._decorate_and_save(fig, "spectrum_one_row", chip)

    def orderlet_flux_ratios(self, chip):
        """Per-order orderlet flux ratios (SCI1/SCI2, SCI3/SCI2, SCI1/SCI3,
        SKY/SCI2, CAL/SCI2) vs order-center wavelength, one panel each, with
        the all-order median annotated."""
        chip = chip.upper()
        if not self._has_chip(chip):
            return None

        pairs = [
            ("SCI1", "SCI2"),
            ("SCI3", "SCI2"),
            ("SCI1", "SCI3"),
            ("SKY", "SCI2"),
            ("CAL", "SCI2"),
        ]
        wave = self._require_wave(chip)
        x = wave[:, wave.shape[1] // 2]
        xlabel = "Wavelength (Ang) at order center"

        fig, axes = plt.subplots(
            len(pairs),
            1,
            figsize=(11, 1.8 * len(pairs)),
            sharex=True,
            tight_layout=True,
        )
        for ax, (a, b) in zip(axes, pairs, strict=True):
            fa, fb = self._flux(chip, a), self._flux(chip, b)
            if fa is None or fb is None:
                ax.set_visible(False)
                continue
            ratio = np.nanmedian(fa, axis=1) / np.nanmedian(fb, axis=1)
            med = np.nanmedian(ratio[np.isfinite(ratio)])
            ax.plot(x, ratio, marker=".", linewidth=1)
            ax.axhline(med, color="r", linestyle="--", linewidth=0.8)
            ax.set_ylabel(f"{a}/{b}", fontsize=10)
            ax.annotate(
                f"median={med:.3f}",
                xy=(0.99, 0.9),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=8,
                color="r",
            )
            ax.grid(alpha=0.3)

        axes[0].set_title(
            f"L2 Orderlet Flux Ratios - {chip.capitalize()} CCD: "
            f"{self.obs_id} - {self.name}",
            fontsize=14,
        )
        axes[-1].set_xlabel(xlabel, fontsize=14)
        return self._decorate_and_save(fig, "orderlet_flux_ratios", chip)
