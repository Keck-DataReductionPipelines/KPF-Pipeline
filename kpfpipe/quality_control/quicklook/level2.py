"""L2 quicklook plots for extracted KPF 1D spectra.

Ports the extracted-spectrum quicklook plots from the v2.12 ``AnalyzeL1``
class (the old pipeline's "L1" level is vNext's L2). These plots REQUIRE an
attached wavelength solution: the per-fiber ``{chip}_{fiber}_WAVE`` arrays
must be populated (i.e. WavelengthCalibration has run). If they are not, the
plot methods raise rather than silently producing a pixel-axis plot that
could be mistaken for a wavelength-calibrated one (charter: fail loudly).

Pure visualization — no science computation is written back to the product.
"""

import os
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np

from kpfpipe.data_models.headers import HeaderParser
from kpfpipe.quality_control.quicklook._save_png import save_png

_DPI = 200
_FIBERS = ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]
_SCI_FIBERS = ["SCI1", "SCI2", "SCI3"]
_SNR_PERCENTILE = 95
_FLUX_PERCENTILE = 95


class PlotL2:
    """
    Quicklook plots for KPF L2 (extracted 1D spectra) data.

    Args:
        l2_obj: KPF2 data object (post-SpectralExtraction; requires
            WavelengthCalibration to have populated the per-fiber WAVE arrays).
        output_dir: directory to save PNG files. None = return Figure only.
        obs_id: observation ID for titles/filenames. Unlike KPF0/KPF1, KPF2
            has no obs_id attribute, so the recipe passes it explicitly; absent
            that, it is derived from the PRIMARY FILENAME header.
    """

    _PLOT_METHODS = (
        "snr_per_order",
        "peak_flux",
        "spectrum_single_order",
        "spectrum_one_row",
        "orderlet_flux_ratios",
    )

    def __init__(self, l2_obj, output_dir=None, obs_id=None):
        self.l2 = l2_obj
        self.output_dir = output_dir
        self.fibers = _FIBERS

        primary = (
            l2_obj.headers.get("PRIMARY", {}) if hasattr(l2_obj, "headers") else {}
        )
        # obs_id: explicit arg (recipe knows it) > model attr (KPF0/1 only) >
        # FILENAME header, stripped of level/extension suffixes.
        self.obs_id = (
            obs_id
            or getattr(l2_obj, "obs_id", None)
            or self._obsid_from_filename(HeaderParser.get(primary, "FILENAME", ""))
            or ""
        )
        self.name = HeaderParser.get(primary, "OBJECT", "") if primary else ""

    @staticmethod
    def _obsid_from_filename(filename):
        """Derive an obs_id from a FILENAME header (strip path/level/ext)."""
        if not filename:
            return ""
        base = os.path.basename(str(filename))
        for suffix in ("_L0", "_L1", "_L2", "_L4"):
            base = base.replace(suffix, "")
        for ext in (".fits", ".fits.gz"):
            if base.endswith(ext):
                base = base[: -len(ext)]
        return base

    # ------------------------------------------------------------------
    # Data access helpers
    # ------------------------------------------------------------------

    def _flux(self, chip, fiber):
        """Return the (norder, ncol) flux array for one fiber, or None."""
        arr = self.l2.data.get(f"{chip.upper()}_{fiber.upper()}_FLUX")
        arr = np.asarray(arr) if arr is not None else None
        if arr is None or arr.ndim != 2 or arr.size == 0:
            return None
        return arr

    def _var(self, chip, fiber):
        arr = self.l2.data.get(f"{chip.upper()}_{fiber.upper()}_VAR")
        arr = np.asarray(arr) if arr is not None else None
        if arr is None or arr.ndim != 2 or arr.size == 0:
            return None
        return arr

    def _wave(self, chip, fiber):
        """Return the (norder, ncol) wavelength array, or None if not populated."""
        arr = self.l2.data.get(f"{chip.upper()}_{fiber.upper()}_WAVE")
        arr = np.asarray(arr) if arr is not None else None
        if arr is None or arr.ndim != 2 or arr.size == 0:
            return None
        return arr

    def _has_chip(self, chip):
        return self._flux(chip, "SCI2") is not None

    def _require_wave(self, chip, fiber="SCI2"):
        """Return the (norder, ncol) wavelength array for a fiber, or raise.

        PlotL2 requires an attached wavelength solution; failing loudly here
        prevents a wavelength-less plot from being mistaken for a calibrated
        one. Run WavelengthCalibration before PlotL2.
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
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = flux / np.sqrt(np.abs(var))
        return np.where(np.isfinite(snr), snr, 0.0)

    # ------------------------------------------------------------------
    # Common decoration
    # ------------------------------------------------------------------

    def _decorate_and_save(self, fig, plot_name, chip):
        """Add the standard QLP timestamp and save if output_dir is set."""
        current_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.005,
            f"KPF QLP: {current_time} UT",
            fontsize=8,
            color="darkgray",
            ha="right",
            va="bottom",
        )
        if self.output_dir is not None:
            prefix = f"{self.obs_id}_" if self.obs_id else ""
            path = os.path.join(
                self.output_dir, f"{prefix}L2_{plot_name}_{chip.lower()}_zoomable.png"
            )
            save_png(fig, path, dpi=_DPI, compress_level=6)
        return fig

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

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

        # summed science orderlet
        sci_flux = [self._flux(chip, f) for f in _SCI_FIBERS]
        sci_var = [self._var(chip, f) for f in _SCI_FIBERS]
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

        sci_flux = [self._flux(chip, f) for f in _SCI_FIBERS]
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

        Args:
            chip: 'GREEN' or 'RED'.
            order: 1-indexed spectral order. Defaults to a representative
                order near the middle of the chip.
        """
        chip = chip.upper()
        if not self._has_chip(chip):
            return None
        norder = self._flux(chip, "SCI2").shape[0]
        if order is None:
            order = norder // 2  # representative middle order
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
            with np.errstate(divide="ignore", invalid="ignore"):
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

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run(self, which):
        """Generate the requested plot(s) for every chip that has data,
        saving each to ``output_dir`` and closing the matplotlib figure.

        Args:
            which: 'all' to run every implemented plot, or the name of a
                single plot method (one of ``self._PLOT_METHODS``).

        Returns:
            dict mapping ``{method_name}_{chip}`` to matplotlib.Figure
            (closed; useful for tests/introspection).
        """
        if which == "all":
            names = self._PLOT_METHODS
        elif which in self._PLOT_METHODS:
            names = (which,)
        else:
            raise ValueError(
                f"unknown plot {which!r}; expected 'all' or one of {self._PLOT_METHODS}"
            )

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

        figures = {}
        for chip in ("green", "red"):
            if not self._has_chip(chip):
                continue
            for name in names:
                fig = getattr(self, name)(chip)
                if fig is None:
                    continue
                figures[f"{name}_{chip}"] = fig
                plt.close(fig)
        return figures
