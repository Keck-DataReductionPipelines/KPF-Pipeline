"""Diagnostics for the KPF Level 0 exposure meter extensions."""

import numpy as np

from kpfpipe.quality_control.diagnostics.base import Diagnostics


class ExposureMeter(Diagnostics):
    """Diagnostics from the EXPMETER_SCI and EXPMETER_SKY tables."""

    LEVEL = "L0"

    def _expmeter_flux(self, ext):
        """One EM fiber's channel wavelengths [nm] and raw flux, readings x channels.

        The numeric column labels are the wavelength channels, in nm at L0 --
        ImageAssembly renames them to Angstroms only at the L0 -> L1 boundary. The
        Date* columns are not channels.
        """
        table = self.kpf_obj.data[ext]
        waves, channels = [], []
        for name in table.colnames:
            try:
                wave = float(name)
            except ValueError:
                continue
            waves.append(wave)
            channels.append(np.asarray(table[name], dtype=float))
        return np.array(waves), np.column_stack(channels)

    @staticmethod
    def _longest_run(mask):
        """Longest run of adjacent True values in a 1D channel mask."""
        longest = run = 0
        for flagged in mask:
            run = run + 1 if flagged else 0
            longest = max(longest, run)
        return longest

    def expmeter_channel_metrics(self):
        """EM{SCI,SKY}{SAT,NEG,INF}: per-fiber exposure meter channel metrics.

        Each fiber is judged on its own. SAT is saturated elements per reading --
        elements above 90% of the 1.93e6 reduced-spectrum saturation level, over
        the interior readings (the first and last are partial and are dropped
        when there are 3+). NEG is the longest run of adjacent channels whose
        time-summed flux is negative, the signature of bias over-subtraction in
        the raw EM images; INF is the same run length for channels holding a
        non-finite reading.
        """
        values = {}
        for ext, fiber in (("EXPMETER_SCI", "SCI"), ("EXPMETER_SKY", "SKY")):
            _, flux = self._expmeter_flux(ext)
            interior = flux[1:-1] if len(flux) >= 3 else flux
            values[f"EM{fiber}SAT"] = round(
                float(np.count_nonzero(interior > 0.9 * 1.93e6) / len(interior)), 6
            )
            values[f"EM{fiber}NEG"] = self._longest_run(flux.sum(axis=0) < 0)
            values[f"EM{fiber}INF"] = self._longest_run(~np.isfinite(flux).all(axis=0))
        return self._tag(**values)

    expmeter_channel_metrics._diag_name = "expmeter_channel_metrics"

    def expmeter_counts(self):
        """EM{SC,SK}CT{48,45,56,67,78}: cumulative EM counts [ADU] per band.

        Raw counts summed over every reading and over the channels of each band,
        per fiber. The 445-870 nm total spans the EM's full range and the four
        sub-bands partition it at the 551.25, 657.50 and 763.75 nm edges, so the
        sub-bands always add up to the total.
        """
        values = {}
        for ext, fiber in (("EXPMETER_SCI", "SC"), ("EXPMETER_SKY", "SK")):
            waves, flux = self._expmeter_flux(ext)
            per_channel = np.nansum(flux, axis=0)
            for band, mask in (
                ("48", (waves >= 445.0) & (waves < 870.0)),
                ("45", (waves >= 445.0) & (waves < 551.25)),
                ("56", (waves >= 551.25) & (waves < 657.50)),
                ("67", (waves >= 657.50) & (waves < 763.75)),
                ("78", (waves >= 763.75) & (waves < 870.0)),
            ):
                values[f"EM{fiber}CT{band}"] = int(np.nansum(per_channel[mask]))
        return self._tag(**values)

    expmeter_counts._diag_name = "expmeter_counts"

    def sky_sci_flux_ratio(self):
        """SKYSCIMS: SKY/SCI flux ratio in the main spectrometer, scaled from EM.

        Total SKY counts over total SCI counts, the SKY side divided by the 14.1
        SKY-to-SCI flux ratio measured on bright twilight observations.
        """
        sci = np.nansum(self._expmeter_flux("EXPMETER_SCI")[1])
        sky = np.nansum(self._expmeter_flux("EXPMETER_SKY")[1])
        return self._tag(SKYSCIMS=round(float(sky / 14.1 / sci), 6))

    sky_sci_flux_ratio._diag_name = "sky_sci_flux_ratio"
