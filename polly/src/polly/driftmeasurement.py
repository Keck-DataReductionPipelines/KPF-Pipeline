"""
polly

driftmeasurement

PeakDrift objects are used to track (and optionally fit) the drift of a singl etalon
peak over time, from a series of mask files (saved outputs from polly.etalonanalysis).

GroupDrift objects contain a list of arbitrary PeakDrift objects within them. These
drifts can be then fit as a group (the group for instance binning the data in
wavelength, or across several orderlets).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property, partial
from operator import attrgetter
from pathlib import Path
from typing import TYPE_CHECKING, overload

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from collections.abc import Callable

    from astropy.units import Quantity
    from numpy.typing import ArrayLike

from polly.misc import savitzky_golay
from polly.parsing import parse_filename, parse_yyyymmdd
from polly.plotting import plot_style

plt.style.use(plot_style)


@dataclass
class PeakDrift:
    """
    A class that tracks the drift of a single wavelength peak over time, and optionally
    fits a linear slope to the drift. The peak is tracked starting from a single peak in
    a reference mask (single wavelength value). The peak is then tracked in time across
    a series of masks, with the closest wavelength found in those masks taken to be the
    new peak position (within a given window). Tracking peaks in this way allows for
    drifts that exceed the search window to be tracked over time, even to the point of
    a peak drifting by a full etalon FSR, just as long as it does not drift between two
    adjacent masks by such a large amount.
    """

    reference_mask: str  # Filename of the reference mask
    reference_wavelength: float  # Starting wavelength of the single peak
    local_spacing: float  # Local distance between wavelengths in reference mask

    masks: list[str]  # List of filenames to search
    dates: list[datetime] | None = None

    # After initialisation, the single peak will be tracked as it appears in
    # each successive mask. The corresponding wavelengths at which it is found
    # will populate the `wavelengths` list.
    _reference_date: datetime | None = None
    wavelengths: list[float | None] = field(default_factory=list)
    sigmas: list[float] = field(default_factory=list)
    valid: list[bool] | None = None

    auto_fit: bool = True

    fit: Callable[[ArrayLike], ArrayLike] | None = None
    fit_err: list[float] | None = None
    fit_slope: Quantity | None = None
    fit_slope_err: Quantity | None = None

    drift_file: str | Path | None = None
    force_recalculate: bool = False
    recalculated: bool = False

    def __post_init__(self) -> None:
        if self.drift_file is not None:
            if isinstance(self.drift_file, str):
                self.drift_file = Path(self.drift_file)

            if self.drift_file.exists():
                # print(f"File exists for λ={self.reference_wavelength:.2f}")

                # First check if there is an existing file. If so, check its length.
                # If this is within 10% of the length of self.masks, don't recalculate
                # unless self.force_recalculate == True?

                if self.force_recalculate:
                    # Then proceed as normal, track the drift from the masks
                    self.track_drift()

                else:
                    self.load_from_file()

        nanwavelength = np.where(~np.isnan(self.wavelengths), True, False)
        nansigma = np.where(~np.isnan(self.sigmas), True, False)
        self.valid = np.logical_and(nanwavelength, nansigma)

        if sum(self.valid) <= 3:  # noqa: PLR2004
            print(
                "Too few located peaks for λ="
                + f"{self.reference_wavelength:.2f} ({len(self.valid)})"
            )

        # self.track_drift()

        if self.auto_fit:
            self.linear_fit()

    def load_from_file(self) -> None:
        """
        Loads pre-traced drifts from a file on disk to allow for faster recomputation,
        eg. for iterative plot generation.
        """

        # Then load all the information from the file
        # print(f"Loading drifts from file: {self.drift_file}")
        file_dates, file_wls, file_sigmas = np.transpose(np.loadtxt(self.drift_file))

        file_dates = [parse_yyyymmdd(d) for d in file_dates]

        self.dates = np.array(file_dates)
        self.wavelengths = np.array(file_wls)
        self.sigmas = np.array(file_sigmas)
        self.sigmas[self.sigmas == 0] = np.nan

    @cached_property
    def valid_wavelengths(self) -> list[float]:
        """
        Returns all wavelengths for dates that satisfy the validity condition (see the
        `valid` property)
        """
        if self.valid is None:
            return self.wavelengths

        return list(np.array(self.wavelengths)[self.valid])

    @cached_property
    def valid_sigmas(self) -> list[float]:
        """
        Returns all sigma values for dates that satisfy the validity condition (see the
        `valid` property)
        """
        if self.valid is None:
            return self.sigmas

        return list(np.array(self.sigmas)[self.valid])

    @cached_property
    def valid_dates(self) -> list[datetime]:
        """
        Returns all measurement dates that satisfy the validity condition (see the
        `valid` property)
        """
        if self.valid is None:
            return self.dates

        return list(np.array(self.dates)[self.valid])

    @cached_property
    def reference_date(self) -> datetime:
        """
        Simply parses the date from the reference mask filename and returns it in a
        datetime object
        """
        if self._reference_date is not None:
            return self._reference_date

        return parse_filename(self.reference_mask).date

    @cached_property
    def days_since_reference_date(self) -> list[float]:
        """
        Returns an array containing the number of days since the reference date for each
        valid date in the dataset. This is used for fitting the drift over time, where
        the absolute date is not important, but the relative time steps are.
        """
        return [(d - self.reference_date).days for d in self.valid_dates]

    @cached_property
    def timesofday(self) -> list[str]:
        """
        Returns the (valid) time of day for each mask in the dataset. This is useful for
        identifying any systematic drifts that occur at certain times of day. See also
        the `valid` property.
        """
        if self.valid is None:
            valid_masks = self.masks

        else:
            valid_masks = list(np.array(self.masks)[self.valid])

        return [parse_filename(m).timeofday for m in valid_masks]

    @cached_property
    def smoothed_wavelengths(self) -> list[float]:
        """
        Returns a smoothed version of the (valid) wavelengths using a Savitzky-Golay
        filter. Dsed in plitting and to exclude outliers.
        """
        return savitzky_golay(y=self.valid_wavelengths, window_size=21, order=3)

    @cached_property
    def deltas(self) -> list[float]:
        """
        Returns the difference between the valid wavelengths and the reference
        wavelength -- this is the drift of the peak over time.
        """
        return [wl - self.reference_wavelength for wl in self.valid_wavelengths]

    @overload
    def get_delta_at_date(self, date: datetime) -> float: ...

    @overload
    def get_delta_at_date(self, date: list[datetime]) -> list[float]: ...

    def get_delta_at_date(
        self,
        date: datetime | list[datetime],
    ) -> float | list[float]:
        """
        Returns the delta value for a given date. If there is no data for a given date,
        returns NaN.

        If more than one matching date is present (why?), it returns the delta value for
        the first occurrance of the date.
        """

        if isinstance(date, list):
            return [self.get_delta_at_date(date=d) for d in date]

        assert isinstance(date, datetime)

        for i, d in enumerate(self.valid_dates):
            if d == date:
                return self.deltas[i]
        return np.nan

    @cached_property
    def fractional_deltas(self) -> list[float]:
        """
        Returns the drift of the peak over time __as a fraction of the reference
        wavelength__. This is more useful than the absolute drift for analysing drift
        as a function of wavelength. Also used to compute the radial velocity-space
        shift over time (multiplying by the speed of light in the desired units).
        """
        return [d / self.reference_wavelength for d in self.deltas]

    @overload
    def get_fractional_delta_at_date(self, date: datetime) -> float: ...

    @overload
    def get_fractional_delta_at_date(self, date: list[datetime]) -> list[float]: ...

    def get_fractional_delta_at_date(
        self,
        date: datetime | list[datetime],
    ) -> float | list[float]:
        """
        Returns the fractional delta value for a given date. If there is no data for a
        given date, returns NaN.

        Args:
            date (datetime | list[datetime]): the requested date(s)

        Returns:
            float | list[float]: the fractional delta value(s) for the requested date(s)
        """
        return self.get_delta_at_date(date) / self.reference_wavelength

    @cached_property
    def smoothed_deltas(self) -> list[float]:
        """
        Returns a smoothed version of the (valid) deltas using a Savitzky-Golay filter.
        """
        return savitzky_golay(y=self.deltas, window_size=21, order=3)

    def track_drift(self) -> PeakDrift:
        """
        Starting with the reference wavelength, track the position of the matching peak
        in successive masks. This function uses the last successfully found peak
        wavelength as the centre of the next search window, so can track positions even
        if they greatly exceed any reasonable search window over time.
        """

        print(f"Tracking drift for λ={self.reference_wavelength:.2f}...")

        self.dates = []

        last_wavelength: float = None

        if self.masks is not None:
            for m in self.masks:
                self.dates.append(parse_filename(m).date)
                peaks, sigmas = np.transpose(np.loadtxt(m))

                if last_wavelength is None:
                    # First mask
                    last_wavelength = self.reference_wavelength

                try:  # Find the closest peak in the mask
                    closest_index = np.nanargmin(np.abs(peaks - last_wavelength))
                except ValueError:  # What would give us a ValueError here?
                    self.wavelengths.append(None)
                    self.sigmas.append(None)
                    continue

                wavelength = peaks[closest_index]
                sigma = sigmas[closest_index]

                # Check if the new peak is within a search window around the last
                if abs(last_wavelength - wavelength) <= self.local_spacing / 50:
                    self.wavelengths.append(wavelength)
                    last_wavelength = wavelength
                    self.sigmas.append(sigma)
                else:  # No peak found within the window!
                    self.wavelengths.append(None)
                    self.sigmas.append(None)
                    # Don't update last_wavelength: we will keep searching at the
                    # same wavelength as previously.

        # Assign self.valid as a mask where wavelengths were successfully found
        self.valid = np.where(self.wavelengths, True, False)
        self.recalculated = True

        return self

    def linear_fit(
        self,
        fit_fractional: bool = False,
    ) -> PeakDrift:
        """
        Fits a linear trend to the tracked (optionally fractional) drift over time.
        The fit is done using the scipy.optimize.curve_fit function, which fits a
        linear model to the data, with the slope as the only free parameter (i.e. it
        assumes that the drift is starting from zero at the reference date).

        This method assigns the self.fit property with a (Callable) function, and
        self.fit_slope with the slope of that function in relevant units.

        What unit should be used by default? Absolute pm/day? RV mm/s/day?
        """

        if len(self.valid_wavelengths) == 0:
            print(f"No valid wavelengths found for {self.reference_wavelength}")
            print("Running PeakDrift.track_drift() first.")
            self.track_drift()

        deltas_to_use = self.fractional_deltas if fit_fractional else self.deltas

        def linear_model(x: float | list[float], slope: float) -> float | list[float]:
            if isinstance(x, list):
                x = np.array(x)

            return list(slope * x)

        try:
            p, cov = curve_fit(
                f=linear_model,
                xdata=self.days_since_reference_date,
                ydata=deltas_to_use,
                sigma=self.valid_sigmas,
                p0=[0],
                absolute_sigma=True,
            )

            self.fit = partial(linear_model, slope=p[0])
            self.fit_err = np.sqrt(cov)
            self.fit_slope = p[0] / u.day
            self.fit_slope_err = np.sqrt(cov[0][0]) / u.day

        except (ValueError, RuntimeError):
            self.fit = lambda x: np.nan  # noqa: ARG005
            self.fit_err = np.nan
            self.fit_slope = np.nan / u.day
            self.fit_slope_err = np.nan / u.day

        return self

    def fit_residuals(self, fractional: bool = False) -> list[float]:
        """
        Returns the residuals between the linear fit and the tracked drift for each
        (valid) date.
        """
        if fractional:
            return (
                self.fractional_deltas - self.fit(self.days_since_reference_date).value
            )

        return self.deltas - self.fit(self.days_since_reference_date).value

    def save_to_file(self, path: str | Path | None = None) -> PeakDrift:
        """
        Saves the tracked drifts to a file on disk.
        """

        if not path:
            if self.drift_file:
                path = self.drift_file
            else:
                raise Exception("No file path passed in and no drift_file specified")

        if isinstance(path, str):
            path = Path(path)

        if path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not self.recalculated:
            # Then don't need to save the file again
            return self

        datestrings = [f"{d:%Y%m%d}" for d in self.valid_dates]
        wlstrings = [f"{wl}" for wl in self.valid_wavelengths]
        sigmastrings = [f"{wl}" for wl in self.valid_sigmas]

        try:
            np.savetxt(
                f"{path}",
                np.transpose([datestrings, wlstrings, sigmastrings]),
                fmt="%s",
            )
        except FileExistsError:
            # print(e)
            ...

        return self


@dataclass
class GroupDrift:
    """
    A class that tracks the drift of a group of peaks and fits their slope
    together. Rather than computing the drift (and fitting a linear slope) for
    individual peaks, here we can consider a block of wavelengths all together.
    """

    peakDrifts: list[PeakDrift]

    group_fit: Callable | None = None
    group_fit_err: list[float] | None = None
    group_fit_slope: float | None = None
    group_fit_slope_err: float | None = None

    def __post_init__(self) -> None:
        self.peakDrifts = sorted(
            self.peakDrifts, key=attrgetter("reference_wavelength")
        )

    @cached_property
    def mean_wavelength(self) -> float:
        """
        Returns the mean wavelength of the peaks in the group.
        """
        return np.mean([pd.reference_wavelength for pd in self.peakDrifts])

    @cached_property
    def min_wavelength(self) -> float:
        """
        Returns the smallest wavelength of the peaks in the group.
        """
        return min(self.peakDrifts[0].reference_wavelength)

    @cached_property
    def max_wavelength(self) -> float:
        """
        Returns the largest wavelength of the peaks in the group.
        """
        return max(self.peakDrifts[-1].reference_wavelength)

    @property
    def all_dates(self) -> list[datetime]:
        """
        Returns a list of all dates for which drifts have been measured in the group.
        There may be some PeakDrifts that have no data for a given date, so this list
        may be longer than the list of unique dates.
        """
        all_dates = []

        for pd in self.peakDrifts:
            all_dates.extend(pd.valid_dates)

        return all_dates

    @cached_property
    def reference_date(self) -> datetime:
        """
        Returns the earliest date for which drifts have been measured in the group.
        """
        return min(self.all_dates)

    @cached_property
    def all_days_since_reference_date(self) -> list[float]:
        """
        Returns an array containing the number of days since the reference date for each
        date in the dataset. This is used for fitting the drift over time, where the
        absolute date is not important, but the relative time steps are.
        """
        return [(d - self.reference_date).days for d in self.all_dates]

    @cached_property
    def unique_dates(self) -> list[datetime]:
        """
        Returns a list of unique dates for which drifts have been measured in the group.
        """
        return sorted(set(self.all_dates))

    @cached_property
    def all_deltas(self) -> list[float]:
        """
        Returns all deltas for all peaks in the group.
        """
        all_deltas = []

        for pd in self.peakDrifts:
            all_deltas.extend(pd.fractional_deltas)

        return all_deltas

    @cached_property
    def all_sigmas(self) -> list[float]:
        """
        Returns all sigma values for all peaks in the group.
        """
        all_sigmas = []

        for pd in self.peakDrifts:
            all_sigmas.extend(pd.valid_sigmas)

        return all_sigmas

    @cached_property
    def all_relative_sigmas(self) -> list[float]:
        """
        Returns all __relative__ sigma values for all peaks in the group.
        """
        all_relative_sigmas = []

        for pd in self.peakDrifts:
            all_relative_sigmas.extend(pd.valid_sigmas / pd.reference_wavelength)

        return all_relative_sigmas

    @property
    def mean_deltas(self) -> list[float]:
        """
        Returns the mean drift across all the wavelengths in the group. The output is a
        list of drifts as a function of time.
        """
        all_deltas = [pd.deltas for pd in self.peakDrifts]
        return list(np.mean(all_deltas, axis=1))

    def fit_group_drift(self, verbose: bool = False) -> None:
        """
        Fits a linear trend to the group drift over time. This is done by fitting a
        linear model to the data, with the slope as the only free parameter (i.e. it
        assumes that the drift is starting from zero at the reference date).
        """
        if verbose:
            print(f"{self.all_days_since_reference_date}")
            print(f"{self.all_deltas}")

        def linear_model(x: float | list[float], slope: float) -> float | list[float]:
            if isinstance(x, list):
                x = np.array(x)

            return list(slope * x)

        try:
            p, cov = curve_fit(
                f=linear_model,
                xdata=self.all_days_since_reference_date,
                ydata=self.all_deltas,
                sigma=self.all_relative_sigmas,
                p0=[0],
                absolute_sigma=True,
            )

            self.group_fit = partial(linear_model, slope=p[0])
            self.group_fit_err = np.sqrt(cov)
            self.group_fit_slope = p[0]
            self.group_fit_slope_err = np.sqrt(cov[0][0])

        except Exception as e:
            print(e)
            self.group_fit = None
            self.group_fit_err = None
            self.group_fit_slope = None
            self.group_fit_slope_err = None
