"""
polly

Polly put the Ketalon?

etalonanalysis

Etalon analysis tools for KPF data products. Contains a class structure to be used for
general analysis of etalon spectra from KPF. A short description of the three levels
and what happens within each, from the top down:

Spectrum
    A Spectrum object represents data corresponding to a single FITS file, including all
    (or a subset of) the SKY / SCI1 / SCI2 / SCI3 / CAL orderlets.

    It can load flux data from either a single FITS file or a list of FITS files (the
    data from which are then median-combined).

    Wavelength solution data can be loaded independently from a separate FITS file. If
    the parameter `wls_file' is not specified, the code will try to find the matching
    WLS file from available daily masters.

    The Spectrum class is where the user interacts with the data. It contains a list of
    Order objects (which each contain a list of Peak objects), but all functionality can
    be initiated at the Spectrum level.

Order
    An Order object represents the L1 data for a single orderlet and spectral order. It
    contains the data arrays `spec' and `wave', loaded directly from the FITS file(s) of
    a Spectrum object.

    The rough (pixel-scale) location of peaks is done at the Order level (using
    `scipy.signal.find_peaks`, which is called from the `locate_peaks()` method)

    Orders contain a list of Peak objects, wherein the fine-grained fitting with
    analytic functions is performed, see below.

Peak
    A Peak object represents a small slice of flux and wavelength data around a single
    located Etalon peak. A Peak is initialised with `speclet' and `wavelet' arrays and a
    roughly identified wavelength position of the peak.

    After initialisation, the `.fit()` method fits a (chosen) analytic function to the
    Peak's contained data. This results gives sub-pixel estimation of the central
    wavelength of the Peak.

    The fitting is typically initiated from the higher level (Spectrum object), and
    inversely all of the Peak's data is also passed upward to be accessible from Order
    and Spectrum objects.
"""

from __future__ import annotations

import ast
import logging
import weakref
from dataclasses import dataclass, field
from operator import attrgetter
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from astropy import constants
from astropy import units as u
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.interpolate import BSpline, interp1d, splrep
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from tqdm import tqdm

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

from polly.kpf import (
    ALL_ORDER_INDICES,
    LFC_ORDER_INDICES,
    MASTERS_DIR,
    ORDERLETS,
    TEST_ORDER_INDICES,  # noqa: F401
    THORIUM_ORDER_INDICES,
)
from polly.log import logger
from polly.parsing import get_orderlet_index, get_orderlet_name
from polly.plotting import plot_style, stroke, wavelength_to_rgb

plt.style.use(plot_style)
T = TypeVar("T", float, list[float])


@dataclass
class Peak:
    """
    Contains information about a single identified or fitted etalon peak

    Properties:
        parent_ref: weakref
            A reference to the parent Order object for the Peak. Used to populate the
            relevant information (orderlet, order_i)

        coarse_wavelength: float [Angstrom]
            The centre pixel of an initially identified peak
        speclet: ArrayLike [ADU]
            A short slice of the `spec` array around the peak
        wavelet: ArrayLike [Angstrom]
            A short slice of the `wave` array around the peak

        orderlet: str ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]
            Automatically inherited from parent Order object
        order_i: int [Index starting from zero]
            The index number of the order that contains the peak, automatically
            inherited from parent Order object

        center_wavelength: float [Angstrom] = None
            The centre wavelength resulting from the function fitting routine
        distance_from_order_center: float [Angstrom] = None
            The absolute difference of the Peak's wavelength from the mean wavelength of
            the containing order, used for selecting between (identical) peaks appearing
            in adjacent orders

        fit_type: str = None
            The name of the function that was used to fit the peak
        amplitude: float = None
            Fit parameter describing the height of the function (see the function
            definitions)
        sigma: float = None
            Fit parameter describing the width of the Gaussian (part of the) function
            (see the function definitions)
        boxhalfwidth: float = None
            Fit parameter describing the width of the top-hat part of the function (see
            `conv_gauss_tophat' in `fit_erf_to_ccf_simplified.py')
        offset: float = None
            Fit parameter describing the vertical offset of the function, allowing for a
            flat bias (see the function definitions)

        wl: float [Angstrom]
            alias for central_wavelength (if it is defined), otherwise it returns the
            value of coarse_wavelength
        i: int [Index starting from zero]
            alias for order_i
        d: float [Angstrom]
            alias for distance_from_order_center

    Methods:
        fit(type: str = "conv_gauss_tophat"):
            Calls the relevant fitting function

        _fit_gaussian():
            Fits a Gaussian function (see top-level `_gaussian()` function) to the data,
            with data-driven initial guesses and bounds for the parameters. Updates the
            Peak object parameters, returns nothing.

        _fit_conv_gauss_tophat():
            Fits an analytic form of a Gaussian function convolved with a top-hat
            function, constructed from two sigmoid "error" functions (see
            `conv_gauss_tophat()` in `fit_erf_to_ccf_simplified.py` module. Fitting
            routine has data-driven initial guesses and bounds for the parameters.
            Updates the Peak object parameters, returns nothing.

        output_parameters():
            TODO: a function to return the parameters to be saved to an output (JSON?)
            file

        has(prop: str):
            Used for repr generation. Returns a checked box if the Peak has `speclet' or
            `wavelet' arrays, else returns an empty box.

        __repr__():
            Returns a one-line summary of the Peak object. May be expanded in the future

        plot_fit(ax: plt.Axes):
            Plot of data and fit with vertical lines showing the coarse center and the
            fine (fitted) center.
            Optionally accepts an axis object in which to plot, for calling the function
            in batch from some higher level.

    """

    parent_ref: weakref.ReferenceType

    coarse_wavelength: float
    speclet: ArrayLike | list
    wavelet: ArrayLike | list

    starting_pixel: int | None = None

    orderlet: str | None = None
    order_i: int | None = None
    distance_from_order_center: float | None = None

    # Fitting results
    fit_space: str | None = None
    fit_type: str | None = None
    # Fit parameters
    _center_wavelength: float | None = None
    _center_pixel: float | None = None
    amplitude: float | None = None
    sigma: float | None = None
    boxhalfwidth: float | None = None
    offset: float | None = None
    # Fit errors
    center_wavelength_stddev: float | None = None
    center_pixel_stddev: float | None = None
    amplitude_stddev: float | None = None
    sigma_stddev: float | None = None
    boxhalfwidth_stddev: float | None = None
    offset_stddev: float | None = None

    def __post_init__(self) -> None:
        # Set order_i and orderlet from parent Order
        self.order_i = self.parent.i
        self.orderlet = self.parent.orderlet

    @property
    def center_wavelength(self) -> float:
        """
        The central wavelength of the peak. If a fit has already been done (and the
        _center_wavelength defined), this is returned. Otherwise, the coarse_wavelength
        is returned.
        """
        if self._center_wavelength is None:
            return self.coarse_wavelength
        return self._center_wavelength

    @property
    def center_pixel(self) -> float:
        """
        The central pixel of the peak. If a fit has already been done (and the
        _center_pixel defined), this is returned. Otherwise, the rough central pixel
        value (starting_pixel + len(wavelet) // 2) is returned.
        """
        if self._center_pixel is None:
            return self.starting_pixel + len(self.wavelet) // 2
        return self._center_pixel

    @property
    def pixlet(self) -> list[int]:
        """
        A list of pixel numbers corresponding to the `speclet` and `wavelet` arrays.
        """

        return [self.starting_pixel + i for i in range(len(self.wavelet))]

    @property
    def parent(self) -> Order:
        """
        Return the `Order` object to which this Peak belongs.
        """

        try:
            return self.parent_ref()
        except NameError:
            return None

    @property
    def wl(self) -> float:
        """
        Alias for `center_wavelength`.
        """
        return self.center_wavelength

    @property
    def i(self) -> int:
        """
        Alias for `order_i`.
        """

        return self.order_i

    @property
    def d(self) -> float:
        """
        Alias for `distance_from_order_center`.
        """

        return self.distance_from_order_center

    @property
    def scaled_rms(self) -> float:
        """
        Intended as a simple metric to measure the quality of the fit.
        """

        if not self.center_wavelength:
            return None

        return np.sqrt(np.mean(np.power(self.residuals / self.amplitude, 2)))

    @property
    def fwhm(self) -> float:
        """
        Convenience function to get the FWHM of a fit from its sigma value.
        """

        if self.sigma is None:
            return None

        return self.sigma * (2 * np.sqrt(2 * np.log(2)))

    def fit(
        self,
        fit_type: str = "conv_gauss_tophat",
        space: str = "pixel",
    ) -> Peak:
        """
        Initiates the fitting routine. Essentially a wrapper around either
        `_fit_gaussian` or `_fit_conv_gauss_tophat` depending on `fit_type`
        """

        if fit_type not in ["gaussian", "conv_gauss_tophat"]:
            raise NotImplementedError

        if space not in ["wavelength", "pixel"]:
            raise NotImplementedError

        self.fit_type = fit_type
        self.fit_space = space

        if fit_type == "gaussian":
            self._fit_gaussian(space=space)

        elif fit_type == "conv_gauss_tophat":
            self._fit_conv_gauss_tophat(space=space)

        return self

    def _fit_gaussian(self, space: str = "pixel") -> None:
        """
        `scipy.optimize.curve_fit` wrapper, with initial guesses `p0` and their `bounds`
        coming from properties of the data themselves

        First centres the wavelength range about zero

        See top-level `_gaussian` for function definition
        """

        mean_dwave = np.mean(np.diff(self.wavelet))

        if space == "wavelength":
            x0 = np.mean(self.wavelet)
            x = self.wavelet - x0  # Centre about zero
            mean_dx = np.abs(mean_dwave)

        elif space == "pixel":
            x0 = np.mean(self.pixlet)
            x = self.pixlet - x0  # Centre about zero
            mean_dx = 1

        maxy = max(self.speclet)
        y = self.speclet / maxy
        offset_guess = float((y[0] + y[-1]) / 2)

        #       amplitude,     center,  sigma,       offset
        p0 = [1 - offset_guess, 0, 2.5 * mean_dx, offset_guess]
        bounds = [
            [0, -5 * mean_dx, 0, -np.inf],
            [2, 5 * mean_dx, 10 * mean_dx, np.inf],
        ]

        try:
            p, cov = curve_fit(
                f=_gaussian,
                xdata=x,
                ydata=y,
                p0=p0,
                bounds=bounds,
            )
        except ValueError:
            # Initial guess is outside of provided bounds
            # logger.warning(e)
            self.remove_fit(fill_with_nan=True)
            return
        except RuntimeError:
            # logger.warning(e)
            self.remove_fit(fill_with_nan=True)
            return

        amplitude, center, sigma, offset = p

        self.remove_fit()

        # Populate the fit parameters
        stddev = np.sqrt(np.diag(cov))

        if space == "wavelength":
            self._center_wavelength = x0 + center
            self.center_wavelength_stddev = float(stddev[0])

            # Also interpolate to pixel space
            def wavelength_to_pixel(wavelength_value: T) -> T:
                mapping = interp1d(self.wavelet, self.pixlet)

                if isinstance(wavelength_value, list):
                    return [float(mapping(pv)[()]) for pv in wavelength_value]

                return float(mapping(wavelength_value)[()])

            try:
                self._center_pixel = wavelength_to_pixel(self.center_wavelength)
            except ValueError:
                self._center_pixel = np.nan
            try:
                self.center_pixel_stddev = abs(
                    wavelength_to_pixel(
                        self.center_wavelength + self.center_wavelength_stddev
                    )
                    - self.center_pixel
                )
            except ValueError:
                self.center_pixel_stddev = np.nan

            self.sigma = sigma
            self.sigma_stddev = stddev[2]

        elif space == "pixel":
            self._center_pixel = x0 + center
            self.center_pixel_stddev = stddev[0]

            # Also interpolate to wavelength space
            def pixel_to_wavelength(pixel_value: T) -> T:
                mapping = interp1d(self.pixlet, self.wavelet)

                if isinstance(pixel_value, list):
                    return [float(mapping(pv)[()]) for pv in pixel_value]

                return float(mapping(pixel_value)[()])

            try:
                self._center_wavelength = pixel_to_wavelength(self.center_pixel)
            except ValueError:
                self._center_wavelength = np.nan

            try:
                self.center_wavelength_stddev = abs(
                    pixel_to_wavelength(self.center_pixel + self.center_pixel_stddev)
                    - self.center_wavelength
                )
            except ValueError:
                self.center_wavelength_stddev = np.nan

            try:
                self.sigma = abs(
                    pixel_to_wavelength(self.center_pixel + sigma)
                    - self.center_wavelength
                )
            except ValueError:
                self.sigma = np.nan

            try:
                self.sigma_stddev = abs(
                    pixel_to_wavelength(self.center_pixel + stddev[2])
                    - self.center_wavelength
                )
            except ValueError:
                self.sigma_stddev = np.nan

        # Populate the fit parameters
        self.amplitude = float(amplitude * maxy)
        self.amplitude_stddev = float(stddev[0])
        self.offset = float(offset * maxy)
        self.offset_stddev = float(stddev[3])

    def _fit_conv_gauss_tophat(self, space: str = "pixel") -> None:
        """
        `scipy.optimize.curve_fit` wrapper, with initial guesses `p0` and their `bounds`
        coming from properties of the data themselves

        First centres the wavelength range about zero

        See top-level `_conv_gauss_tophat` for function definition
        """

        mean_dwave = np.mean(np.diff(self.wavelet))

        if space == "wavelength":
            x0 = np.mean(self.wavelet)
            x = self.wavelet - x0  # Centre about zero
            mean_dx = np.abs(mean_dwave)

        elif space == "pixel":
            x0 = np.mean(self.pixlet)
            x = self.pixlet - x0  # Centre about zero
            mean_dx = 1

        maxy = max(self.speclet)
        # Normalise
        y = self.speclet / maxy
        offset_guess = float((y[0] + y[-1]) / 2)

        # center,     amp,            sigma,     boxhalfwidth,    offset
        p0 = [0, 1 - offset_guess, 2.5 * mean_dx, 3 * mean_dx, offset_guess]
        bounds = [
            [-5 * mean_dx, 0, 0, 0, -np.inf],
            [5 * mean_dx, 10, 10 * mean_dx, 6 * mean_dx, np.inf],
        ]
        try:
            p, cov = curve_fit(
                f=_conv_gauss_tophat,
                xdata=x,
                ydata=y,
                p0=p0,
                bounds=bounds,
            )
        except ValueError:
            # Initial guess is outside of provided bounds
            # logger.warning(e)
            self.remove_fit(fill_with_nan=True)
            return
        except RuntimeError:
            # logger.warning(e)
            self.remove_fit(fill_with_nan=True)
            return

        center, amplitude, sigma, boxhalfwidth, offset = p

        self.remove_fit()

        # Populate the fit parameters
        stddev = np.sqrt(np.diag(cov))

        if space == "wavelength":
            self._center_wavelength = x0 + center
            self.center_wavelength_stddev = float(stddev[0])

            # Also interpolate to pixel space
            def wavelength_to_pixel(wavelength_value: T) -> T:
                mapping = interp1d(self.wavelet, self.pixlet)

                if isinstance(wavelength_value, list):
                    return [float(mapping(pv)[()]) for pv in wavelength_value]

                return float(mapping(wavelength_value)[()])

            try:
                self._center_pixel = wavelength_to_pixel(self.center_wavelength)
            except ValueError:
                self._center_pixel = np.nan
            try:
                self.center_pixel_stddev = abs(
                    wavelength_to_pixel(
                        self.center_wavelength + self.center_wavelength_stddev
                    )
                    - self.center_pixel
                )
            except ValueError:
                self.center_pixel_stddev = np.nan

            self.sigma = sigma
            self.sigma_stddev = stddev[2]
            self.boxhalfwidth = boxhalfwidth
            self.boxhalfwidth_stddev = stddev[3]

        elif space == "pixel":
            self._center_pixel = float(x0 + center)
            self.center_pixel_stddev = float(stddev[0])

            # Also interpolate to wavelength space
            def pixel_to_wavelength(pixel_value: T) -> T:
                mapping = interp1d(self.pixlet, self.wavelet)

                if isinstance(pixel_value, list):
                    return [float(mapping(pv)[()]) for pv in pixel_value]

                return float(mapping(pixel_value)[()])

            try:
                self._center_wavelength = pixel_to_wavelength(self.center_pixel)
            except ValueError:
                self._center_wavelength = np.nan

            try:
                self.center_wavelength_stddev = abs(
                    pixel_to_wavelength(self.center_pixel + self.center_pixel_stddev)
                    - self.center_wavelength
                )
            except ValueError:
                self.center_wavelength_stddev = np.nan

            try:
                self.sigma = abs(
                    pixel_to_wavelength(self.center_pixel + sigma)
                    - self.center_wavelength
                )
            except ValueError:
                self.sigma = np.nan

            try:
                self.sigma_stddev = abs(
                    pixel_to_wavelength(self.center_pixel + stddev[2])
                    - self.center_wavelength
                )
            except ValueError:
                self.sigma_stddev = np.nan

            try:
                self.boxhalfwidth = abs(
                    pixel_to_wavelength(self.center_pixel + boxhalfwidth)
                    - self.center_wavelength
                )
            except ValueError:
                self.boxhalfwidth = np.nan

            try:
                self.boxhalfwidth_stddev = abs(
                    pixel_to_wavelength(self.center_pixel + stddev[3])
                    - self.center_wavelength
                )
            except ValueError:
                self.boxhalfwidth_stddev = np.nan

        self.amplitude = float(amplitude * maxy)
        self.amplitude_stddev = float(stddev[1])
        self.offset = float(offset * maxy)
        self.offset_stddev = float(stddev[4])

    def remove_fit(self, fill_with_nan: bool = False) -> Peak:
        """
        Reset any previously fitted parameters, used in the case of fitting again,
        perhaps with a different function, or in pixel space instead of wavelength space
        """

        ___ = np.nan if fill_with_nan else None

        self._center_wavelength = ___
        self._center_pixel = ___
        self.amplitude = ___
        self.sigma = ___
        self.boxhalfwidth = ___
        self.offset = ___

        self.center_wavelength_stddev = ___
        self.center_pixel_stddev = ___
        self.amplitude_stddev = ___
        self.sigma_stddev = ___
        self.boxhalfwidth_stddev = ___
        self.offset_stddev = ___

        return self

    @property
    def fit_parameters(self) -> dict:
        """
        Construct a dictionary of the fit parameters and their standard deviations
        """

        return {
            "fit_type": self.fit_type,
            "fit_space": self.fit_space,
            "center_wavelength": self.center_wavelength,
            "center_pixel": self.center_pixel,
            "amplitude": self.amplitude,
            "sigma": self.sigma,
            "boxhalfwidth": self.boxhalfwidth,
            "offset": self.offset,
            "center_wavelength_stddev": self.center_wavelength_stddev,
            "center_pixel_stddev": self.center_pixel_stddev,
            "amplitude_stddev": self.amplitude_stddev,
            "sigma_stddev": self.sigma_stddev,
            "boxhalfwidth_stddev": self.boxhalfwidth_stddev,
            "offset_stddev": self.offset_stddev,
        }

    def output_parameters(self) -> str:
        """
        Construct a string with the parameters we want to save to an output (JSON?) file
        """

        raise NotImplementedError

    def evaluate_fit(
        self,
        x: list[float],
        about_zero: bool = False,
    ) -> list[float] | None:
        """
        A function to evaluate the function fit to the peak across a wavelength array.
        Used for computing residuals, and for plotting the fit across a finer wavelength
        grid than the native pixels.
        """

        if self.fit_type is None:
            return None

        if about_zero:
            center = 0
        elif self.center_wavelength:
            center = self.center_wavelength
        elif self.center_pixel:
            center = self.center_pixel

        if self.fit_type == "gaussian":
            yfit = _gaussian(
                x=x,
                amplitude=self.amplitude,
                center=center,
                sigma=self.sigma,
                offset=self.offset,
            )

        elif self.fit_type == "conv_gauss_tophat":
            yfit = _conv_gauss_tophat(
                x=x,
                center=center,
                amp=self.amplitude,
                sigma=self.sigma,
                boxhalfwidth=self.boxhalfwidth,
                offset=self.offset,
            )

        return yfit

    @property
    def residuals(self) -> ArrayLike:
        """
        If a fit exists, return the residuals between the raw data and the fit, after
        normalising to the max value of the fit.
        """

        if self.fit_type is None:
            return [np.nan]

        xfit = np.linspace(min(self.wavelet), max(self.wavelet), 100)
        yfit = self.evaluate_fit(x=xfit)
        maxy = max(yfit)
        coarse_yfit = self.evaluate_fit(x=self.wavelet)

        return (self.speclet - coarse_yfit) / maxy

    @property
    def rms_residuals(self) -> float:
        """
        If a fit exists, return the RMS of the residuals.
        """

        return np.std(self.residuals)

    def plot_fit(self, ax: plt.Axes | None = None) -> None:
        """
        Generates a plot of the (normalised) wavelet and speclet raw data, with the
        functional fit overplotted on a denser grid of wavelengths.

        The central wavelength and RMS of the residuals are labelled.
        """

        if ax is None:
            fig = plt.figure(figsize=(3, 3))
            ax = fig.gca()
            show = True
        else:
            show = False

        x = self.wavelet - self.center_wavelength
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            return

        if ax.get_xlim() == (0.0, 1.0):
            ax.set_xlim(min(x), max(x))

        ax.set_ylim(-0.1, 1.5)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels([f"{yt:.2g}" for yt in ax.get_yticks()])

        xfit = np.linspace(min(x), max(x), 100)
        yfit = self.evaluate_fit(x=xfit, about_zero=True)

        if yfit is None:
            return

        maxy = max(yfit)

        ax.step(
            x,
            self.speclet / maxy,
            where="mid",
            color=wavelength_to_rgb(self.center_wavelength),
            lw=2.5,
            path_effects=stroke(4),
            label="Peak data",
        )

        ax.plot(
            xfit,
            yfit / maxy,
            color="k",
            label=f"{self.fit_type}\nRMS(residuals)={self.rms_residuals:.2e}",
        )

        ax.axvline(
            x=0,
            color="r",
            ls="--",
            alpha=0.5,
            label=f"{self.center_wavelength:.2f}$\AA$",
        )

        ax.set_xlabel("$\lambda$ [$\AA$]")
        ax.set_ylabel("")
        ax.legend(loc="upper center", fontsize="small", frameon=True)

        if show:
            plt.show()

    def has(self, prop: str) -> str:
        """String generation"""

        if prop == "speclet":
            if self.speclet is None:
                return "[ ]"
            return "[x]"

        if prop == "wavelet":
            if self.wavelet is None:
                return "[ ]"
            return "[x]"

        if prop == "fit":
            if self.fit_type is None:
                return "[ ]"
            return "[x]"

        return f"[{prop}?]"

    def __repr__(self) -> str:
        return (
            "Peak("
            f"order_i={self.order_i:<2}, "
            f"coarse_wavelength={self.coarse_wavelength:.3f}, "
            f"speclet={self.speclet}, "
            f"wavelet={self.wavelet})"
        )

    def __str__(self) -> str:
        return (
            "\nPeak("
            f"order_i {self.order_i:<2}, "
            f"coarse_wavelength {self.coarse_wavelength:.3f}, "
            f"{self.has('speclet')} speclet, "
            f"{self.has('wavelet')} wavelet, "
            f"{self.has('fit')} fit: "
            f"center_wavelength {self.center_wavelength:.3f})"
        )

    def __add__(self, other: float | Peak) -> bool:
        if isinstance(other, Peak):
            return self.wl + other.wl
        if isinstance(other, float | int):
            return self.wl + other
        raise ValueError

    def __sub__(self, other: float | Peak) -> bool:
        if other is None:
            return self.wl
        if isinstance(other, Peak):
            return self.wl - other.wl
        if isinstance(other, float | int):
            return self.wl - other
        raise ValueError

    def __eq__(self, other: float | Peak) -> bool:
        if isinstance(other, Peak):
            return self.wl == other.wl
        if isinstance(other, float | int):
            return self.wl == other
        raise ValueError

    def __lt__(self, other: float | Peak) -> bool:
        if isinstance(other, Peak):
            return self.wl < other.wl
        if isinstance(other, float | int):
            return self.wl < other
        raise ValueError

    def __gt__(self, other: float | Peak) -> bool:
        if isinstance(other, Peak):
            return self.wl > other.wl
        if isinstance(other, float | int):
            return self.wl > other
        raise ValueError

    def __contains__(self, wl: float) -> bool:
        return min(self.wavelet) <= wl <= max(self.wavelet)

    def __hash__(self) -> hash:
        return hash(f"{self.i}{self.wl}")


@dataclass
class Order:
    """
    Contains data arrays read in from KPF L1 FITS files

    Properties:
        parent_ref: weakref
            A reference to the parent Spectrum object for the Order. Used to populate
            the relevant information (orderlet, order_i)

        orderlet: str
            The name of the orderlet for which data should be loaded. Valid options:
                SKY, SCI1, SC2, SCI3, CAL
        spec: ArrayLike [ADU]
            An array of flux values as loaded from the parent Spectrum object's
            `spec_file` FITS file(s)
        i: int [Index starting from zero]
            Index of the echelle order in the full spectrum

        wave: ArrayLike [Angstrom] | None
            An array of wavelength values as loaded from the parent Spectrum object's
            `wls_file` FITS file

        peaks: list[Peak]
            A list of Peak objects within the order. Originally populated when the
            `locate_peaks()` method is called.

        parent
            Returns the parent Spectrum object to which this Order belongs.

        peak_wavelengths: ArrayLike [Angstrom]
            Returns a list of the central wavelengths of all contained Peaks.

        mean_wave: float [Angstrom]
            Returns the mean wavelength of the Order in Agnstroms.

    Methods:
        apply_wavelength_solution(wls: ArrayLike):
            A simple `setter' function to apply wavelength values to the `wave' array.

        locate_peaks():
            Uses `scipy.sigal.find_peaks` to roughly locate peak positions. See function
            docstring for more detail. Returns itself so methods can be chained.

        fit_peaks(type: str = "conv_gauss_tophat"):
            Wrapper function which calls peak fitting function for each contained peak.
            Returns the Order itself so methods can be chained.

        has(prop: str):
            Used for repr generation. Returns a checked box if the Order has `spec' or
            `wave' arrays, else returns an empty box.

        __repr__():
            Returns a one-line summary of the Order object. May be expanded in future.

    """

    parent_ref = weakref.ReferenceType

    orderlet: str  # SCI1, SCI2, SCI3, CAL, SKY
    i: int
    spec: ArrayLike | None = None
    wave: ArrayLike | None = None

    spec_file: str | None = None
    wls_file: str | None = None
    _wls_source: str | None = None

    peaks: list[Peak] = field(default_factory=list)

    @property
    def wls_source(self) -> str:
        """
        This function returns a string that can be used to identify which wavelength
        calibration source was used to generate the underlying wavelength solution (wls)
        that the order's `wave` array data is based on.

        The source can be set explicitly by addressing the Order._wls_source property
        """
        if self._wls_source is not None:
            return self._wls_source

        if self.wls_file is None:
            return "unknown"

        if "lfc" in self.wls_file:
            return "lfc"

        if "thar" in self.wls_file:
            return "thorium"

        return "unknown"

    @property
    def parent(self) -> Spectrum:
        """
        Return the Spectrum object to which this Order belongs
        """

        try:
            return self.parent_ref()
        except (NameError, TypeError):
            return None

    @property
    def peak_wavelengths(self) -> ArrayLike:
        """
        Return a list containing all the central wavelengths of the contained Peaks
        """

        return [p.wl for p in self.peaks()]

    @property
    def mean_wave(self) -> float:
        """
        Return the mean wavelength of the Order's `wave` array (a float in Angstroms)
        """

        return np.mean(self.wave)

    def apply_wavelength_solution(self, wls: ArrayLike, wls_file: str) -> Order:
        """
        Basically a setter function to apply wavelength values to the `wave` array
        """

        self.wls_file = wls_file
        self.wave = wls
        return self

    def locate_peaks(
        self,
        threshold: float | None = None,
        height: float | None = 0.01,
        distance: float | None = 10,
        prominence: float | None = None,
        width: float | None = 3,
        wlen: int | None = None,
        rel_height: float = 0.5,
        plateau_size: float | None = None,
        fractional_heights: bool = True,
        window_to_save: int = 16,
    ) -> Order:
        """
        Short description

        A function using `scipy.signal.find_peaks` to roughly locate peaks within the
        Order.spec flux array, and uses the corresponding wavelengths in the Order.wave
        array to populate a list of Peak objects.

        Args:
            threshold (float, optional). Defaults to None. Here simply to expose the
                underling scipy.signal.find_peaks parameter
            height (float, optional). Defaults to 0.01.
                The minimum height of the peak as a fraction of the maxmium value in the
                flux array. Should account for the expected blaze efficiency curve
            distance (float, optional). Defaults to 10.
                The minimum distance between peaks (here in pixels)
            prominence (float, optional). Defaults to None. Here simply to expose the
                underling scipy.signal.find_peaks parameter
            width (float, optional). Defaults to 3.
                The minimum width of the peaks themselves. Setting this higher than 1-2
                will avoid location of single-pixel noise spikes or cosmic rays, but the
                setting should not exceed the resolution element sampling in the
                spectrograph.
            wlen (float, optional). Defaults to None. Here simply to expose the
                underling scipy.signal.find_peaks parameter
            rel_height (float, optional). Defaults to None. Here simply to expose the
                underling scipy.signal.find_peaks parameter
            plateau_size (float, optional). Defaults to None. Here simply to expose the
                underling scipy.signal.find_peaks parameter
            fractional_heights (bool, optional). Defaults to True. If True, the
                `height', `prominence', and `rel_height' parameters are treated as
                fractions of the maximum value of the `spec' array.
            window_to_save (int, optional). Defaults to 16.
                The total number of pixels to save into each Peak object. A slice of
                both the `wave` and `spec` arrays is stored in each Peak, where an
                analytic function is fit to this data.

        Returns:
            Order: The Order object itself so methods may be chained
        """

        if self.spec is None:
            logger.error(f"No `spec` array for order {self}")
            return self

        if self.wave is None:
            logger.error(f"No `wave` array for order {self}")
            return self

        if fractional_heights:
            if height is not None:
                height = height * np.nanmax(self.spec)
            if prominence is not None:
                prominence = prominence * (np.nanmax(self.spec) - np.nanmin(self.spec))
            if rel_height is not None:
                rel_height = rel_height * np.nanmax(self.spec)

        y = self.spec - np.nanmin(self.spec)
        y = y[~np.isnan(y)]
        p, _ = find_peaks(
            y,
            threshold=threshold,
            height=height,
            distance=distance,
            prominence=prominence,
            width=width,
            wlen=wlen,
            rel_height=rel_height,
            plateau_size=plateau_size,
        )

        self.peaks = [
            Peak(
                parent_ref=weakref.ref(self),
                coarse_wavelength=self.wave[_p],
                order_i=self.i,
                speclet=self.spec[
                    _p - window_to_save // 2 : _p + window_to_save // 2 + 1
                ],
                wavelet=self.wave[
                    _p - window_to_save // 2 : _p + window_to_save // 2 + 1
                ],
                starting_pixel=_p - window_to_save // 2,
                distance_from_order_center=abs(self.wave[_p] - self.mean_wave),
            )
            for _p in p
            # ignore peaks that are too close to the edge of the order
            if _p >= window_to_save // 2 and _p <= len(self.spec) - window_to_save // 2
        ]

        return self

    def fit_peaks(
        self,
        fit_type: str = "conv_gauss_tophat",
        space: str = "pixel",
    ) -> Order:
        """
        Call the fitting routine for all contained Peak objects.

        Args:
            fit_type (str, optional): The function with which peaks should be fit.
                Defaults to "conv_gauss_tophat" (other option is "gaussian")
            space (str, optional):
                Can be either "wavelength" or "pixel". This determines whether the
                fitting routine is done in wavelength space or pixel space. In general,
                a resolved peak should be fit in wavelength space, while an unresolved
                (delta function) peak should be fit in pixel space. Defaults to "pixel".

        Returns:
            Order: The Order object itself so methods may be chained
        """

        for p in self.peaks:
            p.fit(fit_type=fit_type, space=space)

        return self

    @property
    def num_peaks(self) -> int:
        """
        Simply returns the number of peaks in the Order
        """

        return len(self.peaks)

    @property
    def spec_fit(self) -> ArrayLike:
        """
        Stitch together all peak fits where they exist, leaving `spec' values not coverd
        by any wavelengths untouched. Useful for plotting the fit (and/or residuals) at
        the full-order level.
        """

        spec_fit = self.spec.copy()
        for p in self.peaks:
            min_wl = min(p.wavelet)
            max_wl = max(p.wavelet)

            lowmask = min_wl <= self.wave
            highmask = self.wave <= max_wl
            mask = lowmask & highmask

            spec_fit[mask] = p.evaluate_fit(self.wave[mask])

        return spec_fit

    def spec_residuals(self, normalised: bool = True) -> ArrayLike:
        """
        Return the full-order residuals between the original `spec' array and the
        stitched `spec_fit' array of all of the peak fits. See `spec_fit' for more
        details.
        """

        res = np.zeros_like(self.spec)
        for p in self.peaks:
            min_wl = min(p.wavelet)
            max_wl = max(p.wavelet)

            lowmask = min_wl <= self.wave
            highmask = self.wave <= max_wl
            mask = lowmask & highmask

            if normalised:
                res[mask] = p.residuals
            else:
                res[mask] = p.speclet - p.evaluate_fit(p.wavelet)

        return res

    def residuals_rms_split(
        self,
        threshold: float = 0.05,
    ) -> tuple[list[Peak], list[Peak]]:
        """
        This function splits the peaks into two lists, one containing peaks with RMS
        residuals below or equal to a certain threshold, and the other containing peaks
        with RMS residuals above that threshold.

        This is useful for identifying which peaks are well-fit and which are not,
        particularly in the context of fitting peaks in a thorium-argon spectrum (not
        the primary intended use for this code, but one of the always-envisaged
        use-cases).
        """

        good_peaks = []
        bad_peaks = []

        for p in self.peaks:
            if p.rms_residuals <= threshold:
                good_peaks.append(p)
            else:
                bad_peaks.append(p)

        return good_peaks, bad_peaks

    def discard_poorly_fit_peaks(self, threshold: float = 0.05) -> Order:
        """
        Discard peaks with RMS residuals above a certain threshold. This wraps and
        extends the method `residuals_rms_split`, keeping only those peaks that have
        residuals RMS values below the threshold.
        """

        good_peaks, _ = self.residuals_rms_split(threshold=threshold)
        self.peaks = good_peaks

        return self

    @property
    def fit_parameters(self) -> dict:
        """
        Construct a dictionary of the fit parameters and their standard deviations
        """

        return {
            "fit_type": [p.fit_type for p in self.peaks],
            "center_wavelength": [p.center_wavelength for p in self.peaks],
            "amplitude": [p.amplitude for p in self.peaks],
            "sigma": [p.sigma for p in self.peaks],
            "boxhalfwidth": [p.boxhalfwidth for p in self.peaks],
            "offset": [p.offset for p in self.peaks],
            "center_wavelength_stddev": [
                p.center_wavelength_stddev for p in self.peaks
            ],
            "amplitude_stddev": [p.amplitude_stddev for p in self.peaks],
            "sigma_stddev": [p.sigma_stddev for p in self.peaks],
            "boxhalfwidth_stddev": [p.boxhalfwidth_stddev for p in self.peaks],
            "offset_stddev": [p.offset_stddev for p in self.peaks],
        }

    def has(self, prop: str) -> str:
        """
        String generation

        Args:
            prop (str): The property to check for, can be either `spec' or `wave'

        Returns:
            str: A checked box if the Order has the property, else an empty box
        """

        if prop == "spec":
            if self.spec is None:
                return "[ ]"
            return "[x]"
        if prop == "wave":
            if self.wave is None:
                return "[ ]"
            return "[x]"
        return f"[{prop}?]"

    def __str__(self) -> str:
        return (
            "Order("
            f"orderlet={self.orderlet}, i={self.i:<2}, "
            f"{self.has('spec')} spec, {self.has('wave')} wave, "
            f"{len(self.peaks)} peaks, "
            f"spec_file={self.spec_file}, "
            f"wls_file={self.wls_file})"
        )

    def __repr__(self) -> str:
        return (
            "Order("
            f"orderlet={self.orderlet}, i={self.i:<2}, "
            f"spec={self.spec}, "
            f"wave={self.wave}, "
            f"spec_file={self.spec_file}, "
            f"wls_file={self.wls_file})"
        )

    def __contains__(self, wl: float) -> bool:
        """
        Determine if a given wavelength is within the bounds of the Order.

        Args:
            wl (float): Wavelength value in Angstroms

        Returns:
            boolean evaluation if the wavelength is within the Order's bounds

        Example usage::
            order = Order(...)
            query_wavelength = 6328.0

            if (query_wavelength in order):
                print(f"Yes, {query_wavelength} is in the Order!")
        """

        return min(self.wave) <= wl <= max(self.wave)


@dataclass
class Spectrum:
    """
    Contains data and metadata corresponding to a loaded KPF FITS file and optionally a
    wavelength solution loaded from a separate FITS file. Contains a list of Order
    objects (where the loaded L1 data is stored), each of which can contain a list of
    Peak objects. All interfacing can be done to the Spectrum object, which initiates
    function calls in the child objects, and which receives output data passed upward to
    be accessed again at the Spectrum level.

    Properties:
        spec_file: Path | str | list[Path] | list[str]
            The path (or a list of paths) of the L1 file(s) containing flux data to be
            loaded. If a list of files is passed, the flux data is median-combined.
        wls_file: Path | str
            The path of a single file to draw the wavelength solution (WLS) from. This
            is typically the master L1 WLS file for the same date as the flux data.
        orderlets_to_load: str | list[str]
            Which orderlets should be loaded into the Spectrum (and Orders).

        reference_mask: str
            [Not yet implemented], path to a file containing a list of wavelengths
            corresponding to etalon line locations in a reference file. Rather than
            locating peaks in each order, the code should take these reference
            wavelengths as its starting point.
        reference_peaks: list[float]
            [Not yet implemented], the list of wavelengths as parsed from
            `reference_mask'

        _orders: list[Order]
            A list of Order objects (see Order definition)
            See also the .orders() method, the main interface to the Order
            objects.

        orderlets: list[str]
            Returns all unique orderlets that the contained Order objects correspond to.

        date: str
            The date of observation, as read from the FITS header of `spec_file'
        sci_obj: str
            The SCI-OBJ keyword from the FITS header of `spec_file'
        cal_obj: str
            The CAL-OBJ keyword from the FITS header of `spec_file'
        object: str
            The OBJECT keyword from the FITS header of `spec_file'

        filtered_peaks: list[Peak]
            A list of Peak objects after locating, fitting, and filtering.

        pp: str = ""
            A prefix to add to any print or logging statements, for a nicer command line
            interface.

        peaks: list[Peak]
            Traverses the list of Orders, and each Order's list of Peaks. Returns a
            compiled list of all Peaks, the grandchildren of this Spectrum object.

        timeofday: str
            Returns the time of day of the `specfile' FITS file
            Possible values: "morn", "eve", "night", "midnight"

        summary: str
            Create a text summary of the Spectrum


    Methods:
        orders(orderlet: str, i: int) -> list[Order]
            returns a list of orders matching either or both of the input parameters.
            This is the main interface to the Order objects.

        num_located_peaks(orderlet: str) -> int
            Returns the total number of located peaks in all Orders

        num_successfully_fit_peaks(orderlet: str) -> int
            Returns the total number of peaks that have a non-NaN center_wavelength
            property

        parse_reference_mask
            Reads in a reference mask (as output from the `save_peak_locations` method)
            and populates `self.reference_peaks` with the wavelengths. The rest of the
            functionality of using this mask is not yet implemented

        apply_reference_mask
            Once a reference mask is parsed (its wavelengths read into a list), these
            can be applied with this method, which passes the relecant wavelengths down
            to each Order, where a list of Peaks is initialised

        load_spec
            If `spec_file' is a string, this method loads the flux data from the file,
            as well as the DATE, SCI-OBJ, CAL-OBJ and OBJECT keywords. If `spec_file' is
            a list of strings, this method loads the flux data from all of the files,
            checks that their SCI-OBJ, CAL-OBJ and OBJECT match one another, and if so,
            combines the fluxes by taking the median value for each pixel. Flux data is
            stored per-order in a list of Orders: self._orders

        find_wls_file
            If no WLS file is passed in, this method is called. It looks in the
            /data/kpf/masters/ directory for the same date as the `spec_file', and finds
            the corresponding wavelength solution file. If the `spec_file' was taken at
            "night" (from OBJECT keyword string), the corresponding "eve" WLS file is
            located, likewise for "midnight".

        load_wls
            Loads the `wls_file' file, and stores its wavelength data per-order in
            self._orders.

        locate_peaks
            Initiates locating peaks for each order. Parameters here are passed to the
            Order-level functions

        fit_peaks
            Initiates fitting peaks at the Peak level. The `type' parameter here is
            passed down to the Peak-level functions

        filter_peaks
            Filters identical peaks that appear in the overlap regions of two adjacent
            orders. Within a given `window` [Angstroms], if two peaks are identified, it
            removes the one that is further away from _its_ order's central wavelength.
            This must be done at the Spectrum level, where many Orders' Peaks can be
            accessed at the same time.

        save_peak_locations
            Outputs the filtered peaks to a csv file to be used as a mask for either
            further iterations of peak-fitting processing, or for measuring the etalon's
            RVs. If peaks have not been filtered yet, it first calls the `filter_peaks`
            method.

        plot_spectrum
            Generates a colour-coded plot of the spectrum. Optionally can use a
            `matplotlib.pyplot.axes` object passed in as `ax` to allow tweaking in the
            script that calls the class.

        delta_nu_FSR
            Compute and return an array of FSR (the spacing between peaks) in units of
            GHz. Nominally the etalon has ~30GHz FSR, in practice there is an absolute
            offset, a global tilt, and smaller-scale bumps and wiggles as a function of
            wavelength

        plot_FSR
            Creates an FSR plot of the located (and fit) peaks, across all Orders.

        save_config_file
            [Not yet implemented], will save the properties and parameters for this
            Spectrum (and its Orders and their Peaks) to an external file.

        TODO: Use a .cfg file as input as well, parsing parameters to run the
        analysis. Unclear if this should go here or in a script that calls the
        Spectrum class. Parameters needed:
         * spec_file
         * wls_file
         * orderlet
         * reference_mask
         * ???
    """

    spec_file: Path | str | list[Path] | list[str] | None = None
    wls_file: Path | str | dict[str, Path] | None = None
    auto_load_wls: bool = True
    orders_to_load: list[int] | None = None
    orderlets_to_load: str | list[str] = None

    reference_mask: Path | str | None = None
    reference_peaks: list[float] | None = None

    _orders: list[Order] = field(default_factory=list)

    # Hold basic metadata from the FITS file
    date: str | None = None
    # DATE-OBS in FITS header (without dashes), eg. 20240131
    sci_obj: str | None = None  # SCI-OBJ in FITS header
    cal_obj: str | None = None  # CAL-OBJ in FITS header
    object: str | None = None  # OBJECT in FITS header

    filtered_peaks: dict[str, list[Peak]] | None = None

    pp: str = ""  # Print prefix

    def __post_init__(self) -> None:
        if isinstance(self.spec_file, str):
            self.spec_file = Path(self.spec_file)

        elif isinstance(self.spec_file, list):
            self.spec_file = [Path(f) for f in self.spec_file]

        if isinstance(self.wls_file, str):
            self.wls_file = Path(self.wls_file)

        if isinstance(self.wls_file, dict):
            self.wls_file = {
                k: Path(f) for k, f in self.wls_file.items() if isinstance(f, str)
            }

        if isinstance(self.orderlets_to_load, str):
            self.orderlets_to_load = [self.orderlets_to_load]

        if self.orderlets_to_load is None:
            self.orderlets_to_load = ORDERLETS

        if isinstance(self.orders_to_load, int):
            self.orders_to_load = [self.orders_to_load]

        if self.orders_to_load is None:
            self.orders_to_load = ALL_ORDER_INDICES

        self.filtered_peaks = {ol: None for ol in self.orderlets_to_load}

        if self._orders:
            ...

        else:
            if self.spec_file:
                self.load_spec()
            if self.wls_file:
                self.load_wls()
            elif self.auto_load_wls:
                self.find_wls_files()
                if self.wls_file:
                    self.load_wls()
            if self.reference_mask:
                self.parse_reference_mask()
            if self.orders() is not None and self.reference_mask is not None:
                self.apply_reference_mask()

    def __add__(self, other: Spectrum) -> Spectrum:
        """
        I've never used this, but it's here so that two Spectrum objects can be added
        together and it just adds the contained `spec` values together
        """

        if not isinstance(other, Spectrum):
            raise TypeError("Can only add two Spectrum objects together")

        return Spectrum(
            file=None,
            orders=[
                Order(i=o1.i, wave=o1.wave, spec=o1.spec + o2.spec)
                for o1, o2 in zip(self.orders(), other.orders(), strict=True)
            ],
        )

    @property
    def timeofday(self) -> str:
        """
        Get the Spectrum's `timeofday` (time of day) property.

        For a master etalon spectrum file, the time of day is recorded in the FITS
        header as part of the OBJECT keyword, eg "etalon_autocal_all_night".

        Returns:
            str: One of `morn`, `eve`, `night`, or `midnight`
        """

        return self.object.split("-")[-1]

    @property
    def orderlets(self) -> list[str]:
        """
        Loops through the contained Order objects and returns a list of the orderlets
        that the data corresponds to.
        """

        return list({o.orderlet for o in self.orders()})

    def orders(self, orderlet: str | None = None, i: int | None = None) -> list[Order]:
        """
        Get a list of all contained Order objects, optionally filtered by orderlet or i

        Args:
            orderlet (str, optional): The orderlet to filter by. Defaults to None.
                Options: "SCI1", "SCI2", "SCI3", "CAL", "SKY"
            i (int, optional): The index of the order to filter by. Defaults to None.
                Options: integers from 0 to 66

        Returns:
            list[Order]: A list of Order objects that match the input parameters
        """

        if (orderlet is not None) and (i is not None):
            result = [o for o in self._orders if o.orderlet == orderlet and o.i == i]

            if len(result) == 1:
                return result[0]
            if len(result) > 1:
                logger.info(
                    f"{self.pp}"
                    f"More than one Order matching orderlet={orderlet} and i={i}!"
                )
                logger.info(f"{self.pp}{result}")
                return result

            logger.info(f"{self.pp}No matching order found for {orderlet=} and {i=}")
            return None

        if orderlet is not None:
            return sorted(
                [o for o in self._orders if o.orderlet == orderlet],
                key=(attrgetter("i")),
            )

        if i is not None:
            return sorted(
                [o for o in self._orders if o.i == i], key=(attrgetter("orderlet"))
            )

        # neither orderlet nor i is specified!
        return sorted(self._orders, key=(attrgetter("orderlet", "i")))

    def num_orders(self, orderlet: str = "SCI2") -> int:
        """
        Get the number of contained Order objects, filtered by orderlet

        Args:
            orderlet (str, optional): The orderlet to filter by. Defaults to "SCI2".

        Returns:
            int: the number of Order objects that match the input orderlet
        """
        return len(self.orders(orderlet=orderlet))

    def peaks(self, orderlet: str | list[str] | None = None) -> list[Peak]:
        """
        Find all peaks matching a particular orderlet

        Args:
            orderlet (str | list[str] | None, optional): The orderlet to filter by.
                Defaults to None. In that case, the peaks for all orderlets are returned

        Returns:
            list[Peak]: A list of Peak objects that match the (optional) filter criteria
        """

        if isinstance(orderlet, str):
            orderlet = [orderlet]

        if orderlet is None:
            orderlet = self.orderlets

        result = [
            p for ol in orderlet for o in self.orders(orderlet=ol) for p in o.peaks
        ]

        if not result:
            return None

        return result

    def num_located_peaks(self, orderlet: str | list[str] | None = None) -> int:
        """
        Get the number of located peaks in the Spectrum's Orders.

        Args:
            orderlet (str | list[str] | None, optional): The orderlet to optionally
            filter by. Defaults to None, in which case all orderlets are considered.

        Returns:
            int: The number of located peaks in the Spectrum's Orders
        """
        if isinstance(orderlet, str):
            return sum(len(o.peaks) for o in self.orders(orderlet=orderlet))

        if orderlet is None:
            orderlet = self.orderlets

        return {
            ol: sum(len(o.peaks) for o in self.orders(orderlet=ol)) for ol in orderlet
        }

    def num_successfully_fit_peaks(
        self,
        orderlet: str | list[str] | None = None,
    ) -> int:
        """
        Get the number of **successfully fit** peaks in the Spectrum's Orders.

        Args:
            orderlet (str | list[str] | None, optional): The orderlet to optionally
            filter by. Defaults to None, in which case all orderlets are considered.

        Returns:
            int: The number of **successfull fit** peaks in the Spectrum's Orders
        """
        if isinstance(orderlet, str):
            return sum(
                1
                for o in self.orders(orderlet=orderlet)
                for p in o.peaks
                if not np.isnan(p.center_wavelength)
            )

        if orderlet is None:
            orderlet = self.orderlets

        return {
            ol: sum(
                1
                for o in self.orders(orderlet=ol)
                for p in o.peaks
                if not np.isnan(p.center_wavelength)
            )
            for ol in orderlet
        }

    def num_filtered_peaks(
        self,
        orderlet: str | list[str] | None = None,
    ) -> int:
        """
        Get the number of **successfully fit** peaks in the Spectrum's Orders **after
        filtering**.

        Args:
            orderlet (str | list[str] | None, optional): The orderlet to optionally
            filter by. Defaults to None, in which case all orderlets are considered.

        Returns:
            int: The number of **successfully fit** peaks in the Spectrum's Orders
                **after filtering**
        """
        if not self.filtered_peaks:
            logger.warning(
                f"{self.pp}"
                "List of filtered peaks is empty. Call Spectrum.filter_peaks() first."
            )
            return 0

        if isinstance(orderlet, str):
            return len(self.filtered_peaks[orderlet])

        if orderlet is None:
            orderlet = self.orderlets

        return {ol: len(self.filtered_peaks[ol]) for ol in orderlet}

    def parse_reference_mask(self) -> Spectrum:
        """
        Not yet fully implemented. Designed to read in a reference mask file, which is
        then to be used as the starting point for locating peaks in the Orders.

        Returns:
            Spectrum (self): The Spectrum object itself, so methods can be chained.
        """
        p = Path(self.reference_mask)
        with p.open(mode="r") as f:
            lines = f.readlines()

            self.reference_peaks = [float(line.strip().split(" ")[0]) for line in lines]

        return self

    def apply_reference_mask(self) -> Spectrum:
        """
        Applying a reference mask to the Spectrum. Not yet implemented, but this method
        would initiate the peak location process for each Order, using the reference
        wavelengths as the starting point.
        """
        raise NotImplementedError

        if not self.orders():
            logger.warning(
                f"{self.pp}No order data - first load data then apply reference mask"
            )

            return self

        for _o in self.orders():
            """
            Find the wavelength limits
            Loop through reference mask (should be sorted)
            For any wavelengths in the range, create a Peak with that coarse wavelength,
            also need to create slices of the underlying data?

            Maybe it's best done at the Order level, but I just pass the relevant peak
            wavelengths down.

            TODO
            """

        return self

    def load_spec(self) -> Spectrum:
        """
        Load spectrum data

        One of the main construction functions for a Spectrum object, called
        automatically after initialisation. This method reads in the flux data from the
        `spec_file` FITS file(s), and stores it in the `spec` attribute of each Order
        (which this method also creates).

        Raises:
            NotImplementedError: if `spec_file` is not a string or list of strings

        Returns:
            Spectrum (self): The Spectrum object itself, so methods can be chained.

        TODO: Allow for pathlib.Path objects to be passed in
        """
        if isinstance(self.spec_file, Path):
            logger.info(
                f"{self.pp}Loading flux values from a single file: "
                f"{self.spec_file.name}..."
            )

            _orders = []
            for ol in self.orderlets_to_load:
                spec_green = fits.getdata(
                    self.spec_file,
                    f"GREEN_{get_orderlet_name(ol)}_FLUX{get_orderlet_index(ol)}",
                )
                spec_red = fits.getdata(
                    self.spec_file,
                    f"RED_{get_orderlet_name(ol)}_FLUX{get_orderlet_index(ol)}",
                )

                self.date = "".join(fits.getval(self.spec_file, "DATE-OBS").split("-"))
                self.sci_obj = fits.getval(self.spec_file, "SCI-OBJ")
                self.cal_obj = fits.getval(self.spec_file, "CAL-OBJ")
                self.object = fits.getval(self.spec_file, "OBJECT")

                spec = np.append(spec_green, spec_red, axis=0)

                if self.orders_to_load is not None:
                    for i in self.orders_to_load:
                        _orders.append(
                            Order(
                                orderlet=ol,
                                spec=spec[i],
                                spec_file=self.spec_file.name,
                                wave=None,
                                i=i,
                            )
                        )

                else:
                    for i, s in enumerate(spec):
                        _orders.append(
                            Order(
                                orderlet=ol,
                                spec=s,
                                spec_file=self.spec_file.name,
                                wave=None,
                                i=i,
                            )
                        )

            self._orders = _orders

        elif isinstance(self.spec_file, list):
            _orders = []
            for ol in self.orderlets_to_load:
                logger.info(
                    f"{self.pp}Loading {ol} flux values from a list of "
                    f"{len(self.spec_file)} files..."
                )

                spec_green = np.median(
                    [
                        fits.getdata(
                            f,
                            f"GREEN_{get_orderlet_name(ol)}_"
                            + f"FLUX{get_orderlet_index(ol)}",
                        )
                        for f in self.spec_file
                    ],
                    axis=0,
                )
                spec_red = np.median(
                    [
                        fits.getdata(
                            f,
                            f"RED_{get_orderlet_name(ol)}_"
                            + f"FLUX{get_orderlet_index(ol)}",
                        )
                        for f in self.spec_file
                    ],
                    axis=0,
                )

                try:
                    assert all(
                        fits.getval(f, "SCI-OBJ")
                        == fits.getval(self.spec_file[0], "SCI-OBJ")
                        for f in self.spec_file
                    )
                    self.sci_obj = fits.getval(self.spec_file[0], "SCI-OBJ")
                except AssertionError:
                    logger.warning(
                        f"{self.pp}SCI-OBJ did not match between input files!"
                    )
                    logger.warning(f"{self.pp}{self.spec_file}")

                try:
                    assert all(
                        fits.getval(f, "CAL-OBJ")
                        == fits.getval(self.spec_file[0], "CAL-OBJ")
                        for f in self.spec_file
                    )
                    self.cal_obj = fits.getval(self.spec_file[0], "CAL-OBJ")
                except AssertionError:
                    logger.warning(
                        f"{self.pp}CAL-OBJ did not match between input files!"
                    )
                    logger.warning(f"{self.pp}{self.spec_file}")

                try:
                    assert all(
                        fits.getval(f, "OBJECT")
                        == fits.getval(self.spec_file[0], "OBJECT")
                        for f in self.spec_file
                    )
                    self.object = fits.getval(self.spec_file[0], "OBJECT")
                except AssertionError:
                    logger.warning(
                        f"{self.pp}OBJECT did not match between input files!"
                    )
                    logger.warning(f"{self.pp}{self.spec_file}")

                try:
                    assert all(
                        fits.getval(f, "DATE-OBS")
                        == fits.getval(self.spec_file[0], "DATE-OBS")
                        for f in self.spec_file
                    )
                    self.date = "".join(
                        fits.getval(self.spec_file[0], "DATE-OBS").split("-")
                    )
                except AssertionError:
                    logger.warning(
                        f"{self.pp}DATE-OBS did not match between input files!"
                    )
                    logger.warning(f"{self.pp}{self.spec_file}")

                spec = np.append(spec_green, spec_red, axis=0)

                if self.orders_to_load is not None:
                    spec = spec[self.orders_to_load]

                for i, s in enumerate(spec):
                    _orders.append(
                        Order(
                            orderlet=ol,
                            spec=s,
                            spec_file=self.spec_file,
                            wave=None,
                            i=i,
                        )
                    )

            self._orders = _orders

        else:  # self.spec_file is something else entirely
            raise NotImplementedError(
                "`spec_file` must be a single Path or list of Paths"
            )

        return self

    def find_wls_files(self) -> str:
        """
        Find the WLS files matching the date and time of the Spectrum's `spec_file` FITS
        file. This function will locate the matching LFC and ThAr WLS files.

        If the `spec_file` was taken at night, the corresponding "eve" WLS files are
        located, likewise for "midnight". This method is automatically called only if no
        `wls_file` filename is explicitly passed in, and if `auto_load_wls` is True.
        """

        wls_to_find = []

        # Using set intersections to determine if there is any overlap
        if set(self.orders_to_load) & set(LFC_ORDER_INDICES):
            wls_to_find.append("lfc")
        if set(self.orders_to_load) & set(THORIUM_ORDER_INDICES):
            wls_to_find.append("thar")

        wls_files = None

        if self.timeofday in ["night", "midnight"]:
            wls_files: dict[str, Path] = {
                wls_type: (
                    MASTERS_DIR
                    / f"{self.date}"
                    / f"kpf_{self.date}_master_WLS_autocal-{wls_type}-all-eve_L1.fits"
                )
                for wls_type in wls_to_find
            }

        else:
            wls_files: dict[str, Path] = {
                wls_type: (
                    MASTERS_DIR / f"{self.date}" / f"kpf_{self.date}_master_WLS_"
                    f"autocal-{wls_type}-all-{self.timeofday}_L1.fits"
                )
                for wls_type in wls_to_find
            }

        if "lfc" in wls_files:
            try:
                assert "lfc" in fits.getval(wls_files["lfc"], "OBJECT").lower()
            except AssertionError:
                logger.warning(
                    f"{self.pp}"
                    f"'lfc' not found in {self.timeofday} WLS file 'OBJECT' value!"
                )
            except FileNotFoundError:
                logger.warning(f"{self.pp}{self.timeofday} WLS file not found")

        if "thar" in wls_files:
            try:
                assert "thar" in fits.getval(wls_files["thar"], "OBJECT").lower()
            except AssertionError:
                logger.warning(
                    f"{self.pp}"
                    f"'thar' not found in {self.timeofday} WLS file 'OBJECT' value!"
                )
            except FileNotFoundError:
                logger.warning(f"{self.pp}{self.timeofday} WLS file not found")

        if wls_files is not None:
            self.wls_file = wls_files
        else:
            # Use the WLS embedded in the spec_file?
            self.wls_file = self.spec_file

    def load_wls(self) -> Spectrum:
        """
        Load the wavelength solution (WLS) data from the Spectrum's `wls_file` FITS
        file(s). `wls_file` can either be a single file or (especially if the
        corresponding files were automatically located) a dictionary of filenames
        corresponding to two different (LFC and ThAr) WLS files.

        Raises:
            FileNotFoundError:
            NotImplementedError:

        Returns:
            Spectrum (self): The Spectrum object itself, so methods can be chained.
        """
        if self.wls_file is None:
            raise FileNotFoundError("No WLS file specified or found!")

        if isinstance(self.wls_file, list):
            raise NotImplementedError(
                "wls_file must be either a single filename or a dictionary."
            )

        if isinstance(self.wls_file, Path):
            self.load_single_wls(wls_file=self.wls_file)

        elif isinstance(self.wls_file, dict):
            if "lfc" in self.wls_file:
                self.load_partial_wls(
                    wls_file=self.wls_file["lfc"],
                    valid_i=LFC_ORDER_INDICES,
                )
            if "thar" in self.wls_file:
                self.load_partial_wls(
                    wls_file=self.wls_file["thar"],
                    valid_i=THORIUM_ORDER_INDICES,
                )

        return self

    def load_single_wls(self, wls_file: Path) -> Spectrum:
        """
        Load a single wavelength solution file.

        This method is called by `load_wls` if the `wls_file` is a single filename. It
        reads the wavelength solution data from the file, and stores it in the `wave`
        attribute of each Order object.

        Raises:
            FileNotFoundError: If the WLS file is not found

        Returns:
            Spectrum (self): The Spectrum object itself, so methods can be chained.
        """

        if not self.wls_file.is_file():
            raise FileNotFoundError(f"WLS file not found: {wls_file.name}")

        logger.info(
            f"{self.pp}Loading WLS values from a single file: {wls_file.name}..."
        )

        for ol in self.orderlets_to_load:
            wave_green = fits.getdata(
                wls_file,
                f"GREEN_{get_orderlet_name(ol)}_WAVE{get_orderlet_index(ol)}",
            )
            wave_red = fits.getdata(
                wls_file,
                f"RED_{get_orderlet_name(ol)}_WAVE{get_orderlet_index(ol)}",
            )

            wave = np.append(wave_green, wave_red, axis=0)

            # If there are no orders already (for this orderlet), just populate
            # a new set of orders with only the wavelength solution, no flux yet
            if not self.orders(orderlet=ol):
                for i, w in enumerate(wave):
                    self._orders.append(
                        Order(
                            spec=None,
                            wave=w,
                            wls_file=self.wls_file.name,
                            i=i,
                        )
                    )
                continue

            # Otherwise, apply the wavelength solution to the appropriate orders
            if self.orders_to_load is not None:
                for i in self.orders_to_load:
                    self.orders(orderlet=ol, i=i).apply_wavelength_solution(
                        wls=wave[i], wls_file=self.wls_file.name
                    )
                continue

            for i, w in enumerate(wave):
                try:
                    self.orders(orderlet=ol, i=i).apply_wavelength_solution(
                        wls=w, wls_file=self.wls_file.name
                    )
                except AttributeError as e:
                    logger.error(f"{self.pp}{e}")
                    logger.error(f"{self.pp}No order exists: orderlet={ol}, {i=}")

        return self

    def load_partial_wls(
        self,
        wls_file: Path | str,
        valid_i: int | list[int] | None = None,
        orderlet: str | list[str] | None = None,
    ) -> Spectrum:
        """
        Load a wavelength solution and apply it only to a subset of orderlets or orders.

        This used to handle files that need multiple wavelength solutions (eg. using the
        laser frequency comb solution for some orders, and using the thorium-argon
        solution for others).
        """
        if isinstance(wls_file, str):
            wls_file = Path(wls_file)

        if valid_i is None:
            valid_i = self.orders_to_load

        if isinstance(valid_i, int):
            valid_i = [valid_i]

        if orderlet is None:
            orderlet = self.orderlets

        if isinstance(orderlet, str):
            orderlet = [orderlet]

        logger.info(f"{self.pp}Loading partial WLS from file: {wls_file.name}...")

        for ol in orderlet:
            wave_green = fits.getdata(
                wls_file,
                f"GREEN_{get_orderlet_name(ol)}_WAVE{get_orderlet_index(ol)}",
            )
            wave_red = fits.getdata(
                wls_file,
                f"RED_{get_orderlet_name(ol)}_WAVE{get_orderlet_index(ol)}",
            )

            wave = np.append(wave_green, wave_red, axis=0)

            for i in valid_i:
                if i in self.orders_to_load:
                    self.orders(orderlet=ol, i=i).apply_wavelength_solution(
                        wls=wave[i], wls_file=wls_file.name
                    )

        return self

    def locate_peaks(
        self,
        orderlet: str | list[str] | None = None,
        threshold: float = None,
        height: float = 0.1,
        distance: float = 10,
        prominence: float = None,
        width: float = 3,
        wlen: int = None,
        rel_height: float = 0.5,
        plateau_size: float = None,
        fractional_heights: bool = True,
        window_to_save: int = 16,
    ) -> Spectrum:
        """
        Initiate the peak location process for each Order.

        This method loops through each of the containted Order objects, and calls the
        `locate_peaks` method for each one.

        Args:
            orderlet (str | list[str] | None, optional): Defaults to None.
            threshold
            height (float, optional): Defaults to 0.1.
            distance (float, optional): Defaults to 10.
            prominence
            width (float, optional): Defaults to 3.
            wlen
            rel_height
            plateau_size
            fractional_heights
            window_to_save (int, optional): Defaults to 16.

        Returns:
            Spectrum (self): The Spectrum object itself, so methods can be chained.
        """

        if isinstance(orderlet, str):
            orderlet = [orderlet]
        elif orderlet is None:
            orderlet = self.orderlets

        if self.reference_mask is None:
            for ol in orderlet:
                logger.info(f"{self.pp}Locating {ol:<4} peaks...")
                for o in self.orders(orderlet=ol):
                    o.locate_peaks(
                        threshold=threshold,
                        height=height,
                        distance=distance,
                        prominence=prominence,
                        width=width,
                        wlen=wlen,
                        rel_height=rel_height,
                        plateau_size=plateau_size,
                        fractional_heights=fractional_heights,
                        window_to_save=window_to_save,
                    )

        else:
            logger.info(
                f"{self.pp}Not locating peaks because a reference mask was passed in."
            )
        return self

    def fit_peaks(
        self,
        orderlet: str | list[str] | None = None,
        fit_type: str = "conv_gauss_tophat",
        space: str = "pixel",
    ) -> Spectrum:
        """
        Fit the Peaks in each contained Order object.

        This method loops through each of the contained Order objects, and calls the
        `fit_peaks` method for each. That method in turn loops through each Peak object
        in the Order, and fits the peak using the specified `fit_type` function. The
        package is designed so that this process can be initiated at any level,
        typically starting at the Spectrum level.

        Args:
            orderlet (str | list[str] | None, optional): Defaults to None.
            fit_type (str, optional): Defaults to "conv_gauss_tophat".
            space (str, optional): Defaults to "wavelength".

        Returns:
            Spectrum (self): The Spectrum object itself, so methods can be chained.

        TODO: Run multiple fits at once, each in a separate process. The fitting
        routine(s) are naturally the most time-intensive part of running the analysis.
        Because they are individually fit, it should be relatively straightforward to
        run this in multiple processes.

        It could be multiplexed at the Order level (67 orders per speclet), or within
        each order at the Peak level.
        """

        if isinstance(orderlet, str):
            orderlet = [orderlet]

        if orderlet is None:
            orderlet = self.orderlets

        for ol in orderlet:
            if self.num_located_peaks is None:
                self.locate_peaks()

            logger.info(
                f"{self.pp}"
                f"Fitting {ol} peaks in {space} space with {fit_type} function..."
            )

            for o in tqdm(
                self.orders(orderlet=ol),
                desc=f"{self.pp}Orders",
                unit="order",
                ncols=100,
            ):
                o.fit_peaks(fit_type=fit_type, space=space)

        return self

    def discard_poorly_fit_peaks(self, threshold: float = 0.05) -> Spectrum:
        """
        Filter poorly fit peaks by the RMS value of the residuals of the fit.

        I'm imagining this to be done after the fit but before filtering peaks that
        appear in multiple orders.
        """

        for o in self.orders():
            o.discard_poorly_fit_peaks(threshold=threshold)

    def filter_peaks(
        self, orderlet: str | list[str] | None = None, window: float = 0.1
    ) -> Spectrum:
        """
        Filter the peaks to remove identical peaks appearing in adjacent orders.

        Filter the peaks such that any peaks of a close enough wavelength, but appearing
        in different echelle orders, are selected so that only one remains. To do this,
        all Orders (and all Peaks) have an order index, so we can tell which order a
        peak was located in. So I just loop through all peaks, and if two fall within
        the wavelength `window' AND have different order indices, I remove the one that
        is further from its order's mean wavelength (`distance_from_order_center' is
        also storedinside each Peak).

        Args:
            orderlet (str | list[str] | None, optional): The orderlet name. Defaults to
                None, in which case all orderlets are considered.
            window (float, optional): Wavelength value in Angstroms. This is the range
                around each peak that will be searched for any other peaks in adjacent
                orders. Defaults to 0.1.

        Returns:
            Spectrum (self): The Spectrum object itself, so methods can be chained.
        """

        if isinstance(orderlet, str):
            orderlet = [orderlet]

        if orderlet is None:
            orderlet = self.orderlets

        for ol in orderlet:
            logger.info(
                f"{self.pp}Filtering {ol:<4} peaks "
                "to remove identical peaks appearing in adjacent orders..."
            )

            peaks = self.peaks(orderlet=ol)

            if peaks is None:
                logger.warning(f"{self.pp}No peaks found")
                return self

            peaks = sorted(peaks, key=attrgetter("wl"))

            to_keep = []
            # TODO: Use itertools pairwise?
            for i, _ in enumerate(peaks):
                p1: Peak = peaks[i]
                p2: Peak | None = peaks[i + 1] if i + 1 < len(peaks) else None

                if p2 is None:
                    to_keep.append(p1)
                    continue

                if p1.i == p2.i:
                    to_keep.append(p1)
                    continue

                if abs(p1 - p2) <= window:
                    # Whichever peak is chosen, it is also removed from the
                    # `peaks` array, so the next iteration will not include
                    # either of these peaks!

                    # If only one of the peaks is in an order whose wavelength
                    # solution is derived from thorium, take the other one!
                    if p1.i in THORIUM_ORDER_INDICES and p2.i not in LFC_ORDER_INDICES:
                        to_keep.append(peaks.pop(i))
                        continue

                    if p2.i in THORIUM_ORDER_INDICES and p1.i in LFC_ORDER_INDICES:
                        to_keep.append(peaks.pop(i + 1))
                        continue

                    # Otherwise take the peak that's closest to its order centre
                    if p1.d < p2.d:
                        to_keep.append(peaks.pop(i))
                        continue

                    to_keep.append(peaks.pop(i + 1))
                    continue

                # Different orders and outside the window
                to_keep.append(p1)
                continue

            self.filtered_peaks[ol] = to_keep

        return self

    def save_peak_locations(
        self,
        filename: str,
        orderlet: str | list[str] | None,
        locations: str = "wavelength",
        filtered: bool = True,
        weights: bool = False,
    ) -> Spectrum:
        """
        Save the identified and fitted peak locations to a CSV file.

        _extended_summary_

        Args:
            filename (str): The destination file.
            orderlet (str | list[str] | None): The orderlet name(s) selecting the peak
                locations to save. In the intended usage, each output file contains
                peaks corresponding to a single orderlet.
            locations (str, optional): Either "pixel" or "wavelength", selecting which
                peak location should be saved to the file. Defaults to "wavelength".
            filtered (bool, optional): Save the filtered peaks?. Defaults to True. If
                False, all fitted peaks are saved, and there will be peaks appearing
                from adjacent spectral orders.
            weights (bool, optional): Save a weight for each peak? Defaults to False.
                If True, this will save the standard deviation of the peak location
                fit in a second column of the CSV file. If False (default behabviour),
                this column is still present, but all weights are set to `1.0'.

        Raises:
            NotImplementedError if `locations` is not either "wavelength" or "pixel"

        Returns:
            Spectrum (self): The Spectrum object itself, so methods can be chained.
        """

        if locations not in ["wavelength", "pixel"]:
            raise NotImplementedError

        if isinstance(orderlet, str):
            orderlet = [orderlet]

        if orderlet is None:
            orderlet = self.orderlets

        for ol in orderlet:
            if filtered:
                if not self.filtered_peaks[ol]:
                    self.filter_peaks(orderlet=ol)
                peaks_to_use = self.filtered_peaks[ol]

            else:
                peaks_to_use = self.peaks(orderlet=ol)

            logger.info(
                f"{self.pp}Saving {ol:<4} peak {locations} locations to {filename}..."
            )
            p = Path(filename)
            with p.open(mode="w") as f:
                for p in peaks_to_use:
                    if locations == "wavelength":
                        location = f"{p.center_wavelength:f}"
                        weight = f"{p.center_wavelength_stddev:f}" if weights else "1.0"

                    elif locations == "pixel":
                        location = f"{p.center_pixel:f}"
                        weight = f"{p.center_pixel_stddev:f}" if weights else "1.0"

                    f.write(f"{location}\t{weight}\n")

        return self

    def plot_spectrum(
        self,
        orderlet: str,
        ax: plt.Axes | None = None,
        plot_peaks: bool = True,
    ) -> plt.Axes:
        """
        Plot the spectrum (flux as a function of wavelength) for a given orderlet.

        Args:
            orderlet (str): The orderlet name. Options: "SCI1", "SCI2", "SCI3", "CAL",
                "SKY"
            ax (plt.Axes | None, optional): A Matplotlib Pyplot axis object in which
                to plot. Defaults to None, in which case a new Pyplot figure and axis
                are created and used.
            plot_peaks (bool, optional): Defaults to True, in which case the location of
                each peak is indicated with a vertical line.

        Returns:
            plt.Axes: The Pyplot axis object in which the plot was made.
        """

        logger.info(f"{self.pp}Plotting {orderlet} spectrum...")

        assert orderlet in self.orderlets

        if ax is None:
            fig = plt.figure(figsize=(12, 4))
            ax = fig.gca()
            ax.set_title(f"{orderlet} {self.date} {self.timeofday}", size=20)

        if ax.get_xlim() == (0.0, 1.0):
            ax.set_xlim(4400, 8800)
        xlims = ax.get_xlim()

        # plot order by order
        for o in self.orders(orderlet=orderlet):
            mask = (o.wave > xlims[0]) & (o.wave < xlims[1])
            ax.plot(
                o.wave[mask],
                o.spec[mask],
                lw=0.5,
                color=wavelength_to_rgb(o.mean_wave),
                path_effects=stroke(2),
            )
        # ax.plot(0, 0, color="k", lw=1.5)

        if plot_peaks:
            if self.filtered_peaks.get(orderlet, None):
                peaks_to_plot = [
                    p
                    for p in self.filtered_peaks[orderlet]
                    if xlims[0] <= p.wl <= xlims[1]
                ]

            else:
                peaks_to_plot = [
                    p
                    for p in self.peaks(orderlet=orderlet)
                    if xlims[0] <= p.wl <= xlims[1]
                ]

            for p in peaks_to_plot:
                ax.axvline(x=p.wl, color="k", alpha=0.1)

        ax.set_ylim(0.1 * ax.get_ylim()[0], ax.get_ylim()[1])
        ax.set_xlabel("Wavelength [Angstroms]")
        ax.set_ylabel("Flux")

        return self

    def plot_residuals(
        self,
        orderlet: str,
        ax: plt.Axes | None = None,
        plot_peaks: bool = True,
        normalised: bool = True,
    ) -> plt.Axes:
        """
        Plot the residuals between the full spectrum and the peak fits.

        Args:
            orderlet (str): The orderlet name.
            ax (plt.Axes | None, optional): A Matplotlib Pyplot axis object in which
                to plot. Defaults to None, in which case a new Pyplot figure and axis
                are created and used.
            plot_peaks (bool, optional): Defaults to True, in which case the location of
                each peak is indicated with a vertical line.

        Returns:
            plt.Axes: The Pyplot axis object in which the plot was made.
        """

        logger.info(f"{self.pp}Plotting {orderlet} residuals...")

        assert orderlet in self.orderlets

        if ax is None:
            fig = plt.figure(figsize=(12, 4))
            ax = fig.gca()
            ax.set_title(
                f"{orderlet} {self.date} {self.timeofday}\n"
                "Residuals after peak fitting",
                size=20,
            )

        if ax.get_xlim() == (0.0, 1.0):
            ax.set_xlim(4400, 8800)
        xlims = ax.get_xlim()

        # plot order by order
        for o in self.orders(orderlet=orderlet):
            mask = (o.wave > xlims[0]) & (o.wave < xlims[1])
            ax.plot(
                o.wave[mask],
                o.spec_residuals(normalised=normalised)[mask],
                lw=0.5,
                color=wavelength_to_rgb(o.mean_wave),
                path_effects=stroke(2),
            )
        # ax.plot(0, 0, color="k", lw=1.5)

        ax.axhline(y=0, ls="--", color="k", alpha=0.25, zorder=-1)

        if plot_peaks:
            if self.filtered_peaks.get(orderlet, None):
                peaks_to_plot = [
                    p
                    for p in self.filtered_peaks[orderlet]
                    if xlims[0] <= p.wl <= xlims[1]
                ]

            else:
                peaks_to_plot = [
                    p
                    for p in self.peaks(orderlet=orderlet)
                    if xlims[0] <= p.wl <= xlims[1]
                ]

            for p in peaks_to_plot:
                ax.axvline(x=p.wl, color="k", alpha=0.1)

        ax.set_xlabel("Wavelength [Angstroms]")
        ax.set_ylabel("Residuals (data $-$ fit)")

        return self

    def delta_nu_FSR(self, orderlet: str, unit: u.core.Unit = u.GHz) -> ArrayLike:
        """
        Calculate and return the FSR of the etalon spectrum in GHz

        The free spectral range (or FSR) of an etalon is the spacing between adjacent
        peaks in the etalon spectrum. This is typically reported in the frequency
        domain, as it is here, where the default unit is GHz.

        Args:
            orderlet (str): The orderlet name. Options: "SCI1", "SCI2", "SCI3", "CAL",
                "SKY"
            unit (u.core.Unit, optional): The frequency-domain unit (in astropy.units)
                which the FSR values should be reported in. Defaults to u.GHz.

        Returns:
            ArrayLike: The calculated free spectral range values in the specified unit.
        """

        assert orderlet in self.orderlets

        if not self.filtered_peaks[orderlet]:
            logger.info(f"{self.pp}You may want to filter peaks before computing FSR")
            peaks_to_use = self.peaks(orderlet=orderlet)
            # self.filter_peaks(orderlet = ol)

        else:
            peaks_to_use = self.filtered_peaks[orderlet]

        # Get peak wavelengths
        wls = np.array([p.wl for p in peaks_to_use]) * u.angstrom
        # Filter out any NaN values
        nanmask = ~np.isnan(wls)
        wls = wls[nanmask]

        return (constants.c * np.diff(wls) / np.power(wls[:-1], 2)).to(unit).value

    def plot_FSR(
        self,
        orderlet: str,
        ax: plt.Axes | None = None,
        name: str = "",
    ) -> Spectrum:
        """
        Plot the etalon FSR as a function of wavelength

        This method takes the FSR values as computed by the `delta_nu_FSR` method, and
        plots them, also fitting a function to the data.

        Args:
            orderlet (str): The orderlet name. Options: "SCI1", "SCI2", "SCI3", "CAL",
                "SKY"
            ax (plt.Axes | None, optional): A Matplotlib Pyplot axis object in which to
                plot the data. Defaults to None, in which case a new Pyplot figure and
                axis are created and used.
            name (str, optional): A name (title) for the plot. Defaults to an empty
                string.

        Returns:
            Spectrum (self): The Spectrum object itself, so methods can be chained.
        """
        logger.info(f"{self.pp}Plotting {orderlet} Etalon FSR...")

        assert orderlet in self.orderlets

        if ax is None:
            fig = plt.figure(figsize=(12, 4))
            ax = fig.gca()
            ax.set_title(f"{orderlet} {self.date} {self.timeofday}", size=20)

        if ax.get_xlim() == (0.0, 1.0):
            ax.set_xlim(4400, 8800)  # Default xlims
        if ax.get_ylim() == (0.0, 1.0):
            ax.set_ylim(30.15, 30.35)  # Default ylims

        if not self.filtered_peaks[orderlet]:
            logger.info(f"{self.pp}Filtering peaks before computing FSR")
            self.filter_peaks(orderlet=orderlet)

        wls = np.array([p.wl for p in self.filtered_peaks[orderlet]])
        nanmask = ~np.isnan(wls)
        wls = wls[nanmask]

        fsr = self.delta_nu_FSR(orderlet=orderlet, unit=u.GHz)
        estimate_fsr = np.nanmedian(fsr)
        # Remove last wls value to make it the same length as FSR array
        wls = wls[:-1]

        # Coarse removal of >= 1GHz outliers
        mask = np.where(np.abs(fsr - estimate_fsr) <= 1)

        try:
            model = _fit_spline(x=wls[mask], y=fsr[mask], knots=21)
            label = f"{name}Spline fit"
        except ValueError as e:
            logger.error(f"{self.pp}{e}")
            logger.error(f"{self.pp}Spline fit failed. Fitting with polynomial.")

            model = np.poly1d(np.polyfit(wls[mask], fsr[mask], 5))
            label = f"{name}Polynomial fit"

        ax.plot(wls, model(wls), label=label, linestyle="--")

        try:
            # Remove >= 250MHz outliers from model
            mask = np.where(np.abs(fsr - model(wls)) <= 0.25)  # noqa: PLR2004
        except ValueError:  # eg. operands could not be broadcast together
            ...

        ax.scatter(
            wls[mask],
            fsr[mask],
            marker=".",
            alpha=0.2,
            label=f"Data (n = {len(mask[0]):,}/{len(fsr):,})",
        )

        ax.legend(loc="lower right")
        ax.set_xlabel("Wavelength [Angstroms]", size=16)
        ax.set_ylabel("Etalon $\Delta\\nu_{FSR}$ [GHz]", size=16)

        return self

    def plot_peak_fits(self, orderlet: str) -> Spectrum:
        """
        Generate a diagnostic plot showing example peak data and the corresponding
        fitted functions.

        Args:
            orderlet (str): The orderlet name. Options: "SCI1", "SCI2", "SCI3", "CAL",
                "SKY".

        Returns:
            Spectrum (self): The Spectrum object itself, so methods can be chained.
        """
        logger.info(f"{self.pp}Plotting fits of {orderlet} etalon peaks...")

        assert orderlet in self.orderlets

        fig, axs = plt.subplots(6, 3, figsize=(9, 18))

        # Green arm - orders 0, 17, 34
        for i, order_i in enumerate([0, 17, 34]):
            o = self.orders(orderlet=orderlet, i=order_i)
            if not o:
                continue
            o.peaks[0].plot_fit(ax=axs[i][0])
            o.peaks[o.num_peaks // 2].plot_fit(ax=axs[i][1])
            o.peaks[o.num_peaks - 1].plot_fit(ax=axs[i][2])

        # Red arm - orders 35, 51, 66
        for i, order_i in enumerate([35, 51, 66], start=3):
            o = self.orders(orderlet=orderlet, i=order_i)
            if not o:
                continue
            o.peaks[0].plot_fit(ax=axs[i][0])
            o.peaks[o.num_peaks // 2].plot_fit(ax=axs[i][1])
            o.peaks[o.num_peaks - 1].plot_fit(ax=axs[i][2])

        return self

    def fit_parameters(
        self,
        which: str = "all",
        orderlet: str | list[str] | None = None,
    ) -> dict:
        """
        Get the fit parameters for all or a subset of the contained Peak objects.

        Args:
            which (str, optional): Which Peaks to return parameters for. Options: "all",
            "filtered". Defaults to "all".
            orderlet (str | list[str] | None, optional): The orderlet name(s) to
            consider. Defaults to None, in which case all orderlets are considered.

        Returns:
            dict: A dictionary containing the fit parameters for the selected Peaks.
        """

        assert which in ["all", "filtered"]

        if isinstance(orderlet, str):
            assert orderlet in self.orderlets
            orderlet = [orderlet]

        elif isinstance(orderlet, list):
            for ol in orderlet:
                assert ol in self.orderlets

        elif orderlet is None:
            orderlet = self.orderlets

        if which == "all":
            peaks_to_use = [p for ol in orderlet for p in self.peaks(orderlet=ol)]
        elif which == "filtered":
            peaks_to_use = [p for ol in orderlet for p in self.filtered_peaks[ol]]

        return {
            "fit_type": [p.fit_type for p in peaks_to_use],
            "center_wavelength": [p.center_wavelength for p in peaks_to_use],
            "amplitude": [p.amplitude for p in peaks_to_use],
            "sigma": [p.sigma for p in peaks_to_use],
            "boxhalfwidth": [p.boxhalfwidth for p in peaks_to_use],
            "offset": [p.offset for p in peaks_to_use],
            "center_wavelength_stddev": [
                p.center_wavelength_stddev for p in peaks_to_use
            ],
            "amplitude_stddev": [p.amplitude_stddev for p in peaks_to_use],
            "sigma_stddev": [p.sigma_stddev for p in peaks_to_use],
            "boxhalfwidth_stddev": [p.boxhalfwidth_stddev for p in peaks_to_use],
            "offset_stddev": [p.offset_stddev for p in peaks_to_use],
        }

    def data2D(
        self,
        orderlet: str,
        data: str = "spec",  # spec, wave, spec_fit, spec_residuals
    ) -> ArrayLike:
        """
        A method that returns a full raw data array corresponding to a single orderlet.

        `data` is used to select which data is returned. This can be either the raw
        flux values (`spec`), the wavelength values (`wave`), the fitted spectrum
        (`spec_fit`), or the residuals between the fit and the original data
        (`spec_residuals`).

        Args:
            orderlet (str): The orderlet name.
            data (str, optional): The data type to return. Options: "spec", "wave",
                "spec_fit", "spec_residuals". Defaults to "spec".

        Raises:
            ValueError if an invalid data type or orderlet is passed.

        Returns:
            _type_: _description_
        """

        if data not in ["spec", "wave", "spec_fit", "spec_residuals"]:
            raise ValueError("Invalid data type")

        if orderlet not in self.orderlets:
            raise ValueError("Invalid orderlet")

        if data == "spec_residuals":
            # spec_residuals is a function call, not a property
            data = "spec_residuals()"

        return np.array(
            [ast.literal_eval(f"o.{data}") for o in self.orders(orderlet=orderlet)]
        )

    def save_config_file(self) -> None:
        """
        Not yet implemented method to save a configuration file for the Spectrum object.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def __str__(self) -> str:
        out_string = f"Spectrum {self.spec_file} with {len(self.orderlets)} orderlets:"

        for ol in self.orderlets:
            out_string += (
                f"\n - {ol:<4}:"
                f"{len(self.orders(orderlet=ol))} Orders"
                f" and {len(self.peaks(orderlet=ol))} total Peaks"
            )

        return out_string

    def __repr__(self) -> str:
        return (
            "Spectrum("
            f"spec_file={self.spec_file}, "
            f"wls_file={self.wls_file}, "
            f"orderlets_to_load={self.orderlets_to_load})"
        )


def _fit_spline(x: ArrayLike, y: ArrayLike, knots: int = 21) -> BSpline:
    """
    Fit a B-spline to the input data with a given number of knots

    This is a wrapper around the `scipy.interpolate.splrep` function, which simply sets
    up the fit for a given number of spline knots. This method is used to fit the free
    spectral range (FSR) of the etalon peaks.

    Args:
        x (ArrayLike): The x-axis data
        y (ArrayLike): The y-axis data
        knots (int, optional): The number of knots to use. Defaults to 21.

    Returns:
        BSpline: The fitted B-spline function, which can then be evaluated at any
            point.
    """

    # model = UnivariateSpline(x, y, k=5)
    x_new = np.linspace(0, 1, knots + 2)[1:-1]
    q_knots = np.quantile(x, x_new)
    t, c, k = splrep(x, y, t=q_knots, s=1)

    return BSpline(t, c, k)


def _gaussian(
    x: ArrayLike,
    amplitude: float = 1,
    center: float = 0,
    sigma: float = 1,
    offset: float = 0,
) -> ArrayLike:
    """
    A parametrised Gaussian function, optionally used for peak fitting.

    Args:
        x (ArrayLike): The x-axis values
        amplitude (float, optional): The height of the Gaussian function. Defaults to 1.
        center (float, optional): The x-axis offset. Defaults to 0.
        sigma (float, optional): The width of the Gaussian fucntion. Defaults to 1.
        offset (float, optional): A constant y-axis offset. Defaults to 0.

    Returns:
        ArrayLike: The Gaussian function evaluated at the input x values.
    """

    return amplitude * np.exp(-(((x - center) / (2 * sigma)) ** 2)) + offset


def _conv_gauss_tophat(
    x: ArrayLike,
    center: float = 0,
    amp: float = 1,
    sigma: float = 1,
    boxhalfwidth: float = 1,
    offset: float = 0,
    normalize: bool = False,
) -> ArrayLike:
    """
    An analytical description of a convolution of a Gaussian with a finite-width tophat

    This is a piecewise analytical description of a convolution of a gaussian with a
    finite-width tophat function (sometimes called a super-Gaussian). This accounts for
    a finite width of the summed fibre cross-disperion profile (~flat-top) as well as
    the optical image quality (~Gaussian).

    Adapted from a script by Sam Halverson, Ryan Terrien & Arpita Roy
    (originally in a script `fit_erf_to_ccf_simplified.py')

    Changes since that script:
      * The arguments have be re-expressed in terms of familiar Gaussian parameters
      * Added a small value, meaning zero `boxhalfwidth' corresponds to convolution
        with ~a delta function, producing a normal Gaussian as expected if zero
        `boxhalfwidth' is passed
      * Optionally normalise the function so that `amp' corresponds to the highest value

    Args:
        x (ArrayLike): The x-axis values at which to evaluate the function
        center (float, optional): The center x-coordinate. Defaults to 0.
        amp (float, optional): The height of the function. Defaults to 1.
        sigma (float, optional): The width of the Gaussian component(s). Defaults to 1.
        boxhalfwidth (float, optional): The width of the top-hat component.
            Defaults to 1.
        offset (float, optional): A constant y-axis ofset. Defaults to 0.
        normalize (bool, optional): Whether or not to normalise the function and make
            `amp' the maximum value. Defaults to False.

    Returns:
        ArrayLike: The function evaluated at the input x values.
    """

    from scipy.special import erf

    arg1 = (x - center + (boxhalfwidth / 2 + 1e-6)) / (2 * sigma)
    arg2 = (x - center - (boxhalfwidth / 2 + 1e-6)) / (2 * sigma)
    partial = 0.5 * (erf(arg1) - erf(arg2))

    if normalize:
        return amp * (partial / np.nanmax(partial)) + offset

    return amp * partial + offset


def test() -> None:
    """
    A basic test case

    This self-contained script will load in data from master files and locate, fit, and
    filter peaks.
    """

    DATE: str = "20240520"

    etalon_file: Path = (
        MASTERS_DIR
        / f"{DATE}"
        / f"kpf_{DATE}_master_WLS_autocal-etalon-all-morn_L1.fits"
    )
    lfc_wls_file: Path = (
        MASTERS_DIR / f"{DATE}" / f"kpf_{DATE}_master_WLS_autocal-lfc-all-morn_L1.fits"
    )
    thorium_wls_file: Path = (
        MASTERS_DIR / f"{DATE}" / f"kpf_{DATE}_master_WLS_autocal-thar-all-morn_L1.fits"
    )

    s: Spectrum = Spectrum(
        spec_file=etalon_file,
        # wls_file=wls_file,
        auto_load_wls=False,
        orderlets_to_load="SCI2",
        # orders_to_load=TEST_ORDER_INDICES,
    )

    s.load_partial_wls(wls_file=lfc_wls_file, valid_i=LFC_ORDER_INDICES)
    s.load_partial_wls(wls_file=thorium_wls_file, valid_i=THORIUM_ORDER_INDICES)

    s.locate_peaks()

    for o in s.orders():
        print(o)

    s.fit_peaks(fit_type="conv_gauss_tophat", space="pixel")
    print(f"{s.num_successfully_fit_peaks() = }")

    s.filter_peaks()
    print(f"{s.num_filtered_peaks() = }")

    s.save_peak_locations(
        filename=f"/scr/jpember/temp/temp_mask_{DATE}.csv", orderlet="SCI2"
    )

    s.plot_spectrum(orderlet="SCI2", plot_peaks=False)
    plt.savefig(f"/scr/jpember/temp/temp_spectrum_{DATE}.png")

    s.plot_residuals(orderlet="SCI2", plot_peaks=False, normalised=True)
    plt.savefig(f"/scr/jpember/temp/temp_residuals_{DATE}.png")

    s.plot_peak_fits(orderlet="SCI2")
    plt.savefig(f"/scr/jpember/temp/temp_peak_fits_{DATE}.png")


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    test()
