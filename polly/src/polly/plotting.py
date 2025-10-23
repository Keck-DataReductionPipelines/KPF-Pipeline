"""
polly

plotting

Contains the main plot_style dictionary, used to define custom defaults for matplotlib
plots, as well as functions used to get an RGB colour value for an input wavelength.
"""

import colorsys
from typing import overload

import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import numpy as np
from pyfonts import load_font

# load Quicksand font
url = (
    "https://github.com/andrew-paglinawan/QuicksandFamily/blob/master/"
    + "fonts/statics/Quicksand-Regular.ttf"
)
quicksand = load_font(font_url=f"{url}?raw=true")
fm.FontManager.addfont(fm.fontManager, path=quicksand._file)

lw = 1.3

plot_style = {
    # Set font to the beautiful Quicksand:
    # https://github.com/andrew-paglinawan/QuicksandFamily
    "font.family": "Quicksand",
    # Also try to use this font for mathtext characters
    "mathtext.default": "regular",
    # Set axis text label sizes
    "axes.labelsize": 12,
    "axes.titlesize": 16,
    # Axis spine line widths
    "axes.linewidth": lw,
    # Set the default axis limits to be round numbers
    # Use plt.rcParams["axes.autolimit_mode"] = "data" to disable
    "axes.autolimit_mode": "round_numbers",
    "axes.unicode_minus": True,
    # x tick properties
    "xtick.top": True,
    "xtick.bottom": True,
    "xtick.major.size": 8,
    "xtick.major.width": lw,
    "xtick.minor.visible": True,
    "xtick.minor.size": 4,
    "xtick.minor.width": lw,
    "xtick.direction": "in",
    "xtick.major.pad": 10,
    # y tick properties
    "ytick.left": True,
    "ytick.right": True,
    "ytick.major.size": 8,
    "ytick.major.width": lw,
    "ytick.minor.visible": True,
    "ytick.minor.size": 4,
    "ytick.minor.width": lw,
    "ytick.direction": "in",
    "ytick.major.pad": 10,
    # Tick text label sizes
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    # Legend properties
    "legend.frameon": False,
    "legend.fontsize": 14,
    "legend.labelspacing": 0.25,
    "legend.handletextpad": 0.25,
    "figure.titlesize": 18,
    # Default figure size and constrined_layout (previously plt.tight_layout())
    "figure.figsize": (14 / 2.54, 10 / 2.54),
    "figure.dpi": 96,
    "figure.constrained_layout.use": True,
    # Default properties for plotting lines, markers, scatterpoints, etc.
    "lines.linewidth": 2,
    "lines.markeredgecolor": "k",
    "lines.markersize": 16,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
    "scatter.marker": ".",
    "scatter.edgecolors": "k",
    "errorbar.capsize": 3,
    "hist.bins": 20,
}


def stroke(thickness: float = 4) -> list:
    """
    A cheap and cheerful wrapper around matplotlib.patheffects objects to provide a
    stroke effect around any line or text object
    """
    return [pe.Stroke(linewidth=thickness, foreground="k"), pe.Normal()]


@overload
def wavelength_to_rgb(
    wavelength: float, gamma: float, fade_factor: float
) -> tuple[float, float, float]: ...


@overload
def wavelength_to_rgb(
    wavelength: list, gamma: float, fade_factor: float
) -> list[tuple[float, float, float]]: ...


def wavelength_to_rgb(
    wavelength: float | list[float],
    gamma: float = 3,
    fade_factor: float = 0.5,
) -> tuple[float, float, float] | list[tuple[float, float, float]]:
    """
    Compute the RGB colour values corresponding to a given wavelength of light.

    Colors are returned for wavelengths in the range from 3800 A to 7500 A, otherwise
    black is returned. Wavelength must be passed in Angstroms.

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    """

    if isinstance(wavelength, list | np.ndarray):
        return [
            wavelength_to_rgb(wavelength=wl, gamma=gamma, fade_factor=fade_factor)
            for wl in wavelength
        ]

    if wavelength >= 3800 and wavelength <= 4400:  # noqa: PLR2004
        attenuation = 0.3 + 0.7 * (wavelength - 3800) / (4400 - 3800)
        r = ((-(wavelength - 4400) / (4400 - 3800)) * attenuation) ** gamma
        g = 0.0
        b = (1.0 * attenuation) ** gamma

    elif wavelength >= 4400 and wavelength <= 4900:  # noqa: PLR2004
        r = 0.0
        g = ((wavelength - 4400) / (4900 - 4400)) ** gamma
        b = 1.0

    elif wavelength >= 4900 and wavelength <= 5100:  # noqa: PLR2004
        r = 0.0
        g = 1.0
        b = (-(wavelength - 5100) / (5100 - 4900)) ** gamma

    elif wavelength >= 5100 and wavelength <= 5800:  # noqa: PLR2004
        r = ((wavelength - 5100) / (5800 - 5100)) ** gamma
        g = 1.0
        b = 0.0

    elif wavelength >= 5800 and wavelength <= 6450:  # noqa: PLR2004
        r = 1.0
        g = (-(wavelength - 6450) / (6450 - 5800)) ** gamma
        b = 0.0

    elif wavelength >= 6450 and wavelength <= 7500:  # noqa: PLR2004
        attenuation = 0.3 + 0.7 * (7500 - wavelength) / (7500 - 6450)
        r = (1.0 * attenuation) ** gamma
        g = 0.0
        b = 0.0

    else:
        attenuation = 0.3
        r = 1.0 * attenuation
        g = 1.0 * attenuation
        b = 1.0 * attenuation

    return fade((r, g, b), fade_factor=fade_factor)


def fade(
    rgb: tuple[float, float, float],
    fade_factor: float = 0.8,
) -> tuple[float, float, float]:
    """
    Applies a saturation fade to a given RGB colour tuple.
    """

    h, s, v = colorsys.rgb_to_hsv(*rgb)

    return colorsys.hsv_to_rgb(h=h, s=fade_factor * s, v=v)


def test_font() -> None:
    """
    A simple test plot to check that the Quicksand font is being correctly loaded.
    """

    import matplotlib.pyplot as plt

    plt.style.use(plot_style)

    fig = plt.figure()
    ax = fig.gca()

    x = np.linspace(0, 10, 100)

    ax.plot(x, np.sin(x), label="y = sin(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Test plot")

    plt.savefig("test_plot.png")


if __name__ == "__main__":
    test_font()
