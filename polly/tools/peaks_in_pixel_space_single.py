#!/usr/bin/env python

# /// script
# dependencies = [
#     "polly-kpf>=0.2.0",
# ]
# [tool.uv.sources]
# polly-kpf = { path = "../", editable = true }
# ///

"""
Single file analysis command-line utility that outputs a CSV with the pixel location of
identified and fit peaks.

Takes a single filename as argument.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

from polly.etalonanalysis import Spectrum
from polly.kpf import TIMESOFDAY
from polly.log import logger
from polly.parsing import parse_bool, parse_orderlets
from polly.plotting import plot_style

plt.style.use(plot_style)


DEFAULT_FILENAME = (
    "/data/kpf/masters/"
    + "20240515/kpf_20240515_master_arclamp_autocal-etalon-all-eve_L1.fits"
)


def main(
    filename: str,
    orderlets: str | list[str] | None,
    fit_plot: bool,
) -> None:
    """
    Main function for the peaks_in_pixel_space_single command-line utility.
    """

    if isinstance(orderlets, str):
        orderlets = [orderlets]

    (OUTDIR / "masks").mkdir(parents=True, exist_ok=True)

    date = "".join(fits.getval(filename, "DATE-OBS").split("-"))
    timeofday = fits.getval(filename, "OBJECT").split("-")[-1]
    assert timeofday in TIMESOFDAY

    pp = f"{f'[{date} {timeofday:>5}]':<20}"  # Print/logging line prefix

    s = Spectrum(
        spec_file=filename,
        wls_file=None,  # It will try to find the corresponding WLS file(s)
        auto_load_wls=True,
        orderlets_to_load=orderlets,
        pp=pp,
    )

    s.locate_peaks()
    s.fit_peaks(space="pixel")

    # Now you can access peak locations with the following:
    center_pixels = [p.center_pixel for p in s.peaks()]
    pixel_std_devs = [p.center_pixel_stddev for p in s.peaks()]

    order_is, center_pixels, pixel_std_devs = np.transpose(
        [(p.i, p.center_pixel, p.center_pixel_stddev) for p in s.peaks()]
    )

    # for i, pix, dpix in zip(order_is, center_pixels, pixel_std_devs, strict=True):
    #     print(f"{pp}Order i={i:<3.0f}| {pix:.3f} +/- {dpix:.4f}")

    # And save the output to a CSV with built-in methods like so:
    for ol in s.orderlets:
        try:
            s.save_peak_locations(
                filename=OUTDIR
                / "masks"
                / f"{date}_{timeofday}_{ol}_etalon_wavelengths.csv",
                orderlet=ol,
                locations="pixel",
                filtered=False,
            )
        except Exception as e:
            print(f"{pp}{e}")
            continue

        if fit_plot:
            (OUTDIR / "fit_plots").mkdir(parents=True, exist_ok=True)
            s.plot_peak_fits(orderlet=ol)
            plt.savefig(
                OUTDIR / "fit_plots" / f"{date}_{timeofday}_{ol}_etalon_fits.png"
            )
            plt.close()


parser = argparse.ArgumentParser(
    prog="polly peaks_in_pixel_space_single",
    description="""A utility to process KPF etalon data from an individual file,
                specified by filename. Produces an output mask file with the pixel
                position of each identified etalon peak, as well as optional diagnostic
                plots.""",
)

parser.add_argument("-f", "--filename", default=DEFAULT_FILENAME)

parser.add_argument("-o", "--orderlets", type=parse_orderlets, default="SCI2")
parser.add_argument("--fit_plot", type=parse_bool, default=False)

parser.add_argument(
    "--outdir", type=lambda p: Path(p).absolute(), default="/scr/jpember/temp"
)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    args = parser.parse_args()
    OUTDIR: Path = args.outdir

    main(
        filename=args.filename,
        orderlets=args.orderlets,
        fit_plot=args.fit_plot,
    )
