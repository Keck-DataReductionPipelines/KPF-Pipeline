#!/usr/bin/env python

# /// script
# dependencies = [
#     "polly-kpf>=0.2.0",
# ]
# [tool.uv.sources]
# polly-kpf = { path = "../", editable = true }
# ///

"""
Single file analysis command-line utility. Can be passed a filename as argument.
"""

import argparse
import logging
from pathlib import Path

from astropy.io import fits
from matplotlib import pyplot as plt

from polly.etalonanalysis import Spectrum
from polly.log import logger
from polly.parsing import parse_bool, parse_orderlets
from polly.plotting import plot_style

plt.style.use(plot_style)


def main(
    filename: str,
    orderlets: str | list[str] | None,
    spectrum_plot: bool,
    fsr_plot: bool,
    fit_plot: bool,
) -> None:
    """
    Main function for the command-line utility. Processes a single file and saves the
    output to a CSV file in the output directory.
    """

    if isinstance(orderlets, str):
        orderlets = [orderlets]

    Path(f"{OUTDIR}/masks/").mkdir(parents=True, exist_ok=True)

    date = "".join(fits.getval(filename, "DATE-OBS").split("-"))
    timeofday = fits.getval(filename, "OBJECT").split("-")[-1]

    pp = f"{f'[{date} {timeofday:>5}]':<20}"  # Print/logging line prefix

    s = Spectrum(
        spec_file=filename,
        wls_file=None,  # It will try to find the corresponding WLS file
        orderlets_to_load=orderlets,
        pp=pp,
    )
    s.locate_peaks()
    s.fit_peaks()
    s.filter_peaks()

    for ol in s.orderlets:
        try:
            s.save_peak_locations(
                filename=f"{OUTDIR}/masks/"
                + f"{date}_{timeofday}_{ol}_etalon_wavelengths.csv",
                orderlet=ol,
            )
        except Exception as e:
            print(f"{pp}{e}")
            continue

        if spectrum_plot:
            Path(f"{OUTDIR}/spectrum_plots").mkdir(parents=True, exist_ok=True)
            s.plot_spectrum(orderlet=ol, plot_peaks=False)
            plt.savefig(f"{OUTDIR}/spectrum_plots/{date}_{timeofday}_{ol}_spectrum.png")
            plt.close()

        if fsr_plot:
            Path(f"{OUTDIR}/FSR_plots").mkdir(parents=True, exist_ok=True)
            s.plot_FSR(orderlet=ol)
            plt.savefig(f"{OUTDIR}/FSR_plots/{date}_{timeofday}_{ol}_etalon_FSR.png")
            plt.close()

        if fit_plot:
            Path(f"{OUTDIR}/fit_plots").mkdir(parents=True, exist_ok=True)
            s.plot_peak_fits(orderlet=ol)
            plt.savefig(f"{OUTDIR}/fit_plots/{date}_{timeofday}_{ol}_etalon_fits.png")
            plt.close()


parser = argparse.ArgumentParser(
    prog="polly run_analysis_single",
    description="""A utility to process KPF etalon data from an individual file,
                specified by filename. Produces an output mask file with the wavelengths
                of each identified etalon peak, as well as optional diagnostic plots""",
)

parser.add_argument("-f", "--filename", type=str)
parser.add_argument("--outdir", default="/scr/jpember/polly_outputs")
parser.add_argument("--orderlets", type=parse_orderlets, default=None)
parser.add_argument("--spectrum_plot", type=parse_bool, default=False)
parser.add_argument("--fsr_plot", type=parse_bool, default=True)
parser.add_argument("--fit_plot", type=parse_bool, default=True)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    args = parser.parse_args()
    OUTDIR = args.outdir

    test_filename = (
        "/data/kpf/masters/20240515/"
        + "kpf_20240515_master_WLS_autocal-etalon-all-eve_L1.fits"
    )

    main(
        filename=test_filename,
        orderlets="SCI2",
        spectrum_plot=False,
        fsr_plot=True,
        fit_plot=True,
    )
