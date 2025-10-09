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
from polly.kpf import TIMESOFDAY
from polly.log import logger
from polly.parsing import parse_bool, parse_orderlets
from polly.plotting import plot_style

plt.style.use(plot_style)


default_filename = (
    "/data/kpf/masters/"
    + "20240515/kpf_20240515_master_arclamp_autocal-etalon-all-eve_L1.fits"
)


def main(
    filename: str,
    orderlets: str | list[str] | None,
    spectrum_plot: bool,
    fsr_plot: bool,
    fit_plot: bool,
) -> None:
    """
    The main script to run the analysis on a single file.
    """

    if isinstance(orderlets, str):
        orderlets = [orderlets]

    Path(f"{OUTDIR}/masks/").mkdir(parents=True, exist_ok=True)

    date = "".join(fits.getval(filename, "DATE-OBS").split("-"))
    timeofday = fits.getval(filename, "OBJECT").split("-")[-1]
    assert timeofday in TIMESOFDAY

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
            plt.savefig(
                f"{OUTDIR}/FSR_plots/" + f"{date}_{timeofday}_{ol}_etalon_FSR.png"
            )
            plt.close()

        if fit_plot:
            Path(f"{OUTDIR}/fit_plots").mkdir(parents=True, exist_ok=True)
            s.plot_peak_fits(orderlet=ol)
            plt.savefig(
                f"{OUTDIR}/fit_plots/" + f"{date}_{timeofday}_{ol}_etalon_fits.png"
            )
            plt.close()


parser = argparse.ArgumentParser(
    prog="polly run_analysis_single",
    description="""A utility to process KPF etalon data from an  individual file,
                specified by filename. Produces an output mask file with the
                wavelengths of each identified etalon peak, as well as optional
                diagnostic plots.""",
)

parser.add_argument("-f", "--filename", default=default_filename)
parser.add_argument("-o", "--orderlets", type=parse_orderlets, default=None)
parser.add_argument(
    "--outdir",
    type=lambda p: Path(p).absolute(),
    default=Path("/scr/jpember/polly_outputs"),
)
parser.add_argument(
    "--spectrum_plot", "--spectrum-plot", type=parse_bool, default=False
)
parser.add_argument("--fsr_plot", "--fsr-plot", type=parse_bool, default=False)
parser.add_argument("--fit_plot", "--fit-plot", type=parse_bool, default=False)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    args = parser.parse_args()
    OUTDIR = args.outdir

    main(
        filename=args.filename,
        orderlets=args.orderlets,
        spectrum_plot=args.spectrum_plot,
        fsr_plot=args.fsr_plot,
        fit_plot=args.fit_plot,
    )
