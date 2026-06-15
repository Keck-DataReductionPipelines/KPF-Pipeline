"""
Build a rough wavelength-solution CSV from a legacy KPF *_1D file.

The vNext WLS module seeds its ThAr line search from a *rough* per-order
wavelength solution (`reference/rough_wls_fallback.csv`). That guess must be
accurate to within the line-fit window (~5 px) of the true solution; a simple
linear ramp between the order endpoints is off by the full echelle curvature
(~2 A ~ 140 px mid-order), which collapses the fit back onto the linear guess.

This script recovers an accurate rough WLS by fitting a Legendre polynomial to
the WAVE arrays of a trusted legacy *_1D file (whose solution is itself an
LFC/ThAr product) and writing the per-order coefficients to a CSV the WLS
module can evaluate directly.

Orientation note: legacy WAVE arrays run red->blue along the dispersion axis
(descending), whereas vNext extracts blue->red (ascending, after the
ImageAssembly dispersion-axis flip). We reverse each legacy order before
fitting so the coefficients evaluate on the vNext pixel grid as-is.

Fibers: the SCI1/SCI2/SCI3 solutions differ by <1 px, so a single fiber
(default SCI2, the central science fiber) is a sufficient rough seed for all
SCI fibers. The CSV is therefore fiber-agnostic (one row per chip/order).

One rough WLS is needed per KPF ERA; rerun this against a representative legacy
file from each era to regenerate the CSV.

Usage
-----
    python scripts/build_rough_wls_from_legacy_wls.py \
        --legacy-file tests/testdata/legacy/20240405/KP.20240405.40113.57_legacy_1D.fits
"""
import argparse

import numpy as np
from astropy.io import fits
from numpy.polynomial import legendre

from kpfpipe import DETECTOR, REPO_ROOT

DEFAULT_OUTPUT = f'{REPO_ROOT}/reference/rough_wls_fallback.csv'
DEFAULT_FIBER  = 'SCI2'
DEFAULT_DEGREE = 9   # degree-9 Legendre reconstructs legacy WAVE to ~1e-11 A


def fit_rough_wls(legacy_file, fiber=DEFAULT_FIBER, degree=DEFAULT_DEGREE):
    """
    Fit per-order Legendre coefficients to a legacy file's WAVE arrays.

    Parameters
    ----------
    legacy_file : str
        Path to a legacy KPF *_1D FITS file with per-chip {CHIP}_SCI_WAVE{n}
        image extensions.
    fiber : str
        SCI fiber to read (e.g. 'SCI2'); selects the {CHIP}_SCI_WAVE{n} HDU.
    degree : int
        Legendre polynomial degree (coefficients C0..C{degree}).

    Returns
    -------
    rows : list of (chip, order, wave_min, wave_max, coeffs)
        Per-order blue/red endpoint wavelengths and Legendre coefficients in
        vNext (ascending) orientation. WAVE_MIN/WAVE_MAX are not used by the
        pipeline; they are written for human readability and to verify the
        coefficients resolve to the expected order endpoints.
    worst_resid : float
        Worst-case per-order max reconstruction residual, in Angstrom.
    """
    fiber_num = fiber[-1]   # 'SCI2' -> '2'
    ncol = DETECTOR['ccd']['ncol']
    x = 2 * np.arange(ncol) / (ncol - 1) - 1   # normalized pixel [-1, 1], ascending

    rows = []
    worst_resid = 0.0
    with fits.open(legacy_file) as h:
        for chip, norder in DETECTOR['norder'].items():
            wave = h[f'{chip}_SCI_WAVE{fiber_num}'].data[:, ::-1]   # reverse to ascending
            for o in range(norder):
                wave_o = wave[o].astype(np.float64)
                coeffs = legendre.legfit(x, wave_o, degree)
                resid = np.max(np.abs(wave_o - legendre.legval(x, coeffs)))
                worst_resid = max(worst_resid, resid)
                rows.append((chip, o, wave_o[0], wave_o[-1], coeffs))

    return rows, worst_resid


def write_csv(rows, output, degree=DEFAULT_DEGREE):
    """Write rough WLS to CSV with columns CHIP,ORDER,WAVE_MIN,WAVE_MAX,C0..Cdegree."""
    header = ','.join(['CHIP', 'ORDER', 'WAVE_MIN', 'WAVE_MAX'] + [f'C{i}' for i in range(degree + 1)])
    with open(output, 'w') as f:
        f.write(header + '\n')
        for chip, order, wave_min, wave_max, coeffs in rows:
            f.write(f'{chip},{order},{float(wave_min)!r},{float(wave_max)!r},'
                    + ','.join(repr(float(c)) for c in coeffs) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--legacy-file', required=True,
                        help='Legacy KPF *_1D FITS file to fit.')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help=f'Output CSV path (default: {DEFAULT_OUTPUT}).')
    parser.add_argument('--fiber', default=DEFAULT_FIBER,
                        help=f'SCI fiber to fit (default: {DEFAULT_FIBER}).')
    parser.add_argument('--degree', type=int, default=DEFAULT_DEGREE,
                        help=f'Legendre polynomial degree (default: {DEFAULT_DEGREE}).')
    args = parser.parse_args()

    rows, worst_resid = fit_rough_wls(args.legacy_file, fiber=args.fiber, degree=args.degree)
    write_csv(rows, args.output, degree=args.degree)

    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"  source:  {args.legacy_file} ({args.fiber}, degree {args.degree} Legendre)")
    print(f"  worst per-order reconstruction residual: {worst_resid:.3e} A")


if __name__ == '__main__':
    main()
