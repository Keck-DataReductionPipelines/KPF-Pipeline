#!/usr/bin/env python
"""
Standalone QLP (quicklook plot) generator.

Usage:
    python scripts/qlp.py --obs_id KP.20240405.03637.74 --level L0 --config configs/kpf_drp_science.toml

    # Or specify paths directly:
    python scripts/qlp.py --input /data/kpf/L0/20240405/KP.20240405.03637.74.fits --level L0 --output_dir ./qlp_output
"""

import argparse
import os
import sys

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.quality_control.quicklook.level0 import PlotL0
from kpfpipe.quality_control.quicklook.level1 import PlotL1
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import build_filepath
from kpfpipe.utils.kpf import get_datecode


def main():
    parser = argparse.ArgumentParser(description="KPF Quicklook Plot Generator")
    parser.add_argument(
        "--obs_id", type=str, help="Observation ID (e.g. KP.20240405.03637.74)"
    )
    parser.add_argument("--input", type=str, help="Direct path to input FITS file")
    parser.add_argument(
        "--level",
        type=str,
        required=True,
        choices=["L0", "L1"],
        help="Data level to plot",
    )
    parser.add_argument("--config", type=str, help="Path to TOML config file")
    parser.add_argument("--output_dir", type=str, help="Output directory for plots")
    args = parser.parse_args()

    # Determine input file path
    if args.input:
        input_file = args.input
    elif args.obs_id and args.config:
        config = ConfigHandler(args.config)
        params = config.get_params(["DATA_DIRS"])
        data_root = params.get("KPF_DATA_INPUT", "/data/kpf/")
        input_file = build_filepath(args.obs_id, args.level, data_root=data_root)
    else:
        parser.error("Provide either --input or both --obs_id and --config")

    if not os.path.isfile(input_file):
        print(f"Error: file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Load data and determine obs_id/datecode
    if args.level == "L0":
        data = KPF0.from_fits(input_file)
    else:  # L1
        data = KPF1.from_fits(input_file)
    obs_id = args.obs_id or data.obs_id

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.config:
        if not obs_id:
            parser.error(
                f"Could not determine obs_id from {input_file}. "
                "Pass --obs_id explicitly or --output_dir to bypass obs_id-based path building."
            )
        config = ConfigHandler(args.config)
        params = config.get_params(["DATA_DIRS"])
        data_root = params.get("KPF_DATA_OUTPUT", "/data/kpf-next/")
        datecode = get_datecode(obs_id)
        output_dir = os.path.join(data_root, "QLP", datecode, obs_id, args.level)
    else:
        output_dir = "."

    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    print(f"Generating {args.level} QLP for {obs_id}")
    if args.level == "L0":
        qlp = PlotL0(data, output_dir=output_dir)
    else:
        qlp = PlotL1(data, output_dir=output_dir)
    figs = qlp.all()
    for _, fig in figs.items():
        fig.clear()
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
