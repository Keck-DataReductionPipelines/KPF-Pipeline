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
from kpfpipe.qlp.plot_l0 import PlotL0
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.kpf import get_datecode
from kpfpipe.utils.pipeline import build_filepath


def main():
    parser = argparse.ArgumentParser(description="KPF Quicklook Plot Generator")
    parser.add_argument("--obs_id", type=str, help="Observation ID (e.g. KP.20240405.03637.74)")
    parser.add_argument("--input", type=str, help="Direct path to input FITS file")
    parser.add_argument("--level", type=str, required=True, choices=["L0"], help="Data level to plot")
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
        input_file = build_filepath(args.obs_id, "L0", data_root=data_root)
    else:
        parser.error("Provide either --input or both --obs_id and --config")

    if not os.path.isfile(input_file):
        print(f"Error: file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.config:
        config = ConfigHandler(args.config)
        params = config.get_params(["DATA_DIRS"])
        data_root = params.get("KPF_DATA_OUTPUT", "/data/kpf-next/")
        obs_id = args.obs_id or KPF0.from_fits(input_file).obs_id
        datecode = get_datecode(obs_id)
        output_dir = os.path.join(data_root, "QLP", datecode, obs_id, "L0")
    else:
        output_dir = "."

    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    if args.level == "L0":
        l0 = KPF0.from_fits(input_file)
        print(f"Generating L0 QLP for {l0.obs_id}")
        qlp = PlotL0(l0, output_dir=output_dir)
        figs = qlp.all()
        for name, fig in figs.items():
            fig.clear()
        print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
