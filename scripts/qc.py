#!/usr/bin/env python
"""
Standalone QC runner.

Usage:
    python scripts/qc.py --input <file> --level L0|L1|L2 [--config <toml>] [--write]
    python scripts/qc.py --obs_id KP.20240405.03637.74 --level L1 --config configs/kpf_drp_science.toml [--write]

Exit codes:
    0 — ISGOOD=1 (all checks passed)
    1 — ISGOOD=0 (one or more checks failed)
    2 — Input / IO error
"""

import argparse
import os
import sys

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.quality_control.qc_booleans import QCL0, QCL1, QCL2
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.pipeline import build_filepath

_LEVEL_MAP = {
    "L0": (KPF0, QCL0),
    "L1": (KPF1, QCL1),
    "L2": (KPF2, QCL2),
}


def main():
    parser = argparse.ArgumentParser(description="KPF QC Runner")
    parser.add_argument("--obs_id", type=str,
                        help="Observation ID (e.g. KP.20240405.03637.74)")
    parser.add_argument("--input", type=str,
                        help="Direct path to input FITS file")
    parser.add_argument("--level", type=str, required=True,
                        choices=["L0", "L1", "L2"],
                        help="Data level to QC")
    parser.add_argument("--config", type=str,
                        help="Path to TOML config file")
    parser.add_argument("--write", action="store_true",
                        help="Persist QC keywords back to the source FITS file")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Resolve input file path (mirrors scripts/qlp.py)
    # ------------------------------------------------------------------ #
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
        sys.exit(2)

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    try:
        DataClass, QCClass = _LEVEL_MAP[args.level]
        data = DataClass.from_fits(input_file)
    except Exception as exc:
        print(f"Error loading {input_file}: {exc}", file=sys.stderr)
        sys.exit(2)

    obs_id = args.obs_id or getattr(data, "obs_id", None) or os.path.basename(input_file)

    # ------------------------------------------------------------------ #
    # Run QC
    # ------------------------------------------------------------------ #
    print(f"Running {args.level} QC for {obs_id}")

    try:
        qc = QCClass(data)
        results = qc.run()
    except RuntimeError as exc:
        print(f"Error: QC raised unexpectedly: {exc}", file=sys.stderr)
        sys.exit(2)

    # ------------------------------------------------------------------ #
    # Print results
    # ------------------------------------------------------------------ #
    pass_count = 0
    total = len(results)
    for keyword, (passed, comment) in results.items():
        marker = "PASS" if passed else "FAIL"
        if passed:
            pass_count += 1
        print(f"  {marker}  {keyword:<8}  {comment}")

    print()

    isgood_raw = data.headers["PRIMARY"].get("ISGOOD", (0,))
    isgood_val = isgood_raw[0] if isinstance(isgood_raw, tuple) else isgood_raw
    isgood_label = "PASS" if isgood_val else "FAIL"
    print(f"ISGOOD: {isgood_label}  ({pass_count} of {total} checks passed)")

    # ------------------------------------------------------------------ #
    # Optional write-back
    # ------------------------------------------------------------------ #
    if args.write:
        try:
            data.to_fits(input_file)
            print(f"QC keywords written to {input_file}")
        except Exception as exc:
            print(f"Error writing to {input_file}: {exc}", file=sys.stderr)
            sys.exit(2)

    sys.exit(0 if isgood_val else 1)


if __name__ == "__main__":
    main()
