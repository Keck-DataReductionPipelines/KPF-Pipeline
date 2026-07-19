#!/usr/bin/env python
"""Standalone QC runner.

Usage:
    python scripts/quality_control/qc.py \
        --input <file> --level L0|L1|L2 [--config <toml>] [--write]
    python scripts/quality_control/qc.py \
        --obs_id <obs_id> --level L1 --config <toml> [--write]

Exit codes:
    0 -- ISGOOD=1 (all checks passed)
    1 -- ISGOOD=0 (one or more checks failed)
    2 -- Input / IO error
"""

import argparse
import os
import sys

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.modules.astro_query import AstroQuery
from kpfpipe.quality_control.checkpoints import CheckpointL0, CheckpointL1, CheckpointL2
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import kpf_filepath

# CheckpointL{n}.run() folds in the paired Diagnostics + QC, so the standalone
# runner drives the whole QC stack through the Checkpoint class alone.
_LEVEL_MAP = {
    "L0": (KPF0, CheckpointL0),
    "L1": (KPF1, CheckpointL1),
    "L2": (KPF2, CheckpointL2),
}


def main():
    parser = argparse.ArgumentParser(description="KPF QC Runner")
    parser.add_argument(
        "--obs_id", type=str, help="Observation ID (e.g. KP.20240405.03637.74)"
    )
    parser.add_argument("--input", type=str, help="Direct path to input FITS file")
    parser.add_argument(
        "--level",
        type=str,
        required=True,
        choices=["L0", "L1", "L2"],
        help="Data level to QC",
    )
    parser.add_argument("--config", type=str, help="Path to TOML config file")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist QC keywords back to the source FITS file",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Resolve input file path (mirrors scripts/quality_control/qlp.py)
    # ------------------------------------------------------------------ #
    if args.input:
        input_file = args.input
    elif args.obs_id and args.config:
        config = ConfigHandler(args.config)
        params = config.get_params(["DATA_DIRS"])
        data_root = params["KPF_DATA_INPUT"]
        input_file = kpf_filepath(args.obs_id, args.level, data_root=data_root)
    else:
        parser.error("Provide either --input or both --obs_id and --config")

    if not os.path.isfile(input_file):
        print(f"Error: file not found: {input_file}", file=sys.stderr)
        sys.exit(2)

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    try:
        DataClass, CheckpointClass = _LEVEL_MAP[args.level]
        data = DataClass.from_fits(input_file)
    except Exception as exc:
        print(f"Error loading {input_file}: {exc}", file=sys.stderr)
        sys.exit(2)

    obs_id = args.obs_id or getattr(data, "obs_id", None) or "unknown"

    # AstroQuery resolves target astrometry into data's CATALOG_RECORD extension,
    # which the L0 pointing-offset diagnostics consume; run it before the checkpoint.
    # Only L0 needs it, and AstroQuery requires a science (IMTYPE 'Object') frame.
    if args.level == "L0":
        try:
            astro_config = ConfigHandler(args.config) if args.config else None
            AstroQuery(data, astro_config).perform()
        except Exception as exc:
            print(f"Error: AstroQuery failed: {exc}", file=sys.stderr)
            sys.exit(2)

    # ------------------------------------------------------------------ #
    # Run QC (via the Checkpoint stage, which folds in Diagnostics + QC and
    # then warns or raises -- e.g. on an unregistered keyword or a fatal flag).
    # A raised checkpoint is a structural failure -> exit 2.
    # ------------------------------------------------------------------ #
    print(f"Running {args.level} QC for {obs_id}")

    try:
        checkpoint = CheckpointClass(data)
        checkpoint.run()
    except Exception as exc:
        print(f"Error: QC/checkpoint failed: {exc}", file=sys.stderr)
        sys.exit(2)

    results = checkpoint.qc_results

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

    # ISGOOD is routed to the QUALITY_CONTROL header (its registry home) by QC.run;
    # once checkpoint.run() has folded in QC the flag is always present, so read it
    # directly -- a missing one is a broken invariant, not a default-to-FAIL case.
    isgood_val = data.headers["QUALITY_CONTROL"]["ISGOOD"]
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
