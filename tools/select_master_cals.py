#!/usr/bin/env python3
"""select_master_cals.py -- pick calibration frames to build masters.

Run this where the L0 tree and kpfpipe are available (typically on the remote
data server). It takes the same obs_id-list input as fetch_l0.sh, infers every
unique datecode in that list, and for each night selects one calibration
sequence (cluster) of each type (bias, dark, flat, thar) and emits its frames.

The header scan and clustering are done by the pipeline's own
``kpfpipe.utils.io.FileHandler`` (``build_mini_database`` +
``build_calibration_stacks``), so this tool always matches the masters recipe
exactly: same OBJECT typing, gap splitting, HST-midnight handling, observer-junk
exclusion, and per-type thresholds. Of the eligible clusters for each type the
largest (earliest on a tie) is chosen, and up to --count of its frames are
emitted.

Per-type settings match recipes/kpf_drp_masters.py (see _TYPE_KWARGS):
bias/flat/thar require 5 frames; darks require 3, merge undersized clusters, and
ignore the HST-midnight split (their sparse sequences legitimately span a
night's HST midnight).

Output is an obs_id list in the exact format of the input (index<TAB>obs_id,
one per line), ready to feed straight back into fetch_l0.sh. A human-readable
selection summary is written to stderr.

Usage:
    python3 select_master_cals.py <science_obs_ids.txt> -o <cal_obs_ids.txt>
    python3 select_master_cals.py obs_ids.txt --root /data/kpf/L0 -o cals.txt
    # then feed the output to fetch_l0.sh to retrieve the calibration frames.
"""

import argparse
import os
import sys

from kpfpipe.utils.io import FileHandler
from kpfpipe.utils.kpf_utils import get_datecode, get_obs_id, get_timestamp, is_obs_id

# Per-type kwargs for build_calibration_stacks, matching recipes/kpf_drp_masters.py.
# Darks are sparse long exposures whose sequences straddle HST midnight, so they
# drop the threshold to 3 and group the whole night into one stack (obs_night);
# bias/thar use the default time_of_day (one stack per morn/eve session).
_TYPE_KWARGS = {
    "bias": {},
    "dark": {"min_file_count": 3, "groupby": "obs_night"},
    "flat": {},
    "thar": {},
}


def datecodes_from_list(path):
    """Unique datecodes (sorted) from the obs_ids in an input list file."""
    with open(path) as fh:
        codes = {get_datecode(t) for t in fh.read().split() if is_obs_id(t)}
    if not codes:
        sys.exit(f"error: no obs_ids found in {path}")
    return sorted(codes)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("obs_id_list", help="input list of science obs_ids (as 10700.txt)")
    ap.add_argument(
        "-o",
        "--out",
        default="master_calibs.txt",
        help="output obs_id list (default: master_calibs.txt; '-' = stdout)",
    )
    ap.add_argument(
        "--root",
        default="/data/kpf/L0",
        help="L0 root holding {datecode}/ subdirs (default: "
        "/data/kpf/L0, the standard KPF data-server layout)",
    )
    ap.add_argument(
        "--count",
        type=int,
        default=5,
        help="max frames to take per type per night (fewer if the "
        "chosen cluster is smaller, e.g. sparse darks)",
    )
    ap.add_argument(
        "--gap",
        type=int,
        default=7200,
        help="cluster-splitting gap in seconds (default 2h)",
    )
    args = ap.parse_args()

    # FileHandler keys the L0 tree off KPF_DATA_INPUT and appends L0/{datecode}
    # itself, so --root (the L0 dir) maps to KPF_DATA_INPUT = its parent.
    file_handler = FileHandler({"KPF_DATA_INPUT": os.path.dirname(args.root)})

    datecodes = datecodes_from_list(args.obs_id_list)
    print(
        f"# {len(datecodes)} unique datecode(s) from {args.obs_id_list}: "
        f"{', '.join(datecodes)}\n",
        file=sys.stderr,
    )

    all_ids = []
    for dc in datecodes:
        try:
            db = file_handler.build_mini_database(dc, cache="r")
        except ValueError as e:
            print(f"{dc}: {e} -- skipped", file=sys.stderr)
            continue
        exptime_by_fn = dict(zip(db["FILENAME"], db["EXPTIME"], strict=True))
        print(f"{dc}: scanned {len(db)} L0 frames", file=sys.stderr)

        for cal_type in ("bias", "dark", "flat", "thar"):
            try:
                clusters = file_handler.build_calibration_stacks(
                    cal_type, cluster_gap_seconds=args.gap, **_TYPE_KWARGS[cal_type]
                )
            except ValueError as e:
                print(f"    {cal_type:5s}: {e} -- skipped", file=sys.stderr)
                continue

            # Largest eligible cluster; max returns the first on a tie and the
            # clusters are chronological, so ties resolve to the earliest.
            best = max(clusters, key=len)
            chosen = best[: args.count]
            obs_ids = [get_obs_id(fn) for fn in chosen]
            exps = sorted(
                {exptime_by_fn[fn] for fn in chosen if exptime_by_fn[fn] is not None}
            )
            t0, t1 = get_timestamp(chosen[0]), get_timestamp(chosen[-1])
            print(
                f"    {cal_type:5s}: {len(clusters)} eligible cluster(s), "
                f"chose size {len(best)} -> {len(obs_ids)} frames "
                f"[{t0} .. {t1}] EXPTIME={exps}",
                file=sys.stderr,
            )
            all_ids.extend(obs_ids)

    lines = [f"{i}\t{oid}" for i, oid in enumerate(all_ids, start=1)]
    out_text = "\n".join(lines) + "\n" if lines else ""

    if args.out == "-":
        sys.stdout.write(out_text)
    else:
        with open(args.out, "w") as fh:
            fh.write(out_text)
        print(
            f"\n# wrote {len(all_ids)} calibration obs_ids to {args.out}",
            file=sys.stderr,
        )

    if not all_ids:
        sys.exit("error: no calibration frames selected")


if __name__ == "__main__":
    main()
