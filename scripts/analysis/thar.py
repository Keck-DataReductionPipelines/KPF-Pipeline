#!/usr/bin/env python3
"""Track which ThAr lamp was in use, night by night (``kpfpipe analyze thar``).

KPF's ThAr hollow-cathode lamps are swapped as they wear out, and a dying lamp is a
real failure mode: it dims ~10x before replacement, starving the WLS line fits until
the masters build crashes (``notes/thar_lamp_health.md``). The lamp is named per
frame by the PRIMARY card ``HCLSN`` ("S/N of lamp in use"), so a swap is visible in
the raw headers -- but only to someone looking across nights.

Over an inclusive datecode range this writes one row per ThAr frame to
``{analysis_dir}/thar/thar_lamps_{START}_{END}.csv``. Frames come from the L0
mini-database (``--cache``), so only the ThAr ones are reopened for their lamp
cards; nothing is reduced and no data product written.

    kpfpipe analyze thar --date_range 20240727 20241022 \
        --input_dir /data/kpf --output_dir /data/kpf-next
"""

import argparse
import csv
import logging
import os
import sys

from astropy.io import fits

import kpfpipe
from kpfpipe.utils.io import datecode_dirs_in_range
from kpfpipe.utils.kpf import get_datecode, get_obs_id, is_datecode
from kpfpipe.utils.logger import build_run_log_dir, setup_batch_logging
from scripts.processing._argparse import analysis_parser, resolve_dir_shortcuts
from scripts.processing._dispatch import _default_science_jobs
from scripts.processing._scan import scan_datecodes, scan_night_to_cache

logger = logging.getLogger(__name__)

# Mini-database columns carried into the report: OBJECT's morn/eve suffix and the
# exposure times bear on how bright a frame should be (notes/thar_lamp_health.md).
_MINI_DB_COLUMNS = ["OBJECT", "EXPTIME", "ELAPSED"]

# PRIMARY cards read per frame: the lamp's serial, which lamp position was selected,
# and how long each had been powered -- a cold lamp is dim without being faulty.
_LAMP_KEYS = ["OCTAGON", "HCLSN", "THDAYON", "THDAYTON", "THAUON", "THAUTON"]

CSV_COLUMNS = ["OBS_ID", "DATECODE"] + _MINI_DB_COLUMNS + _LAMP_KEYS


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="kpfpipe analyze thar",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[analysis_parser()],
    )
    args = ap.parse_args(argv)

    start, end = args.date_range
    for dc in (start, end):
        if not is_datecode(dc):
            ap.error(f"--date_range value is not a valid datecode: {dc!r}")
    if start > end:
        ap.error(f"--date_range START must be <= END (got {start} > {end})")
    if args.jobs is not None and args.jobs < 1:
        ap.error("--jobs must be >= 1")

    args = resolve_dir_shortcuts(args)
    if not args.analysis_dir or not args.log_dir:
        ap.error("pass --output_dir, or both --analysis_dir and --log_dir")
    return args


def _frame_row(record):
    """One CSV row for the frame in mini-database `record`, reopening its header.

    A card the header lacks becomes an empty field: ``HCLSN`` is genuinely blank on
    some early nights, which is itself worth reporting.
    """
    path = record["FILENAME"]
    header = fits.getheader(path, ext=0)
    row = {"OBS_ID": get_obs_id(path), "DATECODE": get_datecode(path)}
    row.update({k: record[k] for k in _MINI_DB_COLUMNS})
    row.update({k: str(header.get(k, "")).strip() for k in _LAMP_KEYS})
    return row


def scan_lamp_serial_numbers(data_input, start, end, jobs, cache="rw"):
    """Lamp rows for every ThAr frame under ``{data_input}/L0`` in [start, end].

    Nights fan out over ``scan_datecodes``' thread pool. Exits loudly when the tree
    is missing, the range covers no nights, or nothing was found -- an empty report
    would read as "no lamp changes" rather than "nothing was looked at".
    """
    l0_root = os.path.join(data_input, "L0")
    if not os.path.isdir(l0_root):
        sys.exit(f"error: L0 input directory not found: {l0_root}")

    nights = datecode_dirs_in_range(l0_root, start, end)
    if not nights:
        sys.exit(f"error: no datecode dirs under {l0_root} in range {start}..{end}")

    def _scan_night(dc):
        df = scan_night_to_cache(data_input, dc, cache=cache)
        if df is None:  # empty/absent night -- already warned, skip it
            return [], ""
        # Every 'autocal-thar*' frame, not just the masters stack inputs: the
        # fiber-specific cals light the same lamp, and early nights have only those.
        is_thar = df["OBJECT"].astype(str).str.strip().str.startswith("autocal-thar")
        frames = df.loc[is_thar & ~df["ISJUNK"].astype(bool)]
        if frames.empty:
            return [], ": no ThAr frames"
        rows = [_frame_row(f) for _, f in frames.iterrows()]
        serials = sorted({r["HCLSN"] or "(blank)" for r in rows})
        return rows, f": {len(rows)} ThAr frame(s), HCLSN {', '.join(serials)}"

    results = scan_datecodes(nights, jobs, _scan_night, label="scanning ThAr lamps,")
    # Nights complete out of order; an obs_id sorts chronologically.
    rows = sorted((r for night in results for r in night), key=lambda r: r["OBS_ID"])
    if not rows:
        sys.exit(f"error: no ThAr frames under {l0_root} in range {start}..{end}")
    return rows


def write_csv(rows, out_dir, start, end):
    """Write `rows` to ``{out_dir}/thar_lamps_{start}_{end}.csv``; return the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"thar_lamps_{start}_{end}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv=None):
    args = parse_args(argv)
    start, end = args.date_range

    log_dir = args.log_run_dir or build_run_log_dir(args.log_dir, "analyze_thar")
    log_path = setup_batch_logging(
        log_dir, "analyze_thar", level=args.log_level or "INFO"
    )

    logger.info("kpfpipe %s thar lamp analysis starting", kpfpipe.__version__)
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info("date_range: %s..%s", start, end)
    logger.info("analysis log: %s", log_path)

    jobs = args.jobs or _default_science_jobs()
    rows = scan_lamp_serial_numbers(
        args.kpf_data_input, start, end, jobs, cache=args.cache
    )
    path = write_csv(rows, os.path.join(args.analysis_dir, "thar"), start, end)
    logger.info("done: %d ThAr frame(s) -> %s", len(rows), path)


if __name__ == "__main__":
    main()
