#!/usr/bin/env python3
"""Track which ThAr lamp was in use, night by night (``kpfpipe analyze thar``).

KPF's ThAr hollow-cathode lamps are swapped as they wear out, and a dying lamp is a
real failure mode: it dims ~10x before replacement, starving the WLS line fits until
the masters build crashes (``notes/thar_lamp_health.md``). The lamp is named per
frame by the PRIMARY card ``HCLSN`` ("S/N of lamp in use"), so a swap is visible in
the raw headers -- but only to someone looking across nights.

Over an inclusive datecode range this writes, under ``{analysis_dir}/thar/``:

- ``thar_exposures_{START}_{END}.csv`` -- one row per ThAr frame, daily and gold,
  junk included: the mini-database's ThAr rows plus each frame's ``HCLSN``;
- ``thar_lamp_history_{START}_{END}.csv`` -- one row per consecutive stretch of a
  lamp serial over every daily frame, whatever its fibers (a reinstalled lamp gets
  a new row; a blank ``HCLSN`` is reported as ``(none)``).

Frames come from the L0 mini-database (``--cache``), so only the ThAr ones are
reopened for their lamp serial; nothing is reduced and no data product written.
``--date_range`` defaults to the whole archive: first light through tomorrow.

    kpfpipe analyze thar --date_range 20240727 20241022 \
        --input_dir /data/kpf --output_dir /data/kpf-next
"""

import argparse
import csv
import logging
import os
import sys

import pandas as pd
from astropy.io import fits

import kpfpipe
from kpfpipe.utils.kpf import get_datecode, get_obs_id
from kpfpipe.utils.logger import setup_batch_logging
from scripts._argparse import (
    analysis_parser,
    cache_parser,
    pool_parser,
    resolve_dates,
    resolve_dir_shortcuts,
)
from scripts._dispatch import _default_science_jobs
from scripts._scan import datecodes_in_range, scan_datecodes, scan_night_to_cache

logger = logging.getLogger(__name__)

# The frame's identity, its mini-database columns (UTC/HST_TIME aside), then the
# native PRIMARY HCLSN, the serial of the lamp in use.
_EXPOSURE_CSV_COLUMNS = [
    "FILENAME",
    "OBS_ID",
    "DATECODE",
    "DRP_OBJECT",
    "DRP_OBSTYPE",
    "DRP_OBSMODE",
    "SKYSRC",
    "SCISRC",
    "CALSRC",
    "GREEN",
    "RED",
    "CA_HK",
    "EXPTIME",
    "ISJUNK",
    "COMMENTS",
    "HCLSN",
]

# The lamp history follows the daily lamp through all its frames.
_LAMP_HISTORY_CSV_COLUMNS = ["HCLSN", "FIRST_DATECODE", "LAST_DATECODE", "N_OBS"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="kpfpipe analyze thar",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[
            analysis_parser("analyzes every L0 night in it"),
            pool_parser(jobs_help="max concurrent per-night header scans"),
            cache_parser(default="rw"),
        ],
    )
    ap.add_argument(
        "--kpf_data_input",
        "--input_dir",
        dest="kpf_data_input",
        required=True,
        help="root of the raw L0 input tree to read (--input_dir is an alias)",
    )
    args = resolve_dates(ap, ap.parse_args(argv))

    if args.jobs is not None and args.jobs < 1:
        ap.error("--jobs must be >= 1")

    args = resolve_dir_shortcuts(args)
    if not args.analysis_dir or not args.log_dir:
        ap.error("pass --output_dir, or both --analysis_dir and --log_dir")
    return args


def _frame_row(record):
    """One CSV row for mini-database `record`, plus ``HCLSN`` from its header.

    An absent ``HCLSN`` (genuinely blank on some early nights, itself worth
    reporting) or a mini-database NaN (e.g. ``COMMENTS`` on a warning-free frame)
    becomes an empty field.
    """
    path = record["FILENAME"]
    row = {
        "OBS_ID": get_obs_id(path),
        "DATECODE": get_datecode(path),
        "HCLSN": str(fits.getheader(path, ext=0).get("HCLSN", "")).strip(),
    }
    for k in _EXPOSURE_CSV_COLUMNS:
        if k not in row:
            value = record.get(k)
            row[k] = value if pd.notna(value) else ""
    return row


def scan_lamp_serial_numbers(data_input, start, end, jobs, cache="rw"):
    """Lamp rows for every ThAr frame under ``{data_input}/L0`` in [start, end].

    Nights fan out over ``scan_datecodes``' thread pool. Exits loudly when the tree
    is missing, the range covers no nights, or nothing was found -- an empty report
    would read as "no lamp changes" rather than "nothing was looked at".
    """
    nights = datecodes_in_range(data_input, start, end)

    def _scan_night(dc):
        df = scan_night_to_cache(data_input, dc, cache=cache)
        if df is None:  # empty/absent night -- already warned, skip it
            return [], ""
        # Every ThAr frame from either lamp, junk included -- a full accounting of
        # the ThAr frames on disk, not just the masters stack inputs.
        frames = df.loc[df["DRP_OBJECT"].isin(["ThAr_daily", "ThAr_gold"])]
        if frames.empty:
            return [], ": no ThAr frames"
        rows = [_frame_row(f) for _, f in frames.iterrows()]
        serials = sorted({r["HCLSN"] or "(blank)" for r in rows})
        return rows, f": {len(rows)} ThAr frame(s), HCLSN {', '.join(serials)}"

    results = scan_datecodes(nights, jobs, _scan_night, label="scanning ThAr lamps,")
    # Nights complete out of order; an obs_id sorts chronologically.
    rows = sorted((r for night in results for r in night), key=lambda r: r["OBS_ID"])
    if not rows:
        sys.exit(f"error: no ThAr frames under {data_input}/L0 in range {start}..{end}")
    return rows


def lamp_history(rows):
    """Consecutive-HCLSN stretches over every daily ThAr frame in `rows`.

    The serial names the installed lamp whichever fibers a frame lit, so every
    ``ThAr_daily`` frame counts. `rows` must be chronological, as
    ``scan_lamp_serial_numbers`` returns them; a new stretch starts wherever the
    serial changes from one frame to the next.
    """
    history = []
    for r in rows:
        if r["DRP_OBJECT"] != "ThAr_daily":
            continue
        lamp = r["HCLSN"] or "(none)"
        if history and history[-1]["HCLSN"] == lamp:
            history[-1]["LAST_DATECODE"] = r["DATECODE"]
            history[-1]["N_OBS"] += 1
        else:
            history.append(
                {
                    "HCLSN": lamp,
                    "FIRST_DATECODE": r["DATECODE"],
                    "LAST_DATECODE": r["DATECODE"],
                    "N_OBS": 1,
                }
            )
    return history


def write_csv(path, rows, columns):
    """Write `rows` under header `columns` to `path`, creating its directory."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    start, end = args.date_range

    _, log_path = setup_batch_logging(
        args.log_dir, "analyze_thar", args.run_id, level=args.log_level or "INFO"
    )

    logger.info("kpfpipe %s thar lamp analysis starting", kpfpipe.__version__)
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info("date_range: %s..%s", start, end)
    logger.info("analysis log: %s", log_path)

    jobs = args.jobs or _default_science_jobs()
    rows = scan_lamp_serial_numbers(
        args.kpf_data_input, start, end, jobs, cache=args.cache
    )
    out_dir = os.path.join(args.analysis_dir, "thar")
    path = os.path.join(out_dir, f"thar_exposures_{start}_{end}.csv")
    write_csv(path, rows, _EXPOSURE_CSV_COLUMNS)
    logger.info("%d ThAr frame(s) -> %s", len(rows), path)

    history = lamp_history(rows)
    if not history:
        logger.warning("no daily ThAr frames: lamp history is empty")
    path = os.path.join(out_dir, f"thar_lamp_history_{start}_{end}.csv")
    write_csv(path, history, _LAMP_HISTORY_CSV_COLUMNS)
    logger.info("done: %d lamp stretch(es) -> %s", len(history), path)


if __name__ == "__main__":
    main()
