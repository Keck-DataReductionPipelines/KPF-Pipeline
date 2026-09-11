#!/usr/bin/env python3
"""Track which ThAr lamp was in use, night by night (``kpfpipe analyze thar``).

KPF's ThAr hollow-cathode lamps are swapped as they wear out, and a dying lamp is a
real failure mode: it dims ~10x before replacement, starving the WLS line fits until
the masters build crashes (``notes/thar_lamp_health.md``). The lamp is named per
frame by the PRIMARY card ``HCLSN`` ("S/N of lamp in use"), so a swap is visible in
the raw headers -- but only to someone looking across nights.

Over an inclusive datecode range this writes, under ``{analysis_dir}/thar/``:

- ``thar_exposures_{START}_{END}.csv`` -- one row per ThAr frame;
- ``thar_lamp_history_{START}_{END}.csv`` -- one row per consecutive stretch of a
  lamp serial in the daily all-fiber frames (a reinstalled lamp gets a new row;
  a blank ``HCLSN`` is reported as ``(none)``).

Frames come from the L0
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
from kpfpipe.utils.logger import setup_batch_logging
from scripts._argparse import analysis_parser, resolve_dir_shortcuts
from scripts._dispatch import _default_science_jobs
from scripts._scan import scan_datecodes, scan_night_to_cache

logger = logging.getLogger(__name__)

# Mini-database columns carried into the report: OBJECT's morn/eve suffix and the
# exposure times bear on how bright a frame should be (notes/thar_lamp_health.md).
_MINI_DB_COLUMNS = ["OBJECT", "EXPTIME", "ELAPSED"]

# PRIMARY cards read per frame: the lamp's serial, which lamp position was selected,
# and how long each had been powered -- a cold lamp is dim without being faulty.
_LAMP_KEYS = ["OCTAGON", "HCLSN", "THDAYON", "THDAYTON", "THAUON", "THAUTON"]

CSV_COLUMNS = ["OBS_ID", "DATECODE"] + _MINI_DB_COLUMNS + _LAMP_KEYS

# The lamp history follows the daily lamp through its all-fiber frames; OBJECT is a
# prefix because later nights suffix it with -morn/-eve/-night.
_HISTORY_OBJECT = "autocal-thar-all"
_HISTORY_OCTAGON = "Th_daily"
HISTORY_COLUMNS = ["HCLSN", "FIRST_DATECODE", "LAST_DATECODE", "N_OBS"]


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


def lamp_history(rows):
    """Consecutive-HCLSN stretches over the daily all-fiber frames in `rows`.

    `rows` must be chronological, as ``scan_lamp_serial_numbers`` returns them; a
    new stretch starts wherever the serial changes from one frame to the next.
    """
    history = []
    for r in rows:
        if not (
            str(r["OBJECT"]).strip().startswith(_HISTORY_OBJECT)
            and r["OCTAGON"] == _HISTORY_OCTAGON
        ):
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
    write_csv(path, rows, CSV_COLUMNS)
    logger.info("%d ThAr frame(s) -> %s", len(rows), path)

    history = lamp_history(rows)
    if not history:
        logger.warning(
            "no OBJECT=%s*, OCTAGON=%s frames: lamp history is empty",
            _HISTORY_OBJECT,
            _HISTORY_OCTAGON,
        )
    path = os.path.join(out_dir, f"thar_lamp_history_{start}_{end}.csv")
    write_csv(path, history, HISTORY_COLUMNS)
    logger.info("done: %d lamp stretch(es) -> %s", len(history), path)


if __name__ == "__main__":
    main()
