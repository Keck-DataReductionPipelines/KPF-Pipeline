#!/usr/bin/env python3
"""
Script name: slowtouch.py

This script 'touches' a list of KPF L0 files with names like
KP.YYYYMMDD.12345.67.fits to trigger reprocessing in the KPF DRP.

Ways to provide filenames (any combination works):
  1) As positional arguments on the command line.
  2) With -f <csv>, reading the first column (quotes removed; header 'observation_id' skipped).
  3) With -d <dir>, adding every file name in that directory.

Date range mode (Docker only):
  If you pass exactly two positional arguments that are valid datecodes
  (YYYYMMDD) and you do NOT use -f/--csv or -d/--dir, the script switches to
  'date range mode'. This mode is available **only when running inside a Docker
  container**. In date range mode it:
    • Validates the two YYYYMMDD values and sorts them into start_date/end_date.
    • Uses the time series database (TSDB) to query for ObsIDs in that date window,
      optionally filtering by:
        --only-object <name>   (exact match or DB-implemented rule for OBJECT)
        --only-source <name>   (e.g., Star, Etalon, Dark, etc.)
    • Touches each matched ObsID's L0 file under the resolved L0 base path.

  If you attempt date range mode outside Docker, the script prints an error and exits.

Options (all optional):
  -f, --csv <filename>       CSV with L0 filenames in the first column (can be used multiple times)
  -d, --dir <directory>      Directory to scan for filenames (can be used multiple times)
  -p, --path <path>          L0 base path (default: automatic)
                             automatic -> /data/L0 when in Docker, /data/kpf/L0 otherwise
  -s, --sleep <seconds>      Sleep interval between touches (default: 0.2)
  -e, --echo                 Echo touch commands instead of executing
      --only-object <name>   (Date range mode) filter TSDB rows to this OBJECT (e.g., autocal-bias)
      --only-source <name>   (Date range mode) filter TSDB rows to this SOURCE (e.g., Star, Etalon, Dark)

Examples:
  slowtouch.py KP.20230623.12345.67.fits KP.20230623.12345.68.fits # touch two files (matched to L0 dir by ObsID)
  slowtouch.py -f filenames.csv                                    # touch files in first col of csv
  slowtouch.py -d /path/to/directory                               # touch files in dir (matched to L0 dir by ObsID)
  slowtouch.py KP.20230623.12345.67.fits -p /new/L0/path -s 0.5    # specify L0 path and sleep interval 
  slowtouch.py KP.20230623.12345.67.fits -e                        # echo touch commands
  slowtouch.py 20241001 20241015 --only-object autocal-dark        # touch matching object name in date range
  slowtouch.py 20241001 20241015 --only-source Star                # touch matching source type in date range
  slowtouch.py 20241001 20241015 --only-source Etalon              # touch matching source type in date range
"""

import argparse
import csv
import re
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List

DOCKER_L0_DEFAULT = "/data/L0"
HOST_L0_DEFAULT   = "/data/kpf/L0"


def parse_args(argv=None):
    description = __doc__ or "Touch KPF L0 files (KP.YYYYMMDD.xxxxx.xx.fits) to trigger reprocessing."
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("filenames", nargs="*", help="KPF L0 filenames (e.g., KP.20230623.12345.67.fits) OR two YYYYMMDD dates")
    p.add_argument("-f", "--csv", action="append", default=[], help="CSV file with filenames in first column")
    p.add_argument("-d", "--dir", dest="dirs", action="append", default=[], help="Directory containing filenames")
    p.add_argument("-p", "--path", dest="l0_path", default='automatic',
                   help="Base L0 path. 'automatic' -> /data/L0 in Docker, /data/kpf/L0 otherwise")
    p.add_argument("-s", "--sleep", dest="sleep_interval", type=float, default=0.2,
                   help="Sleep interval between touches (seconds; default: 0.2)")
    p.add_argument("-e", "--echo", dest="echo", action="store_true", help="Echo touch commands instead of executing")
    p.add_argument("--only-object", "--only_object", dest="only_object", default=None,
                   help="(Date range mode) filter DB query to OBJECT (exact match or DB-implemented rule)")
    p.add_argument("--only-source", "--only_source", dest="only_source", default=None,
                   help="(Date range mode) filter DB query to SOURCE (e.g., Star, ThAr, LFC, etc.)")
    return p.parse_args(argv)


def is_running_in_docker():
    # Method 1: Check /.dockerenv presence
    if os.path.exists('/.dockerenv'):
        return True
    # Method 2: Check for Docker or container-specific strings in /proc/self/cgroup
    try:
        with open('/proc/self/cgroup', 'rt') as f:
            cgroup_content = f.read()
            if 'docker' in cgroup_content or 'kubepods' in cgroup_content or 'containerd' in cgroup_content:
                return True
    except Exception:
        pass
    return False


def resolve_l0_base(args) -> Path:
    """Return the L0 base path respecting --path and auto-detecting Docker."""
    if args.l0_path and args.l0_path != "automatic":
        return Path(args.l0_path)
    auto_base = DOCKER_L0_DEFAULT if is_running_in_docker() else HOST_L0_DEFAULT
    return Path(auto_base)


def is_valid_datecode(s: str) -> bool:
    """Return True if s is YYYYMMDD and represents a real calendar date."""
    if not re.fullmatch(r"\d{8}", s):
        return False
    try:
        datetime.strptime(s, "%Y%m%d")
        return True
    except ValueError:
        return False


def read_csv_first_column(csv_path: Path):
    """Yield cleaned filename strings from the first column of a CSV."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"File {csv_path} does not exist.")
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            cell = row[0].strip().replace('"', "")
            if cell.lower().startswith("observation_id"):
                continue
            if cell:
                yield cell


def read_directory_entries(dir_path: Path):
    """Yield bare filenames from a directory listing (not full paths)."""
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory {dir_path} does not exist.")
    for p in sorted(dir_path.iterdir()):
        if p.is_file():
            yield p.name


def normalize_ObsID(name: str) -> str:
    """Remove quotes and ensure it's an ObsID; strip trailing '.fits' for compatibility."""
    text = name.replace('"', "")
    if text.lower().startswith("observation_id"):
        return ""
    text = Path(text).name
    if text.endswith(".fits"):
        text = text[:-5]
    return text


def build_ObsIDs(args) -> List[str]:
    """Collect filenames from positional args, CSVs, and directories."""
    names: List[str] = []

    # From CSV(s)
    for csv_path in args.csv:
        for entry in read_csv_first_column(Path(csv_path)):
            ObsID = normalize_ObsID(entry)
            if ObsID:
                names.append(ObsID)

    # From directory/ies
    for d in args.dirs:
        for entry in read_directory_entries(Path(d)):
            ObsID = normalize_ObsID(entry)
            if ObsID:
                names.append(ObsID)

    # From positional arguments
    for entry in args.filenames:
        ObsID = normalize_ObsID(entry)
        if ObsID:
            names.append(ObsID)

    return names


def touch_file(fullpath: Path, echo: bool, sleep_interval: float):
    if echo:
        print(f"touch {fullpath}")
        return
    print(f"{datetime.now().strftime('%H:%M:%S')} touching file: {fullpath}")
    fullpath.parent.mkdir(parents=True, exist_ok=True)  # ensure date dir exists
    fullpath.touch(exist_ok=True)
    time.sleep(sleep_interval)


def main(argv=None):
    args = parse_args(argv)
    base = resolve_l0_base(args)

    # --- Date range mode: exactly two YYYYMMDD positional args (real dates), and no -f/-d used.
    if not args.csv and not args.dirs and len(args.filenames) == 2 \
       and all(is_valid_datecode(x) for x in args.filenames):

        if not is_running_in_docker():
            print("Error: date range mode is only available inside a Docker container.", file=sys.stderr)
            return 2

        start_date, end_date = sorted(args.filenames)
        try:
            # Query TSDB for matching ObsIDs
            from database.modules.utils.tsdb import TSDB
            myDB = TSDB(backend='psql')
            cols = ['ObsID', 'Source', 'Object']
            df = myDB.dataframe_from_db(
                columns=cols,
                only_object=args.only_object,
                only_source=args.only_source,
                start_date=start_date,
                end_date=end_date
            )
            ObsIDs = df['ObsID'].astype(str).tolist()
            print(f"# matched ObsIDs: {len(ObsIDs)}")
            for ObsID in ObsIDs:
                datecode = ObsID[3:11] if len(ObsID) >= 11 else ""
                fullpath = base / datecode / f"{ObsID}.fits"
                touch_file(fullpath, args.echo, args.sleep_interval)
        except Exception as e:
            print(f"Error querying DB: {e}", file=sys.stderr)
            return 3

        return 0

    # --- Normal touch mode
    ObsIDs = build_ObsIDs(args)
    if not ObsIDs:
        print("No inputs found. Provide -f filename.csv, -d directory, or KPF filenames as arguments.", file=sys.stderr)
        return 1

    for ObsID in ObsIDs:
        datecode = ObsID[3:11] if len(ObsID) >= 11 else ""
        if not is_valid_datecode(datecode):
            print(f"Warning: cannot parse valid datecode from '{ObsID}'. Skipping.", file=sys.stderr)
            continue

        fullpath = base / datecode / f"{ObsID}.fits"
        touch_file(fullpath, args.echo, args.sleep_interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
