#!/usr/bin/env python3
"""
Script name: slowtouch.py

This script 'touches' a list of KPF L0 files with names like
KP.YYYYMMDD.12345.67.fits to trigger reprocessing in the KPF DRP.

Ways to provide filenames (any combination works):
  1) As positional arguments on the command line.
  2) With -f <csv>, reading the first column (quotes removed; header 'observation_id' skipped).
  3) With -d <dir>, adding every file name in that directory.

Date range mode:
  If you pass exactly two positional arguments that are valid datecodes
  (YYYYMMDD) and you do NOT use -f/--csv or -d/--dir, the script switches to
  'date range mode'. In this mode it:
    • Validates the two YYYYMMDD values and sorts them into start_date/end_date.
    • Use the time series database (TSDB) to query for ObsIDs
      in that date window, optionally filtering by:
        --only-object <name>   (exact match or DB-implemented rule for OBJECT)
        --only-source <name>   (e.g., Star, Etalon, Dark, etc.)
    • Prints the matched ObsIDs (one per line).
    • Does NOT touch any files in this mode.

Options (all optional):
  -f, --csv <filename>       CSV with L0 filenames in the first column (can be used multiple times)
  -d, --dir <directory>      Directory to scan for filenames (can be used multiple times)
  -p, --path <path>          L0 base path (default: /data/kpf/L0)
  -s, --sleep <seconds>      Sleep interval between touches (default: 0.2)
  -e, --echo                 Echo touch commands instead of executing
      --only-object <name>   (Date range mode) filter TSDB rows to this OBJECT (e.g., autocal-bias)
      --only-source <name>   (Date range mode) filter TSDB rows to this SOURCE (e.g., Star, Etalon, Dark)

Examples:
  slowtouch.py KP.20230623.12345.67.fits KP.20230623.12345.68.fits # filenames using command line arguments
  slowtouch.py -f filenames.csv # filenames using a .csv file 
  slowtouch.py -d /path/to/directory # touch all files in a directory 
  slowtouch.py KP.20230623.12345.67.fits -p /new/path -s 0.5  # set path for L0 directory (-p) and sleep interval (-s)
  slowtouch.py KP.20230623.12345.67.fits -e (only echo the touch commands)
  slowtouch.py 20241001 20241015 --only-object autocal-dark   # date range mode, only 'autocal-dark' observations
  slowtouch.py 20241001 20241015 --only-source Star   # date range mode, only observations of stars (with a variety of object names)
  slowtouch.py 20241001 20241015 --only-source Etalon # date range mode, only Etalon observations
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List


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


def parse_args(argv=None):
    description = __doc__ or "Touch KPF L0 files (KP.YYYYMMDD.xxxxx.xx.fits) to trigger reprocessing."
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("filenames", nargs="*", help="KPF L0 filenames (e.g., KP.20230623.12345.67.fits) OR two YYYYMMDD dates")
    p.add_argument("-f", "--csv", action="append", default=[], help="CSV file with filenames in first column")
    p.add_argument("-d", "--dir", dest="dirs", action="append", default=[], help="Directory containing filenames")
    p.add_argument("-p", "--path", dest="l0_path", default='/data/kpf/L0',
                   help=f"Base L0 path (default: /data/L0 (docker) /data/kpf/L0 (not in docker))")
    p.add_argument("-s", "--sleep", dest="sleep_interval", type=float, default=0.2,
                   help=f"Sleep interval between touches (seconds; default: 0.2)")
    p.add_argument("-e", "--echo", dest="echo", action="store_true", help="Echo touch commands instead of executing")
    p.add_argument("--only-object", "--only_object", dest="only_object", default=None,
                   help="(Date range mode) filter DB query to OBJECT (exact match or DB-implemented rule)")
    p.add_argument("--only-source", "--only_source", dest="only_source", default=None,
                   help="(Date range mode) filter DB query to SOURCE (e.g., Star, ThAr, LFC, etc.)")
    return p.parse_args(argv)


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
    text = Path(text).name  # keep only the filename
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
    # Create file if missing; update mtime otherwise
    fullpath.parent.mkdir(parents=True, exist_ok=True)  # ensure date dir exists
    fullpath.touch(exist_ok=True)
    time.sleep(sleep_interval)


def main(argv=None):
    args = parse_args(argv)

    # --- Date range mode: exactly two YYYYMMDD positional args (real dates), and no -f/-d used.
    if not args.csv and not args.dirs and len(args.filenames) == 2 \
       and all(is_valid_datecode(x) for x in args.filenames):
        start_date, end_date = sorted(args.filenames)
        try:
            # myTS must already exist in your environment
            cols = ['ObsID', 'Source', 'Object']
            df = myTS.db.dataframe_from_db(
                columns=cols,
                only_object=args.only_object,
                only_source=args.only_source,
                start_date=start_date,
                end_date=end_date
            )
            ObsIDs = df['ObsID'].astype(str).tolist()
            # For now, just print them (one per line). No touching in date-range mode.
            print(f"# matched ObsIDs: {len(ObsIDs)}")
            for obs in ObsIDs:
                print(obs)
        except NameError:
            print("myTS is not defined in this environment; skipping DB query.", file=sys.stderr)
        except Exception as e:
            print(f"Error querying DB: {e}", file=sys.stderr)

        return 0

    # --- Normal touch mode
    ObsIDs = build_ObsIDs(args)
    if not ObsIDs:
        print("No inputs found. Provide -f filename.csv, -d directory, or KPF filenames as arguments.", file=sys.stderr)
        return 1

    base = Path(args.l0_path)

    for ObsID in ObsIDs:
        # ObsID is without ".fits" at this point
        yyyymmdd = ObsID[3:11] if len(ObsID) >= 11 else ""
        if not is_valid_datecode(yyyymmdd):
            # Mirror bash script permissiveness; warn and skip if date slice is wrong
            print(f"Warning: cannot parse valid datecode from '{ObsID}'. Skipping.", file=sys.stderr)
            continue

        fullpath = base / yyyymmdd / f"{ObsID}.fits"
        touch_file(fullpath, args.echo, args.sleep_interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
