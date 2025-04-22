#!/usr/bin/env python3
"""
watch_file_events.py  PATTERN [PATTERN ...]  [-i SECONDS]

Wildcards accepted (e.g.  *.fits  /data/kpf/2D/**/*.fits ).

Events printed:
    created   – file appeared
    deleted   – file vanished
    replaced  – inode changed (atomic overwrite)
    grew      – size increased
    truncated – size decreased
    modified  – mtime changed, inode & size unchanged
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Set

Stat = Dict[str, object]                # {'mtime': float, 'size': int, 'inode': (dev, ino)}


# ────────────────────────── helpers ──────────────────────────
def get_stat(path: Path) -> Optional[Stat]:
    try:
        s = os.stat(path)
    except FileNotFoundError:
        return None
    return {"mtime": s.st_mtime, "size": s.st_size, "inode": (s.st_dev, s.st_ino)}


def classify(prev: Optional[Stat], curr: Optional[Stat]) -> Optional[str]:
    if prev is None and curr is not None:
        return "created"
    if prev is not None and curr is None:
        return "deleted"
    if prev is None and curr is None:
        return None
    if prev["inode"] != curr["inode"]:
        return "replaced"
    if curr["size"] > prev["size"]:
        return "grew"
    if curr["size"] < prev["size"]:
        return "truncated"
    if curr["mtime"] != prev["mtime"]:
        return "modified"
    return None


def expand_patterns(patterns) -> Set[Path]:
    """Return *absolute* Paths matching any of the glob patterns."""
    paths: Set[Path] = set()
    for patt in patterns:
        paths.update(Path(p).expanduser().resolve() for p in glob.glob(patt, recursive=True))
    return paths


# ─────────────────────────── main ────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Print timestamp, event, and file whenever a match changes.")
    ap.add_argument("patterns", nargs="+", help="file patterns to watch (wildcards allowed)")
    ap.add_argument("-i", "--interval", type=float, default=0.5,
                    help="polling interval in seconds (default 0.5)")
    args = ap.parse_args()

    seen: Dict[Path, Optional[Stat]] = {}          # last known state per file
    poll = args.interval

    print(f"Watching patterns: {', '.join(args.patterns)} (poll {poll}s) — Ctrl‑C to stop")
    try:
        while True:
            # current set of files matching the patterns
            current_paths = expand_patterns(args.patterns)

            # union with previously seen paths to detect deletions & recreations
            for path in current_paths | set(seen):
                prev = seen.get(path)
                curr = get_stat(path)              # may be None if deleted

                event = classify(prev, curr)
                if event:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(f"{ts}  {event:<9}  {path}")
                    sys.stdout.flush()

                seen[path] = curr                  # update / add entry
            time.sleep(poll)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
