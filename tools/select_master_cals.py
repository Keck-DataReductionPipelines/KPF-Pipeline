#!/usr/bin/env python3
"""select_master_cals.py -- pick calibration frames to build masters.

Run this where the L0 tree and astropy are available (typically on the remote
data server). It takes the same obs_id-list input as fetch_l0.sh, infers every
unique datecode in that list, scans the L0 headers for each night under --root,
and -- reproducing the KPF pipeline's own clustering logic from
kpfpipe/utils/io.py::build_l0_file_lists -- selects one calibration sequence
(cluster) of each type (bias, dark, flat, thar) per night and emits its frames.

Frames are typed by the PRIMARY-header OBJECT keyword (autocal-bias, etc., the
same map io.py uses). Frames of one type are grouped by OBJECT, sorted in time,
and split into clusters wherever consecutive frames are more than --gap seconds
apart OR fall on different HST calendar days (KPF's morning vs. evening cal
sequences). Only clusters meeting the per-type minimum are eligible; of those,
the largest (earliest on a tie) is chosen so all frames come from a single
sequence. Per-type thresholds match recipes/kpf_drp_masters.py (see
_TYPE_CONFIG): bias/flat/thar require 5 frames, darks require 3 and merge
undersized clusters into a same-HST-day neighbor.

Output is an obs_id list in the exact format of the input (index<TAB>obs_id,
one per line), ready to feed straight back into fetch_l0.sh. A human-readable
selection summary is written to stderr.

Usage:
    python3 select_master_cals.py <science_obs_ids.txt> -o <cal_obs_ids.txt>
    python3 select_master_cals.py obs_ids.txt --root /data/kpf/L0 -o cals.txt
    # then feed the output to fetch_l0.sh to retrieve the calibration frames.
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta

from astropy.io import fits

# --- mirrored verbatim from kpfpipe/utils/{io,kpf}.py -----------------------
_OBS_ID_RE = re.compile(r"(KP\.\d{8}\.\d{5}\.\d{2})")
_TS_RE = re.compile(r"(\d{8}\.\d{5}\.\d{2})")
_HST_UTC_OFFSET_SECONDS = 36000

# Calibration type -> the PRIMARY-header OBJECT values that mark it (io.py).
_OBJECT_MAP = {
    "bias": ["autocal-bias"],
    "dark": ["autocal-dark"],
    "flat": ["autocal-flat-all"],
    "thar": [
        "autocal-thar-all-morn",
        "autocal-thar-all-midday",
        "autocal-thar-all-eve",
        "autocal-thar-all-night",
        "autocal-thar-all-midnight",
    ],
}

# Per-type clustering settings, matching recipes/kpf_drp_masters.py exactly:
# bias/flat/thar use build_l0_file_lists defaults (min 5, no merge); darks are
# sparse (long 1200 s exposures scattered across a night), so the recipe drops
# the threshold to 3 and merges undersized clusters into a same-HST-day
# neighbor. Each type still yields a single sequence, as requested.
_TYPE_CONFIG = {
    "bias": {"min_count": 5, "merge": False},
    "dark": {"min_count": 3, "merge": True},
    "flat": {"min_count": 5, "merge": False},
    "thar": {"min_count": 5, "merge": False},
}


def get_timestamp(s):
    m = _TS_RE.search(os.path.basename(s))
    if not m:
        raise ValueError(f"No KPF timestamp in: {s}")
    return m.group(1)


def ts_to_datetime(ts):
    """KPF UTC timestamp 'YYYYMMDD.SSSSS.FF' -> naive UTC datetime."""
    date_str, sec_str, _ = ts.split(".")
    return datetime.strptime(date_str, "%Y%m%d") + timedelta(seconds=int(sec_str))


def hst_day(ts):
    """HST (UTC-10) calendar day 'YYYYMMDD' for a KPF UTC timestamp."""
    date_str, sec_str, _ = ts.split(".")
    hst = int(sec_str) - _HST_UTC_OFFSET_SECONDS
    date = datetime.strptime(date_str, "%Y%m%d")
    if hst < 0:
        date -= timedelta(days=1)
    return date.strftime("%Y%m%d")


def datecodes_from_list(path):
    """Unique datecodes (sorted) from the obs_ids in an input list file."""
    with open(path) as fh:
        text = fh.read()
    codes = {m.split(".")[1] for m in _OBS_ID_RE.findall(text)}
    if not codes:
        sys.exit(f"error: no obs_ids found in {path}")
    return sorted(codes)


def scan_night(data_dir):
    """Return {filename: (object, exptime, timestamp)} for one L0 datecode dir.

    Only frames whose OBJECT is a known calibration type are kept.
    """
    cal_objects = {o for objs in _OBJECT_MAP.values() for o in objs}
    frames = {}
    files = sorted(glob.glob(os.path.join(data_dir, "*.fits")))
    for fn in files:
        try:
            hdr = fits.getheader(fn, ext=0)
        except Exception as e:  # noqa: BLE001 -- log and skip unreadable frames
            print(f"  WARN: could not read header {fn}: {e}", file=sys.stderr)
            continue
        obj = hdr.get("OBJECT")
        if obj in cal_objects:
            frames[fn] = (obj, hdr.get("EXPTIME"), get_timestamp(fn))
    return frames, len(files)


def _merge_small(clusters, min_count):
    """Fold undersized clusters into a same-HST-day neighbor (io.py logic).

    Mirrors build_l0_file_lists(merge_small_clusters=True): repeatedly take the
    smallest undersized cluster and merge it into its nearer chronological
    neighbor on the same HST day; an isolated undersized cluster (no same-day
    neighbor) is dropped. Cluster entries are (datetime, ts, fn) tuples.
    """
    def day(c):
        return hst_day(c[0][1])

    inf = float("inf")
    while len(clusters) > 1 and any(len(c) < min_count for c in clusters):
        i = min((k for k, c in enumerate(clusters) if len(c) < min_count),
                key=lambda k: len(clusters[k]))
        di = day(clusters[i])
        prev_gap = ((clusters[i][0][0] - clusters[i - 1][-1][0]).total_seconds()
                    if i > 0 and day(clusters[i - 1]) == di else inf)
        next_gap = ((clusters[i + 1][0][0] - clusters[i][-1][0]).total_seconds()
                    if i < len(clusters) - 1 and day(clusters[i + 1]) == di else inf)
        if prev_gap == inf and next_gap == inf:
            clusters.pop(i)
            continue
        j = i - 1 if prev_gap <= next_gap else i + 1
        merged = sorted(clusters[i] + clusters[j])  # by datetime (tuple order)
        for idx in sorted((i, j), reverse=True):
            clusters.pop(idx)
        clusters.append(merged)
    clusters.sort(key=lambda c: c[0][0])
    return clusters


def cluster(frames, cal_type, gap, min_count, merge):
    """Clusters of a given cal type, mirroring io.py::build_l0_file_lists.

    frames: {fn: (object, exptime, ts)}. Groups by OBJECT, splits on >gap gaps
    and HST-midnight, optionally merges undersized clusters, then returns those
    with >= min_count frames, sorted chronologically. Each cluster is a list of
    (datetime, ts, fn) tuples.
    """
    wanted = set(_OBJECT_MAP[cal_type])
    by_obj = defaultdict(list)
    for fn, (obj, _exp, ts) in frames.items():
        if obj in wanted:
            by_obj[obj].append((ts_to_datetime(ts), ts, fn))

    clusters = []
    for group in by_obj.values():
        timed = sorted(group)  # by datetime
        cur = [timed[0]]
        for (pdt, pts, _pfn), (dt, ts, fn) in zip(timed, timed[1:]):
            if (dt - pdt).total_seconds() > gap or hst_day(ts) != hst_day(pts):
                clusters.append(cur)
                cur = [(dt, ts, fn)]
            else:
                cur.append((dt, ts, fn))
        clusters.append(cur)
    clusters.sort(key=lambda c: c[0][0])

    if merge:
        clusters = _merge_small(clusters, min_count)
    return [c for c in clusters if len(c) >= min_count]


def obs_id_of(fn):
    m = _OBS_ID_RE.search(os.path.basename(fn))
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obs_id_list", help="input list of science obs_ids (as 10700.txt)")
    ap.add_argument("-o", "--out", default="master_calibs.txt",
                    help="output obs_id list (default: master_calibs.txt; '-' = stdout)")
    ap.add_argument("--root", default="/data/kpf/L0",
                    help="L0 root holding {datecode}/ subdirs (default: "
                         "/data/kpf/L0, the standard KPF data-server layout)")
    ap.add_argument("--count", type=int, default=5,
                    help="max frames to take per type per night (fewer if the "
                         "chosen cluster is smaller, e.g. sparse darks)")
    ap.add_argument("--gap", type=int, default=7200,
                    help="cluster-splitting gap in seconds (default 2h)")
    args = ap.parse_args()

    datecodes = datecodes_from_list(args.obs_id_list)
    print(f"# {len(datecodes)} unique datecode(s) from {args.obs_id_list}: "
          f"{', '.join(datecodes)}\n", file=sys.stderr)

    selected = []  # list of (datecode, cal_type, [obs_ids])
    for dc in datecodes:
        data_dir = os.path.join(args.root, dc)
        if not os.path.isdir(data_dir):
            print(f"{dc}: MISSING remote dir {data_dir} -- skipped", file=sys.stderr)
            continue
        frames, n_all = scan_night(data_dir)
        print(f"{dc}: scanned {n_all} L0 frames, {len(frames)} calibration frames",
              file=sys.stderr)
        for cal_type in ("bias", "dark", "flat", "thar"):
            cfg = _TYPE_CONFIG[cal_type]
            min_count = cfg["min_count"]
            clusters = cluster(frames, cal_type, args.gap, min_count, cfg["merge"])
            if not clusters:
                print(f"    {cal_type:5s}: no cluster with >= {min_count} "
                      f"frames -- skipped", file=sys.stderr)
                continue
            # largest cluster (earliest on a tie) -> take the first `count`
            best = max(clusters, key=lambda c: (len(c), -c[0][0].timestamp()))
            chosen = best[:args.count]
            obs_ids = [obs_id_of(fn) for _dt, _ts, fn in chosen]
            exps = {frames[fn][1] for _dt, _ts, fn in chosen}
            t0, t1 = chosen[0][1], chosen[-1][1]
            print(f"    {cal_type:5s}: {len(clusters)} eligible cluster(s), "
                  f"chose size {len(best)} -> {len(obs_ids)} frames "
                  f"[{t0} .. {t1}] EXPTIME={sorted(e for e in exps if e is not None)}",
                  file=sys.stderr)
            selected.append((dc, cal_type, obs_ids))

    all_ids = [oid for _dc, _t, ids in selected for oid in ids]
    lines = [f"{i}\t{oid}" for i, oid in enumerate(all_ids, start=1)]
    out_text = "\n".join(lines) + "\n" if lines else ""

    if args.out == "-":
        sys.stdout.write(out_text)
    else:
        with open(args.out, "w") as fh:
            fh.write(out_text)
        print(f"\n# wrote {len(all_ids)} calibration obs_ids to {args.out}",
              file=sys.stderr)

    if not all_ids:
        sys.exit("error: no calibration frames selected")


if __name__ == "__main__":
    main()
