#!/usr/bin/env python3
"""Reduce a single star's RV timeseries over a datecode range (``kpfpipe timeseries``).

A lightweight, fail-loud wrapper that discovers a target's raw science frames,
then hands the work off to the masters and science orchestrators. Given a target
star and an inclusive datecode range, it

  1. combs the L0 input tree for that target's raw science frames (obs_ids),
  2. infers the unique nights (datecodes) those frames span,
  3. builds the nightly calibration masters (``kpfpipe masters``),
  4. reduces every science frame end-to-end, L0 -> L4 (``kpfpipe science``),
  5. plots the RV timeseries from those L4 products (``plot_timeseries``).

It reimplements no pipeline logic: this script owns only discovery (steps 1-2) and
dispatch; steps 3-5 each run one subprocess (the two orchestrators, which fan out
one ``reduce`` per unit, and the standalone plotter), so the robust fan-out,
timeout, and interrupt handling live in the orchestrators. The plotter is handed
the already-discovered obs_ids, so it does no second scan.

All three stages run by default; ``--no-masters`` / ``--no-science`` / ``--no-plots``
skip any (discovery always runs). The stages are fail-soft and independent: every
discovered frame is handed to science regardless of the masters result (a frame
missing its masters just fails in science, reported per-frame -- no gating here),
and plotting runs whatever L4 is on disk (``--no-science`` plots the existing L4),
writing to ``{KPF_SCIENCE_OUTPUT}/QLP/timeseries`` unless ``--plot_dir`` overrides
it. The run exits nonzero if any stage that ran reported a failure.

    kpfpipe timeseries --target 10700 --date_range 20240101 20240131
"""

import argparse
import concurrent.futures
import logging
import os
import subprocess
import sys

import kpfpipe
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import FileHandler, datecode_dirs_in_range
from kpfpipe.utils.kpf_utils import get_datecode, get_obs_id, is_datecode
from kpfpipe.utils.logger import setup_batch_logging
from scripts.processing import (
    DEFAULT_MASTERS_CONFIG,
    DEFAULT_MASTERS_RECIPE,
    DEFAULT_SCIENCE_CONFIG,
    DEFAULT_SCIENCE_RECIPE,
)
from scripts.processing._argparse import (
    data_dirs_parser,
    logging_parser,
    pool_parser,
    resolve_dir_shortcuts,
)
from scripts.processing._dispatch import _default_science_jobs

logger = logging.getLogger(__name__)

# --jobs help is timeseries-specific: it is forwarded to both stages (each of
# which otherwise picks its own default) and also sizes the discovery scan pool.
_JOBS_HELP = (
    "max concurrent recipe subprocesses, forwarded to both the masters and "
    "science stages; left unset, each stage uses its own default (masters its "
    "fixed cap, science a cores-based value) and the discovery scan uses a "
    "cores-based worker count"
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="kpfpipe timeseries",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[
            data_dirs_parser(science_output=True),
            logging_parser(),
            pool_parser(jobs_help=_JOBS_HELP),
        ],
    )
    ap.add_argument(
        "--target",
        required=True,
        help="star id as it appears in the L0 OBJECT header, e.g. 10700",
    )
    ap.add_argument(
        "--date_range",
        nargs=2,
        metavar=("START", "END"),
        required=True,
        help="inclusive datecode range, e.g. --date_range 20240101 20240131",
    )
    # Stage toggles (which *stage* to run) -- unrelated to reduce's
    # --masters/--science recipe shortcuts, which live only on that leaf's parser.
    ap.add_argument(
        "--masters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the masters build stage (default: on; pass --no-masters to skip)",
    )
    ap.add_argument(
        "--science",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the science reduction stage (default: on; pass --no-science to skip)",
    )
    ap.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the RV-timeseries plotting stage (default: on; pass --no-plots "
        "to skip)",
    )
    ap.add_argument(
        "--plot_dir",
        default=None,
        help="directory for the RV plots (default: "
        "{KPF_SCIENCE_OUTPUT}/QLP/timeseries)",
    )
    # Two recipe/config pairs (masters + science), so the shared single -r/-c
    # parser does not fit; each is forwarded to its stage only when set.
    ap.add_argument(
        "--masters_recipe",
        default=None,
        help="masters recipe .py forwarded to the masters stage (default: the "
        "masters stage's own default recipe)",
    )
    ap.add_argument(
        "--masters_config",
        default=None,
        help="masters recipe TOML forwarded to the masters stage (default: the "
        "masters stage's own default config)",
    )
    ap.add_argument(
        "--science_recipe",
        default=None,
        help="science recipe .py forwarded to the science stage (default: the "
        "science stage's own default recipe)",
    )
    ap.add_argument(
        "--science_config",
        default=None,
        help="science recipe TOML forwarded to the science stage (default: the "
        "science stage's own default config)",
    )
    args = ap.parse_args(argv)

    start, end = args.date_range
    for dc in (start, end):
        if not is_datecode(dc):
            ap.error(f"--date_range value is not a valid datecode: {dc!r}")
    if start > end:
        ap.error(f"--date_range START must be <= END (got {start} > {end})")
    if args.job_timeout < 1:
        ap.error("--job_timeout must be >= 1")
    if args.jobs is not None and args.jobs < 1:
        ap.error("--jobs must be >= 1")
    return resolve_dir_shortcuts(args)


def _scan_nights(nights, target, worker, jobs):
    """Fan `worker(dc)` out over `nights` in a thread pool; return the pooled hits.

    Scanning a night reads every PRIMARY header -- I/O bound, so a thread pool
    overlaps the NFS latency (``getheader`` releases the GIL during I/O).
    `worker(dc)` returns ``(hits, note)``: that night's obs_ids and an optional
    note, logged as a per-night heartbeat in completion order.
    """
    logger.info(
        "scanning %d night(s) for target %s (%d workers)...",
        len(nights),
        target,
        min(jobs, len(nights)),
    )
    hits_all = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(worker, dc): dc for dc in nights}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            dc = futures[future]
            hits, note = future.result()
            hits_all.extend(hits)
            logger.info(
                "  [%d/%d] %s: %d matching frame(s)%s (total %d)",
                i,
                len(nights),
                dc,
                len(hits),
                note,
                len(hits_all),
            )
    return hits_all


def discover_science_obs_ids(data_input, target, start, end, jobs):
    """Raw science obs_ids for `target` over [start, end], from the L0 tree.

    Scans each night (datecode dir) under {data_input}/L0 via
    FileHandler.build_mini_database (cache="rw": reuse the night's on-disk CSV when
    present, else write it), keeping frames whose IMTYPE is 'Object' and OBJECT
    matches `target`. Non-datecode entries are skipped with a note; observer-flagged
    junk frames (the ISJUNK column) are dropped. Exits loudly when the tree is
    missing or nothing matches.
    """
    l0_root = os.path.join(data_input, "L0")
    if not os.path.isdir(l0_root):
        sys.exit(f"error: L0 input directory not found: {l0_root}")

    non_datecode = [
        e
        for e in sorted(os.listdir(l0_root))
        if os.path.isdir(os.path.join(l0_root, e)) and not is_datecode(e)
    ]
    if non_datecode:
        logger.info(
            "  note: ignoring %d non-datecode entr(y/ies) under %s: %s",
            len(non_datecode),
            l0_root,
            ", ".join(non_datecode),
        )

    nights = datecode_dirs_in_range(l0_root, start, end)
    if not nights:
        sys.exit(f"error: no datecode dirs under {l0_root} in range {start}..{end}")

    def _scan_night(dc):
        # A FileHandler per night (not shared): it carries the scanned night on
        # self._mini_db, so one instance across these pooled threads would race.
        file_handler = FileHandler({"KPF_DATA_INPUT": data_input})
        try:
            df = file_handler.build_mini_database(dc, cache="rw")
        except ValueError as e:
            # e.g. a datecode dir with no FITS files -- skip it, don't abort.
            logger.warning("  skipping night %s: %s", dc, e)
            return [], ""
        is_object = df["IMTYPE"].astype(str).str.strip() == "Object"
        is_target = df["OBJECT"].astype(str).str.strip() == str(target)
        matched = df.loc[is_object & is_target]
        good = matched.loc[~matched["ISJUNK"].astype(bool)]
        n_junk = len(matched) - len(good)
        note = f", {n_junk} junk skipped" if n_junk else ""
        return [get_obs_id(fn) for fn in good["FILENAME"]], note

    obs_ids = _scan_nights(nights, target, _scan_night, jobs)

    obs_ids = sorted(set(obs_ids))
    if not obs_ids:
        sys.exit(
            f"error: no science frames for target {target!r} in range "
            f"{start}..{end} under {l0_root}"
        )
    return obs_ids


def _orchestrator_argv(module, unit_flag, units, forward, recipe=None, config=None):
    """Build the argv for one orchestrator subprocess (masters or science).

    Runs ``python -m scripts.processing.{module} {unit_flag} {units...}`` with the
    dir/log/pool overrides in `forward`, plus ``-r``/``-c`` when a recipe/config is
    given. The unit list precedes `forward` so the ``--dates``/``--obs_ids``
    ``nargs`` stops at the first forwarded flag.
    """
    argv = [sys.executable, "-m", f"scripts.processing.{module}"]
    if recipe:
        argv += ["-r", recipe]
    if config:
        argv += ["-c", config]
    argv += [unit_flag, *units, *forward]
    return argv


def main(argv=None):
    args = parse_args(argv)
    start, end = args.date_range

    # Resolve each stage's recipe+config (its default unless overridden) and pass
    # both explicitly, so timeseries owns exactly what runs. The science config
    # also supplies this wrapper's L0 input root + log dir/level.
    masters_recipe = args.masters_recipe or DEFAULT_MASTERS_RECIPE
    masters_config = args.masters_config or DEFAULT_MASTERS_CONFIG
    science_recipe = args.science_recipe or DEFAULT_SCIENCE_RECIPE
    science_config = args.science_config or DEFAULT_SCIENCE_CONFIG
    science_dirs = ConfigHandler(science_config).get_params(["DATA_DIRS"])
    logger_params = ConfigHandler(science_config).get_params(["LOGGER"])
    data_input = args.kpf_data_input or science_dirs["KPF_DATA_INPUT"]
    log_dir = args.log_dir or logger_params.get("log_dir")
    if not log_dir:
        sys.exit(
            "error: no log directory configured; set [LOGGER] log_dir in the "
            "config file or pass --log_dir"
        )

    # The batch-summary log: this wrapper's own DRP-RUN-08 decision trail
    # (discovery, dispatch), echoed to stdout and persisted alongside each stage's
    # own batch log and each unit's reduction log.
    level = args.log_level or logger_params.get("log_level", "INFO")
    log_path = setup_batch_logging(log_dir, "timeseries", level=level)

    # Overrides forwarded to the orchestrators. Both parsers accept the common set;
    # --jobs and --kpf_science_output ride science_forward only (see below).
    common_forward = []
    for value, flag in (
        (args.kpf_data_input, "--kpf_data_input"),
        (args.kpf_masters_output, "--kpf_masters_output"),
        (args.log_dir, "--log_dir"),
        (args.log_level, "--log_level"),
    ):
        if value:
            common_forward += [flag, value]
    common_forward += ["--job_timeout", str(args.job_timeout)]

    # --jobs sizes only the science fan-out: forwarding a large --jobs to masters
    # would defeat its fixed _MASTERS_JOBS safety cap (see _dispatch.py), so --jobs
    # rides science_forward and masters keeps its own default sizing.
    science_forward = list(common_forward)
    if args.jobs is not None:
        science_forward += ["--jobs", str(args.jobs)]
    if args.kpf_science_output:
        science_forward += ["--kpf_science_output", args.kpf_science_output]

    # The plotting stage reads L4 from the science output root, and writes to
    # {KPF_SCIENCE_OUTPUT}/QLP/timeseries unless --plot_dir overrides it.
    science_output = args.kpf_science_output or science_dirs["KPF_SCIENCE_OUTPUT"]
    plot_dir = args.plot_dir or os.path.join(science_output, "QLP", "timeseries")

    logger.info("kpfpipe %s timeseries batch starting", kpfpipe.__version__)
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info("target: %s  date_range: %s..%s", args.target, start, end)
    logger.info("batch log: %s", log_path)

    # Steps 1-2: discover the target's science frames -> the nights they span.
    scan_workers = args.jobs or _default_science_jobs()
    obs_ids = discover_science_obs_ids(
        data_input, args.target, start, end, scan_workers
    )
    datecodes = sorted({get_datecode(o) for o in obs_ids})
    logger.info(
        "target %s: %d science frame(s) across %d night(s) [%s..%s]",
        args.target,
        len(obs_ids),
        len(datecodes),
        start,
        end,
    )

    # Step 3: build every night's masters, unless --no-masters. A skipped stage's
    # exit code stays None (not a failure), so the final exit reflects only the
    # stages that ran.
    masters_rc = None
    if args.masters:
        masters_argv = _orchestrator_argv(
            "masters",
            "--dates",
            datecodes,
            common_forward,
            recipe=masters_recipe,
            config=masters_config,
        )
        logger.info("dispatching masters for %d night(s)", len(datecodes))
        masters_rc = subprocess.run(masters_argv, cwd=kpfpipe.REPO_ROOT).returncode
    else:
        logger.info("skipping masters stage (--no-masters)")

    # Step 4: reduce every discovered science frame, unless --no-science. Every
    # frame is handed over regardless of the masters result -- a frame whose masters
    # failed simply fails here, reported per-frame; no gating happens.
    science_rc = None
    if args.science:
        science_argv = _orchestrator_argv(
            "science",
            "--obs_ids",
            obs_ids,
            science_forward,
            recipe=science_recipe,
            config=science_config,
        )
        logger.info("dispatching science for %d frame(s)", len(obs_ids))
        science_rc = subprocess.run(science_argv, cwd=kpfpipe.REPO_ROOT).returncode
    else:
        logger.info("skipping science stage (--no-science)")

    # Step 5: plot the RV timeseries, unless --no-plots. The plotter builds L4 paths
    # from the already-discovered obs_ids (no second scan); it runs independently of
    # science -- with --no-science it plots whatever L4 is already on disk.
    plots_rc = None
    if args.plots:
        plot_argv = [
            sys.executable,
            "-m",
            "scripts.plots.plot_timeseries",
            "--target",
            args.target,
            "--data_dir",
            science_output,
            "--plot_dir",
            plot_dir,
            "--obs_ids",
            *obs_ids,
        ]
        logger.info("dispatching plots for %d frame(s) -> %s", len(obs_ids), plot_dir)
        plots_rc = subprocess.run(plot_argv, cwd=kpfpipe.REPO_ROOT).returncode
    else:
        logger.info("skipping plots stage (--no-plots)")

    logger.info(
        "done: masters exit %s, science exit %s, plots exit %s",
        "skipped" if masters_rc is None else masters_rc,
        "skipped" if science_rc is None else science_rc,
        "skipped" if plots_rc is None else plots_rc,
    )
    if masters_rc or science_rc or plots_rc:
        sys.exit(1)


if __name__ == "__main__":
    main()
