#!/usr/bin/env python3
"""Reduce a set of science frames end-to-end, L0 -> L4 (``kpfpipe science``).

A lightweight, fail-loud orchestrator. Given an explicit list of obs_ids, it
dispatches each frame as a separate
``python -m scripts.processing.reduce --science -o <obs_id>`` subprocess (the
``kpfpipe run`` leaf), so every frame gets its own log file, clean process state,
and an independent exit code.

It reimplements no pipeline logic and does not decide *which* frames to reduce:
the caller supplies the obs_ids (discovering a target's frames from the L0 tree
lives in the orchestrator, not here). The single input form is:

    kpfpipe science --obs_id_list KP.20240405.40113.57 KP.20240405.40237.36

Reductions run in a bounded process pool: the first frame runs alone as a canary
to warm the shared on-disk caches, then the rest fan out (paced apart, since the
per-frame L0 pointing QC queries SIMBAD/Gaia). The run is fail-soft -- a frame
that fails to reduce is reported and the run continues with the others -- but the
script exits nonzero if any frame failed, so a caller gets a meaningful exit code.
"""

import argparse
import sys

from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.kpf_utils import is_obs_id
from scripts.processing._common import shortcut_paths
from scripts.processing._fanout import _default_jobs, configure_runtime, run_stage

# Minimum spacing (seconds) between fanned-out subprocess launches. The per-frame
# L0 pointing QC fires rapid SIMBAD/Gaia catalog queries at startup, so a burst of
# pool workers launching at once would hammer those services; pacing the launches
# rate-limits them. Passed to run_stage as its launch_interval.
_LAUNCH_INTERVAL = 1.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="kpfpipe science",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--obs_id_list",
        nargs="+",
        required=True,
        metavar="OBS_ID",
        help="one or more obs_ids to reduce, e.g. --obs_id_list "
        "KP.20240405.40113.57 KP.20240405.40237.36",
    )
    ap.add_argument(
        "-c",
        "--config",
        default=None,
        help="science recipe TOML override (default: the --science shortcut's "
        "configs/kpf_drp_science.toml)",
    )
    ap.add_argument("--kpf_data_input", help="override [DATA_DIRS] KPF_DATA_INPUT")
    ap.add_argument(
        "--kpf_masters_output", help="override [DATA_DIRS] KPF_MASTERS_OUTPUT"
    )
    ap.add_argument(
        "--kpf_science_output", help="override [DATA_DIRS] KPF_SCIENCE_OUTPUT"
    )
    ap.add_argument("--log_dir", help="override [LOGGER] log_dir")
    ap.add_argument(
        "--log_level", help="override [LOGGER] log_level for each recipe (e.g. DEBUG)"
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="max concurrent science reductions; left unset, defaults to a "
        "cores-based value (~25%% of CPUs, but up to 16)",
    )
    ap.add_argument(
        "--job_timeout",
        type=int,
        default=600,
        help="per-job wall-clock limit (seconds) for each fanned-out recipe "
        "subprocess (default: %(default)s). A recipe normally runs in ~2 min, so a "
        "job exceeding this is treated as wedged: its process group is killed and "
        "the job counts as a failure, rather than hanging the whole batch. The "
        "serial canary uses a larger, separate limit since it warms cold caches on "
        "the first run",
    )
    args = ap.parse_args(argv)

    for obs_id in args.obs_id_list:
        if not is_obs_id(obs_id):
            ap.error(f"not a valid obs_id: {obs_id!r}")
    args.obs_id_list = sorted(set(args.obs_id_list))

    if args.job_timeout < 1:
        ap.error("--job_timeout must be >= 1")
    if args.jobs is None:
        args.jobs = _default_jobs()
    elif args.jobs < 1:
        ap.error("--jobs must be >= 1")
    return args


def _cli_task(obs_id, forward, config=None):
    """Build a (tag, argv) task reducing one frame via the ``kpfpipe run`` leaf.

    Runs ``python -m scripts.processing.reduce --science -o <obs_id>`` (the leaf
    resolves the science recipe + default config from the --science shortcut);
    `config` overrides the shortcut's default config only when given. `forward` is
    the resolved dir/log overrides appended to every invocation. The tag is the
    obs_id itself, which the sentinel/log paths key on.
    """
    argv = [sys.executable, "-m", "scripts.processing.reduce", "--science"]
    if config:
        argv += ["-c", config]
    argv += ["-o", obs_id, *forward]
    return obs_id, argv


def main(argv=None):
    configure_runtime()
    args = parse_args(argv)

    # Read the config the subprocesses will use (the --science shortcut default
    # unless overridden) only for the log dir (used in the failure sentinels); the
    # subprocess gets the config via the shortcut, plus -c only when overridden
    # (see _cli_task).
    config = ConfigHandler(args.config or shortcut_paths("science")[1])
    log_dir = args.log_dir or config.get_params(["LOGGER"]).get("log_dir")

    forward = []
    for value, flag in (
        (args.kpf_data_input, "--kpf_data_input"),
        (args.kpf_masters_output, "--kpf_masters_output"),
        (args.kpf_science_output, "--kpf_science_output"),
        (args.log_dir, "--log_dir"),
        (args.log_level, "--log_level"),
    ):
        if value:
            forward += [flag, value]

    obs_ids = args.obs_id_list
    print(f"reducing {len(obs_ids)} science frame(s): {', '.join(obs_ids)}")

    tasks = [_cli_task(o, forward, config=args.config) for o in obs_ids]
    failed = run_stage(
        "science",
        tasks,
        args.jobs,
        log_dir,
        job_timeout=args.job_timeout,
        abort_on_failure=False,
        launch_interval=_LAUNCH_INTERVAL,
    )

    reduced = len(obs_ids) - len(failed)
    print(f"\ndone: reduced {reduced}/{len(obs_ids)} frame(s)")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
