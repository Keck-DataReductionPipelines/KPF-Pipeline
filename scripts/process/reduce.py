#!/usr/bin/env python3
"""Run one recipe on one unit (the ``kpfpipe run`` leaf).

Loads the recipe, reads the config, applies CLI overrides, configures logging,
clears the unit's stale products (see ``clear_stale_outputs``), and execs the
recipe's ``main(config, args)``. It is both the in-process target of
``kpfpipe run`` and the leaf the orchestrators (``masters.py``/``science.py``/
``engineering.py``) fan out as ``python -m scripts.process.reduce`` subprocesses.

The recipe comes from exactly one of ``--masters``/``--science``/``--engineering``
(repo-relative shortcuts, usable from any cwd) or ``-r``. A shortcut also supplies
the default config, which ``-c`` overrides (``--science -c my.toml``); ``-r``
requires ``-c``. The recipe's declared ``RECIPE_TYPE`` alone decides the target
selector (an obs_id ``-o`` for science and engineering, a datecode ``-d`` for
masters), its ``RECIPE_STAGES`` the valid ``--stages``, and which stale products
are cleared. A custom recipe may omit ``RECIPE_TYPE``: it then runs with any or no
selector, but nothing is checked, gated, or cleared on its behalf.

The following pairs of invocations are equivalent:

    kpfpipe run --masters -d 20240405
    kpfpipe run -r {REPO_ROOT}/recipes/kpf_drp_masters.py \\
        -c {REPO_ROOT}/configs/kpf_drp_masters.toml -d 20240405

    kpfpipe run --science -o KP.20240405.40113.57
    kpfpipe run -r {REPO_ROOT}/recipes/kpf_drp_science.py \\
        -c {REPO_ROOT}/configs/kpf_drp_science.toml \\
        -o KP.20240405.40113.57
"""

import argparse
import glob
import importlib.util
import logging
import os
import shutil
import sys

import kpfpipe
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.header import HeaderParser
from kpfpipe.utils.io import kpf_directory, kpf_filepath, load_junk_obs_ids
from kpfpipe.utils.kpf import is_obs_id
from kpfpipe.utils.logger import setup_logging
from scripts._argparse import (
    config_parser,
    data_dirs_parser,
    logging_parser,
    resolve_dir_shortcuts,
    stages_override,
    stages_parser,
)
from scripts.process import (
    DEFAULT_ENGINEERING_CONFIG,
    DEFAULT_ENGINEERING_RECIPE,
    DEFAULT_MASTERS_CONFIG,
    DEFAULT_MASTERS_RECIPE,
    DEFAULT_SCIENCE_CONFIG,
    DEFAULT_SCIENCE_RECIPE,
)

logger = logging.getLogger("kpfpipe.cli")


def main(argv=None):
    # Before the recipe is exec'd and pulls matplotlib in: this runs headless, and an
    # unset backend resolves to macosx.
    os.environ.setdefault("MPLBACKEND", "Agg")

    parser = argparse.ArgumentParser(
        prog="kpfpipe run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[
            config_parser(),
            data_dirs_parser(science_output=True, engineering_output=True),
            logging_parser(),
            # Choices are the loaded recipe's RECIPE_STAGES, checked after parsing.
            stages_parser(),
        ],
    )
    recipe_choice = parser.add_mutually_exclusive_group(required=True)
    recipe_choice.add_argument(
        "--masters",
        dest="shortcut",
        action="store_const",
        const=(DEFAULT_MASTERS_RECIPE, DEFAULT_MASTERS_CONFIG),
        help="shorthand for the default masters recipe/config pair",
    )
    recipe_choice.add_argument(
        "--science",
        dest="shortcut",
        action="store_const",
        const=(DEFAULT_SCIENCE_RECIPE, DEFAULT_SCIENCE_CONFIG),
        help="shorthand for the default science recipe/config pair",
    )
    recipe_choice.add_argument(
        "--engineering",
        dest="shortcut",
        action="store_const",
        const=(DEFAULT_ENGINEERING_RECIPE, DEFAULT_ENGINEERING_CONFIG),
        help="shorthand for the default engineering recipe/config pair",
    )
    recipe_choice.add_argument(
        "-r",
        "--recipe",
        default=None,
        help="recipe .py to run (requires -c/--config)",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "-d",
        "--datecode",
        default=None,
        help="datecode, e.g. 20240405 (masters; mutually exclusive with -o)",
    )
    target.add_argument(
        "-o",
        "--obs_id",
        default=None,
        help="obs_id, e.g. KP.20240405.40113.57 (science, engineering; exclusive "
        "with -d)",
    )
    args = resolve_dir_shortcuts(parser.parse_args(argv))

    if args.shortcut:
        # -c overrides the shortcut's default config.
        args.recipe, config_def = args.shortcut
        args.config = args.config or config_def
    elif not args.config:
        parser.error("-r/--recipe requires -c/--config")

    if not os.path.isfile(args.recipe):
        raise SystemExit(f"Recipe file not found: {args.recipe}")
    spec = importlib.util.spec_from_file_location("recipe", args.recipe)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    if not hasattr(recipe, "main"):
        raise SystemExit(f"Recipe {args.recipe!r} has no main() function")

    # The recipe's declared type is the sole source of truth for what it reduces
    # (science/engineering one obs_id, masters one night) and which products it
    # owns. A custom recipe may declare none; it is then run as-is (warned below).
    recipe_type = getattr(recipe, "RECIPE_TYPE", None)
    if recipe_type not in (None, "science", "masters", "engineering"):
        parser.error(f"recipe {args.recipe!r} declares unknown RECIPE_TYPE")
    if recipe_type == "masters" and not args.datecode:
        parser.error("masters recipe takes -d/--datecode, not -o/--obs_id")
    if recipe_type in ("science", "engineering") and not args.obs_id:
        parser.error(f"{recipe_type} recipe takes -o/--obs_id, not -d/--datecode")
    stages = getattr(recipe, "RECIPE_STAGES", ())
    unknown = [s for s in args.stages or () if s not in stages]
    if unknown:
        parser.error(
            f"--stages {' '.join(unknown)} not in this recipe's stages: "
            f"{', '.join(stages)}"
        )

    overrides = {}
    data_dirs = {
        key: value
        for key, value in (
            ("KPF_DATA_INPUT", args.kpf_data_input),
            ("KPF_MASTERS_OUTPUT", args.kpf_masters_output),
            ("KPF_SCIENCE_OUTPUT", args.kpf_science_output),
            ("KPF_ENGINEERING_OUTPUT", args.kpf_engineering_output),
        )
        if value
    }
    if data_dirs:
        overrides["DATA_DIRS"] = data_dirs
    if args.log_dir or args.log_level:
        overrides["LOGGER"] = {}
        if args.log_dir:
            overrides["LOGGER"]["log_dir"] = args.log_dir
        if args.log_level:
            overrides["LOGGER"]["log_level"] = args.log_level
    overrides |= stages_override(args, stages)

    config = ConfigHandler(args.config, overrides=overrides or None)

    try:
        log_params = resolve_logging(config, args.recipe, args.obs_id, args.datecode)
    except ValueError as e:
        parser.error(str(e))
    # One run, one log directory: an orchestrator forwards its run id to pull this
    # reduction into its run; unset, setup_logging mints this run its own.
    log_path = setup_logging(run_id=args.run_id, **log_params)

    # Invocation banner: the start of the DRP-RUN-08 reduction-step trail.
    logger.info("kpfpipe %s starting", kpfpipe.__version__)
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info("recipe: %s", args.recipe)
    logger.info("config: %s", args.config)
    logger.info("data dirs: %s", config.get_params(["DATA_DIRS"]))
    logger.info("log file: %s", log_path)
    if recipe_type is None:
        logger.warning(
            "recipe declares no RECIPE_TYPE: no target checks, frame gating, or "
            "stale-output clearing are applied"
        )

    # Junk and single-CCD frames never enter the science recipe.
    if recipe_type == "science":
        data_input = config.get_params(["DATA_DIRS"])["KPF_DATA_INPUT"]
        if args.obs_id in load_junk_obs_ids(data_input):
            logger.info("skipping %s: junk frame", args.obs_id)
            return
        drp = HeaderParser(kpf_filepath(args.obs_id, "L0", data_root=data_input))
        if not (drp.DRP_KEYWORDS.get("GREEN") and drp.DRP_KEYWORDS.get("RED")):
            logger.warning(
                "skipping %s: DRP does not yet support single-CCD exposures",
                args.obs_id,
            )
            return

    # Clear prior products so this run's outputs stand alone. After the load, never
    # before: a mistyped -r must not destroy a night's products on its way out.
    clear_stale_outputs(config, args, recipe)

    try:
        recipe.main(config, args)
    except Exception:
        # The one sanctioned catch-log-reraise point:
        # ensure the traceback reaches the log before the nonzero exit. The
        # SystemExit usage errors above bypass this on purpose.
        logger.critical("uncaught exception; pipeline aborted", exc_info=True)
        raise
    logger.info("pipeline completed successfully")


def resolve_logging(config, recipe_path, obs_id, datecode):
    """Resolve the [LOGGER] config into setup_logging keyword arguments.

    Derives the recipe token from the recipe filename (``kpf_drp_science.py`` ->
    ``science``) and the target token from the obs_id/datecode (or ``run`` when
    neither applies). Raises ValueError when no log directory is configured -- the
    log location must be explicit (DRP-RUN-07), never a hidden default.

    Parameters
    ----------
    config : kpfpipe.utils.config.ConfigHandler
        The loaded pipeline config (CLI overrides already applied).
    recipe_path : str
        Path to the recipe file.
    obs_id, datecode : str or None
        The reduction target selectors from the command line.

    Returns
    -------
    dict
        Keyword arguments for ``setup_logging``.
    """
    params = config.get_params(["LOGGER"])
    log_dir = params.get("log_dir")
    if not log_dir:
        raise ValueError(
            "no log directory configured: set [LOGGER] log_dir in the "
            "config file or pass --log_dir"
        )
    recipe_name = os.path.splitext(os.path.basename(recipe_path))[0]
    recipe_name = recipe_name.removeprefix("kpf_drp_")
    return {
        "log_dir": log_dir,
        "recipe_name": recipe_name,
        "target": obs_id or datecode or "run",
        "level": params.get("log_level", "INFO"),
        "console": params.get("console", True),
    }


# Glob patterns for one masters stage's products in the night's directory (KOAID
# prefix wildcarded): the FITS masters, and the order trace's CSV. The WLS L2
# master's per-frame ThAr L2s and diagnostics HDF5 live in a thar_L2/ subdirectory
# that clear_stale_outputs removes wholesale.
_MASTER_OUTPUT_GLOBS = ("*_master_{stage}_L*.fits", "*_master_{stage}.csv")


def clear_stale_outputs(config, args, recipe):
    """Delete the recipe's existing products for this unit before it is reduced.

    Re-reducing should overwrite, never silently co-exist with, a prior run's
    products. Some are not always rewritten (the science recipe leaves L1 in
    memory), and a master's filename is keyed on its stack's first frame, so a
    changed input set can orphan a stale product under a name the rebuild never
    reuses -- clearing up front guarantees the run's results stand alone.

    Keyed on ``recipe.RECIPE_TYPE``. Science (``-o obs_id``): the three per-obs_id
    L1/L2/L4 paths under ``KPF_SCIENCE_OUTPUT``. Masters (``-d datecode``): the
    master products in the night's directory under ``KPF_MASTERS_OUTPUT`` (KOAID
    prefix wildcarded) of the earliest ``RECIPE_STAGES`` stage enabled in config
    [STAGES] and every stage after
    it, since each is built from the ones before; plus, with the thar stage, the
    whole WLS ``thar_L2/`` subdirectory (per-frame ThAr L2s and the diagnostics
    HDF5), so a rebuilt thar master can never co-exist with a prior run's frame
    L2s or diagnostics. Earlier stages' masters, which a staged batch run builds
    on, are kept. Engineering (``-o obs_id``): the obs_id's ``kpf_EngL2`` under
    ``KPF_ENGINEERING_OUTPUT``. A recipe declaring no ``RECIPE_TYPE`` owns nothing
    here, so nothing is cleared. A no-op when nothing matches; raises KeyError
    if the relevant output root is absent from [DATA_DIRS] (the recipe requires
    the same key, so failing here surfaces a broken config one step earlier).
    """
    recipe_type = getattr(recipe, "RECIPE_TYPE", None)
    data_dirs = config.get_params(["DATA_DIRS"])
    paths = []
    if recipe_type == "science" and is_obs_id(args.obs_id):
        data_root = data_dirs["KPF_SCIENCE_OUTPUT"]
        paths = [
            kpf_filepath(args.obs_id, level, data_root=data_root)
            for level in ("L1", "L2", "L4")
        ]
    elif recipe_type == "engineering" and is_obs_id(args.obs_id):
        data_root = data_dirs["KPF_ENGINEERING_OUTPUT"]
        paths = [kpf_filepath(args.obs_id, "L2", data_root=data_root, engineering=True)]
    elif recipe_type == "masters":
        data_root = data_dirs["KPF_MASTERS_OUTPUT"]
        masters_dir = kpf_directory(
            kind="masters", data_root=data_root, datecode=args.datecode
        )
        all_stages = recipe.RECIPE_STAGES
        enabled = [s for s in all_stages if config.get_params(["STAGES"])[s]]
        stages = all_stages[all_stages.index(enabled[0]) :] if enabled else ()
        for stage in stages:
            for pattern in _MASTER_OUTPUT_GLOBS:
                paths += glob.glob(
                    os.path.join(masters_dir, pattern.format(stage=stage))
                )
        stack_subdir = kpf_directory(
            kind="cal_stack",
            data_root=data_root,
            level="L2",
            datecode=args.datecode,
            cal_type="thar",
        )
        if "thar" in stages and os.path.isdir(stack_subdir):
            shutil.rmtree(stack_subdir)
            logger.info("removed stale output dir: %s", stack_subdir)

    for path in paths:
        if os.path.isfile(path):
            os.remove(path)
            logger.info("removed stale output: %s", path)


if __name__ == "__main__":
    main()
