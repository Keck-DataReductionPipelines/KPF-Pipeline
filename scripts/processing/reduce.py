#!/usr/bin/env python3
"""Run one recipe on one unit (the ``kpfpipe run`` leaf).

The single-recipe, single-unit runner: read the config, apply the CLI overrides,
configure logging, and exec the recipe's ``main(config, args)``. It is the leaf
the batch orchestrators (``masters.py``/``science.py``) fan out as
``python -m scripts.processing.reduce -r <recipe> -c <config>`` subprocesses, and
the in-process target of ``kpfpipe run``.

Recipe + config come from ``--masters``/``--science`` (which resolve the
repo-relative recipe/config pair, so they work from any cwd) or an explicit
``-r/-c`` pair; the shortcuts supply defaults an explicit ``-r/-c`` overrides, so
``--science -c my.toml`` runs the science recipe against a custom config. Provide
an obs_id (``-o``) for science recipes, a datecode (``-d``) for masters.

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
import importlib.util
import logging
import os
import sys

import kpfpipe
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.logger import setup_logging
from scripts.processing import (
    DEFAULT_MASTERS_CONFIG,
    DEFAULT_MASTERS_RECIPE,
    DEFAULT_SCIENCE_CONFIG,
    DEFAULT_SCIENCE_RECIPE,
)
from scripts.processing._argparse import (
    data_dirs_parser,
    logging_parser,
    recipe_parser,
)

logger = logging.getLogger("kpfpipe.cli")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="kpfpipe run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[
            recipe_parser(),
            data_dirs_parser(science_output=True),
            logging_parser(),
        ],
    )
    shortcut = parser.add_mutually_exclusive_group()
    shortcut.add_argument(
        "--masters",
        dest="shortcut",
        action="store_const",
        const=(DEFAULT_MASTERS_RECIPE, DEFAULT_MASTERS_CONFIG),
        help="shorthand for the default masters recipe/config pair",
    )
    shortcut.add_argument(
        "--science",
        dest="shortcut",
        action="store_const",
        const=(DEFAULT_SCIENCE_RECIPE, DEFAULT_SCIENCE_CONFIG),
        help="shorthand for the default science recipe/config pair",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "-d",
        "--datecode",
        default=None,
        help="datecode, e.g. 20240405 (masters only; mutually exclusive with -o)",
    )
    target.add_argument(
        "-o",
        "--obs_id",
        default=None,
        help="obs_id, e.g. KP.20240405.40113.57 (science only; exclusive with -d)",
    )
    args = parser.parse_args(argv)

    if args.shortcut:
        # The shortcut supplies a default (recipe, config) pair; an explicit
        # -r/-c overrides the corresponding half (so `--science -c foo.toml`
        # keeps the science recipe but swaps its config).
        recipe_def, config_def = args.shortcut
        args.recipe = args.recipe or recipe_def
        args.config = args.config or config_def

    if not args.recipe or not args.config:
        parser.error(
            "must specify --masters, --science, or both -r/--recipe and -c/--config"
        )

    # Guard the default recipes against the wrong target selector.
    if args.recipe == DEFAULT_MASTERS_RECIPE and args.obs_id:
        parser.error("masters recipe takes -d/--datecode, not -o/--obs_id")
    if args.recipe == DEFAULT_SCIENCE_RECIPE and args.datecode:
        parser.error("science recipe takes -o/--obs_id, not -d/--datecode")

    overrides = {}
    if args.kpf_data_input or args.kpf_masters_output or args.kpf_science_output:
        overrides["DATA_DIRS"] = {}
        if args.kpf_data_input:
            overrides["DATA_DIRS"]["KPF_DATA_INPUT"] = args.kpf_data_input
        if args.kpf_masters_output:
            overrides["DATA_DIRS"]["KPF_MASTERS_OUTPUT"] = args.kpf_masters_output
        if args.kpf_science_output:
            overrides["DATA_DIRS"]["KPF_SCIENCE_OUTPUT"] = args.kpf_science_output
    if args.log_dir or args.log_level:
        overrides["LOGGER"] = {}
        if args.log_dir:
            overrides["LOGGER"]["log_dir"] = args.log_dir
        if args.log_level:
            overrides["LOGGER"]["log_level"] = args.log_level

    config = ConfigHandler(args.config, overrides=overrides or None)

    try:
        log_params = resolve_logging(config, args.recipe, args.obs_id, args.datecode)
    except ValueError as e:
        parser.error(str(e))
    log_path = setup_logging(**log_params)

    # Invocation banner: the start of the DRP-RUN-08 reduction-step trail.
    logger.info("kpfpipe %s starting", kpfpipe.__version__)
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info("recipe: %s", args.recipe)
    logger.info("config: %s", args.config)
    logger.info("data dirs: %s", config.get_params(["DATA_DIRS"]))
    logger.info("log file: %s", log_path)

    if not os.path.isfile(args.recipe):
        raise SystemExit(f"Recipe file not found: {args.recipe}")

    spec = importlib.util.spec_from_file_location("recipe", args.recipe)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    if not hasattr(recipe, "main"):
        raise SystemExit(f"Recipe {args.recipe!r} has no main() function")

    try:
        recipe.main(config, args)
    except Exception:
        # The one sanctioned catch-log-reraise point (style guide section 6):
        # guarantee the traceback is in the log before the nonzero exit.
        # SystemExit usage errors above bypass this on purpose.
        logger.critical("uncaught exception; pipeline aborted", exc_info=True)
        raise
    logger.info("pipeline completed successfully")


def resolve_logging(config, recipe_path, obs_id, datecode):
    """Resolve the [LOGGER] config into setup_logging keyword arguments.

    Derives the recipe token from the recipe filename (``kpf_drp_science.py``
    -> ``science``) and the target token from the obs_id/datecode (or
    ``run`` when neither applies). Raises ValueError when no log directory
    is configured -- the log location must be explicit (DRP-RUN-07), never
    a hidden default.

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


if __name__ == "__main__":
    main()
