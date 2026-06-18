"""
KPF Pipeline CLI entry point.

Recipe + config must be specified via --masters, --science, or an explicit
-r/-c pair. The shortcuts resolve repo-relative, so they work from any cwd.

When running science recipes, provide an obs_id.
When running masters recipes, provide a datecode.

The following pairs of invocations are equivalent:

    kpfpipe --masters -d 20240405
    kpfpipe -r {REPO_ROOT}/recipes/kpf_drp_masters.py -c {REPO_ROOT}/configs/kpf_drp_masters.toml -d 20240405

    kpfpipe --science -o KP.20240405.40113.57
    kpfpipe -r {REPO_ROOT}/recipes/kpf_drp_science.py  -c {REPO_ROOT}/configs/kpf_drp_science.toml  -o KP.20240405.40113.57
"""
import argparse
import importlib.util
import os

from kpfpipe.utils.config import ConfigHandler


_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTDATA_DIR = os.path.join(_REPO_ROOT, "tests", "testdata")

_SHORTCUTS = {
    "masters": ("recipes/kpf_drp_masters.py", "configs/kpf_drp_masters.toml"),
    "science": ("recipes/kpf_drp_science.py", "configs/kpf_drp_science.toml"),
}


def main():
    parser = argparse.ArgumentParser(
        prog="kpfpipe",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-r", "--recipe",   default=None, help="path to recipe.py file")
    parser.add_argument("-c", "--config",   default=None, help="path to TOML config file")
    shortcut = parser.add_mutually_exclusive_group()
    shortcut.add_argument("--masters", dest="shortcut", action="store_const", const="masters",
                          help="shorthand for default -r/-c masters pair")
    shortcut.add_argument("--science", dest="shortcut", action="store_const", const="science",
                          help="shorthand for default -r/-c science pair")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("-d", "--datecode", default=None, help="datecode, e.g. 20240405 (masters recipe only; mutually exclusive with -o)")
    target.add_argument("-o", "--obs_id",   default=None, help="obs_id, e.g. KP.20240405.40113.57 (science recipe only; mutually exclusive with -d)")
    parser.add_argument("--data_input",  default=None, help="override KPF_DATA_INPUT directory")
    parser.add_argument("--data_output", default=None, help="override KPF_DATA_OUTPUT directory")
    parser.add_argument("--test", action="store_true",
                        help="shorthand to use tests/testdata/ for input and output")
    args = parser.parse_args()

    if args.shortcut:
        if args.recipe or args.config:
            parser.error("--masters/--science cannot be combined with -r/--recipe or -c/--config")
        recipe_rel, config_rel = _SHORTCUTS[args.shortcut]
        args.recipe = os.path.join(_REPO_ROOT, recipe_rel)
        args.config = os.path.join(_REPO_ROOT, config_rel)

    if not args.recipe or not args.config:
        parser.error("must specify --masters, --science, or both -r/--recipe and -c/--config")

    recipe_kind = {os.path.basename(r): k for k, (r, _) in _SHORTCUTS.items()}.get(
        os.path.basename(args.recipe)
    )
    if recipe_kind == "masters" and args.obs_id:
        parser.error("masters recipe takes -d/--datecode, not -o/--obs_id")
    if recipe_kind == "science" and args.datecode:
        parser.error("science recipe takes -o/--obs_id, not -d/--datecode")

    if args.test:
        args.data_input  = args.data_input  or _TESTDATA_DIR
        args.data_output = args.data_output or _TESTDATA_DIR

    overrides = {}
    if args.data_input or args.data_output:
        overrides["DATA_DIRS"] = {}
        if args.data_input:
            overrides["DATA_DIRS"]["KPF_DATA_INPUT"] = args.data_input
        if args.data_output:
            overrides["DATA_DIRS"]["KPF_DATA_OUTPUT"] = args.data_output

    config = ConfigHandler(args.config, overrides=overrides or None)

    if not os.path.isfile(args.recipe):
        raise SystemExit(f"Recipe file not found: {args.recipe}")

    spec = importlib.util.spec_from_file_location("recipe", args.recipe)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    if not hasattr(recipe, "main"):
        raise SystemExit(f"Recipe {args.recipe!r} has no main() function")
    recipe.main(config, args)
