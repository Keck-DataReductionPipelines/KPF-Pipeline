"""
KPF pipeline CLI entry point.

Usage:
    kpfpipe -r recipes/kpf_drp_masters.py -c configs/kpf_drp_masters.toml -d 20240405
"""
import argparse
import importlib.util
import os

from kpfpipe.utils.config import ConfigHandler


_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTDATA_DIR = os.path.join(_REPO_ROOT, 'tests', 'testdata')


def main():
    parser = argparse.ArgumentParser(
        prog='kpfpipe',
        description='KPF Data Reduction Pipeline',
    )
    parser.add_argument('-r', '--recipe',   required=True, help='path to recipe .py file')
    parser.add_argument('-c', '--config',   required=True, help='path to TOML config file')
    parser.add_argument('-d', '--datecode', required=True, help='datecode, e.g. 20240405')
    parser.add_argument('--data-input',  default=None, help='override KPF_DATA_INPUT directory')
    parser.add_argument('--data-output', default=None, help='override KPF_DATA_OUTPUT directory')
    parser.add_argument('--test', action='store_true',
                        help='use tests/testdata/ for input and output')
    args = parser.parse_args()

    if args.test:
        args.data_input  = args.data_input  or _TESTDATA_DIR
        args.data_output = args.data_output or _TESTDATA_DIR

    overrides = {}
    if args.data_input or args.data_output:
        overrides['DATA_DIRS'] = {}
        if args.data_input:
            overrides['DATA_DIRS']['KPF_DATA_INPUT'] = args.data_input
        if args.data_output:
            overrides['DATA_DIRS']['KPF_DATA_OUTPUT'] = args.data_output

    config = ConfigHandler(args.config, overrides=overrides or None)

    spec = importlib.util.spec_from_file_location('recipe', args.recipe)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    recipe.main(config, args)
