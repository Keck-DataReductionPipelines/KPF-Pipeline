#!/usr/bin/env python

import sys
import os
import argparse
import traceback

from keckdrpframework.core.framework import Framework
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.config.framework_config import ConfigClass

sys.path.append('..')
from kpfpipe.level0 import KPF0
from kpfpipe.level1 import KPF1
from kpfpipe.level2 import KPF2


def _parseArguments(in_args):
    description = "KPF Pipeline CLI"

    parser = argparse.ArgumentParser(description=description, prog='kpf')
    parser.add_argument('-r', dest="recipe_file", type=str, help="Recipe file with list of actions to take.")
    parser.add_argument('-c', dest="config_file", type=str, help="Run Configuration file")
    parser.add_argument("-o", "--output_dir", dest="dirname", type=str, default='./', help="Output directory")

    parser.add_argument('--pipeline', dest="pipeline_name", type=str,
                        default="KPF_pipeline", help="Name of the pipeline class")

    args = parser.parse_args(in_args[1:])

    return args


def main():
    args = _parseArguments(sys.argv)

    try:
        pipeline_name = args.pipeline_name
        config = ConfigClass(args.config_file)
    except Exception as e:
        print(e)
        args.print_help()
        sys.exit(1)

    try:
        framework = Framework(pipeline_name, config.framework_config)
    except Exception as e:
        print("Failed to initialize framework, exiting ...", e)
        traceback.print_exc()
        sys.exit(1)

    framework.config.run = config
    framework.logger.info("Framework initialized")

    for fn in config.config_files:
        basename = os.path.basename(fn).replace('.cfg', '')
        setattr(framework.config, basename, ConfigClass(fn))

    for fname, lvl in config.file_list.items():

        if lvl not in [0, 1, 2]:
            framework.logger.error("{} is an invalid data level. Specify level 0, 1, or 2".format(lvl))
            continue

        if lvl == 0:
            kpf0 = KPF0.from_fits(fname)
        else:
            kpf0 = None

        if lvl == 1:
            kpf1 = KPF1.from_fits(fname)
        else:
            kpf1 = None

        if lvl == 2:
            kpf2 = KPF2.from_fits(fname)
        else:
            kpf2 = None

        arg = Arguments(recipe=args.recipe_file, level0=kpf0, level1=kpf1, level2=kpf2)

        framework.append_event('execute_recipe', arg)

    framework.append_event('exit', arg)
    framework.start()


