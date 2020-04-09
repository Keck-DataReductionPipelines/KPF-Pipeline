#!/usr/bin/env python


import sys
import os
import argparse
import traceback
import configparser
import logging
import copy

from keckdrpframework.core.framework import Framework
from keckdrpframework.models.arguments import Arguments

from kpfpipe.pipelines.kpfpipeline import KPFPipeline
from kpfpipe.logger import start_logger

# This is the default framework configuration file path
framework_config = 'configs/framework.cfg'
framework_logcfg= 'configs/framework_logger.cfg'


def _parseArguments(in_args: list) -> argparse.Namespace:
    description = "KPF Pipeline CLI"

    parser = argparse.ArgumentParser(description=description, prog='kpf')
    parser.add_argument('recipe', type=str, help="Recipe file with list of actions to take.")
    parser.add_argument('config_file', type=str, help="Run Configuration file")

    args = parser.parse_args(in_args[1:])

    return args

def main():
    '''
    This is executed when 'kpfpipe' is called from commandline 
    '''
    args = _parseArguments(sys.argv)
    # Set the pipeline and read the config file.
    # Using configparser for any configuration reading on the pipeline's
    # level and below. 
    pipe_config = args.config_file
    pipe = KPFPipeline
    recipe = args.recipe

    # Setup a pipeline logger
    # This is to differentiate between the loggers of framework and pipeline
    # and individual modules.
    # The configs related to the logger is under the section [LOGGER]

    # Try to initialize the framework 
    try:
        framework = Framework(pipe, framework_config)
        # Overwrite the framework logger with this instance of logger
        # using framework default logger creates some obscure problem
        framework.logger = start_logger('DRPFrame', framework_logcfg)
        framework.pipeline.start(pipe_config)
    except Exception as e:
        print("Failed to initialize framework, exiting ...", e)
        traceback.print_exc()
        sys.exit(1)
    

    # python code
    arg = Arguments() # Placeholder. actual arguments are set in the pipeline
    arg.recipe = recipe
    framework.append_event('start_recipe', arg)
    framework.append_event('exit', arg)
    framework.start()


