#!/usr/bin/env python


import sys
import os
import argparse
import traceback
import configparser
import logging
import copy

## This is an ugly fix for local package dependencies 
# --TODO-- find a better way to do this 
sys.path.insert(0, os.path.abspath('../KeckDRPFramework'))


from keckdrpframework.core.framework import Framework
from keckdrpframework.models.arguments import Arguments

from kpfpipe.pipelines.kpfpipeline import KPFPipeline
from kpfpipe.logger import start_logger

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
    pipe_config = configparser.ConfigParser()
    res = pipe_config.read(args.config_file)
    # res is a list containing the name of the configfile
    # res is empty if config failed to read
    if len(res) == 0:
        raise IOError('failed to read config file {}'.format(args.config_file))
    pipe = KPFPipeline
    recipe = args.recipe

    # Setup a pipeline logger
    # This is to differentiate between the loggers of framework and pipeline
    # and individual modules.
    # The configs related to the logger is under the section [LOGGER]

    # Try to initialize the framework 
    try:
        framework_config  = pipe_config.get('FRAMEWORK', 'config_path')
        framework_logcfg = pipe_config.get('FRAMEWORK', 'log_config')

        framework = Framework(pipe, framework_config)
        # Overwrite the framework logger with this instance of logger
        # using framework default logger creates some obscure problem
        log_cfg = configparser.ConfigParser()
        res = log_cfg.read(framework_logcfg)

        framework.logger = start_logger('DRPFrame', log_cfg['LOGGER'])
        framework.pipeline.start(pipe_config)
    except Exception as e:
        print("Failed to initialize framework, exiting ...", e)
        traceback.print_exc()
        sys.exit(1)
    

    # python code
    arg = Arguments() # Placeholder. actual arguments are set in the pipeline
    arg.recipe = recipe
    framework.append_event('evaluate_recipe', arg)
    framework.append_event('exit', arg)
    framework.start()


