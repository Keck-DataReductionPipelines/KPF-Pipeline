#!/usr/bin/env python


import sys
import os
import argparse
import traceback
import configparser
import logging

from keckdrpframework.core.framework import Framework
from keckdrpframework.models.arguments import Arguments

from kpfpipe.pipelines.TFAPipe import TFAPipeline

def 

def start_logger(pipe_name: str, log_config: dict) -> logging.Logger:

    log_path = log_config.get('log_path')
    log_lvl = log_config.get('level')
    log_verbose = log_config.getboolean('verbose')

    # basic logger instance
    self.logger = logging.getLogger(self.name)
    self.logger.setLevel(self.get_level(log_lvl))

    formatter = logging.Formatter('[%(name)s] - %(levelname)s - %(message)s')
    f_handle = logging.FileHandler(log_path, mode='w') # logging to file
    f_handle.setLevel(self.get_level(log_lvl))
    f_handle.setFormatter(formatter)
    self.logger.addHandler(f_handle)

    if log_verbose: 
        # also print to terminal 
        s_handle = logging.StreamHandler()
        s_handle.setLevel(self.get_level(log_lvl))
        s_handle.setFormatter(formatter)
        self.logger.addHandler(s_handle)
    self.running = True


class Logger: 
    '''
    A class used for logging for the KPF Pipeline
    '''
    def __init__(self, pipe_name: str, log_config: dict) -> None:
        '''
        Initializer
        '''
        # Note that one may not want to log anything. 
        # If this is the case, Logger is initalized, but will not actually run
        # This can be determined from the config file
        self.running = False
        self.name = pipe_name
        if log_config.getboolean('start_log'):
            self.start(log_config)
        
    
    def get_level(self, lvl:str) -> int:
        '''
        read the logging level (string) from config file and return 
        the corresponding logging level
        '''
        if lvl == 'debug': return logging.DEBUG
        elif lvl == 'info': return logging.INFO
        elif lvl == 'warning': return logging.WARNING
        elif lvl == 'error': return logging.ERROR
        elif lvl == 'critical': return logging.CRITICAL
        else: return logging.NOTSET

    def start(self, log_config: dict) -> None:
        '''
        starting the logging instance
        Logger is only started if specified in the config file 
        '''
        log_path = log_config.get('log_path')
        log_lvl = log_config.get('level')
        log_verbose = log_config.getboolean('verbose')

        # basic logger instance
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.get_level(log_lvl))

        formatter = logging.Formatter('[%(name)s] - %(levelname)s - %(message)s')
        f_handle = logging.FileHandler(log_path, mode='w') # logging to file
        f_handle.setLevel(self.get_level(log_lvl))
        f_handle.setFormatter(formatter)
        self.logger.addHandler(f_handle)

        if log_verbose: 
            # also print to terminal 
            s_handle = logging.StreamHandler()
            s_handle.setLevel(self.get_level(log_lvl))
            s_handle.setFormatter(formatter)
            self.logger.addHandler(s_handle)
        self.running = True
        self.log("jello", 'info')
    
    def warn(self, msg: str) -> None: 
        self.
    def log(self, msg: str, lvl: str):
        '''
        log message
        '''
        if self.running: 
            if lvl == 'debug':
                self.logger.debug(msg)
            elif lvl == 'info':
                self.logger.info(msg)
            elif lvl == 'warning':
                self.logger.warning(msg)
            elif lvl == 'error':
                self.logger.error(msg)
            elif lvl == 'critical':
                self.logger.critical(msg)
            else:
                raise ValueError('Logger does not recognize level: {}'.format(lvl))

def _parseArguments(in_args):
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
    pipe_config.read(args.config_file)
    pipe = TFAPipeline
    recipe = args.recipe

    # Setup a pipeline logger
    # This is to differentiate between the loggers of framework and pipeline
    # The configs related to the logger is under the section [LOGGER]
    print(pipe_config['LOGGER'])
    pipe_logger = Logger(pipe.name, pipe_config['LOGGER'])

    # Try to initialize the framework 
    try:
        framework_config  = pipe_config.get('FRAMEWORK', 'config_path')
        framework = Framework(pipe, framework_config)
    except Exception as e:
        print("Failed to initialize framework, exiting ...", e)
        traceback.print_exc()
        sys.exit(1)

    ## Set up argument
    # To begin, argument is set to be the absolute path of a folder
    # containing all the files that requires prcessing.
    # Note that the member of Argument() changes dynamically (UGH WHY) so 
    # as the data progress in the pipe, arg will contain different types of data
    arg = Arguments()
    # Adding a few members that keeps track of where the argument is in the pipeline. 
    arg.name = 'KPF Test run ' # --TODO-- change this accordingly
    arg.state = 'Initialized'  # This keeps track of the previous action 
    arg.recipe = recipe
    # This is the actial input to the pipeline
    # this should be a folder path 
    arg.input_path = pipe_config.get('PIPELINE', 'input')

    ## Setup the context
    # The context class should be passed down the pipeline along with the argument.
    # It contains any relevant information that primitives might need, 
    # such as config path
    # Note that context of the pipeline is already set by the framework.
    # see KeckDRPFramework/keckrpframework/core/framework.py line 89
    # Overwrite the logger member to seperate framework and pipeline logger
    # Overwrite the config member to use configparser
    framework.pipeline.logger = pipe_logger
    framework.pipeline.context.module_config = pipe_config['MODULES']
    

    # python code
    framework.append_event('evaluate_recipe', arg)

    framework.append_event('exit', arg)
    pipe_logger.log('what?', 'info')
    framework.start()


