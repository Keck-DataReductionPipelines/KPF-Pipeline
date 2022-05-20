#!/usr/bin/env python


import sys
import os
import argparse
import traceback
import configparser
import logging
import copy
import time
import threading
from multiprocessing import Process, cpu_count

from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, PatternMatchingEventHandler

from keckdrpframework.core.framework import Framework
from keckdrpframework.models.arguments import Arguments

from kpfpipe.pipelines.kpfpipeline import KPFPipeline
from kpfpipe.logger import start_logger

# This is the default framework configuration file path
framework_config = 'configs/framework.cfg'
framework_logcfg= 'configs/framework_logger.cfg'

update_lock = threading.Lock()

def _parseArguments(in_args: list) -> argparse.Namespace:
    description = "KPF Pipeline CLI"

    parser = argparse.ArgumentParser(description=description, prog='kpf')
    parser.add_argument('--watch', dest='watch', type=str, default=None, help="Watch for new data arriving in a directory and run the recipe and config on each file.")
    parser.add_argument('recipe', type=str, help="Recipe file with list of actions to take.")
    parser.add_argument('config_file', type=str, help="Run configuration file")

    args = parser.parse_args(in_args[1:])

    return args

class FileAlarm(PatternMatchingEventHandler):
    def __init__(self, framework, arg, patterns=["*"], cooldown=1):
        PatternMatchingEventHandler.__init__(self, patterns=patterns,
                                             ignore_patterns=['*/.*', '*/*~'])
        self.framework = framework
        self.arg = arg
        self.logging = framework.pipeline.logger

        self.file_cache = {}

    def check_redundant(self, event):
        # ignore multiple triggers that happen within 1 second of each other
        seconds = int(time.time())
        # key = (seconds, event.src_path)
        key = event.src_path
        if key in self.file_cache:
            last_update = self.file_cache[key]
            if time.time() - last_update < 2 * self.cooldown:
                logging.debug("Ignoring duplicate file event: {}".format(event.src_path))
                return False

        self.file_cache[key] = time.time()
        return True            

    def process(self, event):
        if os.path.basename(event.src_path).startswith('.'):
            final_file = '.'.join(os.path.basename(event.src_path).split('.')[1:-1])
            logging.debug("Temporary rsync file detected. Waiting for transfer of {} to complete.".format(final_file))
            while not os.path.exists(final_file):
                time.sleep(1)
            os.environ['INPUT_FILE'] = final_file
        else:
            os.environ['INPUT_FILE'] = event.src_path
        logging.debug("Executing recipe with INPUT_FILE={}".format(os.environ['INPUT_FILE']))
        os.environ['DATE_DIR'] = os.path.basename(os.path.dirname(os.environ['INPUT_FILE']))
        if os.environ['INPUT_FILE'].endswith('.fits') and self.check_redundant(event):
            self.framework.append_event('start_recipe', self.arg)

    def on_modified(self, event):
        logging.debug("File modification event: {}".format(event.src_path))
        self.process(event)

    def on_moved(self, event):
        logging.debug("File move event: {}".format(event.src_path))
        self.process(event)

    def on_created(self, event):
        logging.debug("File creation event: {}".format(event.src_path))
        self.process(event)

    def on_deleted(self, event):
        logging.debug("File removal event: {}".format(event.src_path))

    def stop(self):
        os._exit(0)


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
        framework.pipeline.start(pipe_config)

        # root = logging.getLogger()
        # map(root.removeHandler, root.handlers[:])
        # map(root.removeFilter, root.filters[:])

        # Overwrite the framework logger with this instance of logger
        # using framework default logger creates some obscure problem
        # framework.logger = start_logger('DRPFrame', framework_logcfg)

    except Exception as e:
        framework.pipeline.logger.error("Failed to initialize framework, exiting ...", e)
        traceback.print_exc()
        sys.exit(1)
    
    arg = Arguments() # Placeholder. actual arguments are set in the pipeline
    arg.recipe = recipe

    # watch mode
    if args.watch != None:
        framework.pipeline.logger.info("Waiting for files to appear in {}".format(args.watch))
        observer = Observer()
        al = FileAlarm(framework, arg, patterns=[args.watch+"*.fits*"])
        observer.schedule(al, path=args.watch, recursive=True)
        observer.start()

        framework.start_action_loop()
        framework.wait_for_ever()

    else:
        framework.append_event('start_recipe', arg)
        framework.append_event('exit', arg)
        framework.start()
