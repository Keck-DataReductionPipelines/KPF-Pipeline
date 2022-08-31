#!/usr/bin/env python


import sys
import os
from glob import glob
from copy import copy
import argparse
import traceback
from datetime import datetime
import time
import threading
from multiprocessing import Process, cpu_count

from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver, PollingObserverVFS
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
    parser.add_argument('--watch', dest='watch', type=str, default=None,
                        help="Watch for new data arriving in a directory and run the recipe and config on each file.")
    parser.add_argument('-r', '--recipe', required=True, dest='recipe', type=str, help="Recipe file with list of actions to take.")
    parser.add_argument('-c', '--config', required=True, dest="config_file", type=str, help="Configuration file")
    parser.add_argument('--date', dest='date', type=str, default=None,
                        help="Date for the data to be processed.")

    args = parser.parse_args(in_args[1:])

    return args

class FileAlarm(PatternMatchingEventHandler):
    def __init__(self, framework, arg, patterns=["*"], cooldown=1):
        PatternMatchingEventHandler.__init__(self, patterns=patterns,
                                             ignore_patterns=['*/.*', '*/*~'])
        
        self.framework = framework
        self.arg = arg
        self.logging = framework.pipeline.logger
        self.cooldown = 0.5

        self.file_cache = {}

    def check_redundant(self, event):
        # ignore multiple triggers that happen within 1 second of each other
        seconds = int(time.time())
        # key = (seconds, event.src_path)
        key = event.src_path
        if key in self.file_cache:
            last_update = self.file_cache[key]
            if time.time() - last_update < 2 * self.cooldown:
                print("Ignoring duplicate file event: {}".format(event.src_path))
                return False

        self.file_cache[key] = time.time()
        return True            

    def process(self, event):
        if os.path.basename(event.src_path).startswith('.'):
            final_file = os.path.dirname(event.src_path) + "/" + \
                '.'.join(os.path.basename(event.src_path).split('.')[1:-1])
            while not os.path.exists(final_file):
                print("Temporary rsync file detected. Waiting for transfer of {} to complete.".format(final_file))
                time.sleep(1)
            self.arg.file_path = final_file
        else:
            self.arg.file_path = event.src_path
        print("Executing recipe with file_path={}".format(self.arg.file_path))

        self.arg.date_dir = os.path.basename(os.path.dirname(self.arg.file_path))
        if self.arg.file_path.endswith('.fits') and self.check_redundant(event):
            self.framework.append_event('next_file', self.arg)

    def on_modified(self, event):
        print("File modification event: {}".format(event.src_path))
        self.process(event)

    def on_moved(self, event):
        print("File move event: {}".format(event.src_path))
        self.process(event)

    def on_created(self, event):
        print("File creation event: {}".format(event.src_path))
        self.process(event)

    def on_deleted(self, event):
        print("File removal event: {}".format(event.src_path))

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
    datestr = datetime.now().strftime(format='%Y%m%d')

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
    arg = Arguments(name='action_args')
    arg.recipe = recipe
    # watch mode
    if args.watch != None:
        framework.start_action_loop()
        framework.pipeline.logger.info("Waiting for files to appear in {}".format(args.watch))
        framework.pipeline.logger.info("Getting existing file list.")
        infiles = sorted(glob(args.watch + "*.fits"), reverse=True) + \
                    sorted(glob(args.watch + "20*/*.fits"), reverse=True)
        framework.pipeline.logger.info("Found {:d} files to process.".format(len(infiles)))

        frameworks = []
        for fname in infiles:
            fm = Framework(pipe, framework_config)
            fm.pipeline.start(pipe_config)
            frameworks.append(fm)
            fm.start_action_loop()

            arg = arg
            arg.date_dir = datestr
            arg.file_path = fname
            arg.watch = True
            fm.append_event('next_file', arg)

        observer = PollingObserver(framework.config.monitor_interval)
        al = FileAlarm(framework, arg, patterns=[args.watch+"*.fits*",
                                                 args.watch+"20*/*.fits"])
        observer.schedule(al, path=args.watch, recursive=True)
        observer.start()

        while True:
            time.sleep(300)
    else:
        arg.watch = False
        if hasattr(args, 'date') and args.date:
            datedir = args.date
            arg.date_dir = os.path.basename(datedir[0:-1] if str.endswith(datedir, "/") else datedir)
            arg.file_path = datedir
        else:
            arg.date_dir = datestr
            arg.file_path = datestr

        framework.append_event('start_recipe', arg)
        framework.append_event('exit', arg)
        framework.start()
