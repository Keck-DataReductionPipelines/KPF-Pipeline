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
import logging
from multiprocessing import Process, cpu_count

from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver, PollingObserverVFS
from watchdog.events import LoggingEventHandler, PatternMatchingEventHandler

from keckdrpframework.core.framework import Framework
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.utils.drpf_logger import getLogger
from keckdrpframework.config.framework_config import ConfigClass

from kpfpipe.pipelines.kpfpipeline import KPFPipeline
from kpfpipe.logger import start_logger

# This is the default framework configuration file path
framework_config = 'configs/framework.cfg'
framework_logcfg= 'configs/framework_logger.cfg'
pipeline_logcfg = 'configs/logger.cfg'

update_lock = threading.Lock()

def _parseArguments(in_args: list) -> argparse.Namespace:
    description = "KPF Pipeline CLI"

    parser = argparse.ArgumentParser(description=description, prog='kpf')
    parser.add_argument('--watch', dest='watch', type=str, default=None,
                        help="Watch for new data arriving in a directory and run the recipe and config on each file.")
    parser.add_argument('--reprocess', dest='reprocess', action='store_true',
                        help="For use in watch mode. Process any existing files found in the watch mode path.")
    parser.add_argument('-r', '--recipe', required=True, dest='recipe', type=str, help="Recipe file with list of actions to take.")
    parser.add_argument('-c', '--config', required=True, dest="config_file", type=str, help="Configuration file")
    parser.add_argument('--date', dest='date', type=str, default=None, help="Date for the data to be processed.")
    parser.add_argument('-n', '--ncpus', dest='ncpus', type=int, default=1, help="Number of CPU cores to utilize.")


    args = parser.parse_args(in_args[1:])

    return args


def worker(worker_num, pipeline_config, framework_logcfg_file, framework_config_file):
    """The worker framework instances that will execute items from the queue

    Parameters
    ----------
    worker_num : int
        Number of this worker (just nice for printing out)
    pipeline_config : ConfigClass
        Pipeline config
    framework_logcfg_file : str
        Logger config file
    framework_config_file : str
        Framework config file
    """
    # Initialize the framework however you normally do
    try:
        framework = Framework(KPFPipeline, framework_config_file)
        # logging.config.fileConfig(framework_logcfg_file)
        framework.config.instrument = pipeline_config
    except Exception as e:
        print("Failed to initialize framework, exiting ...", e)
        traceback.print_exc()
        sys.exit(1)

    # Create a logger to use for this instance
    framework.logger = start_logger(f'KPFPipe-{worker_num}', framework_logcfg_file)
    framework.logger.info("Framework initialized")

    # Start the framework. We set wait_for_event and continous to true, which
    # tells this instance to wait for something to happen, forever
    # qm_only=False, ingest_data_only=False,
    framework.pipeline.start(pipeline_config)
    framework.start(wait_for_event=True, continuous=True)


class FileAlarm(PatternMatchingEventHandler):
    def __init__(self, framework, arg, patterns=["*"], cooldown=1):
        PatternMatchingEventHandler.__init__(self, patterns=patterns,
                                             ignore_patterns=['*/.*', '*/*~'])
        
        self.framework = framework
        self.arg = arg
        self.arg.watch = True
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
                self.logging.info("Ignoring duplicate file event: {}".format(event.src_path))
                return False

        self.file_cache[key] = time.time()
        return True            

    def process(self, event):
        if os.path.basename(event.src_path).startswith('.'):
            final_file = os.path.dirname(event.src_path) + "/" + \
                '.'.join(os.path.basename(event.src_path).split('.')[1:-1])
            while not os.path.exists(final_file):
                self.logging.info("Temporary rsync file detected. Waiting for transfer of {} to complete.".format(final_file))
                time.sleep(1)
            self.arg.file_path = final_file
        else:
            self.arg.file_path = event.src_path
        self.logging.info("Executing {} with context.file_path={}".format(self.arg.recipe, self.arg.file_path))

        self.arg.date_dir = os.path.basename(os.path.dirname(self.arg.file_path))
        if self.arg.file_path.endswith('.fits') and self.check_redundant(event):
            self.framework.append_event('next_file', self.arg)

    def on_modified(self, event):
        self.logging.info("File modification event: {}".format(event.src_path))
        self.process(event)

    def on_moved(self, event):
        self.logging.info("File move event: {}".format(event.src_path))
        self.process(event)

    def on_created(self, event):
        self.logging.info("File creation event: {}".format(event.src_path))
        self.process(event)

    def on_deleted(self, event):
        self.logging.info("File removal event: {}".format(event.src_path))

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

    # Using the multiprocessing library, create the specified number of instances
    if args.watch and args.ncpus > 1:
        frame_config = 'configs/framework_multi.cfg'
        for i in range(args.ncpus):
            # This could be done with a careful use of subprocess.Popen, if that's more your style
            p = Process(target=worker, args=(i, pipe_config, framework_logcfg, frame_config))
            p.start()
    else:
        frame_config = framework_config

    # Try to initialize the framework
    try:
        framework = Framework(pipe, frame_config)
        framework.pipeline.start(pipe_config)
    except Exception as e:
        framework.pipeline.logger.error("Failed to initialize framework, exiting ...", e)
        traceback.print_exc()
        sys.exit(1)
    arg = Arguments(name='action_args')
    arg.recipe = recipe
    # watch mode


    if args.watch != None:
        framework.logger.info("Starting queue manager only, no processing")
        framework._get_queue_manager(ConfigClass(frame_config))

        framework.pipeline.logger.info("Waiting for files to appear in {}".format(args.watch))
        framework.pipeline.logger.info("Getting existing file list.")
        infiles = sorted(glob(args.watch + "*.fits"), reverse=True) + \
                    sorted(glob(args.watch + "20*/*.fits"), reverse=True)

        observer = PollingObserver(framework.config.monitor_interval)
        al = FileAlarm(framework, arg, patterns=[args.watch+"*.fits*",
                                                 args.watch+"20*/*.fits"])
        observer.schedule(al, path=args.watch, recursive=True)
        observer.start()

        if args.reprocess:
            framework.pipeline.logger.info("Found {:d} files to process.".format(len(infiles)))

            for fname in infiles:
                arg = arg
                arg.date_dir = datestr
                arg.file_path = fname
                arg.watch = True
                framework.append_event('next_file', arg)
                time.sleep(0.2)
                            
            while len(framework.get_pending_events()[0]) > 0:
                framework.pipeline.logger.debug("Waiting for event queue to clear {}".format(framework.get_pending_events()[0]))
                time.sleep(3)

            framework.append_event('exit', arg)

        if args.ncpus > 1:
            framework.start(qm_only=True)
        else:
            framework.start(wait_for_event=True, continuous=True)

    else:
        arg.watch = False
        if hasattr(args, 'date') and args.date:
            datedir = args.date
            arg.date_dir = os.path.basename(datedir[0:-1] if str.endswith(datedir, "/") else datedir)
            arg.file_path = datedir
        else:
            arg.date_dir = datestr
            arg.file_path = datestr

        framework.pipeline.start(pipe_config)
        framework.append_event('start_recipe', arg)
        framework.append_event('exit', arg)
        framework.start()
