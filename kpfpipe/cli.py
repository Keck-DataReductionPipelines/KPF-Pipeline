#!/usr/bin/env python

import sys
import os
import gc
from glob import glob
import argparse
import traceback
import random
from datetime import datetime
import time
import threading
import logging
import tempfile
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
from kpfpipe.tools.git_tools import get_git_tag, get_git_branch

# This is the default framework configuration file path
framework_config = 'configs/framework.cfg'
framework_config_multi = 'configs/framework_multi.cfg'
framework_logcfg= 'configs/framework_logger.cfg'
pipeline_logcfg = 'configs/logger.cfg'

update_lock = threading.Lock()

def _parseArguments(in_args: list) -> argparse.Namespace:
    """
    Parse command-line arguments for the KPF Pipeline CLI.

    This function sets up the argument parser, including mutually exclusive --watch and --reprocess modes,
    and returns the parsed arguments namespace. Required and optional arguments are documented in the CLI help.

    Args:
        in_args (list): List of command-line arguments (typically sys.argv).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    git_tag = get_git_tag()
    git_branch = get_git_branch()
    description = (
        f"KPF Pipeline CLI ({git_tag}, branch: {git_branch})\n"
        "A command-line interface for running the KPF data reduction pipeline in batch or watch mode.\n\n"
        "Examples:\n"
        "  kpf --watch /data/kpf/ -r my_recipe.txt -c my_config.cfg\n"
        "  kpf --reprocess /data/kpf/20240806/ -r my_recipe.txt -c my_config.cfg\n"
        "  kpf --reprocess /data/kpf/filelist.txt -r my_recipe.txt -c my_config.cfg\n"
        "  kpf --reprocess /data/kpf/singlefile.fits -r my_recipe.txt -c my_config.cfg\n"
    )

    parser = argparse.ArgumentParser(description=description, prog='kpf', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', action='version', version=f'%(prog)s {git_tag}',
                        help="Show version number and exit.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--watch', dest='watch', type=str, default=None,
                        help="Watch a directory for new FITS files and process them as they arrive. The pipeline will continue running until stopped. Default: None.")
    group.add_argument('--reprocess', dest='reprocess', type=str, default=None,
                        help="Process existing files and exit. Accepts a directory (processes all .fits files in reverse order), a single .fits file, or a text file containing a list of .fits files (processed in listed order). Mutually exclusive with --watch. Default: None.")
    parser.add_argument('--masters', dest='masters', action='store_true',
                        help="When used with --watch or --reprocess, only process files that do not contain 'WLS', 'L1', or 'L2' in the filename (useful for master calibration files). Default: False.")
    parser.add_argument('--date', dest='date', type=str, default=None, 
                        help="Date string for the data to be processed (used in non-watch mode). Default: None.")
    parser.add_argument('-n', '--ncpus', dest='ncpus', type=int, default=1, 
                        help="Number of CPU cores to use for parallel processing. Default: 1.")
    parser.add_argument('-r', '--recipe', required=True, dest='recipe', type=str, 
                        help="Path to the recipe file specifying the pipeline actions to perform. (Required)")
    parser.add_argument('-c', '--config', required=True, dest="config_file", type=str, 
                        help="Path to the pipeline configuration file. (Required)")

    args = parser.parse_args(in_args[1:])

    # If both are set, argparse will error due to mutually exclusive group
    return args


def worker(worker_num, pipeline_config, framework_logcfg_file, framework_config_file):
    """
    Worker process for running a pipeline instance.

    Each worker initializes a Framework instance and starts the pipeline, typically used for parallel processing.

    Args:
        worker_num (int): Worker index (for logging and identification).
        pipeline_config (ConfigClass): Pipeline configuration object.
        framework_logcfg_file (str): Path to the logger configuration file.
        framework_config_file (str): Path to the framework configuration file.
    """
    try:
        framework = Framework(KPFPipeline, framework_config_file)
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
    framework.pipeline.start(pipeline_config)
    framework.start()


class FileAlarm(PatternMatchingEventHandler):
    """
    Event handler for file system events in watch mode.

    This class is used with watchdog to monitor a directory for new or modified FITS files.
    When a relevant file event is detected, it triggers the pipeline to process the file.
    Duplicate or temporary file events are filtered to avoid redundant processing.
    """
    def __init__(self, framework, arg, patterns=["*"], cooldown=1):
        """
        Initialize the FileAlarm event handler.

        Args:
            framework: The pipeline framework instance.
            arg: Arguments object for the pipeline event.
            patterns (list): List of file patterns to watch.
            cooldown (float): Minimum time (seconds) between processing the same file.
        """
        PatternMatchingEventHandler.__init__(self, patterns=patterns,
                                             ignore_patterns=['*/.*', '*/*~'])

        self.framework = framework
        self.arg = arg
        self.arg.watch = True
        self.logging = framework.pipeline.logger
        self.cooldown = 0.5
        self.file_cache = {}

    def check_redundant(self, event):
        """
        Check if a file event is redundant (e.g., multiple triggers in quick succession).
        Returns True if the event should be processed, False if it should be ignored.
        """
        key = event.src_path
        if key in self.file_cache:
            last_update = self.file_cache[key]
            if time.time() - last_update < 2 * self.cooldown:
                self.logging.info("Ignoring duplicate file event: {}".format(event.src_path))
                return False
        self.file_cache[key] = time.time()
        return True

    def process(self, event):
        """
        Process a file event, triggering the pipeline if appropriate.
        Handles temporary/partial files and ensures only complete files are processed.
        """
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
        """Handle file modification events."""
        self.logging.info("File modification event: {}".format(event.src_path))
        self.process(event)

    def on_moved(self, event):
        """Handle file move events."""
        self.logging.info("File move event: {}".format(event.src_path))
        self.process(event)

    def on_created(self, event):
        """Handle file creation events."""
        self.logging.info("File creation event: {}".format(event.src_path))
        self.process(event)

    def on_deleted(self, event):
        """Handle file deletion events."""
        self.logging.info("File removal event: {}".format(event.src_path))

    def stop(self):
        """Stop the event handler and exit the process."""
        os._exit(0)


def main():
    """
    Entry point for the KPF Pipeline CLI.

    Supports two main modes:
      --watch: Watches a directory for new FITS files and processes them as they arrive. The pipeline continues running until stopped.
      --reprocess: Processes a specified set of files and exits. Accepts a directory (all .fits files in reverse order), a single .fits file, or a file containing a list of .fits files (processed in listed order).

    Use --masters to restrict processing to master calibration files (excludes files with 'WLS', 'L1', or 'L2' in the filename).

    Defaults:
      --watch: None
      --reprocess: None
      --masters: False
      --date: None
      --ncpus: 1

    See --help for more details and usage examples.
    """
    args = _parseArguments(sys.argv)
    # Set the pipeline and read the config file.
    # Using configparser for any configuration reading on the pipeline's
    # level and below.
    pipe_config = args.config_file
    pipe = KPFPipeline
    recipe = args.recipe
    datestr = datetime.now().strftime(format='%Y%m%d')

    # randomize queue manager port to avoid crosstalk between pipeline instances
    port = random.randint(50101, 60101)
    frame_config = ConfigClass(framework_config_multi)
    print(f"Setting queue manager port to {port}")
    frame_config.set('DEFAULT', 'queue_manager_portnr', str(port))
    frame_config.set('DEFAULT', 'queue_manager_auth_code', str(hash(port)))

    if args.reprocess or (hasattr(args, 'date') and args.date):
        print(f"Setting queue manager to shutdown after reprocessing.")
        frame_config.set('DEFAULT', 'event_timeout', '10')
        frame_config.set('DEFAULT', 'no_event_event', 'exit')
        frame_config.set('DEFAULT', 'no_event_wait_time', '5')
        # For downstream logic, treat as watch mode
        args.watch = True
    elif args.watch:
        frame_config.set('DEFAULT', 'event_timeout', '1200')
        frame_config.set('DEFAULT', 'no_event_event', 'Event("wait", None)')
        frame_config.set('DEFAULT', 'no_event_wait_time', '900')

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='cfg') as tp:
        frame_config.write(tp)

    frame_config = tp.name

    # Using the multiprocessing library, create the specified number of instances
    if args.watch:
        for i in range(args.ncpus):
            p = Process(target=worker, args=(i, pipe_config, framework_logcfg, frame_config))
            p.start()
    else:
        frame_config = ConfigClass(framework_config)


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


    if args.watch is not None:
        framework.pipeline.logger.info("Waiting for files to appear in {}".format(args.watch if not args.reprocess else args.reprocess))
        framework.pipeline.logger.info("Getting existing file list.")

        # Determine infiles based on reprocess or watch
        infiles = []
        if args.reprocess:
            reproc_path = args.reprocess
            if os.path.isdir(reproc_path):
                # Directory: all .fits files, reverse order
                infiles = sorted(glob(os.path.join(reproc_path, "*.fits")), reverse=True)
            elif os.path.isfile(reproc_path):
                if reproc_path.lower().endswith('.fits'):
                    # Single fits file
                    infiles = [reproc_path]
                else:
                    # Assume file list
                    with open(reproc_path, 'r') as f:
                        listed_files = [line.strip() for line in f if line.strip()]
                    for fpath in listed_files:
                        if os.path.isfile(fpath):
                            infiles.append(fpath)
                        else:
                            framework.pipeline.logger.warning(f"File listed in {reproc_path} not found: {fpath}")
            else:
                framework.pipeline.logger.error(f"--reprocess argument '{reproc_path}' is not a valid file or directory.")
                sys.exit(1)
            # If masters, filter out L1/L2
            if args.masters:
                infiles = [f for f in infiles if 'L1' not in f and 'L2' not in f]
        else:
            # Watch mode: gather files in directory
            infiles = sorted(glob(args.watch + "*.fits"), reverse=True) + \
                    sorted(glob(args.watch + "20*/*.fits"), reverse=True)
            if args.masters:
                infiles_all_fits = sorted(glob(args.watch + "*.fits"), reverse=True)
                infiles = []
                for f in infiles_all_fits:
                    if "L1" in f:
                        continue
                    if "L2" in f:
                        continue
                    infiles.append(f)

        # Only start observer in watch mode (not reprocess)
        if not args.reprocess:
            observer = PollingObserver(framework.config.monitor_interval)
            al = FileAlarm(framework, arg, patterns=[args.watch+"*.fits*",
                                                     args.watch+"20*/*.fits"])
            observer.schedule(al, path=args.watch, recursive=True)
            observer.start()

        if args.reprocess:
            framework.pipeline.logger.info("Found {:d} files to process.".format(len(infiles)))
            for fname in infiles:
                if args.masters and ('L1' in fname or 'L2' in fname):
                    framework.pipeline.logger.info("Skipping reduced file {}.".format(fname))
                    continue
                arg = arg
                arg.date_dir = datestr
                arg.file_path = fname
                arg.watch = True
                framework.append_event('next_file', arg)
                time.sleep(0.1)
            # After processing, exit
            return
        else:
            framework.start(qm_only=True)
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
        framework.start()
