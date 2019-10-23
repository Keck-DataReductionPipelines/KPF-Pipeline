#!/usr/bin/env python

import sys
import os
import argparse
import traceback

from keckdrpframework.core.framework import Framework
from keckdrpframework.models.arguments import Arguments

sys.path.append('..')
from kpfpipe.level0 import KPF0
from kpfpipe.level1 import KPF1
from kpfpipe.level2 import KPF2
import numpy as np

# make test data
kpf0 = KPF0()
kpf0.data['green'] = np.ones((32,32), dtype=np.float)
kpf0.data['red'] = np.ones((32,32), dtype=np.float)
kpf0.bias['green'] = np.ones((32,32), dtype=np.float)*0.4
kpf0.bias['red'] = np.ones((32,32), dtype=np.float)*0.5
kpf0.flat['green'] = np.ones((32,32), dtype=np.float)*0.2
kpf0.flat['red'] = np.ones((32,32), dtype=np.float)*0.1

kpf1 = KPF1()
kpf2 = KPF2()

def _parseArguments(in_args):
    description = "Test using the Keck DRP Framework"
    usage = "\n{} pipeline config_file [-w] [-W] [-s] [-i] [-f file [files ...]]|[-d dirname]\n".format(in_args[0])
    epilog = "\nRuns the given pipeline using the given configuration\n"

    parser = argparse.ArgumentParser(prog=f"{in_args[0]}", description=description, usage=usage, epilog=epilog)

    parser.add_argument(dest="pipeline_name", type=str, help="Name of the pipeline class")
    parser.add_argument(dest="config_file", type=str, help="Configuration file")
    # parser.add_argument(dest="infiles", help="Input files", nargs="*")

    parser.add_argument("-w", "--wait_for_event", dest="wait_for_event", action='store_true', help="Wait for events")
    parser.add_argument("-W", "--continue", dest="continuous", action='store_true',
                        help="Continue processing, wait for ever")

    parser.add_argument("-s", "--start_queue_manager_only", dest="queue_manager_only", action='store_true',
                        help="Starts queue manager only, no processing")

    parser.add_argument("-i", "--ingest_data_only", dest="ingest_data_only", action='store_true',
                        help="Ingest data and terminate")
    #
    # parser.add_argument("-d", "--directory", dest="dirname", type=str, help="Input directory")

    args = parser.parse_args(in_args[1:])
    return args


if __name__ == "__main__":

    args = _parseArguments(sys.argv)

    try:
        pipeline_name = args.pipeline_name
        config = args.config_file
    except Exception as e:
        print (e)
        args.print_help()
        sys.exit(1)

    try:
        framework = Framework(pipeline_name, config)
    except Exception as e:
        print ("Failed to initialize framework, exiting ...", e)
        traceback.print_exc()
        sys.exit(1)

    framework.logger.info("Framework initialized")

    if args.queue_manager_only:
        # The queue manager runs for ever.
        framework.logger.info("Starting queue manager only, no processing")
        framework.start_queue_manager()
    # else:
    #     print (args.infiles, args.dirname)
    #     if (len(args.infiles) > 0) or args.dirname is not None:
    #         # Ingest data and terminate
    #         framework.ingest_data(args.dirname, args.infiles)
    #
        # framework.start(args.queue_manager_only, args.ingest_data_only, args.wait_for_event, args.continuous)

    # run on the test arrays
    arg = Arguments(level0=kpf0, level1=kpf1, level2=kpf2,
                    chips=True, orders=[10,11], regions=True)

    framework.append_event('reduce_level0', arg)
    framework.append_event('reduce_level1', arg)
    framework.append_event('exit', arg)

    framework.start(args.queue_manager_only, args.ingest_data_only, args.wait_for_event, args.continuous)
