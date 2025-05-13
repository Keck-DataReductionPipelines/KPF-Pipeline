#!/usr/bin/env python3
import argparse
import datetime
import logging
import multiprocessing
import os
import subprocess
import sys
from tqdm import tqdm

# Import the repository version
try:
    from kpfpipe import __version__
except ImportError:
    __version__ = 'unknown'


def parse_args():
    parser = argparse.ArgumentParser(description='Reprocess datecoded directories.')
    parser.add_argument('startdate', type=str, help='Start date in YYYYMMDD format')
    parser.add_argument('enddate', type=str, help='End date in YYYYMMDD format')
    parser.add_argument('--ncpu', type=int, default=max(1, multiprocessing.cpu_count() // 2),
                        help='Number of CPUs to use')
    parser.add_argument('--logfile', type=str, default='reprocess.log', help='Log file path')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing them')
    parser.add_argument('--nonice', action='store_true', help='Do not apply nice (priority) modifications')
    return parser.parse_args()


def daterange(start, end):
    current = start
    while current <= end:
        yield current
        current += datetime.timedelta(days=1)


def main():
    args = parse_args()

    log_exists = os.path.isfile(args.logfile)
    logging.basicConfig(filename=args.logfile, level=logging.INFO,
                        format='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if not log_exists:
        with open(args.logfile, 'w') as f:
            f.write(f"{'Datecode':<10}  {'Start Time':<19}  {'End Time':<19}  {'Compute Time':<15}  {'Version':<10}\n")

    start_date = datetime.datetime.strptime(args.startdate, '%Y%m%d')
    end_date = datetime.datetime.strptime(args.enddate, '%Y%m%d')

    dates = list(daterange(start_date, end_date))[::-1]  # reversed order

    nice_prefix = [] if args.nonice else ['nice', '-n', '10']

    for single_date in tqdm(dates, desc="Reprocessing Dates"):
        datecode = single_date.strftime('%Y%m%d')

        src_dir = f'/data/L0/{datecode}/'
        d2_dir = f'/data/2D/{datecode}/'
        l1_dir = f'/data/L1/{datecode}/'
        l2_dir = f'/data/L2/{datecode}/'
        qlp_dir = f'/data/QLP/{datecode}/'

        dirs_to_remove = [d2_dir, l1_dir, l2_dir, qlp_dir]

        cmds_rm = [
            ['rm', '-rf', f'{directory}*'] for directory in dirs_to_remove if os.path.exists(directory)
        ]

        cmd_kpf = [
            'kpf', '--ncpu', str(args.ncpu), '--reprocess',
            src_dir, '-c', 'configs/kpf_drp.config', '-r', 'recipes/kpf_drp.recipe'
        ]

        if args.dry_run:
            for cmd_rm in cmds_rm:
                print(' '.join(nice_prefix + cmd_rm))
            print(' '.join(nice_prefix + cmd_kpf))
        else:
            for cmd_rm in cmds_rm:
                subprocess.run(nice_prefix + cmd_rm, check=False)

            start_time = datetime.datetime.now()
            result = subprocess.run(nice_prefix + cmd_kpf)
            end_time = datetime.datetime.now()

            compute_time = end_time - start_time
            compute_time_str = str(compute_time).split('.')[0]  # remove microseconds

            if result.returncode == 0:
                logging.info(f"{datecode:<10}  {start_time.strftime('%Y-%m-%d %H:%M:%S')}  {end_time.strftime('%Y-%m-%d %H:%M:%S')}  {compute_time_str:<15}  {__version__:<10}")
            else:
                logging.info(f"{datecode:<10}  {start_time.strftime('%Y-%m-%d %H:%M:%S')}  {end_time.strftime('%Y-%m-%d %H:%M:%S')}  {compute_time_str:<15}  FAILED    {__version__:<10}")
                print(f"Error processing date {datecode}", file=sys.stderr)


if __name__ == '__main__':
    main()