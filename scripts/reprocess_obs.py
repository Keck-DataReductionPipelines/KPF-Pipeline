#!/usr/bin/env python3
import argparse
import datetime
import logging
import multiprocessing
import os
import subprocess
import sys
import pytz
from tqdm import tqdm
from kpfpipe.tools.git_tools import get_git_tag, get_git_branch
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries, add_one_month


def parse_args():
    parser = argparse.ArgumentParser(description='Reprocess KPF data over a date range. The results are logged in reprocess_obs.log with errors on a particular date written to <datecode>_error.log. Dates that have been successfully reprocessed with the current pipeline version (according to reprocess_obs.log) are skipped.')
    parser.add_argument('startdate', type=str, help='Start date in YYYYMMDD format')
    parser.add_argument('enddate', type=str, help='End date in YYYYMMDD format')
    parser.add_argument('--ncpu', type=int, default=max(1, multiprocessing.cpu_count() // 2),
                        help='Number of CPUs to use')
    parser.add_argument('--force', action='store_true', help='Process even if datecode/version are listed in the logfile')
    parser.add_argument('--logfile', type=str, default='reprocess_obs.log', help='Log file path')
    parser.add_argument('--forward', action='store_true', help='Process datecodes in chronological order (reverse is default)')
    parser.add_argument('--not-nice', action='store_true', help='Do not apply standard nice (=15) deprioritization')
    parser.add_argument('--delete', action='store_true', help='Delete existing 2D/L1/L2/QLP/outliers/logs/logs_QLP files before reprocessing')
    parser.add_argument('--qlp-regen', action='store_true', help='Regenerate Quicklook plots after processing')
    parser.add_argument('--only-drift', action='store_true', help='Only do drift correction (not spectral extraction)')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing them')
    parser.add_argument('--local-tz', type=str, default='America/Los_Angeles',
                        help='Local timezone for logfile lines (default: America/Los_Angeles)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed messages during execution')
    return parser.parse_args()


def daterange(start, end):
    current = start
    while current <= end:
        yield current
        current += datetime.timedelta(days=1)


def load_processed_dates(logfile, version):
    processed_dates = set()
    if os.path.isfile(logfile):
        with open(logfile, 'r') as f:
            for line in f:
                if line.startswith("Datecode") or "FAILED" in line:
                    continue
                datecode = line[:10].strip()
                log_version = line[67:77].strip()
                if log_version == version:
                    processed_dates.add(datecode)
    return processed_dates


def main():
    args = parse_args()
    git_tag = get_git_tag()
    git_branch = get_git_branch()
    local_tz = pytz.timezone(args.local_tz)

    log_exists = os.path.isfile(args.logfile)
    logging.basicConfig(filename=args.logfile, level=logging.INFO,
                        format='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if not log_exists:
        with open(args.logfile, 'w') as f:
            f.write(f"{'Datecode':<10}  {'Start Time':<19}  {'End Time':<19}  {'Run Time':<11}  {'Version':<10}\n")
        os.chmod(args.logfile, 0o666)

    start_date = datetime.datetime.strptime(args.startdate, '%Y%m%d')
    end_date = datetime.datetime.strptime(args.enddate, '%Y%m%d')

    dates = list(daterange(start_date, end_date))
    if not args.forward:
        dates = dates[::-1]  # reversed order

    valid_dates = [single_date for single_date in dates if os.path.exists(f'/data/L0/{single_date.strftime("%Y%m%d")}/')]

    if not valid_dates:
        print("No valid datecodes found in the specified range.")
        sys.exit(0)

    processed_dates = set()
    if not args.force:
        processed_dates = load_processed_dates(args.logfile, git_tag)

    nice_prefix = [] if args.not_nice else ['nice', '-n', '15']

    for single_date in tqdm(valid_dates, desc="Reprocessing Dates", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}"):
        datecode = single_date.strftime('%Y%m%d')

        if datecode in processed_dates:
            tqdm.write(f"Skipping previously processed datecode: {datecode}")
            continue

        tqdm.write(f"Reprocessing {datecode} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with {git_tag} (branch: {git_branch})")

        dirs_to_remove = [
            f'/data/2D/{datecode}/', 
            f'/data/L1/{datecode}/', 
            f'/data/L2/{datecode}/',
            f'/data/outliers/{datecode}/',
            f'/data/logs/{datecode}/', 
            f'/data/logs_QLP/{datecode}/'
        ]
        dirs_to_remove_qlp = [
            f'/data/QLP/{datecode}/2D/', 
            f'/data/QLP/{datecode}/L1/', 
            f'/data/QLP/{datecode}/L2/', 
        ]
        if args.qlp_regen:
            dirs_to_remove.extend(dirs_to_remove_qlp)

        cmd_kpf = [
            'kpf', '--ncpu', str(args.ncpu), '--reprocess', f'/data/L0/{datecode}/',
            '-c', 'configs/kpf_drp.cfg', '-r', 'recipes/kpf_drp.recipe'
        ]

        if args.dry_run:
            if args.delete:
                for directory in dirs_to_remove:
                    print(f'find {directory} -mindepth 1 -delete')
            print(' '.join(nice_prefix + cmd_kpf))
        else:
            if args.delete:
                for directory in dirs_to_remove:
                    if os.path.exists(directory):
                        result = subprocess.run(['find', directory, '-mindepth', '1', '-delete', '-print'], capture_output=True, text=True)
                        deleted_files = result.stdout.strip().split('\n') if result.stdout else []
                        if args.verbose:
                            print(f"Deleted {len(deleted_files)} files from {directory}")
            if args.delete:
                # Remove TSDB rows for this datecode.  
                # They will be added back when the data is regenerated.
                myTS = AnalyzeTimeSeries(backend='psql')
                null = myTS.db.delete_by_datecode(datecode)
                if args.verbose:
                    if 'tsdb_base' in null:
                        print(f"Deleted {null['tsdb_base']} observations from the TSDB.")

            start_time = datetime.datetime.now(local_tz)

            if not args.only_drift and args.verbose:
                print(f"Processing {datecode} with standard recipe.")
            if not args.only_drift:
                result = subprocess.run(
                    nice_prefix + cmd_kpf,
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL, 
                    text=True,
                    check=False
                )
            
            # Ingest all observations for this date (in case observations any were missed with file-event based ingestion)
            if args.verbose:
                print(f"Ingesting observations for {datecode} into TSDB.")
            myTS = AnalyzeTimeSeries(backend='psql')
            myTS.db.ingest_dates_to_db(datecode, datecode, force_ingest=True, quiet=False)

            if args.only_drift or result.returncode == 0:
                # Now do drift correction only since the initial L2s should be 
                # ingested into the TSDB by this point.
                cmd_kpf = [
                    'kpf', '--ncpu', str(args.ncpu), '--reprocess', f'/data/L1/{datecode}/',
                    '-c', 'configs/kpf_drp_do_only_drift.cfg', '-r', 'recipes/kpf_drp.recipe'
                ]
    
                if args.verbose:
                    print(f"Processing {datecode} with standard drift-correction recipe.")
                result = subprocess.run(
                    nice_prefix + cmd_kpf,
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL, 
                    text=True,
                    check=False
                )

                # Ingest all observations for this date (in case any observations were missed with file-event based ingestion)
                if args.verbose:
                    print(f"Ingesting observations for {datecode} into TSDB.")
                myTS = AnalyzeTimeSeries(backend='psql')
                myTS.db.ingest_dates_to_db(datecode, datecode, force_ingest=True, quiet=False)

            if args.delete and args.qlp_regen:
                # Regenerate time series plots for this datecode.
                # Other QLP plots should be regenerated by quicklook threads running.
                if args.verbose:
                    print(f"Regenerating Quicklook plots and yaml files for {datecode}.")
                day = datetime.datetime.strptime(datecode, "%Y%m%d")
                savedir = f'/data/QLP/{datecode}/Time_Series/'
                myTS = AnalyzeTimeSeries(backend='psql')
                # Make day plots
                myTS.plot_all_quicklook(day, interval='day', fig_dir=savedir)
                # Make month and year plots
                for plotdict in ['drptag', 'drphash']:# , 'files_missing', 'master_age']:
                    for plot_range in ['month', 'year']:
                        if plot_range == 'month':
                            start_date = datetime.datetime(day.year, day.month, 1)
                            end_date = add_one_month(start_date)
                        elif plot_range == 'year':
                            start_date = datetime.datetime(day.year, 1, 1)
                            end_date = datetime.datetime(day.year+1, 1, 1)
                        fig_path = savedir + f'kpf_{datecode}_ts_{plotdict}.png'
                        myTS.plot_time_series_multipanel(plotdict, start_date=start_date, end_date=end_date, fig_path=fig_path, clean=True)

            end_time = datetime.datetime.now(local_tz)
            compute_time = end_time - start_time
            compute_time_str = str(compute_time).split('.')[0]

            if result.returncode != 0:
                error_message = f"Error processing date {datecode}\nError details:\n{result.stderr}"
                print(error_message, file=sys.stderr)
                with open(f"{datecode}_error.log", "w") as err_log:
                    err_log.write(error_message)

            status = "" if result.returncode == 0 else " FAILED"
            logging.info(f"{datecode:<10}  {start_time.strftime('%Y-%m-%d %H:%M:%S')}  {end_time.strftime('%Y-%m-%d %H:%M:%S')}  {compute_time_str:<11}  {git_tag:<10}{status}")
            os.chmod(args.logfile, 0o666)
            
            if args.verbose:
                print(f"Processing complete for {datecode}.")


if __name__ == '__main__':
    main()
