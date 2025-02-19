#!/usr/bin/env python3

"""
Script Name: make_plots_kpf_tsdb.py

Description:
    This script generates standard KPF Time Series plots from the database. 
    It supports plotting data over specific intervals such as day, month, year, 
    decade, or custom ranges like the last N days. The plots are saved to 
    predefined directories for further use or analysis.

Features:
    - Creates time series plots for various intervals (day, month, year, decade).
    - Supports custom ranges like "last N days".
    - Saves plots in a structured directory format.
    - Includes configurable wait times for process orchestration.

Usage:
    Run this script with required arguments to specify the database path, 
    interval, and wait time:

        python make_plots_kpf_tsdb.py --db_path /path/to/database.db --interval <interval> --wait_time <seconds>

Options:
    --db_path       Path to the time series database file. 
                    Default: /data/time_series/kpf_ts.db
    --interval      Interval for plotting. Supported values:
                    - 'day', 'month', 'year', 'decade'
                    - 'last_<N>_days' (e.g., 'last_7_days' for the last 7 days)
    --wait_time     Time (in seconds) to wait before exiting the script.

Examples:
    1. Generate plots for the current year:
        python make_plots_kpf_tsdb.py --interval year --wait_time 10

    2. Generate plots for the last 30 days:
        python make_plots_kpf_tsdb.py --interval last_30_days --wait_time 10

    3. Generate daily plots and specify a custom database path:
        python make_plots_kpf_tsdb.py --db_path /custom/path/to/kpf_ts.db --interval day --wait_time 5
"""

import os
import time
import argparse
from datetime import datetime
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

def plot_data(db_path, interval, wait_time):
    if interval in ['day', 'month', 'year', 'decade']:
        date = datetime.now()
        date_str = date.strftime('%Y%m%d')
        savedir = f'/data/QLP/{date_str}/Time_Series/'
        myTS = AnalyzeTimeSeries(db_path=db_path)
        myTS.plot_all_quicklook(datetime(2024, 1, 1), interval=interval, fig_dir=savedir)
    elif interval.startswith('last'):
        savedir = f'/data/QLP/{interval}/Time_Series/'
        n_days = int(interval.replace('last_', '').replace('_days', '').replace('_day', ''))
        myTS = AnalyzeTimeSeries(db_path=db_path)
        myTS.plot_all_quicklook(last_n_days=n_days, fig_dir=savedir)
    time.sleep(wait_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create standard KPF Time Series plots.')
    parser.add_argument('--db_path', type=str, default='/data/time_series/kpf_ts.db',
                        help='path to database file; default = /data/time_series/kpf_ts.db')
    parser.add_argument('--interval', type=str, required=True,
                        help='Interval for plotting.')
    parser.add_argument('--wait_time', type=int, required=True,
                        help='Time to wait before exiting script.')

    args = parser.parse_args()
    plot_data(args.db_path, args.interval, args.wait_time)
