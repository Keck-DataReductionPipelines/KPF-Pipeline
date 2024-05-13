#!/usr/bin/env python3

import os
import time
import argparse
from datetime import datetime
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

def plot_data(db_path, interval, wait_time):
    if interval in ['day', 'month', 'year', 'decade']:
        date = datetime.now()
        date_str = date.strftime('%Y%m%d')
        savedir = f'/data/QLP/{date_str}/Masters/'
        myTS = AnalyzeTimeSeries(db_path=db_path)
        myTS.plot_all_quicklook(datetime(2024, 1, 1), interval=interval, fig_dir=savedir)
    elif interval.startswith('last'):
        savedir = f'/data/QLP/{interval}/Masters/'
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
