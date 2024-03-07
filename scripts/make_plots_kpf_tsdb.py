#!/usr/bin/env python3
#!/usr/bin/env python3

import os
import time
import threading
import argparse
from datetime import datetime
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

def plot_data(db_path, interval, wait_time):
    """
    Function to plot data at a specified interval and wait time.
    """
    while not stop_event.is_set():
        myTS = AnalyzeTimeSeries(db_path=db_path)
        if interval in ['day', 'month', 'year', 'decade']:
            date = datetime.now()
            date_str = date.strftime('%Y%m%d')
            savedir = f'/data/QLP/{date_str}/Masters/'
            myTS.plot_all_quicklook(datetime(2024, 1, 1), interval=interval, fig_dir=savedir)
        else:
            savedir = f'/data/QLP/{interval}/Masters/'
            n_days = int(last_n_days.replace('last_', '').replace('_day', '').replace('_days', ''))

            myTS.plot_all_quicklook(last_n_days=n_days, fig_dir=savedir)
        time.sleep(wait_time)  # Wait for the specified time before running again

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create standard KPF Time Series plots on timers.')
    parser.add_argument('--db_path', type=str, default='/data/time_series/kpf_ts.db',
                        help='path to database file; default = /data/time_series/kpf_ts.db')
    args = parser.parse_args()

    stop_event = threading.Event()

    # Define intervals and wait times for each thread
    intervals =  ['day', 'month', 'year', 'decade', 'last_1_day', 'last_3_days', 'last_10_days', 'last_30_days', 'last_100_days']
    wait_times = [3600*4, 3600*4, 3600*4, 3600*24, 600, 600, 600, 600, 600]  # seconds

    # Create and start threads
    threads = []
    for interval, wait_time in zip(intervals, wait_times):
        thread = threading.Thread(target=plot_data, args=(args.db_path, interval, wait_time))
        threads.append(thread)
        thread.start()

    try:
        # Keep the script running until a keyboard interrupt is received
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        print("\nStopping threads... (wait up to 30 seconds for threads to exit)")

    # Wait for all threads to finish
    for thread in threads:
        thread.join(timeout=30)
