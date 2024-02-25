#!/usr/bin/env python3

import time
import argparse
from threading import Thread
from datetime import datetime, timedelta
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

def schedule_task(interval, time_range_type, date_range, thread_name, db_path):
    """
    Schedules the plot generation task to run after an initial delay and then at specified intervals,
    allowing for different arguments for each task.
    
    Args:
        initial_delay [int]: Initial delay in seconds before the task is first executed.
        interval [int]: Interval in minutes between task executions.
        time_range_type [str]: one of: 'day', 'month', 'year', 'decade', 'all'
        date_range: one of 'this_day', 'this_month', 'this_year', 'this_decade', 
                           'last_10_days', (start_date, end_date)
                           where (start_date, end_date) is a tuple of datetime objects 
                           or the string 'today'
    """
    print(f'Starting {thread_name}')
    initial_date_range = date_range

    while True:
        date_range = initial_date_range
        start_time = time.time()
        now = datetime.now()
        if date_range == 'this_day':
            kwargs = {
                "start_date":      (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0),
                "end_date":        (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0),
                "time_range_type": time_range_type
                }
        elif date_range == 'this_month':
            kwargs = {
                "start_date":      (now - timedelta(days=31)).replace(day=1, hour=0, minute=0, second=0, microsecond=0),
                "end_date":        (now + timedelta(days=31)).replace(day=1, hour=0, minute=0, second=0, microsecond=0),
                "time_range_type": time_range_type
                }
        elif date_range == 'this_year':
            kwargs = {
                "start_date":      (now - timedelta(days=365)).replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0),
                "end_date":        (now + timedelta(days=365)).replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0),
                "time_range_type": time_range_type
                }
        elif date_range == 'last_10_days':
            # need to determine where to store the results from this
            pass
        else:
            tomorrow   = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            next_month = (now + timedelta(days=28)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            next_year  = (now + timedelta(days=365)).replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            if (time_range_type == 'day') and (date_range[1] > tomorrow):
                date_range = (date_range[0], tomorrow)
            kwargs = {
                "start_date":      date_range[0],
                "end_date":        date_range[1],
                "time_range_type": time_range_type
                }

        start_time = time.time()
        generate_plots(kwargs, db_path=db_path)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'\Finished pass through {thread_name} in ' + str(execution_time) + ' seconds.\n')
        sleep_time = interval - execution_time
        if sleep_time > 0:
            time.sleep(sleep_time)

def generate_plots(kwargs, db_path='/data/time_series/kpf_ts.db'):
    myTS = AnalyzeTimeSeries(db_path=db_path)
    myTS.plot_all_quicklook_daterange(**kwargs)
#    myTS = None  # Clear memory (needed?)

def monitor_threads(threads, sleep_time):
    time.sleep(10)
    while True:
        print("\n------ Thread Status ------")
        for thread in threads:
            print(f"{thread.name}: {'Alive' if thread.is_alive() else 'Dead'} - Started at {thread.start_time}")
        print("---------------------------\n")
        time.sleep(sleep_time)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Repeatedly generate KPF time series plots.')
    parser.add_argument('--db_path', type=str, default='/data/time_series/kpf_ts.db', 
                        help='path to database file; default = /data/time_series/kpf_ts.db')
    args = parser.parse_args()   

    tasks = [
        {"thread_name": "All Days Thread",    "interval": 24*3600, "time_range_type": "day",    "date_range": (datetime(2024,  1,  1), datetime(2024,  2, 24))},
        {"thread_name": "All Months Thread",  "interval": 12*3600, "time_range_type": "month",  "date_range": (datetime(2023,  1,  1), datetime(2024,  2,  1))},
        {"thread_name": "All Years Thread",   "interval": 12*3600, "time_range_type": "year",   "date_range": (datetime(2024,  1,  1), datetime(2024,  2,  1))},
        {"thread_name": "All Decades Thread", "interval": 24*3600, "time_range_type": "decade", "date_range": (datetime(2020,  1,  1), datetime(2024,  2,  1))},
        {"thread_name": "Today Thread",       "interval":  1*3600, "time_range_type": "day",    "date_range": 'this_day'},
        {"thread_name": "This Month Thread",  "interval":  1*3600, "time_range_type": "month",  "date_range": 'this_month'},
        {"thread_name": "This Year Thread",   "interval":  3*3600, "time_range_type": "year",   "date_range": 'this_year'},
    ]

    threads = []
    for task in tasks:
        thread = Thread(target=schedule_task, args=(task["interval"], task["time_range_type"], task["date_range"], task["thread_name"], args.db_path), name=task["thread_name"])
        thread.start_time = datetime.now()  
        thread.start()  
        threads.append(thread)
        time.sleep(1)

    monitor_thread_sleep_time = 300 # seconds
    monitor_thread = Thread(target=monitor_threads, args=(threads, monitor_thread_sleep_time))
    monitor_thread.start()

    for thread in threads:
        thread.join() 

    monitor_thread.join() 
