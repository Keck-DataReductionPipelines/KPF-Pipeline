#!/usr/bin/env python3

"""
Script Name: generate_time_series_plots.py

Description:
    This script generates KPF time series plots over various time intervals and
    saves the results to predefined directories. It supports multithreaded execution
    for tasks with different intervals and date ranges, including daily, monthly,
    yearly, and decade-based plots. Additionally, the script monitors the status
    of running threads and reports on their activity.

Features:
    - Generates plots for multiple time intervals (day, month, year, decade).
    - Supports custom date ranges for plot generation.
    - Multithreaded execution for efficiency, allowing simultaneous tasks.
    - Monitors thread status and execution time for each task.
    - Saves results in a structured format for further analysis.

Usage:
    Run this script with optional arguments to specify the database path:

        python generate_time_series_plots.py --db_path /path/to/database.db

Options:
    --db_path   Path to the time series database file. Default: /data/time_series/kpf_ts.db

Examples:
    1. Using the default database path:
        python generate_time_series_plots.py

    2. Specifying a custom database path:
        python generate_time_series_plots.py --db_path /custom/path/to/kpf_ts.db

    3. Monitoring thread statuses:
        The script automatically reports on thread activity and execution times every 5 minutes.
"""

import time
import argparse
from threading import Thread
from datetime import datetime, timedelta
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

def schedule_task(interval, time_range_type, date_range, thread_name, db_path):
    """
    Schedules the plot generation task to run after an initial delay and then 
    at specified intervals, allowing for different arguments for each task.
    
    Args:
        initial_delay [int]: Initial delay in seconds before the task is first executed.
        interval [int]: Interval in minutes between task executions.
        time_range_type [str]: one of: 'day', 'month', 'year', 'decade', 'all'
        date_range: one of 'this_day', 'this_month', 'this_year', 'this_decade', 
                           'last_10_days', (start_date, end_date)
                           where (start_date, end_date) is a tuple of datetime objects 
                           or the string 'today'
    """
    print(f"Starting: {thread_name} to be executed every {interval/3600} hours.")
    initial_date_range = date_range

    while True:
        date_range = initial_date_range
        start_time = time.time()
        now = datetime.now()
        if date_range == 'all_days':
            start_date = None
            end_date   = None # (now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
            time_range_type =  time_range_type
        elif date_range == 'all_months':
            start_date = None
            end_date   = None # (now - timedelta(days=62)).replace(hour=0, minute=0, second=0, microsecond=0)
            time_range_type =  time_range_type
        elif date_range == 'all_years':
            start_date = None
            end_date   = None # (now - timedelta(days=366)).replace(hour=0, minute=0, second=0, microsecond=0)
            time_range_type =  time_range_type
        elif date_range == 'this_day':
            start_date = (now - timedelta(days=15)).replace(hour=0, minute=0, second=0, microsecond=0)
            start_date = (now - timedelta(days=1000)).replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date   =  now
            time_range_type =  time_range_type
        elif date_range == 'this_month':
            start_date = (now - timedelta(days=31)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            start_date = (now - timedelta(days=1000)).replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date   = now
            time_range_type = time_range_type
        elif date_range == 'this_year':
            start_date = (now - timedelta(days=366)).replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            start_date = (now - timedelta(days=1000)).replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date   =  now
            time_range_type = time_range_type
        elif date_range == 'last_10_days':
            # need to determine where to store the results from this so that it doesn't crash Jump
            pass
        else:
            start_date = date_range[0]
            end_date   = date_range[1]
            time_range_type = time_range_type

        start_time = time.time()
        generate_plots(start_date=start_date, end_date=end_date, time_range_type=time_range_type, db_path=db_path)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Finished pass through {thread_name} in ' + str(int(execution_time)) + ' seconds.\n')
        sleep_time = interval - execution_time
        if sleep_time > 0:
            time.sleep(sleep_time)

def generate_plots(start_date=None, end_date=None, 
                   time_range_type='all', clean=True, 
                   db_path='/data/time_series/kpf_ts.db',
                   base_dir='/data/QLP/'):
    """
    Generate all of the standard time series plots for the quicklook for a 
    date range.  Every unique day, month, year, and decade between 
    start_date and end_date will have a full set of plots produced using 
    plot_all_quicklook(). The set of date range types ('day', 'month', 
    'year', 'decade', 'all') is set by the time_range_type parameter.

    Args:
        start_date (datetime object) - start date for plot
        end_date (datetime object) - start date for plot
        time_range_type (string)- one of: 'day', 'month', 'year', 'decade', 'all'
        base_dir (string) - set to the path for the files to be generated.

    Returns:
        PNG plots in the output directory.
    """
    
    if start_date == None or end_date == None:
        myTS = AnalyzeTimeSeries(db_path=db_path)
        first_last_dates = myTS.get_first_last_dates()
        if start_date == None:
            start_date = first_last_dates[0]
        if end_date == None:
            end_date = first_last_dates[1]
    
    time_range_type = time_range_type.lower()
    if time_range_type not in ['day', 'month', 'year', 'decade', 'all']:
        time_range_type = 'all'

    days = []
    months = []
    years = []
    decades = []
    current_date = start_date
    while current_date <= end_date:
        days.append(current_date)
        months.append(datetime(current_date.year,current_date.month,1))
        years.append(datetime(current_date.year,1,1))
        decades.append(datetime(int(str(current_date.year)[0:3])*10,1,1))
        current_date += timedelta(days=1)
    days    = sorted(set(days),    reverse=True)
    months  = sorted(set(months),  reverse=True)
    years   = sorted(set(years),   reverse=True)
    decades = sorted(set(decades), reverse=True)

    if time_range_type in ['day', 'all']:
        print('Making time series plots for ' + str(len(days)) + ' day(s)')
        for day in days:
            try:
                if base_dir != None:
                    savedir = base_dir + day.strftime("%Y%m%d") + '/Time_Series/'
                else:
                    savedir = None
                myTS = AnalyzeTimeSeries(db_path=db_path)
                myTS.plot_all_quicklook(day, interval='day', fig_dir=savedir)
                myTS = None
            except Exception as e:
                print(e)

    if time_range_type in ['month', 'all']:
        print('Making time series plots for ' + str(len(months)) + ' month(s)')
        for month in months:
            try:
                if base_dir != None:
                    savedir = base_dir + month.strftime("%Y%m") + 'M/Time_Series/'
                else:
                    savedir = None
                myTS = AnalyzeTimeSeries(db_path=db_path)
                myTS.plot_all_quicklook(month, interval='month', fig_dir=savedir)
                myTS = None
            except Exception as e:
                print(e)

    if time_range_type in ['year', 'all']:
        print('Making time series plots for ' + str(len(years)) + ' year(s)')
        for year in years:
            try:
                if base_dir != None:
                    savedir = base_dir + year.strftime("%Y") + 'Y/Time_Series/'
                else:
                    savedir = None
                myTS = AnalyzeTimeSeries(db_path=db_path)
                myTS.plot_all_quicklook(year, interval='year', fig_dir=savedir)
                myTS = None
            except Exception as e:
                print(e)

    if time_range_type in ['decade', 'all']:
        print('Making time series plots for ' + str(len(decades)) + ' decade(s)')
        for decade in decades:
            try:
                if base_dir != None:
                    savedir = base_dir + decade.strftime("%Y")[0:3] + '0D/Time_Series/' 
                else:
                    savedir = None
                myTS = AnalyzeTimeSeries(db_path=db_path)
                myTS.plot_all_quicklook(decade, interval='decade', fig_dir=savedir)
                myTS = None
            except Exception as e:
                print(e)


def monitor_threads(threads, sleep_time):
    time.sleep(10)
    while True:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n------ Thread Status at " + current_time + " ------")
        for thread in threads:
            print(f"{thread.name}: {'Alive' if thread.is_alive() else 'Dead'} - Started at {thread.start_time}")
        print("--------------------------------------------------- \n")
        time.sleep(sleep_time)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Repeatedly generate KPF time series plots.')
    parser.add_argument('--db_path', type=str, default='/data/time_series/kpf_ts.db', 
                        help='path to database file; default = /data/time_series/kpf_ts.db')
    args = parser.parse_args()   

# For now, only one thread is active because of a non-tread-safe issue with fonts in the version of matplotlib that we use
    tasks = [
        {"thread_name": "Today Thread",       "interval":    3600, "time_range_type": "day",    "date_range": 'this_day'},
#        {"thread_name": "This Month Thread",  "interval": 12*3600, "time_range_type": "month",  "date_range": 'this_month'},
#        {"thread_name": "This Year Thread",   "interval": 12*3600, "time_range_type": "year",   "date_range": 'this_year'},
#        {"thread_name": "All Days Thread",    "interval": 96*3600, "time_range_type": "day",    "date_range": 'all_days'},
#        {"thread_name": "All Months Thread",  "interval": 2*3600, "time_range_type": "month",  "date_range": 'all_months'},
#        {"thread_name": "All Years Thread",   "interval": 24*3600, "time_range_type": "year",   "date_range": 'all_years'},
#        {"thread_name": "All Decades Thread", "interval": 24*3600, "time_range_type": "decade", "date_range": (None, None)},
    ]

    threads = []
    for task in tasks:
        thread = Thread(target=schedule_task, args=(task["interval"], task["time_range_type"], task["date_range"], task["thread_name"], args.db_path), name=task["thread_name"])
        thread.start_time = datetime.now()  
        thread.start()  
        threads.append(thread)
        time.sleep(20)

    monitor_thread_sleep_time = 3600 # seconds
    monitor_thread = Thread(target=monitor_threads, args=(threads, monitor_thread_sleep_time))
    monitor_thread.start()

    for thread in threads:
        thread.join() 

    monitor_thread.join() 
