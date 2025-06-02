#!/usr/bin/env python3

"""
Script Name: generate_time_series_plots.py

Description:
    This script generates KPF time series plots over various time intervals and
    saves the results to predefined directories. It supports multiprocess execution
    for tasks with different intervals and date ranges, including daily, monthly,
    yearly, and decade-based plots. Additionally, the script monitors the status
    of running processes and reports on their activity.

Features:
    - Generates plots for multiple time intervals (day, month, year, decade).
    - Supports custom date ranges for plot generation.
    - Multiprocess execution for efficiency, allowing simultaneous tasks.
    - Monitors process status and execution time for each task.
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

    3. Monitoring process statuses:
        The script automatically reports on process activity and execution times every 5 minutes.
"""

import glob
import copy
import time
import argparse
import logging
from multiprocessing import Process
from datetime import datetime, timedelta
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

def schedule_task(interval, time_range_type, date_range, proc_name, db_path):
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
    
    fig_dir_base = f"/output/{proc_name.replace(' ', '_')}/"
    
    print()
    print(f"Starting: {proc_name} to be executed every {interval/3600:.2f} hours.")
    initial_date_range = date_range

    while True:
        date_range = initial_date_range
        start_time = time.time()
        now = datetime.now()
        end_of_today = (now).replace(hour=23, minute=59, second=59, microsecond=0)
        if date_range == 'all_days':
            start_date = None
            end_date   = None
            time_range_type =  time_range_type
        elif date_range == 'all_months':
            start_date = None
            end_date   = None
            time_range_type =  time_range_type
        elif date_range == 'all_years':
            start_date = None
            end_date   = None
            time_range_type =  time_range_type
        elif date_range == 'this_day':
            start_date = (now - timedelta(days=1.0)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = end_of_today
            time_range_type =  time_range_type
        elif date_range == 'this_month':
            start_date = (now - timedelta(days=31)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date   = end_of_today
            time_range_type = time_range_type
        elif date_range == 'this_year':
            start_date = (now - timedelta(days=366)).replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date   = end_of_today
            time_range_type = time_range_type
        elif date_range == 'last_10_days':
            # need to determine where to store the results from this so that it doesn't crash Jump
            pass
        else:
            start_date = date_range[0]
            end_date   = date_range[1]
            time_range_type = time_range_type

        start_time = time.time()
        print(f'    start_date = {str(start_date)}')
        print(f'    end_date = {str(end_date)}')
        print(f'    time_range_type = {str(time_range_type)}')
        print()
        generate_plots(start_date=start_date, end_date=end_date, time_range_type=time_range_type, db_path=db_path)
        end_time = time.time()
        execution_time = end_time - start_time
        sleep_time = interval - execution_time
        if sleep_time < 0:
            sleep_time = 0
        print(f'Finished pass through {proc_name} in {execution_time/3600:.2f} hours.\nStarting again in {sleep_time/3600:.2f} hours.')
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
        
    To-do: 
        Add argument for backend
    """
    
    if start_date == None or end_date == None:
        myTS = AnalyzeTimeSeries(db_path=db_path, backend='psql')
        first_last_dates = myTS.get_first_last_dates()
        if start_date == None:
            start_date = first_last_dates[0].replace(hour=0, minute=0, second=0, microsecond=0)
        if end_date == None:
            end_date = first_last_dates[1].replace(hour=23, minute=59, second=59, microsecond=0)
    
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
                myTS = AnalyzeTimeSeries(db_path=db_path, backend='psql')
                myTS.plot_all_quicklook(day, interval='day', fig_dir=savedir)
                del myTS # free up memory
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
                myTS = AnalyzeTimeSeries(db_path=db_path, backend='psql')
                myTS.plot_all_quicklook(month, interval='month', fig_dir=savedir)
                del myTS # free up memory
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
                myTS = AnalyzeTimeSeries(db_path=db_path, backend='psql')
                myTS.plot_all_quicklook(year, interval='year', fig_dir=savedir)
                del myTS # free up memory
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
                myTS = AnalyzeTimeSeries(db_path=db_path, backend='psql')
                myTS.plot_all_quicklook(decade, interval='decade', fig_dir=savedir)
                del myTS # free up memory
            except Exception as e:
                print(e)


def monitor_processes(tasks, proc_dict, sleep_time, db_path):
    process_start_times = {task["proc_name"]: datetime.now() for task in tasks}

    def print_process_table():
        now = datetime.now()

        header   = f"{'Process Name':<25} {'Interval':>12}   {'Uptime':>12}"
        divider  = "-" * len(header)
        print("\nProcess Status Report:")
        print(divider)
        print(header)
        print(divider)

        for task in tasks:
            proc_name = task["proc_name"]
            interval  = task["interval"]
            proc      = proc_dict.get(proc_name)

            # (Re)spawn a dead process
            if not proc or not proc.is_alive():
                print(f"{proc_name} stopped, restarting...")
                new_proc = Process(
                    target=schedule_task,
                    args=(interval, task["time_range_type"],
                          task["date_range"], proc_name, db_path),
                    name=proc_name
                )
                new_proc.start()
                proc_dict[proc_name]        = new_proc
                process_start_times[proc_name] = datetime.now()
                proc = new_proc

            # ---------- format interval  (DDD HH:MM:SS) ----------
            idays      = interval // 86_400
            iremainder = interval % 86_400
            ihours, iremainder = divmod(iremainder, 3_600)
            iminutes, iseconds = divmod(iremainder, 60)
            interval_str = f"{idays:03} {ihours:02}:{iminutes:02}:{iseconds:02}"

            # ---------- format uptime   (DDD HH:MM:SS) ----------
            uptime_td  = now - process_start_times[proc_name]
            udays      = uptime_td.days
            usecs_tot  = int(uptime_td.total_seconds()) - udays * 86_400
            uhours, uremainder = divmod(usecs_tot, 3_600)
            uminutes, useconds = divmod(uremainder, 60)
            uptime_str  = f"{udays:03} {uhours:02}:{uminutes:02}:{useconds:02}"

            print(f"{proc_name:<25} {interval_str:>12}   {uptime_str:>12}")

        print(divider + "\n")

    while True:
        print_process_table()
        time.sleep(sleep_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Repeatedly generate KPF time series plots.')
    parser.add_argument('--db_path', type=str, default='/data/time_series/kpf_ts.db', 
                        help='path to database file; default = /data/time_series/kpf_ts.db')
    args = parser.parse_args()   

    tasks = [
        {"proc_name": "Today Process",       "interval":     300, "time_range_type": "day",    "date_range": 'this_day'},
        {"proc_name": "This Month Process",  "interval":     600, "time_range_type": "month",  "date_range": 'this_month'},
        {"proc_name": "This Year Process",   "interval":  2*3600, "time_range_type": "year",   "date_range": 'this_year'},
        {"proc_name": "All Days Process",    "interval": 48*3600, "time_range_type": "day",    "date_range": 'all_days'},
        {"proc_name": "All Months Process",  "interval":  3*3600, "time_range_type": "month",  "date_range": 'all_months'},
        {"proc_name": "All Years Process",   "interval":  3*3600, "time_range_type": "year",   "date_range": 'all_years'},
        {"proc_name": "All Decades Process", "interval": 24*3600, "time_range_type": "decade", "date_range": (None, None)},
    ]

    processes = {}
    for task in tasks:
        proc = Process(
            target=schedule_task,
            args=(task["interval"], task["time_range_type"], task["date_range"],
                  task["proc_name"], args.db_path),
            name=task["proc_name"]
        )
        proc.start()
        processes[task["proc_name"]] = proc
        time.sleep(15)

    monitor_processes(tasks, processes, sleep_time=300, db_path=args.db_path)
