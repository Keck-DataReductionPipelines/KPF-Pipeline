#!/usr/bin/env python3

import time
from datetime import datetime
from datetime import timedelta
from threading import Thread
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

def generate_plots(kwargs):
    db_path = '/data/time_series/kpf_ts.db'
    myTS = AnalyzeTimeSeries(db_path=db_path)
    myTS.plot_all_quicklook_daterange(**kwargs)
    myTS = '' # needed?  to clear memory

def schedule_task(initial_delay, interval, time_range_type, date_range):
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
        
        
        #kwargs: keyword arguments for AnalyzeTimeSeries.plot_all_quicklook_daterange()
    """
    time.sleep(initial_delay)  # Initial delay before first execution
    print('starting schedule_task')
    while True:
        start_time = time.time()
        if   date_range == 'this_day':
            pass
        elif date_range == 'this_month':
            pass
        elif date_range == 'this_year':
            pass
        elif date_range == 'this_decade':
            pass
        elif date_range == 'last_10_days':
            pass
        else:
            now = datetime.now()
            tomorrow   = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            next_month = (now + timedelta(days=28)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            next_year  = (now + timedelta(days=365)).replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            if (time_range_type == 'day') and (date_range[1] > tomorrow):
                date_range = (date_range[0], tomorrow)
            #print(date_range)
            kwargs = {
                "start_date":      date_range[0],
                "end_date":        date_range[1],
                "time_range_type": time_range_type
                }
        
        generate_plots(kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        sleep_time = interval * 60 - execution_time
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == "__main__":

    tasks = [
        {"initial_delay": 1, 
         "interval": 3600,
         "time_range_type": "day",
         "date_range": (datetime(2024, 2, 23), datetime(2024, 2, 24))
         #"date_range": (datetime(2024, 2, 23), datetime(2030, 1, 1)) # dates in the future are rounded down to tomorrow
        },
        {"initial_delay": 10, 
         "interval": 3600,
         "time_range_type": "month",
         "date_range": (datetime(2024, 1, 1), datetime(2030, 1, 1)) # dates in the future are rounded down to next month
        },
        ]

    threads = []
    for task in tasks:
        #print('task: ' + str(task["initial_delay"]) + ' ' + str(task["interval"]) + ' ' + str(task["start_date"]) + ' ' +  str(task["end_date"]) + ' ')
        thread = Thread(target=schedule_task, args=(task["initial_delay"], task["interval"], task["time_range_type"], task["date_range"]))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()  # Wait for all threads to complete
