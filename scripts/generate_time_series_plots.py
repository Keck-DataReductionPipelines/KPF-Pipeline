#!/usr/bin/env python3

import time
from threading import Thread
from datetime import datetime, timedelta
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

def schedule_task(initial_delay, interval, time_range_type, date_range, thread_name):
    time.sleep(initial_delay)
    print(f'Starting {thread_name}')
    while True:
        kwargs = {
            "start_date": date_range[0],
            "end_date": date_range[1],
            "time_range_type": time_range_type
        }
        start_time = time.time()
        generate_plots(kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        sleep_time = interval - execution_time
        if sleep_time > 0:
            time.sleep(sleep_time)

def generate_plots(kwargs):
#    print("got to 1 - " + str(kwargs["time_range_type"]))
    myTS = AnalyzeTimeSeries(db_path='/data/time_series/kpf_ts.db')
#    print("got to 2 - " + str(kwargs["time_range_type"]))
#    print(kwargs)
    myTS.plot_all_quicklook_daterange(**kwargs)
#    print("got to 3 - " + str(kwargs["time_range_type"]))
    myTS = ''  # Clear memory (needed?)

def monitor_threads(threads, sleep_time):
    time.sleep(10)
    while True:
        print("\n------ Thread Status ------")
        for thread in threads:
            print(f"{thread.name}: {'Alive' if thread.is_alive() else 'Dead'} - Started at {thread.start_time}")
        print("-------------------------\n")
        time.sleep(sleep_time)

if __name__ == "__main__":
    tasks = [
        {"thread_name": "All Days Thread",    "initial_delay":  1, "interval":  1*3600, "time_range_type": "day",    "date_range": (datetime(2024,  2,  1), datetime(2024,  2, 24))},
        {"thread_name": "All Months Thread",  "initial_delay":  3, "interval":  1*3600, "time_range_type": "month",  "date_range": (datetime(2023,  1,  1), datetime(2024,  2,  1))},
        {"thread_name": "All Years Thread",   "initial_delay":  5, "interval": 10*3600, "time_range_type": "year",   "date_range": (datetime(2024,  1,  1), datetime(2024,  2,  1))},
        {"thread_name": "All Decades Thread", "initial_delay":  7, "interval": 24*3600, "time_range_type": "decade", "date_range": (datetime(2020,  1,  1), datetime(2024,  2,  1))},
    ]

    threads = []
    for task in tasks:
        thread = Thread(target=schedule_task, args=(task["initial_delay"], task["interval"], task["time_range_type"], task["date_range"], task["thread_name"]), name=task["thread_name"])
        thread.start_time = datetime.now()  
        thread.start()  
        threads.append(thread)

    monitor_thread_sleep_time = 30 # seconds
    monitor_thread = Thread(target=monitor_threads, args=(threads, monitor_thread_sleep_time))
    monitor_thread.start()

    for thread in threads:
        thread.join()  # Wait for all threads to complete

    monitor_thread.join()  # Ensure the monitor thread also completes
