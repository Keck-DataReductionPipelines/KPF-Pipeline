#!/usr/bin/env python3

import time
from datetime import datetime, timedelta
from threading import Thread
import psutil  # Import psutil to get CPU usage

def generate_plots(kwargs):
    db_path = '/data/time_series/kpf_ts.db'
    myTS = AnalyzeTimeSeries(db_path=db_path)
    myTS.plot_all_quicklook_daterange(**kwargs)
    myTS = ''  # Clear memory (though just setting it to '' might not be enough for garbage collection)

def monitor_threads(threads):
    while True:
        print("\n--- Thread Status ---")
        for i, thread in enumerate(threads):
            print(f"Thread {i}: {'Alive' if thread.is_alive() else 'Dead'} - Started at {thread.start_time}")
        print(f"System CPU load (last 15 minutes): {psutil.getloadavg()[2]}")
        time.sleep(123)

def schedule_task(initial_delay, interval, time_range_type, date_range):
    time.sleep(initial_delay)
    print('Starting schedule_task')
    while True:
        start_time = time.time()
        # Assuming the date_range and time_range_type logic is handled elsewhere or not relevant for this snippet
        kwargs = {
            "start_date": date_range[0],
            "end_date": date_range[1],
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
        {"initial_delay": 1,  "interval": 3600, "time_range_type": "day",   "date_range": (datetime(2024,  2, 23), datetime(2024,  2, 24))},
        {"initial_delay": 10, "interval": 3600, "time_range_type": "month", "date_range": (datetime(2024,  1,  1), datetime(2030,  1,  1))}
    ]

    threads = []
    for task in tasks:
        thread = Thread(target=schedule_task, args=(task["initial_delay"], task["interval"], task["time_range_type"], task["date_range"]))
        thread.start_time = datetime.now()  # Store the start time in the thread object
        threads.append(thread)

    monitor_thread = Thread(target=monitor_threads, args=(threads,))
    monitor_thread.start()

    for thread in threads:
        thread.join()  # Wait for all threads to complete

    monitor_thread.join()  # Ensure the monitor thread also completes
