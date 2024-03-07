#!/usr/bin/env python3

import subprocess
import time

intervals =  ['day', 'month', 'year', 'decade', 'last_1_day', 'last_3_days', 'last_10_days', 'last_30_days', 'last_100_days']
wait_times = [3600*4, 3600*4, 3600*4, 3600*24, 600, 600, 600, 1200, 1200]  # seconds

db_path = '/data/time_series/kpf_ts.db'

processes = []
for interval, wait_time in zip(intervals, wait_times):
    cmd = ['python', 'scripts/make_plots_kpf_tsdb.py', '--db_path', db_path, '--interval', interval, '--wait_time', str(wait_time)]
    process = subprocess.Popen(cmd)
    processes.append(process)
    time.sleep(2)  # Stagger the start time of each process slightly

# Wait for all processes to complete
for process in processes:
    process.wait()

print("All plotting processes have completed.")
