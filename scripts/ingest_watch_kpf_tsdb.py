#!/usr/bin/env python3

"""
Script Name: ingest_watch_kpf_tsdb.py

Description:
    This script watches directories for new or modified KPF files and ingests their
    data into a KPF Time Series Database. The script utilizes the Watchdog library
    to monitor filesystem events and triggers ingestion processes. Additionally, it 
    performs periodic scans of data directories to ensure all observations are 
    ingested.

Features:
    - Ingests file metadata and telemetry into the database.
    - Watches multiple directories for new or modified KPF files.
    - Performs periodic scans of data directories.
    - Supports multithreaded execution.

Usage:
    Run this script with optional arguments to specify the database path:
    
        python ingest_watch_kpf_tsdb.py --db_path /path/to/database.db

Options:
    --db_path   Path to the time series database file. Default: /data/time_series/kpf_ts.db

Examples:
    1. Using default database path:
        python ingest_watch_kpf_tsdb.py

    2. Specifying a custom database path:
        python ingest_watch_kpf_tsdb.py --db_path /custom/path/to/kpf_ts.db
        
To-do: 
	1. Add backend as an argument
"""

import os
import time
import queue
import argparse
import threading
import subprocess
from collections import deque
from watchdog.observers import Observer
from datetime import datetime, timedelta
from watchdog.events import FileSystemEventHandler
from modules.Utils.kpf_parse import get_ObsID, is_ObsID, get_datecode
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

class Watcher:
    """
    Class to watch for file creations and modifications in a specified directory.  
    Such events are accumulated and passed on to a processing method for ingestion
    into the database.
    """
    def __init__(self, directory_to_watch, db_path, stop_event):
        self.event_queue = queue.Queue()
        self.observer = Observer()
        self.directory_to_watch = directory_to_watch
        self.db_path = db_path
        self.stop_event = stop_event
        self.handler = Handler(self.event_queue)
        threading.Thread(target=process_queue, args=(self.event_queue, self.db_path, self.stop_event), daemon=False).start()

    def run(self):
        self.observer.schedule(self.handler, self.directory_to_watch, recursive=True)
        self.observer.start()
        try:
            while not self.stop_event.is_set():  # Check if the stop event is set
                time.sleep(3)
        except Exception as e:
            self.observer.stop()
        self.observer.stop()  # Stop the observer when the stop event is set or an exception occurs
        self.observer.join()

class Handler(FileSystemEventHandler):
    """
    Class to handle system events (file modifications and creations).
    """
    def __init__(self, event_queue):
        self.event_queue = event_queue

    def on_any_event(self, event):
        if event.is_directory:
            return None
        if event.event_type in ['created', 'modified']:
            filename = os.path.basename(event.src_path)
            if filename.startswith('KP') and filename.endswith('.fits'):
                self.event_queue.put(event.src_path)         
            
def process_queue(event_queue, db_path, stop_event):
    """
    This method processes a set of events (files creations and modifications) by 
    ingesting the headers and telemetry from the corresponding observations.
    """
    processing_interval = 30  # seconds to wait before each processing cycle
    modification_delay = 3  # seconds to ensure file hasn't been modified recently
    last_processed_time = {}  # dictionary to track last processed time for each file
    event_buffer = set()

    while not stop_event.is_set():
        try:
            # Collect unique events over the processing interval
            start_time = time.time()
            while time.time() - start_time < processing_interval:
                try:
                    file_path = event_queue.get(timeout=1)  # timeout set to 1 for responsive checks
                    event_buffer.add(file_path)
                    event_queue.task_done()
                except queue.Empty:
                    continue  # continue checking until processing_interval is reached

            # Process files if they haven't been modified recently
            current_time = time.time()
            L0_path_batch = set()
            for file_path in event_buffer:
                last_mod_time = os.path.getmtime(file_path)
                if (current_time - last_mod_time > modification_delay and
                    (file_path not in last_processed_time or 
                     current_time - last_processed_time[file_path] > modification_delay)):
                    last_processed_time[file_path] = current_time
                    ObsID = get_ObsID(file_path)
                    if is_ObsID(ObsID):
                        L0_path = '/data/L0/' + get_datecode(file_path) + '/' + ObsID + '.fits'
                        L0_path_batch.add(L0_path)

            if len(L0_path_batch) > 0:
                L0_path_batch = sorted(L0_path_batch)
                ObsID_batch = [get_ObsID(L0_path) for L0_path in L0_path_batch]
                myTS = AnalyzeTimeSeries(db_path=db_path, backend='psql')
                myTS.logger.info('Ingesting ' + str(len(L0_path_batch)) + ' observations: ' + ', '.join(ObsID_batch))
                myTS.db.ingest_batch_observations(L0_path_batch)
                myTS.logger.info('Finished ingesting ' + str(len(L0_path_batch)) + ' observations.')
                myTS = [] # clear memory
            
            event_buffer.clear()

        except Exception as e:
            print(f"Error in processing queue: {e}")

def periodic_scan(db_path,stop_event):
    """
    Method to scan the data directories every hour, with the first scan starting 
    20 sec after the script starts.
    """
    time.sleep(20)            
    start_date = '20221201'
    end_date   = '20400101'
    sec_between_scans = 3600*12
    last_run_time = datetime.now() - timedelta(seconds=sec_between_scans)

    while not stop_event.is_set():
        if datetime.now() - last_run_time >= timedelta(seconds=sec_between_scans):
            myTS = AnalyzeTimeSeries(db_path=db_path, backend='psql')
            myTS.logger.info('Starting periodic scan for new or changed files.')
            myTS.db.ingest_dates_to_db(start_date, end_date, batch_size=10000, reverse=True, force_ingest=True)
            myTS.db.print_db_status()
            myTS.logger.info('Ending periodic scan for new or changed files.')
            myTS = [] # clear memory
            last_run_time = datetime.now()
        time.sleep(3) # Wait before checking again

def start_watcher(directory, db_path, stop_event):
    """
    Method that starts a Watcher for the stated directory.  
    This script launches a Watcher for each of the four data directories.
    """
    watcher = Watcher(directory, db_path, stop_event)
    watcher.run()


if __name__ == '__main__':
    print('Starting ingest_watch_kpf_tsdb.py -- Ctrl+C to stop')

    parser = argparse.ArgumentParser(description='Ingest KPF files into an observational database.')
    parser.add_argument('--db_path', type=str, default='/data/time_series/kpf_ts.db', 
                        help='path to database file; default = /data/time_series/kpf_ts.db')
    args = parser.parse_args()   

    # Start threads for each directory
    threads = []
    directories = ['/data/L0/', '/data/2D/', '/data/L1/', '/data/L2/']
    stop_event = threading.Event()
    for directory in directories:
        print('Starting a thread to watch ' + directory)
        thread = threading.Thread(target=start_watcher, args=(directory, args.db_path, stop_event))
        threads.append(thread)
        thread.start()

    # Start the thread to periodically scan for files
    print('Starting a thread to periodically scan all date directories')
    periodic_thread = threading.Thread(target=periodic_scan, args=(args.db_path, stop_event))
    periodic_thread.start()
    threads.append(periodic_thread)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        print("\nStopping threads... (wait up to 30 seconds for threads to exit)")

    for thread in threads:
        thread.join(timeout=30)
