#!/usr/bin/env python3

import os
import time
import queue
import argparse
import threading
import subprocess
from collections import deque
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from modules.Utils.kpf_parse import get_ObsID, is_ObsID, get_datecode
from  modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

class Watcher:
    def __init__(self, directory_to_watch, db_path):
        self.event_queue = queue.Queue()
        self.observer = Observer()
        self.directory_to_watch = directory_to_watch
        self.db_path = db_path
        self.handler = Handler(self.event_queue)
        threading.Thread(target=process_queue, args=(self.event_queue, self.db_path), daemon=True).start()

    def run(self):
        self.observer.schedule(self.handler, self.directory_to_watch, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except Exception as e:
            self.observer.stop()
            print("Observer process stopped due to an error:", e)    
        self.observer.join()

class Handler(FileSystemEventHandler):
    def __init__(self, event_queue):
        self.event_queue = event_queue

    def on_any_event(self, event):
        if event.is_directory:
            return None
        if event.event_type in ['created', 'modified']:
            filename = os.path.basename(event.src_path)
            if filename.startswith('KP') and filename.endswith('.fits'):
                self.event_queue.put(event.src_path)         
            
def process_queue(event_queue, db_path):
    processing_interval = 30  # seconds to wait before each processing cycle
    modification_delay = 3  # seconds to ensure file hasn't been modified recently
    last_processed_time = {}  # dictionary to track last processed time for each file
    event_buffer = set()

    while True:
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

            current_time = time.time()

            # Process files if they haven't been modified recently
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
                myTS = AnalyzeTimeSeries(db_path=db_path)
                myTS.logger.info('Ingesting ' + str(len(L0_path_batch)) + ' observations with recent file creations/modifications.')
                for L0_path in L0_path_batch:
                    myTS.logger.info('    ' + get_ObsID(L0_path))
                myTS.ingest_batch_observation(L0_path_batch)
                myTS.print_db_status()
                myTS = [] # clear memory
            
            event_buffer.clear()

        except Exception as e:
            print(f"Error in processing queue: {e}")

def periodic_scan(db_path):
    time.sleep(5)            
    start_date = '20231201'
    end_date   = '20400101'
    while True:
        myTS = AnalyzeTimeSeries(db_path=db_path)
        myTS.logger.info('Starting periodic scan for new or changed files.')
        myTS.ingest_dates_to_db(start_date, end_date)
        myTS.print_db_status()
        myTS.logger.info('Ending periodic scan for new or changed files.')
        myTS = [] # clear memory
        time.sleep(3600)            
            
def start_watcher(directory, db_path):
    watcher = Watcher(directory, db_path)
    watcher.run()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest KPF files into an observational database.')
    parser.add_argument('--db_path', type=str, default='/data/time_series/kpf_ts.db', 
                        help='path to database file; default = /data/time_series/kpf_ts.db')
    args = parser.parse_args()

    directories = ['/data/L0/', '/data/2D/', '/data/L1/', '/data/L2/']
    threads = []

    # Start threads for each directory
    for directory in directories:
        thread = threading.Thread(target=start_watcher, args=(directory, args.db_path,))
        threads.append(thread)
        thread.start()

    # Start the thread to periodically scan for files
    periodic_thread = threading.Thread(target=periodic_scan, args=(args.db_path,))
    periodic_thread.start()
    threads.append(periodic_thread)

    for thread in threads:
        thread.join()