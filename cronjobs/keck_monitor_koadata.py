import os
import sys
import time
import shutil
import psutil
import signal
from datetime import datetime
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

import keck_utils as utils

BASE_WATCH_DIR = "/koadata/KPF"
BASE_DEST_DIR = "/kpfdata/data_workspace/L0"
LOG_DIR = "/data/logs/DailyRuns/"


class DirectoryWatchHandler(FileSystemEventHandler):
    def __init__(self, current_date):
        self.dest_dir = None
        self.set_dest_dir(current_date)

    def set_dest_dir(self, current_date):
        self.dest_dir = os.path.join(BASE_DEST_DIR, current_date)
        os.makedirs(f"{self.dest_dir}/", exist_ok=True)

    def on_created(self, event):
        if not event.is_directory:
            src_path = event.src_path
            self.copy_file(src_path)

    def copy_file(self, src_path):
        if '.fits' not in src_path:
            return

        try:
            shutil.copy2(src_path, self.dest_dir)
            log.info(f"Copied {src_path} to {self.dest_dir}")
        except Exception as e:
            log.info(f"Error copying {src_path}: {e}")


def get_watch_dir(current_date):
    """
    Get the watch directory based on the current UTC date.
    """
    return os.path.join(BASE_WATCH_DIR, current_date, "lev0")


def wait_watch_dir_exist(current_date):
    """
    Wait for the watch directory to exist before starting.

    Args:
        current_date ():

    Returns:

    """
    current_watch_dir = get_watch_dir(current_date)
    while not os.path.exists(current_watch_dir):
        log.info(f"Directory {current_watch_dir} does not exist. Retrying in 5 minutes.")
        time.sleep(300)

    return current_watch_dir


def monitor_directory():
    """
    Monitor the current watch directory for new files.  It will change date
    when the UT date changes.
    """
    current_date = datetime.utcnow().strftime("%Y%m%d")
    watch_dir = wait_watch_dir_exist(current_date)
    log.info(f"Starting to monitor: {watch_dir}")

    watch_obj = DirectoryWatchHandler(current_date)
    observer = PollingObserver()
    # TODO not working with NFS?
    # observer = Observer()
    observer.schedule(watch_obj, watch_dir, recursive=True)
    observer.start()

    try:
        while True:
            current_date = datetime.utcnow().strftime("%Y%m%d")
            # Check if the date has changed to update the watch directory
            new_watch_dir = get_watch_dir(current_date)
            if new_watch_dir != watch_dir:
                watch_dir = wait_watch_dir_exist(current_date)
                log.info(f"UTC Date changed. Now monitoring: {watch_dir}")
                watch_obj.set_dest_dir(current_date)
                break
            # time.sleep(300)
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def kill_exist_proc(script_name):
    """
    Find and kill processes running the given script name
    """
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            if proc.info['cmdline'] and script_name in proc.info['cmdline'] and proc.info['pid'] != current_pid:
                log.info(f"Killing process {script_name} with PID {proc.info['pid']}")
                os.kill(proc.info['pid'], signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass


if __name__ == "__main__":
    today = datetime.utcnow().strftime("%Y%m%d")
    log = utils.configure_logger(LOG_DIR, f"koadata_monitor_{today}")

    kill_exist_proc(os.path.basename(sys.argv[0]))
    monitor_directory()
