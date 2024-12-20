import os
import time
import shutil
from datetime import datetime
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

import keck_utils as utils

BASE_WATCH_DIR = "/koadata/KPF"
BASE_DEST_DIR = "/kpfdata/data_workspace/L0"
LOG_DIR = "/data/logs/DataMonitor/"


class DirectoryWatchHandler(FileSystemEventHandler):
    def __init__(self, current_date):
        self.dest_dir = None
        self.set_dest_dir(current_date)

    def set_dest_dir(self, current_date):
        self.dest_dir = os.path.join(BASE_DEST_DIR, current_date)
        os.makedirs(f"{self.dest_dir}/", exist_ok=True)

    # def on_created(self, event):
    #     if not event.is_directory:
    #         src_path = event.src_path
    #         self.copy_file(src_path)
    #
    # def copy_file(self, src_path):
    #     if '.fits' not in src_path:
    #         return
    #
    #     try:
    #         shutil.copy2(src_path, self.dest_dir)
    #     except Exception as e:
    #         log.info(f"Error copying {src_path}: {e}")
    def on_created(self, event):
        if not event.is_directory:
            src_path = event.src_path
            self.cp_once_written(src_path)

    def cp_once_written(self, src_path, timeout=300, check_interval=5):
        if '.fits' not in src_path:
            return

        elapsed_time = 0
        previous_size = -1

        while elapsed_time < timeout:
            try:
                current_size = os.path.getsize(src_path)
                if current_size == previous_size:
                    # File size has stabilized, assume it's done writing
                    shutil.copy2(src_path, self.dest_dir)
                    log.info(f"File {src_path} copied successfully to {self.dest_dir}.")
                    return
                previous_size = current_size
            except FileNotFoundError:
                pass
            except Exception as e:
                log.error(f"Error checking file size for {src_path}: {e}")
                return

            time.sleep(check_interval)
            elapsed_time += check_interval


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

    # Copy existing files before starting to monitor the directory
    copy_existing_files(current_date)

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
            time.sleep(2)
    except Exception as err:
        log.info(f"Exiting the monitor: {err}")
        observer.stop()
    observer.join()


def copy_existing_files(current_date):
    """
    Continuously copy existing .fits files from the current watch directory
    to the destination directory.

    It will recursively check for new files after completing each set has been
    copied.
    """
    watch_dir = get_watch_dir(current_date)
    dest_dir = os.path.join(BASE_DEST_DIR, current_date)

    os.makedirs(dest_dir, exist_ok=True)

    # copied_files = set()  # Track files that have already been copied
    copied_files = set(f for f in os.listdir(dest_dir) if f.endswith('.fits'))
    if not os.path.exists(watch_dir):
        log.info(f"Watch directory {watch_dir} does not exist.")
        return

    try:
        current_files = {f for f in os.listdir(watch_dir) if f.endswith('.fits')}

        new_files = current_files - copied_files

        # return, no new files to copy
        if not new_files:
            log.info(f'all current files have been copied.')
            return

        log.info(f'copying {len(new_files)} existing files: {new_files}')

        for file_name in new_files:
            src_path = os.path.join(watch_dir, file_name)
            try:
                shutil.copy2(src_path, dest_dir)
                copied_files.add(file_name)
            except Exception as e:
                log.info(f"Error copying {src_path}: {e}")

        # delay to catch any new files
        time.sleep(1)

    except Exception as err:
        log.info(f"Exiting the copy of existing files: {err}")
        return

    # check to see if any files were created while copying the existing files.
    return copy_existing_files(current_date)


if __name__ == "__main__":
    today = datetime.utcnow().strftime("%Y%m%d")
    log = utils.configure_logger(LOG_DIR, f"koadata_monitor_{today}")

    utils.kill_exist_proc(os.path.basename(__file__), log=log)
    monitor_directory()
