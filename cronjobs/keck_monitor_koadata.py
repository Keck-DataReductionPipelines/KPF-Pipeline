import os
import time
import shutil
import subprocess
from datetime import datetime
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

    def on_created(self, event):
        if not event.is_directory:
            file_loc = event.src_path
            if '.fits' not in file_loc:
                return

            # wait for the files to write,  return if there is an issue
            if not self.wait_for_write(file_loc):
                return

            self.rsync_fullpath(file_loc, self.dest_dir)
            log.info(f"File {file_loc} copied successfully to {self.dest_dir}.")

    def wait_for_write(self, file_loc, timeout=300, check_interval=5):
        """
        Wait for the file to finish writing.  Used for both to wait before
        copying and after copying to ensure the file is fully written.

        Args:
            file_loc (str): path to the file
            timeout (int): time to wait before giving up
            check_interval (int): time to wait between checks

        Returns:

        """
        if '.fits' not in file_loc:
            log.info(f"Extension .fits not in {file_loc}")
            return False

        elapsed_time = 0
        previous_size = -1

        while elapsed_time < timeout:
            try:
                current_size = os.path.getsize(file_loc)
                log.debug(f"file size: {current_size}, {previous_size}")
                if current_size == previous_size:
                    # File size has stabilized, assume it's done writing
                    log.info(f"File {file_loc} written.")
                    return True
                previous_size = current_size
            except FileNotFoundError:
                log.error(f"File not found: {file_loc}")
                return False
            except Exception as e:
                log.error(f"Error checking file size for {file_loc}: {e}")
                return False

            time.sleep(check_interval)
            elapsed_time += check_interval

        return False

    def rsync_fullpath(self, file_loc, dest_dir):
        """
        Copy the full path to the destination directory using rsync to re-copy
        any files that might be different from originally copied.  There was
        originally an issue with the file size not being the same if the copy was
        interrupted.


        Args:
            file_loc (str): path to the file
            dest_dir (str): path to the destination directory

        Returns:

        """
        file_dir = os.path.dirname(file_loc)
        file_name = os.path.basename(file_loc)
        try:
            subprocess.run(
                [
                    "rsync", "--include", "*.fits",
                    "--exclude", "*",
                    file_loc, dest_dir
                ], check=True)

            log.info(f"Copied {file_loc} to {dest_dir} using rsync.")

            dest_file = os.path.join(dest_dir, file_name)
            log.info(f"Waiting on {dest_file}.")
        except subprocess.CalledProcessError as e:
            log.error(f"Issue with rsync {file_loc} to {dest_dir}: {e}")

def get_watch_dir(current_date):
    """
    Get the watch directory based on the current UTC date.
    """
    return os.path.join(BASE_WATCH_DIR, current_date, "lev0")


def wait_watch_dir_exist(current_date):
    """
    Wait for the watch directory to exist before starting.

    Args:
        current_date (str): YYYYMMDD the UT date to monitor.

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
            file_loc = os.path.join(watch_dir, file_name)
            try:
                shutil.copy2(file_loc, dest_dir)
                copied_files.add(file_name)
            except Exception as e:
                log.info(f"Error copying {file_loc}: {e}")

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
