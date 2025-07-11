"""
The keck KPF DRP Data Products Scrubber

-lfuhrman 2024Dec10

"""
import os
import shutil
import argparse
import keck_utils as utils

from datetime import datetime, timezone, timedelta

APP_PATH = os.path.abspath(os.path.dirname(__file__))

class KPFScrubDRP:
    """
    The KPF Data Scrubber
    """
    def __init__(self, today):

        # cfg file parameters
        self.data_drp = None
        self.data_root = None
        self.logs_base = None
        self.data_workspace = None

        # read the configuration
        self.read_cron_cfg()

        # set-up the log
        self.logs_root = f"{self.logs_base}/Scrubber/"
        self.log_name = f'keck_kpfdrp_scrub_{today}'
        os.makedirs(self.logs_root, exist_ok=True)
        self.log = utils.configure_logger(self.logs_root, f"{self.log_name}")
        utils.log_stub('Starting', f'KPF DRP Data Scrubber', today, self.log)

    def read_cron_cfg(self):
        """
        Read the configuration for the data locations
        """
        cfg_name = 'keck_kpfcron.cfg'
        cfg = utils.cfg_init(APP_PATH, cfg_name)

        # work directories
        self.data_workspace = utils.get_cfg(cfg, 'dirs', 'data_workspace')
        self.data_drp = utils.get_cfg(cfg, 'dirs', 'data_drp')
        self.data_root = utils.get_cfg(cfg, 'dirs', 'data_root')
        self.logs_base = utils.get_cfg(cfg, 'dirs', 'logs_base')

    def scrub_workspace(self, utd):
        """
        Scrub all the files that are deemed disposable.  The data workspace
        is an intermediary location for masters calibration and L0 (koadata)
        files to get to the L1/L2 files.

        Args:
            utd (str): the UTD date in format: YYYYMMDD
        """
        dirs_to_scrub = ['L0', '2D', 'logs']
        for direct in dirs_to_scrub:
            scrub_dir = os.path.join(self.data_workspace, direct, utd)
            self.scrub_directory(scrub_dir)

    def scrub_data_drp(self, utd):
        """
        Scrub all the files that are archived.  The data drp location
        is the final location for the DRP output, L1/L2 files.

        Args:
            utd (str): the UTD date in format: YYYYMMDD
        """
        dirs_to_scrub = ['L0', '2D', 'L1', 'L2', 'QLP', 'logs', 'logs/QLP', 'logs/watch']
        for direct in dirs_to_scrub:
            scrub_dir = os.path.join(self.data_drp, direct, utd)
            self.scrub_directory(scrub_dir)

        dated_files_dir = os.path.join(self.data_drp, 'outliers')
        self.scrub_dated_files(dated_files_dir, utd)

    def scrub_directory(self, direct):
        """
        Scrub all files in a directory.

        Args:
            direct (str): the full path to the direct to clean.
        """

        # remove left behind links
        if os.path.islink(direct):
            self.log.info(f'removing symlink: {direct}')
            os.unlink(direct)
            return

        if not os.path.isdir(direct):
            return

        self.log.info(f'removing all files in: {direct}')

        try:
            shutil.rmtree(direct)
        except FileNotFoundError:
            self.log.error(f"Directory {direct} does not exist.")
        except PermissionError:
            self.log.error("Permission denied.")
        except Exception as e:
            self.log.error(f"Error: {e}")

        # remove the empty directory


    def scrub_dated_files(self, direct, utd):
        """
        Scrub all files with filenames that contain the UT date (utd).

        Args:
            direct (str): the full path to the direct to clean.
            utd (str): the UTD date in format: YYYYMMDD
        """
        files = [f for f in os.listdir(direct)
                 if os.path.isfile(os.path.join(direct, f))]

        file_list = []
        for file in files:
            if utd not in file:
                continue
            file_list.append(file)

        if not file_list:
            return

        self.log.info(f'removing all files in: {direct} with date: {utd}')

        for file in file_list:
            if utd not in file:
                continue

            file_path = os.path.join(direct, file)
            try:
                os.remove(file_path)
            except Exception as e:
                self.log.error(f"Error removing {file_path}: {e}")


def get_dates_range(utd1, utd2):
    """
    Create a list of dates in the range from UT Date 1 to UT Date 2.
    Args:
        utd1 (str): the UTD date of the end of the range in format: YYYYMMDD
        utd2 (str): the UTD date of the start of the range in format: YYYYMMDD

    Returns:

    """
    start_date = datetime.strptime(utd1, '%Y%m%d')
    end_date = datetime.strptime(utd2, '%Y%m%d')

    date_list = []

    next_date = start_date
    while next_date <= end_date:
        date_list.append(next_date.strftime('%Y%m%d'))
        next_date += timedelta(days=1)

    return date_list


def read_cmd_line():
    parser = argparse.ArgumentParser(description="Start the KPF DRP Data Scrubber.")

    hstr = " UT date for data scrub: "

    parser.add_argument(
        "--utd1wrk", type=str, required=False, help=f"Start {hstr} workspace.",
        default=(datetime.now(timezone.utc) - timedelta(days=19)).strftime('%Y%m%d')
    )
    parser.add_argument(
        "--utd2wrk", type=str, required=False, help=f"End {hstr} workspace.",
        default=(datetime.now(timezone.utc) - timedelta(days=2)).strftime('%Y%m%d')
    )
    parser.add_argument(
        "--utd1drp", type=str, required=False, help=f"Start {hstr} DRP.",
        default=(datetime.now(timezone.utc) - timedelta(days=24)).strftime('%Y%m%d')
    )
    parser.add_argument(
        "--utd2drp", type=str, required=False, help=f"End {hstr} DRP.",
        default=(datetime.now(timezone.utc) - timedelta(days=4)).strftime('%Y%m%d')
    )

    args = parser.parse_args()

    wrk_space_range = get_dates_range(args.utd1wrk, args.utd2wrk)
    drp_space_range = get_dates_range(args.utd1drp, args.utd2drp)

    return wrk_space_range, drp_space_range


def main():
    wrk_range, drp_range = read_cmd_line()

    today = datetime.now(timezone.utc).strftime('%Y%m%d')
    cron_obj = KPFScrubDRP(today)

    # remove data in date range for workspace-data
    for utd in wrk_range:
        cron_obj.scrub_workspace(utd)

    # remove data in date range for drp-data
    for utd in drp_range:
        cron_obj.scrub_data_drp(utd)


if __name__ == '__main__':
    main()
