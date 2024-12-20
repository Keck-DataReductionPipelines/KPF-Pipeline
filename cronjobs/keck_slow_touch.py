import os
import time
import argparse
from glob import glob
from datetime import datetime, timezone

import keck_utils as utils

APP_PATH = os.path.abspath(os.path.dirname(__file__))


def parse_cmdline():
    parser = argparse.ArgumentParser(
        description="Touch the data in the data workspace for a night."
    )
    parser.add_argument("--date", type=str, required=False, help="The UT date.")
    parser.add_argument("--fits", type=str, required=False, help="The data directory to touch")
    parser.add_argument("--log", type=str, required=False, help="The log location")

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_cmdline()

    # get the date to process and the unique string for docker and logs
    if not args.date:
        procdate = datetime.now(timezone.utc).strftime('%Y%m%d')
    else:
        procdate = args.date

    #get the config file
    cfg = utils.cfg_init(APP_PATH, 'keck_kpfcron.cfg')

    data_workspace = utils.get_cfg(cfg, 'dirs', 'data_workspace')
    data_drp = utils.get_cfg(cfg, 'dirs', 'data_drp')

    if not args.fits:
        fits_dir = f"{data_workspace}/L0/{procdate}/"
    else:
        fits_dir = f"{args.fits}/{procdate}"

    # start the log
    if not args.log:
        logs_root = f"{data_drp}/logs/{procdate}"
    else:
        logs_root = f"{args.log}/{procdate}"

    os.makedirs(logs_root, exist_ok=True)
    log_name = f'keck_slow_touch_{procdate}'

    tm_sleep_before_start = 5

    log = utils.configure_logger(logs_root, f"{log_name}")
    log.info(f'starting slow touch,  sleeping {tm_sleep_before_start}')
    time.sleep(tm_sleep_before_start)

    # Get all FITS files in the directory
    fits_files = glob(os.path.join(fits_dir, '*.fits'))

    # 'Touch' each file with a 1-second interval
    for fits_file in fits_files:
        # Update the access and modification times to the current time
        os.utime(fits_file, None)
        log.info(f"Touched {fits_file}")
        # Wait for 30 second before processing the next file
        time.sleep(30)
