from datetime import datetime, time, timedelta

import subprocess
import sys
import os


def run_script(script_name):
    """Run the specified script."""
    print(f"Starting {script_name}")
    try:
        subprocess.run(script_name, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}")
        sys.exit(1)


def calculate_remaining_time(end_time):
    """
    Calculate remaining time in hours until the specified end time.

    Args:
        end_time (datetime.time): The end time to calculate remaining
    """
    now = datetime.now()
    today_end = datetime.combine(now.date(), end_time)

    # Handle cases where end_time is on the next day (overnight range)
    if now > today_end:
        today_end += timedelta(days=1)

    remaining_time = today_end - now

    return remaining_time.total_seconds() / 3600


def main():
    now = datetime.utcnow().time()

    py_loc = "/usr/local/anaconda/bin/python"
    cron_dir = os.path.join(os.environ["KPFCRONJOB_CODE"], 'cronjobs')

    # Full Pipeline Run
    fullrun_start = time(19, 0)
    fullrun_end = time(1, 50)

    # QLP Pipelines
    qlp_runs_start = time(1, 0)
    qlp_runs_end = time(23, 59, 59)

    # Nightly Watch
    watch_start = time(2, 5)
    watch_end = time(18, 15)

    date_str = datetime.utcnow().strftime("%Y%m%d")

    if fullrun_start <= now or now <= fullrun_end:
        if now < fullrun_start:
            # use yesterday UTD
            date_str = (datetime.utcnow() - timedelta(days=1)).strftime("%Y%m%d")

        remaining = calculate_remaining_time(fullrun_end)

        # skip masters if the nightly processing had started before shutdown
        nightly_log = os.path.join('/data/data_drp/logs/', date_str, f'keck_kpf_nightly_{date_str}.log')
        if not os.path.exists(nightly_log):
            cmd = f"{cron_dir}/keck_run_full.sh --date {date_str} >> /data/logs/DailyRuns/keck_run_daily_{date_str}.log"
            print(f"Running full-run without masters for {remaining} hrs.")
        else:
            cmd = f"{py_loc} {cron_dir}/keck_kpfcron_nightly.py --date {date_str} --timer {remaining}"
            print(f"Running full-run for {remaining} hrs.")

        run_script(cmd)

    elif qlp_runs_start <= now <= qlp_runs_end:
        remaining = calculate_remaining_time(time(23, 59, 59))
        print(f"Running QLP for {remaining} hrs.")

        for lev in ('L0', 'L1', 'L2', 'masters'):
            cmd = f"{py_loc} {cron_dir}/keck_kpfcron_qlp.py --date {date_str} --level {lev} --timer {remaining}"
            run_script(cmd)

    elif watch_start <= now <= watch_end:
        remaining = calculate_remaining_time(watch_end)
        print(f"Running Nightly Watch for {remaining} hrs.")

        cmd = f"{py_loc} {cron_dir}/keck_kpfcron_nightly_watch.py --date {date_str} --timer {remaining}"
        run_script(cmd)

    else:
        print("No script to run at this time.")


if __name__ == "__main__":
    main()
