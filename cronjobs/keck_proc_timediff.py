import os
import argparse
from datetime import datetime

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Compare modification times of files between two directories.")
    parser.add_argument("--date", required=True, help="Specify the date in YYYYMMDD format")
    parser.add_argument("--level", type=str, required=False, help="The level to compare",
                        choices=['L0', 'L1', 'L2', '2D'])
    return parser.parse_args()

# Function to get the modification time of a file
def get_modification_time(file_path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(file_path))
    except FileNotFoundError:
        return None

# Function to calculate time difference
def time_difference(time1, time2):
    return round((time2 - time1).total_seconds() / 60.0, 2)

# Function to log messages to a file and print to the console
def log(message, log_file):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

def main():
    # Parse arguments
    args = parse_args()
    date = args.date
    level = args.level

    # Log file
    current_time = datetime.utcnow().strftime("%Hh%Mm")
    log_dir = "/kpfdata/logs/DailyRuns"
    os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists
    log_file = f"{log_dir}/process_time_{date}_{current_time}_{level}.log"

    # Directories
    dir1 = f"/koadata/KPF/{date}/lev0/"
    dir2 = f"/data/data_drp/{level}/{date}/"

    # Start logging
    log(f"Process started for date: {date}", log_file)
    log("----------------------------------", log_file)

    # Check directories
    if not os.path.isdir(dir1):
        log(f"Directory does not exist: {dir1}", log_file)
        return
    if not os.path.isdir(dir2):
        log(f"Directory does not exist: {dir2}", log_file)
        return

    # Iterate over files in dir1
    for file1 in os.listdir(dir1):
        if file1.endswith(".fits"):
            base_name = file1[:-5]  # Remove the '.fits' extension
            file2 = os.path.join(dir2, f"{base_name}_{level}.fits")
            file1_path = os.path.join(dir1, file1)

            # Get modification times
            time1 = get_modification_time(file1_path)
            time2 = get_modification_time(file2)

            if time1 and time2:
                # Calculate the time difference
                diff = time_difference(time1, time2)
                log(f"File: {base_name}", log_file)
                # log(f"  {file1_path} modified at: {time1}", log_file)
                # log(f"  {file2} modified at: {time2}", log_file)
                # log(f"  Time difference: {diff} minutes", log_file)
                log(f"  {diff} minutes", log_file)
                log("----------------------------------", log_file)
            else:
                log(f"Missing file: {'dir1' if not time1 else 'dir2'} for {base_name}", log_file)
                log("----------------------------------", log_file)

    log("Process completed.", log_file)

if __name__ == "__main__":
    main()