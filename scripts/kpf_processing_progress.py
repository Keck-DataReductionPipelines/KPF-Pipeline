import os
import sys
import glob
import time
import argparse
import subprocess
from astropy.io import fits
from datetime import datetime

def green_red_cahk_present(header):
    """
    # Function to check GREEN, RED, CA_HK keywords - returns True if any camera was used
    """
    if 'GREEN' in header:
        if 'YES' in header['GREEN']:
            return True
    if 'RED' in header:
        if 'YES' in header['RED']:
            return True
    if 'CA_HK' in header:
        if 'YES' in header['CA_HK']:
            return True
    else:
        return False

def not_bias_or_dark(header):
    """
    Method to check if the exposure is not a Bias or Dark image (which don't produce L2)
    """
    if 'IMTYPE' in header:
        isBias = 'Bias' in header['IMTYPE']
        isDark = 'Dark' in header['IMTYPE']
    else:
        return False
    return not (isBias or isDark)

def is_current_version(header, current_version, debug=False):
    """
    Method to check if the file was processed with the current (or newer) version of KPF-Pipeline
    """
    if 'DRPTAG' in header:
        this_version = header['DRPTAG']
        this_version = this_version.lstrip('v').strip() # trip 'v' and whitespace from version string
        this_version_parts = list(map(int, this_version.split('.')))
        current_version_parts = list(map(int, current_version.split('.')))

        # Pad the shorter list with zeros
        max_length = max(len(this_version_parts), len(current_version_parts))
        this_version_parts.extend([0] * (max_length - len(this_version_parts)))
        current_version_parts.extend([0] * (max_length - len(current_version_parts)))

        # Compare each component
        for this_part, current_part in zip(this_version_parts, current_version_parts):
            if this_part > current_part:
                return True
            elif this_part < current_part:
                return False
    else:
        if debug:
            print("DRPTAG not in header")
        return False

    # If all components are equal
    return True


def get_latest_git_version():
    """
    Method to return the latest Git version number of the KPF-Pipeline repository
    """
    original_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)

    # Get the latest Git version number
    latest_version_cmd = "git describe --tags `git rev-list --tags --max-count=1`"
    latest_version = subprocess.check_output(latest_version_cmd, shell=True).decode().strip()

    # Remove the leading 'v' from the version string, if present
    latest_version = latest_version.lstrip('v')

    os.chdir(original_dir)
    return latest_version


def main(start_date, end_date, print_missing, touch_missing, check_version):
    """
    Script Name: kpf_processing_progress.sh
   
    Description:
      This script searches through /data/kpf/L0/YYYYMMDD subdirectories for L0 
      files matching the pattern KP.YYYYMMDD.NNNNN.NN.fits. It records the most 
      recent modification date of any L0 file and checks if corresponding 
      KP.YYYYMMDD.NNNNN.NN_2D.fits, KP.YYYYMMDD.NNNNN.NN_L1.fits, and 
      KP.YYYYMMDD.NNNNN.NN_L2.fits files in respective directories have a file 
      modification date after the L0 file. For any missing 2D, L1, and L2 files, 
      or files of those types with older modification dates, the script checks 
      the 'GREEN' and 'RED' keywords in the FITS header of the L0 file 
      and excludes files from the missing count if the Green and Red cameras 
      are both not selected. The script outputs a summary for each YYYYMMDD 
      directory, showing the count of such files and the most recent 
      L0 modification date. The script takes a starting date (YYYYMMDD) as an 
      argument and optionally an end date and flags to print missing files and
      touch the base L0 files of missing 2D/L1/L2 files.
   
    Options:
      --help           Display this message
      --print_missing  Display missing file names (or files that fail other criteria)
      --touch_missing  Touch the base L0 files of missing 2D/L1/L2 files
      --check_version  Checks that each 2D/L1/L2 file has the latest Git version number for the KPF-Pipeline
   
    Usage:
      python kpf_processing_progress.py YYYYMMDD [YYYYMMDD] [--print_missing]
   
    Example:
      python kpf_processing_progress.sh 20231114 20231231 --print_missing
    """

    base_dir = "/data/kpf"
    missing_L0 = []

    if check_version:
        try: 
            current_version = get_latest_git_version()
        except Exception as e:
            print('Failed to determine latest version of KPF-DRP: ' + str(e))
            current_version = '0.0.0'

    print()
    print(f"{'DATECODE':<8} | {'LAST L0 MOD DATE':<16} | {'2D PROCESSING':<14} | {'L1 PROCESSING':<14} | {'L2 PROCESSING':<14}")
    print("-" * 78)

    for dir_path in glob.glob(f"{base_dir}/L0/{start_date[:4]}????"):
        datecode = os.path.basename(dir_path)
        if not datecode.isdigit() or not start_date <= datecode <= end_date:
            continue

        total_count = {"2D": 0, "L1": 0, "L2": 0}
        match_count = {"2D": 0, "L1": 0, "L2": 0}
        recent_mod_date = 0

        pattern = f"{dir_path}/KP.{datecode}.?????.??.fits"
        date_dict = [
            {
                'L0_filename': f,
                'L0_modtime': os.path.getmtime(f),
                'obsID': '.'.join(os.path.basename(f).split('.')[:-1]),
                '2D_filename': f.replace(f"{base_dir}/L0/", f"{base_dir}/2D/").replace('.fits', '_2D.fits'),
                'L1_filename': f.replace(f"{base_dir}/L0/", f"{base_dir}/L1/").replace('.fits', '_L1.fits'),
                'L2_filename': f.replace(f"{base_dir}/L0/", f"{base_dir}/L2/").replace('.fits', '_L2.fits'),
                '2D_exists': False,
                'L1_exists': False,
                'L2_exists': False,
            } 
            for f in glob.glob(pattern)
        ]

        for i, date in enumerate(date_dict):
            with fits.open(date_dict[i]['L0_filename']) as hdul:
                header = hdul[0].header

                # L0 files
                if date_dict[i]['L0_modtime'] > recent_mod_date:
                    recent_mod_date = date_dict[i]['L0_modtime']

                # 2D files - checks applied:
                #    - file present
                #    - 2D modification time more recent than L0 modification time
                #    - current DRP version number (if check_version option)
                total_count["2D"] += 1
                if os.path.isfile(date_dict[i]['2D_filename']):
                    date_dict[i]['2D_exists'] = True
                    date_dict[i]['2D_modtime'] = os.path.getmtime(date_dict[i]['2D_filename'])
                    if date_dict[i]['2D_modtime'] > date_dict[i]['L0_modtime'] :
                        if check_version:
                            with fits.open(date_dict[i]['2D_filename']) as hdul_2D:
                                header_2D = hdul_2D[0].header
                                if is_current_version(header_2D, current_version):
                                    match_count["2D"] += 1
                                else:
                                    missing_L0.append(date_dict[i]['L0_filename'])
                        else:
                            match_count["2D"] += 1
                    else:
                        missing_L0.append(date_dict[i]['L0_filename'])
                else:
                    missing_L0.append(date_dict[i]['L0_filename'])

                # L1 files - checks applied:
                #    - file present
                #    - L1 modification time more recent than L0 modification time
                #    - current DRP version number (if check_version option)
                total_count["L1"] += 1
                if os.path.isfile(date_dict[i]['L1_filename']):
                    date_dict[i]['L1_exists'] = True
                    date_dict[i]['L1_modtime'] = os.path.getmtime(date_dict[i]['L1_filename'])
                    if date_dict[i]['L1_modtime'] > date_dict[i]['L0_modtime'] :
                        if green_red_cahk_present(header):
                            if check_version:
                                with fits.open(date_dict[i]['L1_filename']) as hdul_2D:
                                    header_2D = hdul_2D[0].header
                                    if is_current_version(header_2D, current_version):
                                        match_count["L1"] += 1
                                    else:
                                        missing_L0.append(date_dict[i]['L0_filename'])
                            else:
                                match_count["L1"] += 1
                        else:
                            missing_L0.append(date_dict[i]['L0_filename'])
                    else:
                        missing_L0.append(date_dict[i]['L0_filename'])
                else:
                    missing_L0.append(date_dict[i]['L0_filename'])

                # L2 files - checks applied:
                #    - file present
                #    - L2 modification time more recent than L0 modification time
                #    - current DRP version number (if check_version option)
                #    - Green, Red, or CaHK extension present
                #    - not a Dark or Bias exposure
                total_count["L2"] += 1
                if os.path.isfile(date_dict[i]['L2_filename']):
                    date_dict[i]['L2_exists'] = True
                    date_dict[i]['L2_modtime'] = os.path.getmtime(date_dict[i]['L2_filename'])
                    if date_dict[i]['L2_modtime'] > date_dict[i]['L0_modtime'] :
                        if green_red_cahk_present(header) and not_bias_or_dark(header):
                            if check_version:
                                with fits.open(date_dict[i]['L2_filename']) as hdul_2D:
                                    header_2D = hdul_2D[0].header
                                    if is_current_version(header_2D, current_version):
                                        match_count["L2"] += 1
                                    else:
                                        missing_L0.append(date_dict[i]['L0_filename'])
                            else:
                                match_count["L2"] += 1
                        else:
                            missing_L0.append(date_dict[i]['L0_filename'])
                    else:
                        missing_L0.append(date_dict[i]['L0_filename'])
                else:
                    missing_L0.append(date_dict[i]['L0_filename'])

        missing_L0_nodupes = []
        for L0 in missing_L0:
            if L0 not in missing_L0_nodupes:
                missing_L0_nodupes.append(L0)

        formatted_recent_mod_date = datetime.fromtimestamp(recent_mod_date).strftime("%Y-%m-%d %H:%M")
        processing_info = []
    
        for file_type in ['2D', 'L1', 'L2']:
            if total_count[file_type] > 0:
                percentage = match_count[file_type] * 100 // total_count[file_type]
                processing_info.append(f"{match_count[file_type]:4d}/{total_count[file_type]:<4d} {percentage:3d}%")
    
        print(f"{datecode:<8} | {formatted_recent_mod_date:<16} | {' | '.join(processing_info)}")
    print("-" * 78)
    print()

    if print_missing:
        if len(missing_L0_nodupes) > 0:
            print("These L0 files have corresponding 2D, L1, or L2 files that are missing or old:")
            for L0_file in missing_L0_nodupes:
                print(L0_file)
        else:
            print("All files are up to date.")

    if touch_missing:
        if len(missing_L0_nodupes) > 0:
            print("These L0 files have corresponding 2D, L1, or L2 files that are missing or old:")
            for L0_file in missing_L0_nodupes:
                print(L0_file)
            confirm = input("Do you want to touch these files? [y/N] ")
            if confirm.lower() == 'y':
                for L0_file in missing_L0_nodupes:
                    print(f"touch {L0_file}")
                    os.utime(L0_file, None)
                    time.sleep(0.2)
        else:
            print("All files are up to date.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process KPF files.')
    parser.add_argument('start_date', type=str, help='Start date in YYYYMMDD format')
    parser.add_argument('end_date', type=str, help='End date in YYYYMMDD format')
    parser.add_argument('--print_missing', action='store_true', help='Print missing file names')
    parser.add_argument('--touch_missing', action='store_true', help='Touch the base L0 files of missing 2D/L1/L2 files')
    parser.add_argument('--check_version', action='store_true', help='Check 2D/L1/L2 files for latest Git version of DRP processing')

    args = parser.parse_args()
    main(args.start_date, args.end_date, args.print_missing, args.touch_missing, args.check_version)