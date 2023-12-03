import os
import sys
import glob
import time
import argparse
import subprocess
import pandas as pd
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
    Method to check if the exposure is not a Bias or Dark image (which don't produce L2).
    Returns False if the object is a Bias or Dark and True otherwise
    """
    if 'IMTYPE' in header:
        isBias = 'Bias' in header['IMTYPE']
        isDark = 'Dark' in header['IMTYPE']
        #print("header['IMTYPE'] = " + str(header['IMTYPE']))
        #print('Bias = ' + str(isBias))
        #print('Dark = ' + str(isBias))
        #print('not (isBias or isDark) = ' + str(not (isBias or isDark)))
    else:
        return True
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


def main(start_date, end_date, print_files, touch_missing, check_version, print_files_2D, print_files_L1, print_files_L2):
    """
    Script Name: kpf_processing_progress.py
   
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
      --help            Display this message
      --print_files     Display missing file names (or files that fail other criteria)
      --print_files_2D  Display missing 2D file names (or files that fail other criteria)
      --print_files_L1  Display missing L1 file names (or files that fail other criteria)
      --print_files_L2  Display missing L2 file names (or files that fail other criteria)
      --touch_missing   Touch the base L0 files of missing 2D/L1/L2 files
      --check_version   Checks that each 2D/L1/L2 file has the latest Git version number for the KPF-Pipeline
   
    Usage:
      python kpf_processing_progress.py YYYYMMDD [YYYYMMDD] [--print_files]
   
    Example:
      python kpf_processing_progress.sh 20231114 20231231 --print_files
    """

    base_dir = "/data/kpf"
    missing_L0 = []

    if check_version:
        try: 
            current_version = get_latest_git_version()
        except Exception as e:
            print('Failed to determine latest version of KPF-DRP: ' + str(e))
            current_version = '0.0.0'

    junk_file_csv = '/data/kpf/reference/Junk_Observations_for_KPF.csv'
    ignore_junk = True
    if ignore_junk:
        if os.path.isfile(junk_file_csv):
            df = pd.read_csv(junk_file_csv, header=1)
            df_junk = df.iloc[:, 0]
            junk_file_missing = False
        else:
            print('File of junked observations not found: ' + str(junk_file_csv))
            print('Junked file not ignored.')
            junk_file_missing = True

    print()
    print(f"{'DATECODE':<8} | {'LAST L0 MOD DATE':<16} | {'2D PROCESSING':<14} | {'L1 PROCESSING':<14} | {'L2 PROCESSING':<14}")
    print("-" * 78)

    # Loop over dates
    dir_paths = glob.glob(f"{base_dir}/L0/{start_date[:4]}????")
    sorted_dir_paths = sorted(dir_paths, key=lambda x: int(x.split('/')[-1]))
    for dir_path in sorted_dir_paths:
        datecode = os.path.basename(dir_path)
        if not datecode.isdigit() or not start_date <= datecode <= end_date:
            continue

        total_count = {"2D": 0, "L1": 0, "L2": 0}
        match_count = {"2D": 0, "L1": 0, "L2": 0}
        recent_mod_date = 0
        
        pattern = f"{dir_path}/KP.{datecode}.?????.??.fits"
        obs_dict = [
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
        sorted_obs_dict = sorted(obs_dict, key=lambda x: x['obsID'])
        obs_dict = sorted_obs_dict

        # Loop over exposures
        for i, obs in enumerate(obs_dict):

#            if obs['obsID'] == 'KP.20231120.12674.58': 
#                print()
#                print()
#                print()
#                print()
#                print()
#                print()
#                print()
#                print()
#                print()
#                print()
#                print()
#                print()
#                print()
#                print()
#                print()
#                print()
            
            # Determine if the file is junk
            if not junk_file_missing: 
                if obs['obsID'] in df_junk.values:
                    obs['isJunk'] = True
                else:
                    obs['isJunk'] = False
            else:
                obs['isJunk'] = False
            
            if not obs['isJunk']:
                with fits.open(obs['L0_filename']) as hdul:
                    header = hdul[0].header
    
                    # L0 files
                    if obs['L0_modtime'] > recent_mod_date:
                        recent_mod_date = obs['L0_modtime']
    
                    # 2D files - checks applied (in order):
                    #    - not junk
                    #    - file present
                    #    - 2D modification time more recent than L0 modification time
                    #    - current DRP version number (if check_version option selected)
                    total_count["2D"] += 1
                    if os.path.isfile(obs['2D_filename']):
                        obs['2D_exists'] = True
                        obs['2D_modtime'] = os.path.getmtime(obs['2D_filename'])
                        if obs['2D_modtime'] > obs['L0_modtime'] :
                            if check_version:
                                with fits.open(obs['2D_filename']) as hdul_2D:
                                    header_2D = hdul_2D[0].header
                                    if is_current_version(header_2D, current_version):
                                        match_count["2D"] += 1
                                    else:
                                        missing_L0.append(obs['L0_filename'])
                                        if print_files_2D:
                                            print('2D with old DRP version: ' + str(obs['2D_filename']))
                            else:
                                match_count["2D"] += 1
                        else:
                            missing_L0.append(obs['L0_filename'])
                            if print_files_2D:
                                print('2D with old modification time: ' + str(obs['2D_filename']))
                    else:
                        missing_L0.append(obs['L0_filename'])
                        if print_files_2D:
                            print('Missing 2D: ' + str(obs['2D_filename']))
    
                    # L1 files - checks applied (in order):
                    #    - not junk
                    #    - file present
                    #    - L1 modification time more recent than L0 modification time
                    #    - current DRP version number (if check_version option selected)
                    if green_red_cahk_present(header):
                        total_count["L1"] += 1
                        if os.path.isfile(obs['L1_filename']):
                            obs['L1_exists'] = True
                            obs['L1_modtime'] = os.path.getmtime(obs['L1_filename'])
                            if obs['L1_modtime'] > obs['L0_modtime'] :
                                if check_version:
                                    with fits.open(obs['L1_filename']) as hdul_2D:
                                        header_2D = hdul_2D[0].header
                                        if is_current_version(header_2D, current_version):
                                            match_count["L1"] += 1
                                        else:
                                            missing_L0.append(obs['L0_filename'])
                                            if print_files_2D:
                                                print('L1 with old DRP version: ' + str(obs['L1_filename']))
                                else:
                                    match_count["L1"] += 1
                            else:
                                missing_L0.append(obs['L0_filename'])
                                if print_files_2D:
                                    print('L2 with old modification time: ' + str(obs['L1_filename']))
                        else:
                            missing_L0.append(obs['L0_filename'])
                            if print_files_L1:
                                print('Missing L1: ' + str(obs['L1_filename']))

                    # L2 files - checks applied (in order):
                    #    - not junk
                    #    - Green, Red, or CaHK extension present
                    #    - not a Dark or Bias exposure
                    #    - file present
                    #    - L2 modification time more recent than L0 modification time
                    #    - current DRP version number (if check_version option selected)
                    if green_red_cahk_present(header) and not_bias_or_dark(header):
                        total_count["L2"] += 1
                        if os.path.isfile(obs['L2_filename']):
                            obs['L2_exists'] = True
                            obs['L2_modtime'] = os.path.getmtime(obs['L2_filename'])
                            if obs['L2_modtime'] > obs['L0_modtime'] :
                                if check_version:
                                    with fits.open(obs['L2_filename']) as hdul_2D:
                                        header_2D = hdul_2D[0].header
                                        if is_current_version(header_2D, current_version):
                                            match_count["L2"] += 1
                                        else:
                                            missing_L0.append(obs['L0_filename'])
                                            if print_files_2D:
                                                print('L2 with old DRP version: ' + str(obs['L2_filename']))
                                else:
                                    match_count["L2"] += 1
                            else:
                                missing_L0.append(obs['L0_filename'])
                                if print_files_2D:
                                    print('L2 with old modification time: ' + str(obs['L2_filename']))
                        else:
                            missing_L0.append(obs['L0_filename'])
                            if print_files_L2:
                                print('Missing L2: ' + str(obs['L2_filename']))

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

    if print_files:
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
            confirm = input("Do you want to touch these " + str(len(missing_L0_nodupes))+ " files? [y/N] ")
            if confirm.lower() == 'y':
                for L0_file in missing_L0_nodupes:
                    try:
                        print(f"touch {L0_file}")
                        os.utime(L0_file, None)
                    except Exception as e:
                        print()
                        print(f"Error: unable to touch {L0_file}")
                        print(str(e))
                        print()
                    time.sleep(0.2)
        else:
            print("All files are up to date.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process KPF files.')
    parser.add_argument('start_date', type=str, help='Start date in YYYYMMDD format')
    parser.add_argument('end_date', type=str, help='End date in YYYYMMDD format')
    parser.add_argument('--print_files',    action='store_true', help='Print file L0 names of missing 2D/L1/L2 files')
    parser.add_argument('--touch_missing',  action='store_true', help='Touch the base L0 files of missing 2D/L1/L2 files')
    parser.add_argument('--check_version',  action='store_true', help='Check 2D/L1/L2 files for latest Git version of DRP processing')
    parser.add_argument('--print_files_2D', action='store_true', help='Print 2D file names where missing and not junk')
    parser.add_argument('--print_files_L1', action='store_true', help='Print L1 file names where missing and not junk')
    parser.add_argument('--print_files_L2', action='store_true', help='Print L2 file names where missing and not junk')

    args = parser.parse_args()
    main(args.start_date, args.end_date, args.print_files, args.touch_missing, args.check_version, args.print_files_2D, args.print_files_L1, args.print_files_L2)