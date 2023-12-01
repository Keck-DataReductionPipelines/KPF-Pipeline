import os
import sys
import glob
import argparse
import subprocess
from astropy.io import fits
from datetime import datetime


# Function to check GREEN, RED, CA_HK keywords - returns True if any camera was used
def green_red_cahk_present(header):
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
    if 'IMTYPE' in header:
        isBias = 'Bias' in header['IMTYPE']
        isDark = 'Dark' in header['IMTYPE']
    else:
        return False

    return not (isBias or isDark)

def main(start_date, end_date, print_missing, touch_missing):
    base_dir = "/data/kpf"
    missing_L0 = []

    print()
    print(f"{'DATECODE':<8} | {'LAST L0 MOD DATE':<16} | {'2D PROCESSING':<14} | {'L1 PROCESSING':<14} | {'L2 PROCESSING':<14}")
    print("-" * 78)


    for dir_path in glob.glob(f"{base_dir}/L0/{start_date[:4]}????"):
        date_code = os.path.basename(dir_path)
        if not date_code.isdigit() or not start_date <= date_code <= end_date:
            continue

        total_count = {"2D": 0, "L1": 0, "L2": 0}
        match_count = {"2D": 0, "L1": 0, "L2": 0}
        recent_mod_date = 0

        pattern = f"{dir_path}/KP.{date_code}.?????.??.fits"
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

                # 2D files
                total_count["2D"] += 1
                if os.path.isfile(date_dict[i]['2D_filename']):
                    date_dict[i]['2D_exists'] = True
                    date_dict[i]['2D_modtime'] = os.path.getmtime(date_dict[i]['2D_filename'])
                    if date_dict[i]['2D_modtime'] > date_dict[i]['L0_modtime'] :
                        match_count["2D"] += 1
                    else:
                        missing_L0.append(date_dict[i]['L0_filename'])

                # L1 files
                total_count["L1"] += 1
                if os.path.isfile(date_dict[i]['L1_filename']):
                    date_dict[i]['L1_exists'] = True
                    date_dict[i]['L1_modtime'] = os.path.getmtime(date_dict[i]['L1_filename'])
                    if date_dict[i]['L1_modtime'] > date_dict[i]['L0_modtime']:
                        if green_red_cahk_present(header):
                            match_count["L1"] += 1
                        else:
                            missing_L0.append(date_dict[i]['L0_filename'])
                    else:
                        missing_L0.append(date_dict[i]['L0_filename'])

                # L2 files
                total_count["L2"] += 1
                if os.path.isfile(date_dict[i]['L2_filename']):
                    date_dict[i]['L2_exists'] = True
                    date_dict[i]['L2_modtime'] = os.path.getmtime(date_dict[i]['L2_filename'])
                    if date_dict[i]['L2_modtime'] > date_dict[i]['L0_modtime']:
                        if green_red_cahk_present(header) and not_bias_or_dark(header):
                            match_count["L2"] += 1
                        else:
                            missing_L0.append(date_dict[i]['L0_filename'])
                    else:
                        missing_L0.append(date_dict[i]['L0_filename'])

        #print(missing_L0)
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
    
        print(f"{date_code:<8} | {formatted_recent_mod_date:<16} | {' | '.join(processing_info)}")

    print("-" * 78)

    if print_missing:
        print("The following L0 files have missing corresponding 2D, L1, or L2 files:")
        for L0 in missing_L0_nodupes:
            print(L0)
#        confirm = input("Do you want to touch these files? [y/N] ")
#        if confirm.lower() == 'y':
#            for file in unique_files:
#                print(f"touch {file}")
#                os.utime(file, None)
#    elif touch_missing:
#        print("\nAll files are up to date.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process KPF files.')
    parser.add_argument('start_date', type=str, help='Start date in YYYYMMDD format')
    parser.add_argument('end_date', type=str, help='End date in YYYYMMDD format')
    parser.add_argument('--print_missing', action='store_true', help='Print missing file names')
    parser.add_argument('--touch_missing', action='store_true', help='Touch the base L0 files of missing 2D/L1/L2 files')

    args = parser.parse_args()
    main(args.start_date, args.end_date, args.print_missing, args.touch_missing)