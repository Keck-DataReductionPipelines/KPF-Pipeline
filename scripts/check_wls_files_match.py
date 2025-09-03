"""
Script to indicate if WLSFILE and WLSFILE2 dates match.
Usage:
    python WLSChecker.py -f /path/to/file.fits
    python WLSChecker.py -d /path/to/fits/files
    Add -l, --list to only print mismatches
    Add -st, --slowtouch to call kpf_slowtouch.sh on files that have flag "no"
"""
import os
import re
import sys
import logging
import argparse
import subprocess
from astropy.io import fits 

def check_file(file_path, quiet_mode=False):
    try:
        wls1 = fits.getval(file_path, "WLSFILE", ext=0)
        wls2 = fits.getval(file_path, "WLSFILE2", ext=0)
    except KeyError:
        if not quiet_mode:
            logging.warning(f"Skipping {file_path}: missing WLSFILE or WLSFILE2 keyword")
        return None
    except Exception as e:
        if not quiet_mode:
            logging.error(f"Error reading {file_path}: {e}")
        return None

    wls1 = str(wls1)
    wls2 = str(wls2)
    m1 = re.search(r"\d{8}", wls1)
    m2 = re.search(r"\d{8}", wls2)
    if not m1 or not m2:
        if not quiet_mode:
            logging.warning(f"Skipping {file_path}: could not find date in WLSFILE/WLSFILE2")
        return None
    date1 = m1.group(0)
    date2 = m2.group(0)

    match_value = "yes" if date1 == date2 else "no"
    try:
        fits.setval(file_path, "WLSMATCH", ext=0, value=match_value,
                    comment="yes if WLSFILE and WLSFILE2 dates match")
    except Exception as e:
        if not quiet_mode:
            logging.error(f"Failed to write WLSMATCH in {file_path}: {e}")
        return None

    if not quiet_mode:
        logging.info(f"Processed {file_path}: WLSFILE={date1}, WLSFILE2={date2}, WLSMATCH='{match_value}'")
    return os.path.basename(file_path) if match_value == "no" else None

def process_directory(directory, quiet_mode=False):
    no_match_files = []
    total_checked = 0

    for dirpath, _, filenames in os.walk(directory):
        for fname in filenames:
            if fname.lower().endswith(".fits"):
                file_path = os.path.join(dirpath, fname)
                result = check_file(file_path, quiet_mode=quiet_mode)
                total_checked += 1
                if result:
                    no_match_files.append(result)

    return no_match_files, total_checked

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if WLSFILE and WLSFILE2 dates match in FITS files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="Path to a single FITS file to check.")
    group.add_argument("-d", "--directory", help="Path to a directory to check recursively.")
    parser.add_argument("-l", "--list", action="store_true", help="Only output names of files where dates do NOT match.")
    parser.add_argument("-st", "--slowtouch", action="store_true", help="Call kpf_slowtouch.sh on all files with WLSMATCH = 'no'")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.list else logging.WARNING,
                        format="%(levelname)s: %(message)s")

    if args.file:
        if not os.path.isfile(args.file):
            logging.error(f"Provided file does not exist: {args.file}")
            sys.exit(1)
        result = check_file(args.file, quiet_mode=args.list)
        if result and args.list:
            print(result)
        elif not result and not args.list:
            print("WLSFILE and WLSFILE2 dates match.")
        elif result and not args.list:
            print(f"WLSFILE and WLSFILE2 dates do NOT match")
        logging.info("WLSMATCH update complete. Files checked: 1")

        if args.slowtouch and result:
            print("\nCalling kpf_slowtouch.sh...")
            base_name = re.sub(r'_L\d+\.fits$', '.fits', result)
            try:
                subprocess.run(["scripts/kpf_slowtouch.sh", base_name], check=True)
            except Exception as e:
                logging.error(f"Failed to execute kpf_slowtouch.sh: {e}")

    elif args.directory:
        if not os.path.isdir(args.directory):
            logging.error(f"Provided directory does not exist: {args.directory}")
            sys.exit(1)

        no_matches, total_checked = process_directory(args.directory, quiet_mode=args.list)

        if args.list:
            for fname in no_matches:
                print(fname)
        else:
            if no_matches:
                print("\nFiles with WLSMATCH = 'no':")
                for fname in no_matches:
                    print(f"- {fname}")
            else:
                print("All files had matching WLSFILE and WLSFILE2 dates.")

        logging.info(f"WLSMATCH update complete. Files checked: {total_checked}")

        if args.slowtouch and no_matches:
            print("\nCalling kpf_slowtouch.sh on mismatched files...")
            cleaned_names = [re.sub(r'_L\d+\.fits$', '.fits', f) for f in no_matches]
            try:
                subprocess.run(["scripts/kpf_slowtouch.sh"] + cleaned_names, check=True)
            except Exception as e:
                logging.error(f"Failed to execute kpf_slowtouch.sh: {e}")
