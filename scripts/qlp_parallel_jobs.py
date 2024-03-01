#!/usr/bin/env python3

import numpy as np
import os
import sys
import glob
import argparse
import subprocess
import tempfile

def main(start_date, end_date, l0, d2, l1, l2, ncpu):
    """
    Script Name: qlp_parallel_jobs.py
   
    Description:
      

    Options:
   

    Usage:
      python kpf_processing_progress.py YYYYMMDD.SSSSS YYYYMMDD.SSSSS --all --L0 --2D --L1 --L2
    
    Example:
      ./scripts/qlp_parallel_jobs.py 20230101.12345.67 20230101.17
    """

    if start_date.count('.') == 2:
        start_date = start_date[:start_date.rfind('.')] + start_date[start_date.rfind('.')+1:]
    if end_date.count('.') == 2:
        end_date = end_date[:end_date.rfind('.')] + end_date[end_date.rfind('.')+1:]
    start_date = float(start_date)
    end_date = float(end_date)
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    base_dir = "/data"
    all_files = []
    if ((not l0) and (not d2) and (not l1) and (not l2)) or l0:
        print("Checking L0 files")
        all_files.extend(glob.glob(f"{base_dir}/L0/????????/*.fits"))
    if ((not l0) and (not d2) and (not l1) and (not l2)) or d2:
        all_files.extend(glob.glob(f"{base_dir}/2D/????????/*_2D.fits"))
        print("Checking 2D files")
    if ((not l0) and (not d2) and (not l1) and (not l2)) or l1:
        all_files.extend(glob.glob(f"{base_dir}/L1/????????/*_L1.fits"))
        print("Checking L1 files")
    if ((not l0) and (not d2) and (not l1) and (not l2)) or l2:
        all_files.extend(glob.glob(f"{base_dir}/L2/????????/*_L2.fits"))
        print("Checking L2 files")
    print("Processing filenames")
    all_files = [item for item in all_files if '-' not in item]  # remove bad files like `KP.20240101.00000.00-1.fits`
    base_names = np.array([os.path.basename(file) for file in all_files], dtype='U')
    base_names = np.where(np.char.startswith(base_names, 'KP.'), np.char.replace(base_names, 'KP.',   '', count=1), base_names)
    base_names = np.where(np.char.endswith(base_names, '.fits'), np.char.replace(base_names, '.fits', '', count=1), base_names)
    base_names = np.where(np.char.endswith(base_names, '_2D'),   np.char.replace(base_names, '_2D',   '', count=1), base_names)
    base_names = np.where(np.char.endswith(base_names, '_L1'),   np.char.replace(base_names, '_L1',   '', count=1), base_names)
    base_names = np.where(np.char.endswith(base_names, '_L2'),   np.char.replace(base_names, '_L2',   '', count=1), base_names)
    base_dates = np.array([item[:item.rfind('.')] + item[item.rfind('.')+1:] for item in base_names])
    base_dates = base_dates.astype(float)
    filtered_indices = np.where((base_dates >= start_date) & (base_dates <= end_date))[0]
    filtered_files = np.array(all_files)[filtered_indices]
    sorted_indices = np.argsort([file.split('/')[-1] for file in filtered_files])
    sorted_paths = filtered_files[sorted_indices]
    sorted_files = sorted_paths.tolist()
    print(f"Number of files for parallel Quicklook processing: {len(filtered_files)}")

    # Create a temporary file and write the sorted file paths to it
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
        tmpfile_name = tmpfile.name
        for file_path in sorted_files:
            tmpfile.write(file_path + '\n')
    command = f"""bash -c "parallel -j {ncpu} -k --bar bash -c 'echo \\"Starting Quicklook instance {{}}\\"; config=\$(mktemp) && sed \\"s|INSERT_FITS_PATH|{{}}|\\" configs/quicklook_parallel.cfg > \\"\\$config\\" && kpf -c \\"\\$config\\" -r recipes/quicklook_match.recipe && rm \\"\\$config\\"' :::: {tmpfile_name}" """
    # note: this version of the command prints on the verbose output with every line from the logger
    #command = f"""bash -c "parallel -j {ncpu} -k --bar --ungroup bash -c 'echo \\"Starting Quicklook instance {{}}\\"; config=\$(mktemp) && sed \\"s|INSERT_FITS_PATH|{{}}|\\" configs/quicklook_parallel.cfg > \\"\\$config\\" && kpf -c \\"\\$config\\" -r recipes/quicklook_match.recipe && rm \\"\\$config\\"' :::: {tmpfile_name}" """
    
    try:
        subprocess.run(command, shell=True, check=True)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch Quicklook pipeline for selected observations.')
    parser.add_argument('start_date', type=str, help='Start date as YYYYMMDD, YYYYMMDD.SSSSS, or YYYYMMDD.SSSSS.SS')
    parser.add_argument('end_date',  type=str, help='End date as YYYYMMDD, YYYYMMDD.SSSSS, or YYYYMMDD.SSSSS.SS')
    parser.add_argument('--ncpu', type=str, default=10, help='Number of cores for parallel processing')
    parser.add_argument('--l0',   action='store_true', help='Select all L0 files')
    parser.add_argument('--2d',   action='store_true', dest='d2', help='Select all 2D files')
    parser.add_argument('--l1',   action='store_true', help='Select all L1 files')
    parser.add_argument('--l2',   action='store_true', help='Select all L2 files')

    args = parser.parse_args()
    main(args.start_date, args.end_date, args.l0, args.d2, args.l1, args.l2, args.ncpu)
    
    # to-do: check if the script is being run in Docker
    # to-do: start_date > end_date and stop, or flip them
    # to-do: add a mode where it will just print out the file names
    # to-do check if there are no matching files and stop if so
