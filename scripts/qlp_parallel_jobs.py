#!/usr/bin/env python3

import numpy as np
import os
import sys
import glob
import argparse
import subprocess
from astropy.io import fits

def main(start_date, end_date):
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

    # to-do: start_date > end_date and stop, or flip them
    # to-do: add print statement about the start and stop dates

    base_dir = "/data"
    missing_L0 = []
    all_files = glob.glob(f"{base_dir}/L1/????????/*.fits")
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
    filtered_files = np.array(all_files)[filtered_indices].tolist()
    print(f"Number of files for parallel Quicklook processing: {len(filtered_files)}")
    
    # to-do check if there are no matching files and stop if so
        
    strings = ' '.join(filtered_files)
    ncpus=100 # to-do - make this a parameter
    command = f"""bash -c "parallel -j {ncpus} -k --bar --ungroup bash -c 'echo \\"Starting Quicklook instance {{}}\\"; config=\$(mktemp) && sed \\"s|INSERT_FITS_PATH|{{}}|\\" configs/quicklook_parallel.cfg > \\"\\$config\\" && kpf -c \\"\\$config\\" -r recipes/quicklook_match.recipe && rm \\"\\$config\\"' ::: {strings}" """

    #print()
    #print()
    #print(len(filtered_files))
    #print()
    ##print(strings)
    #print()
    #print(command)
    #print()

    # to-do: add a mode where it will just print out the file names
    # to-do: add a mode where it only shows the bar and not stdout

    try:
        # Execute the command without capturing the output, allowing real-time display in the terminal
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
    #except Error as e:
    #    print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process KPF files.')
    parser.add_argument('start_date', type=str, help='Start date in YYYYMMDD.SSSSS format')
    parser.add_argument('end_date', type=str, help='End date in YYYYMMDD.SSSSS format')

    args = parser.parse_args()
    main(args.start_date, args.end_date)
    
    # to-do: check if the script is being run in Docker