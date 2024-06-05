#!/usr/bin/env python3

import numpy as np
import re
import os
import sys
import glob
import argparse
import subprocess
import tempfile

def is_running_in_docker():
    try:
        with open('/proc/self/cgroup', 'rt') as ifh:
            return 'docker' in ifh.read()
    except Exception:
        return False


def main(start_date, end_date, l0, d2, l1, l2, master, ncpu, load, print_files):
    """
    Script Name: qlp_parallel.py
   
    Description:
      This script uses the 'parallel' utility to execute the recipe called 
      'recipes/quicklook_match.recipe' to generate standard Quicklook data 
      products.  The script selects all KPF files based on their
      type (L0/2D/L1/L2/master) from the standard data directory using a date 
      range specified by the parameters start_date and end_date.  L0 files are 
      included if the --l0 flag is set or none of the --l0, --2d, --l1, --l2
      flags are set (in which case all data types are included).  The --2d, 
      --l1, and --l2 flags have similar functions.  The script assumes that it
      is being run in Docker and will return with an error message if not. 
      If start_date is later than end_date, the arguments will be reversed 
      and the files with later dates will be processed first.
      
      Invoking the --print_files flag causes the script to print filenames
      but not create QLP data products.
      
      The --ncpu parameter determines the maximum number of cores used.  
      
      The following feature is not operational if this script is run inside of 
      a Docker container: If the --load parameter (a percentage, e.g. 90 = 90%) 
      is set to a non-zero value, this script will be throttled so that no new 
      files will have QLPs processed until the load is below that value.  Note 
      that throttling works in steady state; it is possible to overload the 
      system with the first set of jobs if --ncpu is set too way high.  

    Arguments:
      start_date     Start date as YYYYMMDD, YYYYMMDD.SSSSS, or YYYYMMDD.SSSSS.SS
      end_date       End date as YYYYMMDD, YYYYMMDD.SSSSS, or YYYYMMDD.SSSSS.SS

    Options:
      --l0           Select all L0 files in date range
      --2d           Select all 2D files in date range
      --l1           Select all L1 files in date range
      --l2           Select all L2 files in date range
      --master       Select all master files in date range
      --ncpu         Number of cores used for parallel processing; default=10
      --load         Maximum load (1 min average); default=0 (only activated if !=0)
      --print_files  Display file names matching criteria, but don't generate Quicklook plots
      --help         Display this message
   
    Usage:
      python qlp_parallel.py YYYYMMDD.SSSSS YYYYMMDD.SSSSS --ncpu NCPU --load LOAD --l0 --2d --l1 --l2 --master --print_files
    
    Examples:
      ./scripts/qlp_parallel.py 20230101.12345.67 20230101.17 --ncpu 50 --l0 --2d
      ./scripts/qlp_parallel.py 20240501 20240505 --ncpu 150 --load 90
    """

    if start_date.count('.') == 2:
        start_date = start_date[:start_date.rfind('.')] + start_date[start_date.rfind('.')+1:]
    if end_date.count('.') == 2:
        end_date = end_date[:end_date.rfind('.')] + end_date[end_date.rfind('.')+1:]
    start_date = float(start_date)
    end_date = float(end_date)
    
    if start_date > end_date:
        start_date, end_date = end_date, start_date # swap start_date and end_date
        do_reversed=True
    else:
        do_reversed=False
    
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    base_dir = "/data"
    all_files = []
    if l0 or ((not master) and (not l0) and (not d2) and (not l1) and (not l2)):
        print("Checking L0 files")
        all_files.extend(glob.glob(f"{base_dir}/L0/20??????/*.fits"))
    if d2 or ((not master) and (not l0) and (not d2) and (not l1) and (not l2)):
        all_files.extend(glob.glob(f"{base_dir}/2D/20??????/*_2D.fits"))
        print("Checking 2D files")
    if l1 or ((not master) and (not l0) and (not d2) and (not l1) and (not l2)):
        all_files.extend(glob.glob(f"{base_dir}/L1/20??????/*_L1.fits"))
        print("Checking L1 files")
    if l2 or ((not master) and (not l0) and (not d2) and (not l1) and (not l2)):
        all_files.extend(glob.glob(f"{base_dir}/L2/20??????/*_L2.fits"))
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

    if master or ((not master) and (not l0) and (not d2) and (not l1) and (not l2)):
        print("Adding Master files")
        master_files = []
        master_files.extend(glob.glob(f"{base_dir}/masters/20??????/*.fits"))
        pattern = r'/data/masters/(\d{8})/'
        matching_masters = [
            filename for filename in master_files
            if re.search(pattern, filename) and start_date <= float(re.search(pattern, filename).group(1)) <= end_date
        ]
        sorted_files.extend(matching_masters)

        # Custom sort key function to extract and convert the date part
        def extract_date(filename):
            datepattern = r'/data/.*/(\d{8})/'
            match = re.search(datepattern, filename)
            if match:
                return int(match.group(1))
            return 0  # Default value if the pattern does not match (shouldn't happen if input is consistent)
        
        # Sort again to interleave masters in datecode-sorted list of L0-L2
        sorted_files = sorted(sorted_files, key=extract_date)

    if do_reversed:
        sorted_files = sorted_files[::-1]
 
    print(f"Number of files queued for parallel Quicklook processing: {len(sorted_files)}")
    
    if len(sorted_files) == 0:
        print("Script stopped because no matching files were found.")
    else:
        if print_files:
            print("Matching files:")
            for f in sorted_files:
                print(f)
        else:        
            # Create a temporary file and write the sorted file paths to it
            ncpu_system = os.cpu_count()
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
                tmpfile_name = tmpfile.name
                for file_path in sorted_files:
                    tmpfile.write(file_path + '\n')
            print('Starting parallel with:')
            if int(load) < 0.1:
                print(f'    {ncpu} out of {ncpu_system} cores')
                command = f"""bash -c "parallel -j {ncpu} -k --bar bash -c 'echo \\"Starting Quicklook instance {{}}\\"; config=\$(mktemp) && sed \\"s|INSERT_FITS_PATH|{{}}|\\" configs/quicklook_parallel.cfg > \\"\\$config\\" && kpf -c \\"\\$config\\" -r recipes/quicklook_match.recipe && rm \\"\\$config\\"' :::: {tmpfile_name}" """
            else:
                command = f"""bash -c "parallel -j {ncpu} -k --load {load}% --noswap --bar bash -c 'echo \\"Starting Quicklook instance {{}}\\"; config=\$(mktemp) && sed \\"s|INSERT_FITS_PATH|{{}}|\\" configs/quicklook_parallel.cfg > \\"\\$config\\" && kpf -c \\"\\$config\\" -r recipes/quicklook_match.recipe && rm \\"\\$config\\"' :::: {tmpfile_name}" """
                if is_running_in_docker():
                    print(f'    {ncpu} out of {ncpu_system} cores')
                    print(f'    {load}% maximum load ({int(ncpu_system * float(load)/100)} cores was set) - THIS FEATURE IS NOT OPERATIONAL WHEN RUN INSIDE DOCKER')
                else:
                    print(f'    {ncpu} out of {ncpu_system} cores (initially)')
                    print(f'    {load}% maximum load ({int(ncpu_system * float(load)/100)} cores)')
                    print(f'    no swapping')
            try:
                subprocess.run(command, shell=True, check=True)
            except Exception as e:
                print(e)

if __name__ == "__main__":
    if is_running_in_docker():
        sys.argv = [arg.lower() if arg.startswith('--') else arg for arg in sys.argv] # make command line flags case insensitive
        parser = argparse.ArgumentParser(description='Launch Quicklook pipeline for selected observations.')
        parser.add_argument('start_date', type=str, help='Start date as YYYYMMDD, YYYYMMDD.SSSSS, or YYYYMMDD.SSSSS.SS')
        parser.add_argument('end_date',  type=str, help='End date as YYYYMMDD, YYYYMMDD.SSSSS, or YYYYMMDD.SSSSS.SS')
        parser.add_argument('--ncpu', type=str, default=10, help='Number of cores for parallel processing')
        parser.add_argument('--load', type=str, default=0, help='Maximum load (percent)')
        parser.add_argument('--l0', action='store_true', help='Select all L0 files in date range')
        parser.add_argument('--2d', action='store_true', dest='d2', help='Select all 2D files in date range')
        parser.add_argument('--l1', action='store_true', help='Select all L1 files in date range')
        parser.add_argument('--l2', action='store_true', help='Select all L2 files in date range')
        parser.add_argument('--master', action='store_true', help='Select all master files in date range')
        parser.add_argument('--print_files', action='store_true', help="Display file names matching criteria, but don't generate Quicklook plots")
    
        args = parser.parse_args()
        main(args.start_date, args.end_date, args.l0, args.d2, args.l1, args.l2, args.master, args.ncpu, args.load, args.print_files)
    else:
        print('qlp_parallel.py needs to be run in a Docker environment.')        
        print('Start the KPF-Pipeline instance of Docker before trying again.')        
