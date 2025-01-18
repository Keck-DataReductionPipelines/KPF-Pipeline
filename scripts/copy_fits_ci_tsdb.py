#!/usr/bin/env python3

# This script is used to copy L0/2D/L1/L2 .fits files from their usual 
# directories to a directory used for continuous integration testing of the 
# Time Series Database functions.  It should be run when new fits header keywords
# are added or there are changes to the data model so that current files can be
# tested.  This script should be run from within a Docker container.

import sys
import os
import shutil
import pandas as pd
from modules.Utils.kpf_parse import get_datecode

def main():
    
    ObsID_filename = '/code/KPF-Pipeline/tests/regression/test_analyze_time_series_ObsIDs.csv'
    df = pd.read_csv(ObsID_filename)
    
    # Directories
    dir1 = "/data"
    dir2 = "/code/KPF-Pipeline/kpf/reference_fits/tsdb_data"

 
    for datatype in ['L0', '2D', 'L1', 'L2']:
        for ObsID in df["observation_id"]:
            if datatype == 'L0':
                ending = ''
            else:
                ending = f'_{datatype}'
            path1 = dir1 + f"/{datatype}/" + get_datecode(ObsID) + f"/{ObsID}{ending}.fits"
            path2 = dir2 + f"/{datatype}/" + get_datecode(ObsID) + f"/{ObsID}{ending}.fits"
            
            # Ensure the destination subdirectory exists
            os.makedirs(os.path.dirname(path2), exist_ok=True)
            
            # Copy the file
            try:
                shutil.copy2(path1, path2)
                print(f"Copied: {path1} -> {path2}")
            except FileNotFoundError:
                print(f"ERROR: Source file not found: {path1}")
            except Exception as e:
                print(f"ERROR: Could not copy {path1} -> {path2}. Reason: {e}")

if __name__ == "__main__":
    main()
