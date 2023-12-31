#!/usr/bin/env python3

import sys
from  modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

def main():
    """
    Script Name: ingest_kpf_tsdb.py
   
    Description:
      This script is used to ingest KPF observations into a KPF Time Series Database.

    Options:
      --help            Display this message
   
    Usage:
      python kpf_processing_progress.py YYYYMMDD [YYYYMMDD] [--print_files] [--print_files_2D] [--print_files_L1] [--print_files_L2] [--touch_missing] [--check_version]
   
    Example:
      python kpf_processing_progress.sh 20231114 20231231 --print_files
    """

    if len(sys.argv) > 3:
        print("Usage: python add_to_tsdb.py start_date end_date db_path")
        sys.exit(1)

    start_date = sys.argv[1] # e.g., 20230501
    end_date   = sys.argv[2] # e.g., 20230601
    db_path    = sys.argv[3] # e.g., kpfdb.db

    myTS = AnalyzeTimeSeries()
    myTS.add_dates_to_db(start_date, end_date)
    myTS.print_db_status()

if __name__ == "__main__":
    main()
