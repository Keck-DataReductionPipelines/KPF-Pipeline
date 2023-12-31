#!/usr/bin/env python3

import sys
from  modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

def main():
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
