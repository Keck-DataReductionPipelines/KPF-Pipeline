#!/usr/bin/env python3

import argparse
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

def main(start_date, end_date, db_path):
    """
    Script Name: ingest_kpf_tsdb.py
   
    Description:
      This script is used to ingest KPF observations over a date range into a 
      KPF Time Series Database.

    Options:
      --help        Display help message
      --start_date  Start date in YYYYMMDD format
      --end_date    End date in YYYYMMDD format
   
    Usage:
      ./ingest_dates_kpf_tsdb.py YYYYMMDD YYYYMMDD dbname.db
   
    Example:
      ./ingest_dates_kpf_tsdb.py 20231201 20240101 kpfdb.db
    """

    myTS = AnalyzeTimeSeries(db_path=db_path)
    myTS.print_db_status()
    myTS.ingest_dates_to_db(start_date, end_date)
    myTS.print_db_status()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest KPF files over a date range into an observational database.')
    parser.add_argument('start_date', type=str, help='Start date in YYYYMMDD format, e.g. 20231201')
    parser.add_argument('end_date', type=str, help='End date in YYYYMMDD format, e.g. 20240101')
    parser.add_argument('db_path', type=str, help='path to database file, e.g. kpfdb.db')

    args = parser.parse_args()
    main(args.start_date, args.end_date, args.db_path)
