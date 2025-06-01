#!/usr/bin/env python3

import argparse
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

def main(start_date, end_date, db_path, force):
    """
    Script Name: ingest_dates_kpf_tsdb.py
   
    Description:
      This script is used to ingest KPF observations over a date range into a 
      KPF Time Series Database.

    Options:
      --help        Display help message
      --start_date  Start date in YYYYMMDD format
      --end_date    End date in YYYYMMDD format
      --db_path     Path to database file; default = /data/time_series/kpf_ts.db
      --force       If set, forces re-ingest of existing files
   
    Usage:
      ./ingest_dates_kpf_tsdb.py YYYYMMDD YYYYMMDD --force
   
    Example:
      ./ingest_dates_kpf_tsdb.py 20231201 20240101 --force
    """

    myTS = AnalyzeTimeSeries(db_path=db_path)
    myTS.db.print_db_status()
    myTS.db.ingest_dates_to_db(start_date, end_date, reverse=True, force=force)
    myTS.db.print_db_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest KPF files over a date range into an observational database.')
    parser.add_argument('start_date', type=str, help='Start date in YYYYMMDD format, e.g. 20231201')
    parser.add_argument('end_date', type=str, help='End date in YYYYMMDD format, e.g. 20240101')
    parser.add_argument('--force', action='store_true', help='If set, files are ingested even if old')
    parser.add_argument('--db_path', type=str, default='/data/time_series/kpf_ts.db', 
                        help='path to database file; default = /data/time_series/kpf_ts.db')

    args = parser.parse_args()
    main(args.start_date, args.end_date, args.db_path, args.force)
