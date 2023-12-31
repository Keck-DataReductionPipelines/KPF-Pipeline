import sys
from your_module import modules.quicklook.src.AnalyzeTimeSeries

def main():
    if len(sys.argv) > 2:
        print("Usage: python add_to_tsdb.py start_date end_date")
        sys.exit(1)

    start_date = sys.argv[1]
    end_date   = sys.argv[2]

    myTS = AnalyzeTimeSeries()
    myTS.add_dates_to_db(start_date, end_date)
    myTS.print_db_status()

if __name__ == "__main__":
    main()
