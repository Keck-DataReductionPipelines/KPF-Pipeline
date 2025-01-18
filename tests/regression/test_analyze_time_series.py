import pytest
import warnings
import os
import sys
import shutil
import string
import random
import datetime
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

# Generate a unique DB filename and plot directory
characters = string.ascii_letters + string.digits
random_char = ''.join(random.choice(characters) for _ in range(12))
temp_db_path  = 'temp_kpf_ts_'       + random_char + '.db'
temp_plot_dir = 'temp_kpf_ts_plots_' + random_char
os.mkdir(temp_plot_dir)

# Define paths
#base_dir = '/data/kpf/reference_fits/tsdb_data/L0'
base_dir = '/data/L0'

# Reference FITS files listed in CSV file 
ObsID_filename = '/code/KPF-Pipeline/tests/regression/test_analyze_time_series_ObsIDs.csv'

def test_analyze_time_series():
    
    # Generate a Time Series Database
    myTS = AnalyzeTimeSeries(db_path=temp_db_path, base_dir=base_dir)
    
    # Test metadata table capabilities
    myTS.print_metadata_table()
    df = myTS.metadata_table_to_df()
    
    # Test file ingestion methods
    df_ObsIDs = pd.read_csv(ObsID_filename)
    ObsID_list = df_ObsIDs['observation_id'].tolist()
    myTS.ingest_one_observation(base_dir, ObsID_list[0] + '.fits')
    
    myTS = AnalyzeTimeSeries(db_path=temp_db_path, base_dir=base_dir, drop=True)
    start_date = datetime.datetime(2025,1,12)
    end_date = datetime.datetime(2025,1,13)
    myTS.ingest_dates_to_db(start_date, end_date)

    myTS = AnalyzeTimeSeries(db_path=temp_db_path, base_dir=base_dir, drop=True)
    myTS.add_ObsIDs_to_db(ObsID_list)
    
    myTS = AnalyzeTimeSeries(db_path=temp_db_path, base_dir=base_dir, drop=True)
    myTS.add_ObsID_list_to_db(ObsID_filename)

    # Test plotting
    start_date = datetime.datetime(2025,1,12)
    
    myTS.plot_all_quicklook(start_date=start_date, interval='day', fig_dir=temp_plot_dir)
    myTS.plot_time_series_multipanel('junk_status', fig_path=temp_plot_dir + '/temp.png')
    
    # Test miscellaneous methods
    columns = ['ObsID','GDRXRMS','FIUMODE']
    myTS.display_dataframe_from_db(columns)
    df = myTS.dataframe_from_db(columns=columns)
    myTS.ObsIDlist_from_db('autocal-bias')
    myTS.drop_table()
    
    # Remove the temporary database file and plot directory
    os.remove(temp_db_path)
    shutil.rmtree(temp_plot_dir)

if __name__ == '__main__':
    test_analyze_time_series()