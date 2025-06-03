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
from database.modules.utils.tsdb import convert_to_list_if_array

# Generate a unique DB filename and plot directory
characters = string.ascii_letters + string.digits
random_char = ''.join(random.choice(characters) for _ in range(12))
temp_db_path  = 'temp_kpf_ts_'       + random_char + '.db'
temp_plot_dir = 'temp_kpf_ts_plots_' + random_char
os.mkdir(temp_plot_dir)

# Define path
# Files in base_dir can be copied over from /data/{L0/2D/L1/L2} using 
# the script in scripts/copy_fits_ci_tsdb.py
# It should be run when new fits header keywords are added or there are changes 
# to the data model so that current files can be tested.  
# This script should be run from within a Docker container.
base_dir = '/data/reference_fits/tsdb_data/L0'

# Reference FITS files listed in CSV file 
ObsID_filename = '/code/KPF-Pipeline/tests/regression/test_analyze_time_series_ObsIDs.csv'

def test_analyze_time_series():
    
    # Generate a Time Series Database
    myTS = AnalyzeTimeSeries(db_path=temp_db_path, base_dir=base_dir)
    
    # Test metadata table capabilities
    myTS.db.print_metadata_table()
    df = myTS.db.metadata_table_to_df()
    myTS.db.print_db_status()
    
    # Test file ingestion methods
    df_ObsIDs = pd.read_csv(ObsID_filename)
    ObsID_list = df_ObsIDs['observation_id'].tolist()
    myTS.db.ingest_one_observation(base_dir, ObsID_list[0] + '.fits')
    myTS.db.print_db_status()
    myTS.db.drop_tables()
    
    myTS = AnalyzeTimeSeries(db_path=temp_db_path, base_dir=base_dir)
    start_date = datetime.datetime(2025,1,12)
    end_date = datetime.datetime(2025,1,13)
    myTS.db.ingest_dates_to_db(start_date, end_date)
    myTS.db.print_db_status()
    myTS.db.drop_tables()

    myTS = AnalyzeTimeSeries(db_path=temp_db_path, base_dir=base_dir)
    myTS.db.add_ObsIDs_to_db(ObsID_list)
    myTS.db.print_db_status()
    myTS.db.drop_tables()
    
    myTS = AnalyzeTimeSeries(db_path=temp_db_path, base_dir=base_dir)
    myTS.db.add_ObsID_list_to_db(ObsID_filename)
    myTS.db.print_db_status()

    # Test plotting
    start_date = datetime.datetime(2025,1,12)
    myTS.plot_all_quicklook(start_date=start_date, interval='day', fig_dir=temp_plot_dir)
    myTS.plot_time_series_multipanel('junk_status', fig_path=temp_plot_dir + '/temp.png')
    
    # Test miscellaneous methods
    columns = ['ObsID','GDRXRMS','FIUMODE']
    myTS.db.display_dataframe_from_db(columns)
    df = myTS.db.dataframe_from_db(columns=columns)
    myTS.db.ObsIDlist_from_db('autocal-bias')
    myTS.db.print_db_status()
    myTS.db.drop_tables()
    
    # Remove the temporary database file and plot directory
    os.remove(temp_db_path)
    shutil.rmtree(temp_plot_dir)
    
def test_extra_methods():
    my_string = '["autocal-lfc-all-morn", "autocal-lfc-all-eve"]'
    my_array = ["autocal-lfc-all-morn", "autocal-lfc-all-eve"]
    out1 = convert_to_list_if_array(my_string)
    out2 = convert_to_list_if_array(my_array)
    assert type(out1) == type('abc')
    assert type(out2) == type([1,2])

if __name__ == '__main__':
    test_analyze_time_series()
    test_extra_methods()
    