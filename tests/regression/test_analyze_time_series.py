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

from modules.Utils.kpf_parse import *
from modules.Utils.utils import DummyLogger
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries
from modules.quicklook.src.alg import QuicklookAlg
from modules.quicklook.src.quick_prim import Quicklook
from database.modules.utils.tsdb import convert_to_list_if_array

# Generate a unique DB filename and plot directory
characters = string.ascii_letters + string.digits
random_char = ''.join(random.choice(characters) for _ in range(12))
temp_db_path  = 'temp_kpf_ts_'       + random_char + '.db'
temp_plot_dir = 'temp_kpf_ts_plots_' + random_char + '/'
os.mkdir(temp_plot_dir)
logger = DummyLogger()

# Define path
# Files in base_dir can be copied over from /data/{L0/2D/L1/L2} using 
# the script in scripts/copy_fits_ci_tsdb.py
# It should be run when new fits header keywords are added or there are changes 
# to the data model so that current files can be tested.  
# This script should be run from within a Docker container.
base_dir = '/data/reference_fits/tsdb_data/L0'

# This is for get_kpf_data()
data_dir = '/data/reference_fits/tsdb_data'

# Reference FITS files listed in CSV file 
ObsID_filename = '/code/KPF-Pipeline/tests/regression/test_analyze_time_series_ObsIDs.csv'

def test_analyze_time_series():
    
    # Generate a Time Series Database
    myTS = AnalyzeTimeSeries(db_path=temp_db_path, base_dir=base_dir, backend='sqlite')
    
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
    start_date = datetime(2025,1,12)
    end_date = datetime(2025,1,13)
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
    start_date = datetime(2025,1,12)
    myTS.plot_all_quicklook(start_date=start_date, interval='day', fig_dir=temp_plot_dir)
    myTS.plot_time_series_multipanel('junk_status', fig_path=temp_plot_dir + '/junk_status.png')
    
    # Test methods in kpf_parse.py that depend on having access to files.
    df = myTS.db.dataframe_from_db(columns=['ObsID', 'Object', 'Source', 'DATE-MID'], start_date=start_date, end_date=end_date)  
    spectrum_types = ['Bias', 'Dark', 'Flat', 'LFC', 'Etalon', 'ThAr', 'UNe', 'Sun', 'Star']
    QL = QuicklookAlg()
    for stype in spectrum_types:
        mask = df['Source'] == stype
        if mask.any():  # Checks if at least one row matches
            first_obsid = df.loc[mask, 'ObsID'].iloc[0]
            print(f"First ObsID with Source = {stype}:", first_obsid)
            data_levels_exp = get_data_levels_expected(stype)
            if 'L0' in data_levels_exp:
                L0_fn = get_kpf_data(first_obsid, 'L0', data_dir=data_dir, return_kpf_object=False)
                L0    = get_kpf_data(first_obsid, 'L0', data_dir=data_dir, return_kpf_object=True)
                data_products_L0  = get_data_products_L0(L0)
                data_products_arr = get_data_products_expected(L0, 'L0')
                lev = get_kpf_level(L0)
                primary_header = HeaderParse(L0, 'PRIMARY', logger=logger)
                name = primary_header.get_name()
                if (stype == 'Star'):
                   QL.qlp_L0(L0, temp_plot_dir)
            if '2D' in data_levels_exp:
                D2_fn = get_kpf_data(first_obsid, '2D', data_dir=data_dir, return_kpf_object=False)
                D2    = get_kpf_data(first_obsid, '2D', data_dir=data_dir, return_kpf_object=True)
                data_products_2D  = get_data_products_2D(D2)
                data_products_arr = get_data_products_expected(D2, '2D')
                lev = get_kpf_level(D2)
                primary_header = HeaderParse(D2, 'PRIMARY', logger=logger)
                name = primary_header.get_name()
                last_time = get_latest_receipt_time(D2)
                #test_output = hasattr_with_wildcard(D2, '*WAVE*')
                if (stype == 'Star'):# or (stype == 'Dark') or (stype == 'Bias'):
                    QL.qlp_2D(D2, temp_plot_dir)
            if 'L1' in data_levels_exp:
                L1_fn = get_kpf_data(first_obsid, 'L1', data_dir=data_dir, return_kpf_object=False)
                L1    = get_kpf_data(first_obsid, 'L1', data_dir=data_dir, return_kpf_object=True)
                data_products_L1  = get_data_products_L1(L1)
                data_products_arr = get_data_products_expected(L1, 'L1')
                lev = get_kpf_level(L1)
                primary_header = HeaderParse(L1, 'PRIMARY', logger=logger)
                name = primary_header.get_name()
                if stype == 'Star':
                    QL.qlp_L1(L1, temp_plot_dir)
            if 'L2' in data_levels_exp:
                L2_fn = get_kpf_data(first_obsid, 'L2', data_dir=data_dir, return_kpf_object=False)
                L2    = get_kpf_data(first_obsid, 'L2', data_dir=data_dir, return_kpf_object=True)
                data_products_L2  = get_data_products_L2(L2)
                data_products_arr = get_data_products_expected(L2, 'L2')
                lev = get_kpf_level(L2)
                primary_header = HeaderParse(L2, 'PRIMARY', logger=logger)
                name = primary_header.get_name()
                if stype == 'Star':
                    QL.qlp_L2(L2, temp_plot_dir)
        else:
            print(f"No rows with Source = {stype}")    
    
    # Test miscellaneous methods
    columns = ['ObsID','GDRXRMS','FIUMODE']
    myTS.db.display_data(columns=columns)
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
    