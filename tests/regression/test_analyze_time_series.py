import pytest
import warnings
import os
import sys
import shutil
import string
import random
from dotenv import load_dotenv

from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

## not the best test files but they are available
#wls1_file = "/data/reference_fits/kpf_20230503_master_WLS_cal-LFC-eve_L1.fits"
#wls2_file = "/data/reference_fits/kpf_20230531_master_WLS_autocal-thar-all-eve_L1.fits"
#l1_timestamp = "2023-05-18T08:00:00.0000"
#wls_extensions = ['GREEN_SCI_WAVE1', 'GREEN_SCI_WAVE2']

# Generate a unique DB filename
characters = string.ascii_letters + string.digits
db_path = 'kpfts_temp_' + ''.join(random.choice(characters) for _ in range(12)) + '.db'
data_dir = '/data/kpf/reference_fits/tsdb_data/'

def test_analyze_time_series():
    
    myTS = AnalyzeTimeSeries(db_path=db_path)
    myTS.print_metadata_table()
    
    os.remove(db_path)

    #wls1 = KPF1.from_fits(wls1_file)
    #wls2 = KPF1.from_fits(wls2_file)
    #wls1_ts = wls1.header['PRIMARY']['DATE-BEG']
    #wls2_ts = wls2.header['PRIMARY']['DATE-BEG']
    #wls_timestamps = [wls1_ts, wls2_ts]
    #
    #wls1_arrays = {}
    #wls2_arrays = {}
    #for ext in wls_extensions:
    #    wls1_arrays[ext] = wls1[ext]
    #    wls2_arrays[ext] = wls2[ext]
#
    #wi = WaveInterpolation(l1_timestamp, wls_timestamps, wls1_arrays, wls2_arrays)
#
    #wi.wave_interpolation()

if __name__ == '__main__':
    test_analyze_time_series()