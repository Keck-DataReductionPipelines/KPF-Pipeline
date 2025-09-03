import pytest
import warnings
import os
import sys
import shutil
from dotenv import load_dotenv

from modules.wavelength_cal.src.alg import WaveInterpolation
from kpfpipe.models.level1 import KPF1

# not the best test files but they are available
wls1_file = "/data/reference_fits/kpf_20230503_master_WLS_cal-LFC-eve_L1.fits"
wls2_file = "/data/reference_fits/kpf_20230531_master_WLS_autocal-thar-all-eve_L1.fits"
l1_timestamp = "2023-05-18T08:00:00.0000"
wls_extensions = ['GREEN_SCI_WAVE1', 'GREEN_SCI_WAVE2']

def test_wavelength_interpolation():
    wls1 = KPF1.from_fits(wls1_file)
    wls2 = KPF1.from_fits(wls2_file)
    wls1_ts = wls1.header['PRIMARY']['DATE-BEG']
    wls2_ts = wls2.header['PRIMARY']['DATE-BEG']
    wls_timestamps = [wls1_ts, wls2_ts]
    
    wls1_arrays = {}
    wls2_arrays = {}
    for ext in wls_extensions:
        wls1_arrays[ext] = wls1[ext]
        wls2_arrays[ext] = wls2[ext]

    wi = WaveInterpolation(l1_timestamp, wls_timestamps, wls1_arrays, wls2_arrays)

    wi.wave_interpolation()

if __name__ == '__main__':
    test_wavelength_interpolation()