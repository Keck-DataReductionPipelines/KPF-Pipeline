#imports
from modules.wavelength_calibration.src.alg import WaveCalibrationAlg
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy
import os
import pytest
import configparser
from dotenv import load_dotenv
load_dotenv()
#import test files
filepath = 
#define functions
def f0_and_frep(filepath:str):
    file_ = fits.open(filepath)
    f0 = file_[0].header['LFCF0']
    frep = file_[0].header['LFCFR']
    return f0,frep

def get_needed_data(filepath,wavecal_type):
    test_data_dir = os.getenv('KPFPIPE_TEST_DATA') + '/'
    test_file = fits.open(filepath)
    flux = test_file['CALFLUX'].data
    #lfc: flux + thar
    if wavecal_type = 'lfc':
        thar = test_file['CALWAVE'].data
    #thar: flux, linelists, other_wls
    if wavecal_type = 'thar':
        linelist_path = #shrek
        linelist_subset_path = #shrek
        linelist = np.load(self.linelist_path)
        linelist_sub = np.load(self.linelist_subset_path,allow_pickle=True)
        redman_w = np.array(linelist['redman_w']),dtype=float)
        redman_i = np.array(linelist(['redman_i']),dtype=float)
        wls_ref = 

    #etalon:

    return flux,thar,redman_w,redman_i,linelist_sub,wls_ref

def comb_thar_size(comb:np.ndarray, thar:np.ndarray):
    assert np.shape(comb) == np.shape(thar), "Comb and ThAr sizes not equal"

def initiate_wavecal():
    wavecal = WaveCalibrationAlg()
    return wavecal

def disabled_test_lfc(filepath):
    flux,thar,_,_,_,_ = get_needed_data(filepath,'lfc')
    f0,frep = f0_and_frep(filepath)
    lfc = initiate_wavecal()
    lfc_soln = lfc.open_and_run_lfc(flux,thar,f0,frep,quicklook=False)

def disabled_test_thar(filepath):
    _,_,red_w,red_i,line_sub,wls_ref = get_needed_data(filepath, 'thar')
    thar = initiate_wavecal()
    thar_soln = thar.open_and_run_thar(flux,red_w,red_i,line_sub,wls_ref,plot_toggle = False, quicklook = False)

#def test_etalon():
