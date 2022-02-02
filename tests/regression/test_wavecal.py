#imports
from modules.wavelength_cal.src.alg import WaveCalibration
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy
import os
from scipy import signal
from scipy.signal import find_peaks as peak
from scipy.optimize import curve_fit as cv
from scipy.special import erf
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize.minpack import curve_fit
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre
import configparser
from dotenv import load_dotenv
load_dotenv()
#import test files
test_data_dir = os.getenv('KPFPIPE_TEST_DATA') + '/'
data_type = 'NEID'
cal_orderlette_name = 'CALFLUX'

#define functions
def lfc_run():
    cal_type = 'LFC'
    quicklook = False
    l1_obj = fits.open('/data/NEIDdata/TAUCETI/L1/neidL1_20210719T232216.fits')
    calflux = l1_obj[cal_orderlette_name].data
    #peak_wavelengths_ang = 
    lfc_init = WaveCalibration(cal_type,quicklook)
    master_wls_file = fits.open('/data/NEIDdata/TAUCETI/L2/neidL2_20210714T063111.fits')
    rough_wls = master_wls_file['SCIWAVE'].data
    f0_key = 'LFCF0'
    frep_key = 'LFCFR'
    comb_f0 = float(l1_obj.header['PRIMARY'][f0_key])
    comb_frep = float(l1_obj.header['PRIMARY'][frep_key])
    lfc_allowed_wls = lfc_init.comb_gen(comb_f0,comb_frep)
    wl_soln, wls_and_pixels = lfc_init.run_wavelength_cal(
        calflux, rough_wls=rough_wls, lfc_allowed_wls=lfc_allowed_wls)
    
def thar_run():
    cal_type = 'ThAr'
    master_wls_file = fits.open('/data/KPF-Pipeline-TestData/DRP_V2_Testing/NEID-cals/neidMaster_HR_Wavelength20210218_v003.fits')
    quicklook = False
    l1_obj = fits.open('/data/KPF-Pipeline-TestData/DRP_V2_Testing/NEID-cals/neidL1_20220126T235959')
    calflux = l1_obj[cal_orderlette_name].data
    linelist_path = '/data/KPF-Pipeline-TestData/NEIDdata/neidMaster_ThArLines20210218_v001.npy'
    peak_wavelengths_ang = np.load(linelist_path, allow_pickle=True).tolist()
    th_init = WaveCalibration(cal_type,quicklook)
    rough_wls = master_wls_file['CALWAVE'].data
    wl_soln, wls_and_pixels = th_init.run_wavelength_cal(calflux,peak_wavelengths_ang=peak_wavelengths_ang, 
        rough_wls=rough_wls)
    
# def etalon_run():
#     cal_type = 'Etalon'
#     quicklook = False
#     #peak_wavelengths_ang = 
#     et_init = WaveCalibration(cal_type,quicklook)
#     wl_soln, wls_and_pixels = et_init.run_wavelength_cal()
    
