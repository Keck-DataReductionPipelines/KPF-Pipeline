#imports
from modules.wavelength_cal.src.alg import LFCWaveCalibration
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

#define functions
def f0_and_frep(filepath:str):
    file_ = fits.open(filepath)
    f0 = file_[0].header['LFCF0']
    frep = file_[0].header['LFCFR']
    return f0,frep

def comb_thar_size(comb:np.ndarray, thar:np.ndarray):
    assert np.shape(comb) == np.shape(thar), "Comb and ThAr sizes not equal"

def start_alg():
    config_vals = configparser.ConfigParser()
    config_vals['PARAM'] = {
        'min_wave': 3800,
        'max_wave': 9300,
        'fit_order': 6,
        'min_order':60,
        'max_order':90,
        'n_sections': 20,
        'skip_orders': [84,85,86]
    }
    test_dir = os.getenv('KPFPIPE_TEST_DATA') + '/'
    test_file = test_dir+'NEIDdata/TAUCETI_20191217/L1/neidL1_20191217T023129.fits'
    assert os.path.isfile(test_file), "Test file doesn't exist"
    test_thar = fits.getdata(test_file,ext=6)
    assert test_thar, 'ThAr data not extracted'
    test_comb = fits.getdata(test_file,ext=4)
    assert test_comb, 'Comb data not extracted'
    lfc_start = LFCWaveCalibration(config=config_vals)
    return test_comb,test_thar,lfc_start

def test_run_alg(filepath:str):
    config_vals = configparser.ConfigParser()
    config_vals['PARAM'] = {
        'min_wave': 3800,
        'max_wave': 9300,
        'fit_order': 6,
        'min_order':60,
        'max_order':90,
        'n_sections': 20,
        'skip_orders': [84,85,86]
    }
    combs,thars,algg = start_alg()
    f0,frep = f0_and_frep(filepath)
    #make order list
    orders = algg.remove_orders()
    min_order = config_vals['PARAM']['min_order']
    max_order = config_vals['PARAM']['max_order']
    orderlist = np.arange(min_order,max_order)
    skip_orders = config_vals['PARAM']['skip_orders']
    n_olist = len(orderlist)
    for i in skip_orders:
        if i in orderlist:
            n_olist -= 1
    assert n_olist==len(orders),"Orders improperly removed"

    cl_ang = algg.comb_gen(f0,frep)
    poly_soln = algg.fit_many_orders(combs,thars,cl_ang,orders)
    
def test_rv_acc(filepath:str):
    config_vals = configparser.ConfigParser()
    config_vals['PARAM'] = {
        'min_wave': 3800,
        'max_wave': 9300,
        'fit_order': 6,
        'min_order':60,
        'max_order':90,
        'n_sections': 20,
        'skip_orders': [84,85,86]
    }
    combs,thars,algg = start_alg()
    f0,frep = f0_and_frep(filepath)
    #make order list
    orders = algg.remove_orders()
    cl_ang = algg.comb_gen(f0,frep)
    for order in orders: 
        new_peaks, peaks, peak_heights, gauss_coeffs = algg.find_peaks_in_order(combs[order])
        assert new_peaks.shape != 0, "Find_peaks function has failed"
        good_peak_idx = algg.clip_peaks(combs[order],new_peaks,peaks,gauss_coeffs,peak_heights,thars[order],cl_ang)
        wls, mode_nums = algg.mode_match(combs[order],new_peaks,good_peak_idx,thars[order],cl_ang)
        wl_soln, leg_out = algg.fit_polynomial(wls, gauss_coeffs,good_peak_idx,len(combs[order]),new_peaks)
        precision_cm_s = algg.calculate_rv_precision(new_peaks,good_peak_idx,wls,leg_out)
        assert precision_cm_s < 100, "RV error is greater than 1 m/s"

def test_alg():
    test_dir = os.getenv('KPFPIPE_TEST_DATA') + '/'
    test_file = test_dir+'NEIDdata/TAUCETI_20191217/L1/neidL1_20191217T023129.fits'
    test_run_alg(test_file)

def test_rv():
    test_dir = os.getenv('KPFPIPE_TEST_DATA') + '/'
    test_file = test_dir+'NEIDdata/TAUCETI_20191217/L1/neidL1_20191217T023129.fits'
    test_rv_acc(test_file)

