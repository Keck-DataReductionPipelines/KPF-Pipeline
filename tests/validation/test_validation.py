"""Scientific validation/verification tests go here."""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
load_dotenv()

# Tests below are dummy tests to test the test framework. We will need to construct a test and 
# corresponding reference data to compare.
# importing test files
test_bias = os.getenv('KPFPIPE_TEST_DATA')+ '/NEIDdata/BIAS/neidTemp_Bias20190325.fits'
test_raw = os.getenv('KPFPIPE_TEST_DATA') + '/NEIDdata/TAUCETI_20191217/L0/neidTemp_2D20191217T023129.fits' 

# defining bias subtraction function
# gets data from fits files, subtracts bias array values from raw array values
def test_bias_subtraction(test_raw=os.getenv('KPFPIPE_TEST_DATA') + '/NEIDdata/TAUCETI_20191217/L0/neidTemp_2D20191217T023129.fits', 
                          test_bias=os.getenv('KPFPIPE_TEST_DATA')+ '/NEIDdata/BIAS/neidTemp_Bias20190325.fits'):
    
    biasdata = fits.getdata(test_bias, ext=0)
    rawdata = fits.getdata(test_raw, ext=0)
    #add check to see if both matrices have the same dimensions, Cindy's recommendation
    if biasdata.shape==rawdata.shape:
        print (".Fits Dimensions Equal, Check Passed")
    if biasdata.shape!=rawdata.shape:
        print (".Fits Dimensions NOT Equal! Check Failed")
    raw_minus_bias = rawdata - biasdata

    return raw_minus_bias
