#!/usr/bin/env python
# coding: utf-8

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
load_dotenv()

#importing test files
test_bias = os.getenv('KPFPIPE_TEST_DATA')+ '/NEIDdata/BIAS/neidTemp_Bias20190325.fits'
test_raw= os.getenv('KPFPIPE_TEST_DATA') + '/NEIDdata/TAUCETI_20191217/L0/neidTemp_2D20191217T023129.fits' 

#defining bias subtraction function
#gets data from fits files, subtracts bias array values from raw array values
def bias_subtraction(rawimage, masterbias):
    
    biasdata = fits.getdata(masterbias, ext=0)
    rawdata = fits.getdata(rawimage, ext=0)
    #add check to see if both matrices have the same dimensions, Cindy's recommendation
    if biasdata.shape==rawdata.shape:
        print (".Fits Dimensions Equal, Check Passed")
    if biasdata.shape!=rawdata.shape:
        print (".Fits Dimensions NOT Equal! Check Failed")
    raw_minus_bias=rawdata-biasdata
    return raw_minus_bias

#testing process outside of function 
#start with getting bias data
biasdata = fits.getdata(test_bias, ext=0)


#plotting bias data
plt.figure()
plt.title("Bias Data")
plt.imshow(biasdata)
plt.colorbar()

#getting raw data
rawdata = fits.getdata(test_raw, ext=0)


#plotting raw data
plt.figure()
plt.title("Raw Data")
plt.imshow(rawdata)
plt.colorbar()
biasdata.shape


#testing raw minus bias outside function
rawminusbias=rawdata-biasdata

#plotting raw minus bias outside function
plt.figure()
plt.title("Raw Minus Bias, Not Through Fxn")
plt.imshow(rawminusbias)
plt.colorbar()
biasdata.shape

#getting function data
bias_function_result=bias_subtraction(test_raw,test_bias)

#plotting function result data
plt.figure()
plt.title("Raw Minus Bias, Through Fxn")
plt.imshow(bias_function_result)
plt.colorbar()


#checking to see if there is a difference between the results
difference=rawminusbias-bias_function_result


#appears as though there is no difference
plt.figure()
plt.title("Difference Between Explicit and Fxn Results")
plt.imshow(difference)
plt.colorbar()
biasdata.shape

#as expected, we get the same result
np.where(difference!=0.0)
#0 everywhere means same arrays

