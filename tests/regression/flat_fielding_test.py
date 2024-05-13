import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
load_dotenv()

#importing test files
test_flat = os.getenv('KPFPIPE_TEST_DATA')+ '/NEIDdata/FLAT/neidTemp_2D20191214T001924.fits'
test_raw= os.getenv('KPFPIPE_TEST_DATA') + '/NEIDdata/TAUCETI_20191217/L0/neidTemp_2D20191217T023129.fits' 

#defining flat division function
def flat_div(rawimg, flatimg):
    flatdata = fits.getdata(flatimg, ext=0)
    rawdata = fits.getdata(rawimg, ext=0)
    if flatdata.shape==rawdata.shape:
        print (".Fits Dimensions Equal, Check Passed")
    if flatdata.shape!=rawdata.shape:
        print (".Fits Dimensions NOT Equal! Check Failed")
    raw_div_flat=rawdata/flatdata
    return raw_div_flat

#function plot
field_result=flat_div(test_raw,test_flat)
plt.figure()
plt.title("Flat Division Function Result")
plt.imshow(field_result)
plt.colorbar()

#non-function
flatdata = fits.getdata(test_flat, ext=0)
rawdata = fits.getdata(test_raw, ext=0)
res2=rawdata/flatdata
plt.figure()
plt.title("Flat Division Non-Fxn Result")
plt.imshow(res2)
plt.colorbar()

#original files
plt.figure()
plt.title("Raw Data")
plt.imshow(rawdata)
plt.colorbar()

plt.figure()
plt.title("Flat Data")
plt.imshow(flatdata)
plt.colorbar()

#difference
plt.figure()
plt.title("Difference Between Fxn and Non-Fxn Result")
plt.imshow(field_result-res2)
plt.colorbar()

np.where((field_result-res2)!=0.0)