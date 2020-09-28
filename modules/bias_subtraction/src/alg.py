#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:45:51 2020

@author: paminabby
"""

#packages
from astropy.io import fits

#Subtracting 2D array, function to subtract master bias frame from raw data image

class BiasSubtraction:
    """Args:
    Attributes:
    Raises: """

    def __init__(self,data,config=None, logger=None):
        """constructor"""
         
    def bias_subtraction(self,rawimage, masterbias):
"""steps
    For Bias Subtraction:
        1. Reads in bias frame .fits data
        2. Reads in raw .fits data
        3. Checks whether both data arrays have same dimensions, prints "equal" or "not equal"
        4. Subtracts bias array values from raw array values
        5. Returns array of raw minus bias
        
        In pipeline terms: inputs two L0 files, outputs one L0 file"""

        biasdata = fits.getdata(masterbias, ext=0)
        rawdata = fits.getdata(rawimage, ext=0)
    #check to see if both matrices have the same dimensions, Cindy's recommendation
        if biasdata.shape==rawdata.shape:
            print ("Bias .fits Dimensions Equal, Check Passed")
        if biasdata.shape!=rawdata.shape:
            print ("Bias .fits Dimensions NOT Equal! Check Failed")
        raw_minus_bias=rawdata-biasdata
        return raw_minus_bias

        """ "add save of raw minus bias file as .fits """
