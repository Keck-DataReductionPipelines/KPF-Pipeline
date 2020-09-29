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
    """
    The BiasSubtraction class performs master bias frame subtraction from a raw science frame. 
    Working on file input and export.
    
    """

    def __init__(self,data,config=None, logger=None):
        """[summary]

        Args:
            data ([type]): [description]
            config ([type], optional): [description]. Defaults to None.
            logger ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
    #use KPF0_from_fits?

    def bias_subtraction(rawimage, masterbias):
        """
        Steps:
            Reads in bias frame .fits data
            Reads in raw science .fits data
            Checks whether both data arrays have same dimensions, prints "equal" or "not equal"
            Stacks, creates average of bias frames element-wise - final average results is master bias
            Subtracts master bias array values from raw array values
            Returns array of bias-corrected science frame
        
            In pipeline terms: inputs two L0 files, outputs one L0 file
    
        Args:
            rawimage (str): The string to the raw science frame .fits file
            masterbias (str): The string to master bias .fits file - the result of combining/averaging several bias frames

        Returns:
            raw_bcorrect (array):
        """
        biasdata = fits.getdata(masterbias, ext=0)
        rawdata = fits.getdata(rawimage, ext=0)
    #check to see if both matrices have the same dimensions, Cindy's recommendation
        if biasdata.shape==rawdata.shape:
            print ("Bias .fits Dimensions Equal, Check Passed")
        if biasdata.shape!=rawdata.shape:
            print ("Bias .fits Dimensions NOT Equal! Check Failed")
        raw_bcorrect=rawdata-biasdata
    #to_fits ?
        return raw_bcorrect
