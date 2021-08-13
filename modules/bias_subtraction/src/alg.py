#packages
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
###

from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments

class BiasSubtractionAlg:
    """
    Bias subtraction calculation.

    This module defines 'BiasSubtraction' and methods to perform bias subtraction by subtracting a master bias frame from the raw data frame.  

    Args:
        rawimage (np.ndarray): The FITS raw data with image extensions
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger.
    
    Attributes:
        rawimage (np.ndarray): From parameter 'rawimage'.
    
    Raises:
        Exception: If raw image and bias frame don't have the same dimensions
    """


    def __init__(self,rawimage,ffi_exts,config=None, logger=None):
        """Inits BiasSubtraction class with raw data, config, logger.

        Args:
            rawimage (np.ndarray): The FITS raw data.
            ffi_exts (np.ndarray): The extensions in L0 FITS files where FFIs (full frame images) are stored.
            config (configparser.ConfigParser, optional): Config context. Defaults to None.
            logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.
        """
        self.rawimage=rawimage
        self.ffi_exts=ffi_exts
        self.config=config
        self.logger=logger
        #self.imagesize=
        
    def bias_subtraction(self,masterbias):
        """
            Subtracts bias data from raw data.
            In pipeline terms: inputs two L0 files, produces one L0 file. 

        Args:
            masterbias (np.ndarray): The master bias data.

        Raises:
            Exception: If raw image and bias frame don't have the same dimensions.
        """
        ###for testing purposes###
        #masterbias = fits.open(masterbias)
        #masterbias = masterbias[1].data
        # masterbias = np.zeros_like(frame)
        ###
        for no,ffi in enumerate(self.ffi_exts):
            if self.rawimage[ffi].data.shape==masterbias[no+1].data.shape:
                print ("Bias .fits Dimensions Equal, Check Passed")
            else:
                raise Exception ("Bias .fits Dimensions NOT Equal! Check failed")

            self.rawimage[ffi].data=self.rawimage[ffi].data-masterbias[no+1].data
            #ext no+1 for mflat because there is a primary ext coded into the masterflat currently

    def get(self):
        """Returns bias-corrected raw image result.

        Returns:
            self.rawimage: The bias-corrected data.
        """
        return self.rawimage

    def quicklook(self,masterbias,imagesize=(10,10)):
        """Runs quicklook-version of algorithm for quicklook pipeline.

        Args:
            masterbias (np.ndarray): The FITS master bias data
        """
        #plot of images (master bias and raw image)

        plt.figure(figsize=imagesize)
        plt.title('Master Bias Frame')
        plt.imshow(masterbias.data,aspect='auto')

        for no,ffi in enumerate(self.ffi_exts):
            plt.figure(figsize=imagesize)#size/layout up to config ideally
            plt.title('Raw Frame')
            plt.imshow(self.rawimage[ffi].data,aspect='auto')
            
            plt.figure(figsize=imagesize)
            plt.title('Bias-Corrected L0 Image')
            plt.imshow(self.rawimage[ffi].data-masterbias[no+1],aspect='auto')
        #mean of counts (Std) in reference range


        #raise flag when counts are significantly diff from master bias
        

        #identify bad pixels
